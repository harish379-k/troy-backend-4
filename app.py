import os
import json
import uuid
import base64
import hashlib
import random
from io import BytesIO
from pathlib import Path
from collections import OrderedDict

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

import google.generativeai as genai
from groq import Groq


# =========================================================
# App setup
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)

# Prevent huge images from killing Render memory
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 MB

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
CORS(app, origins=CORS_ORIGINS.split(","))

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Keep low for Render free tier
MAX_BASE64_IMAGE_SIZE = 1_800_000
Image.MAX_IMAGE_PIXELS = 15_000_000

analysis_cache = OrderedDict()
MAX_CACHE_ITEMS = 30

sessions = {}


# =========================================================
# API config
# =========================================================

def get_gemini_api_key():
    return os.environ.get("GEMINI_API_KEY", "").strip()


def get_gemini_model():
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()


def get_groq_api_key():
    return os.environ.get("GROQ_API_KEY", "").strip()


def get_groq_vision_model():
    return os.environ.get(
        "GROQ_VISION_MODEL",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ).strip()


def get_groq_text_model():
    return os.environ.get(
        "GROQ_TEXT_MODEL",
        "llama-3.3-70b-versatile"
    ).strip()


def build_gemini_model():
    api_key = get_gemini_api_key()

    if not api_key:
        return None

    genai.configure(api_key=api_key)

    return genai.GenerativeModel(
        get_gemini_model(),
        generation_config={
            "temperature": 0.75,
            "top_p": 0.95,
            "max_output_tokens": 1500
        }
    )


def build_groq_client():
    api_key = get_groq_api_key()

    if not api_key:
        return None

    return Groq(api_key=api_key)


print("Gemini key:", "FOUND" if get_gemini_api_key() else "NOT FOUND")
print("Gemini model:", get_gemini_model())
print("Groq key:", "FOUND" if get_groq_api_key() else "NOT FOUND")
print("Groq vision model:", get_groq_vision_model())


# =========================================================
# Error handlers
# =========================================================

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):
    return jsonify({
        "error": "Image is too large. Please upload an image below 4 MB."
    }), 413


# =========================================================
# Basic helpers
# =========================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def clean_text(value, fallback=""):
    text = str(value or "").replace("\n", " ").strip()
    text = " ".join(text.split())
    return text or fallback


def ensure_list(value, fallback=None, limit=3):
    fallback = fallback or []
    result = []
    seen = set()

    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str) and value.strip():
        raw_items = [value]
    else:
        raw_items = []

    for item in raw_items:
        text = clean_text(item)
        key = text.lower()

        if text and key not in seen:
            seen.add(key)
            result.append(text)

    if not result:
        result = fallback

    return result[:limit]


def safe_get_dict(data, key):
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def extract_json_block(text):
    if not text:
        return ""

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    first = text.find("{")
    last = text.rfind("}")

    if first != -1 and last != -1 and last > first:
        return text[first:last + 1]

    return text


def parse_json_response(text):
    cleaned = extract_json_block(text)
    return json.loads(cleaned)


def is_rate_limit_error(error_text):
    lower = error_text.lower()
    return (
        "429" in lower
        or "rate limit" in lower
        or "resource_exhausted" in lower
        or "too many requests" in lower
        or "quota" in lower
    )


def is_invalid_key_error(error_text):
    lower = error_text.lower()
    return (
        "401" in lower
        or "403" in lower
        or "invalid api key" in lower
        or "api key not valid" in lower
        or "unauthorized" in lower
        or "forbidden" in lower
    )


def is_temporary_error(error_text):
    lower = error_text.lower()
    return (
        "500" in lower
        or "502" in lower
        or "503" in lower
        or "service unavailable" in lower
        or "temporarily unavailable" in lower
    )


# =========================================================
# Image processing
# =========================================================

def encode_image_to_base64_jpeg(img, quality):
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    raw_bytes = buffer.getvalue()
    encoded = base64.b64encode(raw_bytes).decode("utf-8")
    return raw_bytes, encoded


def prepare_image_for_models(image_file):
    """
    Returns:
    1. PIL image for Gemini
    2. Base64 image data URL for Groq
    3. SHA hash for cache and feedback variation
    """

    img = Image.open(image_file.stream)
    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    # Memory-safe resize
    img.thumbnail((900, 900))

    for quality in [80, 70, 60, 50, 40]:
        raw_bytes, encoded = encode_image_to_base64_jpeg(img, quality)

        if len(encoded.encode("utf-8")) <= MAX_BASE64_IMAGE_SIZE:
            pil_img = Image.open(BytesIO(raw_bytes))
            pil_img.load()
            pil_img = pil_img.convert("RGB")

            image_hash = hashlib.sha256(raw_bytes).hexdigest()
            data_url = f"data:image/jpeg;base64,{encoded}"

            return pil_img, data_url, image_hash

    # More aggressive fallback
    img.thumbnail((700, 700))

    for quality in [60, 50, 40, 35]:
        raw_bytes, encoded = encode_image_to_base64_jpeg(img, quality)

        if len(encoded.encode("utf-8")) <= MAX_BASE64_IMAGE_SIZE:
            pil_img = Image.open(BytesIO(raw_bytes))
            pil_img.load()
            pil_img = pil_img.convert("RGB")

            image_hash = hashlib.sha256(raw_bytes).hexdigest()
            data_url = f"data:image/jpeg;base64,{encoded}"

            return pil_img, data_url, image_hash

    raise ValueError(
        "Image is too large even after compression. Please upload a smaller image."
    )


# =========================================================
# Creativity variation helpers
# =========================================================

def pick_feedback_style(image_hash):
    """
    Picks a consistent creative feedback angle for each image.
    Same image = same style because cache/feedback should stay stable.
    Different image = likely different analysis style.
    """

    styles = [
        {
            "name": "story-builder",
            "instruction": "Focus on what story or pretend-play world this build could become."
        },
        {
            "name": "designer",
            "instruction": "Focus on the child's design choices, shape choices, and arrangement."
        },
        {
            "name": "builder-engineer",
            "instruction": "Focus on balance, support, structure, levels, and how parts hold together."
        },
        {
            "name": "inventor",
            "instruction": "Focus on unusual combinations, hybrid ideas, and creative object guessing."
        },
        {
            "name": "architect",
            "instruction": "Focus on spaces, floors, openings, rooms, height, and layout."
        },
        {
            "name": "movement-maker",
            "instruction": "Focus on whether the build suggests motion, wheels, paths, vehicles, or travel."
        },
        {
            "name": "pattern-finder",
            "instruction": "Focus on repeated blocks, symmetry, spacing, rhythm, and visual patterns."
        }
    ]

    seed_number = int(image_hash[:8], 16)
    return styles[seed_number % len(styles)]


def build_unique_hint(image_hash):
    """
    Adds a small non-secret variation token so the model avoids repeating
    the exact same wording for every upload.
    """

    openings = [
        "Use fresh wording for this image.",
        "Avoid repeating common phrases from previous analyses.",
        "Make this feedback feel specific to this exact build.",
        "Describe this as if seeing the child's build for the first time.",
        "Let the visible shapes guide the guess."
    ]

    seed_number = int(image_hash[8:16], 16)
    return openings[seed_number % len(openings)]


# =========================================================
# Prompt
# =========================================================

def build_troy_prompt(age, image_hash):
    style = pick_feedback_style(image_hash)
    unique_hint = build_unique_hint(image_hash)

    return f"""
You are Troy AI Analyzer.

You are analyzing one uploaded image of a child's Troy wooden-block build.

Child age:
{age if age else "unknown"}

Feedback style for this image:
{style["name"]} — {style["instruction"]}

Uniqueness instruction:
{unique_hint}

Goal:
Give feedback like a careful, creative human teacher who is looking at this exact photo.

Important rules:
- Look at the whole image first.
- Give a creative but realistic guess about what the child may have built.
- Do not force labels like tower, house, bridge, or car.
- If it looks like a hybrid idea, describe the hybrid naturally.
  Examples:
  moving house, house-on-wheels, bridge-house, layered building,
  castle gate, pretend-play scene, animal-like vehicle, abstract machine,
  raised platform, tiny city, block vehicle with a room, walking creature,
  parking garage, lookout post, tunnel path, stage, playground structure.
- If it has floors or sections going upward, do not automatically call it a tower.
  It may be a multi-level building, layered structure, raised house, platform scene,
  parking-garage-like build, lookout station, or pretend-play setup.
- If the child combines multiple ideas, mention the combination.
  Example: "It looks like a little moving home because it has a base that feels vehicle-like and a top section that feels like a room."
- Base every sentence only on visible details.
- Mention visible parts such as base, floors, levels, gaps, supports,
  repeated blocks, stacked sections, roof-like pieces, wheel-like parts,
  curved pieces, openings, paths, bridges, rooms, platforms, loose blocks,
  or upper/lower sections if visible.
- If the image is not a Troy/block build, mark it invalid.
- If you are unsure, use cautious phrases like "looks like", "could be", or "seems to".
- Keep the tone simple, warm, parent-friendly, and encouraging.
- Do not overclaim.
- Avoid generic repeated phrases like:
  "The child practiced creativity",
  "The child learned problem solving",
  "The child used imagination",
  unless you connect them to a visible detail.
- Each learning card must mention something specific from the build.
- Make every title and description feel unique to this photo.
- Return JSON only.

Creativity level:
Be more creative in the guess, but stay grounded in visible evidence.
A good guess can be playful, such as:
"mini treehouse platform", "moving house", "block spaceship base",
"tiny garage", "bridge-home", "castle entrance", "layered lookout tower",
"animal-like machine", or "pretend city corner",
but only if the visible image supports it.

Return this exact JSON shape:

{{
  "status": "success",
  "imageStatus": "valid or invalid",
  "confidenceScore": 0,
  "buildGuess": {{
    "title": "creative but realistic build guess",
    "subtitle": "short reason based on visible image details"
  }},
  "whatWeFound": {{
    "title": "What we found",
    "summary": "2 short sentences describing the visible build with unique details"
  }},
  "whatTheyLearned": [
    {{
      "title": "specific learning skill",
      "description": "specific explanation connected to visible details in this build",
      "color": "cream"
    }},
    {{
      "title": "specific learning skill",
      "description": "specific explanation connected to visible details in this build",
      "color": "green"
    }},
    {{
      "title": "specific learning skill",
      "description": "specific explanation connected to visible details in this build",
      "color": "blue"
    }}
  ],
  "whatWeNoticed": [
    "specific visible observation 1",
    "specific visible observation 2",
    "specific visible observation 3"
  ],
  "suggestionsForParent": [
    "specific parent suggestion based on this build",
    "specific parent suggestion based on this build",
    "specific parent suggestion based on this build"
  ],
  "nextBuildIdeas": [
    "specific next build idea related to this build",
    "specific next build idea related to this build",
    "specific next build idea related to this build"
  ]
}}

Rules for invalid image:
- imageStatus must be "invalid"
- confidenceScore must be below 65
- whatTheyLearned must be []
"""


# =========================================================
# Better fallback feedback
# =========================================================

def contains_any(text, words):
    text = text.lower()
    return any(word in text for word in words)


def build_context_text(build_guess, summary, noticed):
    return " ".join([
        clean_text(build_guess.get("title", "")),
        clean_text(build_guess.get("subtitle", "")),
        clean_text(summary),
        " ".join(noticed or [])
    ]).lower()


def creative_fallback_cards(build_guess, summary, noticed, image_hash):
    """
    Used only when AI gives weak/missing learning cards.
    These cards are generated from the actual guess + observations,
    so they are less repetitive than fixed generic cards.
    """

    context = build_context_text(build_guess, summary, noticed)
    main_detail = noticed[0] if noticed else "the visible block arrangement"

    card_pool = []

    if contains_any(context, ["level", "floor", "platform", "layer", "upper", "lower"]):
        card_pool.extend([
            {
                "title": "Layered Building",
                "description": f"The child explored how one part can sit above another, especially around {main_detail}.",
                "color": "cream"
            },
            {
                "title": "Vertical Planning",
                "description": "The build shows early thinking about how lower sections can support upper sections.",
                "color": "green"
            }
        ])

    if contains_any(context, ["wheel", "vehicle", "moving", "car", "base", "travel"]):
        card_pool.extend([
            {
                "title": "Movement Thinking",
                "description": "The child connected the block shape to the idea of something that could move or travel.",
                "color": "cream"
            },
            {
                "title": "Part-to-Whole Design",
                "description": "The child explored how a base and top section can work together as one bigger idea.",
                "color": "green"
            }
        ])

    if contains_any(context, ["house", "room", "roof", "home", "door", "window"]):
        card_pool.extend([
            {
                "title": "Space Making",
                "description": "The child used blocks to suggest a small space, room, or home-like area.",
                "color": "cream"
            },
            {
                "title": "Pretend-Play Planning",
                "description": "The build can become part of a story about who lives there or what happens inside.",
                "color": "blue"
            }
        ])

    if contains_any(context, ["bridge", "gap", "span", "across", "support"]):
        card_pool.extend([
            {
                "title": "Support and Span",
                "description": "The child explored how blocks can reach across a space or rest on supports.",
                "color": "cream"
            },
            {
                "title": "Testing Stability",
                "description": "The build invites the child to test which parts stay steady and which parts need support.",
                "color": "green"
            }
        ])

    if contains_any(context, ["curve", "arch", "opening", "gate", "entrance", "tunnel"]):
        card_pool.extend([
            {
                "title": "Open-Space Design",
                "description": "The child explored how blocks can create an opening, entrance, or pass-through space.",
                "color": "cream"
            },
            {
                "title": "Shape Experimenting",
                "description": "The build shows curiosity about how different shapes can create a new form.",
                "color": "green"
            }
        ])

    if contains_any(context, ["repeat", "same", "pattern", "symmetry", "line"]):
        card_pool.extend([
            {
                "title": "Pattern Spotting",
                "description": "The child used repeated placement to make the build feel more organized.",
                "color": "cream"
            },
            {
                "title": "Visual Rhythm",
                "description": "The repeated block choices help the child notice spacing and arrangement.",
                "color": "green"
            }
        ])

    if contains_any(context, ["animal", "creature", "head", "legs", "tail", "body"]):
        card_pool.extend([
            {
                "title": "Symbolic Thinking",
                "description": "The child used simple block shapes to suggest a living thing or creature.",
                "color": "cream"
            },
            {
                "title": "Story Imagination",
                "description": "The animal-like form can become a character in the child's pretend play.",
                "color": "blue"
            }
        ])

    if contains_any(context, ["tower", "stack", "tall", "height", "vertical"]):
        card_pool.extend([
            {
                "title": "Height Control",
                "description": "The child explored how the build changes when pieces are placed higher.",
                "color": "cream"
            },
            {
                "title": "Careful Stacking",
                "description": "Placing blocks upward helps the child practice patience and hand control.",
                "color": "green"
            }
        ])

    # Always add some creative non-generic choices
    card_pool.extend([
        {
            "title": "Idea Combining",
            "description": f"The child connected visible parts like {main_detail} into one larger build idea.",
            "color": "cream"
        },
        {
            "title": "Design Choices",
            "description": "The child made choices about where blocks should go, what should be higher, and what should connect.",
            "color": "green"
        },
        {
            "title": "Story Building",
            "description": "The structure can become a small pretend world that the child can explain in their own words.",
            "color": "blue"
        },
        {
            "title": "Spatial Reasoning",
            "description": "The child practiced thinking about beside, above, under, across, and connected spaces.",
            "color": "cream"
        }
    ])

    # Pick varied cards based on image hash
    seed_number = int(image_hash[:10], 16)
    random.Random(seed_number).shuffle(card_pool)

    selected = []
    used_titles = set()

    for card in card_pool:
        title_key = card["title"].lower()
        if title_key in used_titles:
            continue

        selected.append(card)
        used_titles.add(title_key)

        if len(selected) == 3:
            break

    colors = ["cream", "green", "blue"]
    for i, card in enumerate(selected):
        card["color"] = colors[i]

    return selected


def is_weak_learning_card(card):
    title = clean_text(card.get("title")).lower()
    description = clean_text(card.get("description")).lower()

    if not title or not description:
        return True

    if len(description.split()) < 8:
        return True

    too_generic_titles = {
        "creativity",
        "imagination",
        "problem solving",
        "motor skills",
        "engineering",
        "stem learning",
        "critical thinking"
    }

    evidence_words = [
        "block", "base", "level", "floor", "support", "gap", "roof",
        "wheel", "curve", "arch", "opening", "bridge", "stack",
        "repeated", "path", "platform", "room", "shape", "piece"
    ]

    if title in too_generic_titles:
        return not any(word in description for word in evidence_words)

    repeated_phrases = [
        "used creativity",
        "improved problem solving",
        "used imagination",
        "developed motor skills"
    ]

    return any(phrase in description for phrase in repeated_phrases)


def normalize_learning_cards(cards, build_guess, summary, noticed, image_hash):
    allowed_colors = ["cream", "green", "blue"]
    cleaned = []

    if isinstance(cards, list):
        for index, card in enumerate(cards):
            if not isinstance(card, dict):
                continue

            title = clean_text(card.get("title"))
            description = clean_text(card.get("description"))
            color = clean_text(card.get("color", allowed_colors[index % 3])).lower()

            if not title or not description:
                continue

            if color not in allowed_colors:
                color = allowed_colors[index % 3]

            temp_card = {
                "title": title,
                "description": description,
                "color": color
            }

            # Skip weak generic AI cards
            if is_weak_learning_card(temp_card):
                continue

            cleaned.append(temp_card)

    fallback_cards = creative_fallback_cards(build_guess, summary, noticed, image_hash)

    existing_titles = {card["title"].lower() for card in cleaned}

    for fallback in fallback_cards:
        if len(cleaned) >= 3:
            break

        if fallback["title"].lower() not in existing_titles:
            cleaned.append(fallback)
            existing_titles.add(fallback["title"].lower())

    for i, card in enumerate(cleaned[:3]):
        card["color"] = allowed_colors[i]

    return cleaned[:3]


def normalize_analysis_response(parsed, image_hash):
    image_status = clean_text(parsed.get("imageStatus", "invalid")).lower()

    try:
        confidence = int(float(parsed.get("confidenceScore", 0)))
    except Exception:
        confidence = 0

    build_guess = safe_get_dict(parsed, "buildGuess")
    what_found = safe_get_dict(parsed, "whatWeFound")

    noticed = ensure_list(
        parsed.get("whatWeNoticed"),
        [
            "The build shows visible blocks arranged into a structure.",
            "The child used block placement to create a shape or idea.",
            "The structure has details that can be discussed with the child."
        ],
        limit=3
    )

    normalized_build_guess = {
        "title": clean_text(
            build_guess.get("title"),
            "Open-ended Troy block build"
        ),
        "subtitle": clean_text(
            build_guess.get("subtitle"),
            "The child created a visible structure using blocks."
        )
    }

    normalized_summary = clean_text(
        what_found.get("summary"),
        "The image shows a child-made block structure with visible block placement."
    )

    result = {
        "status": "success",
        "imageStatus": "valid" if image_status == "valid" and confidence >= 65 else "invalid",
        "confidenceScore": confidence,
        "buildGuess": normalized_build_guess,
        "whatWeFound": {
            "title": "What we found",
            "summary": normalized_summary
        },
        "whatTheyLearned": normalize_learning_cards(
            parsed.get("whatTheyLearned"),
            normalized_build_guess,
            normalized_summary,
            noticed,
            image_hash
        ),
        "whatWeNoticed": noticed,
        "suggestionsForParent": ensure_list(
            parsed.get("suggestionsForParent"),
            [
                "Ask your child what each part of the build represents.",
                "Invite your child to add one new detail to the build.",
                "Take another photo after your child improves or changes the structure."
            ],
            limit=3
        ),
        "nextBuildIdeas": ensure_list(
            parsed.get("nextBuildIdeas"),
            [
                "Build a version with one extra level or section.",
                "Add a path, door, bridge, or moving part.",
                "Try rebuilding the same idea using fewer blocks."
            ],
            limit=3
        ),
        "session_id": str(uuid.uuid4())
    }

    if result["imageStatus"] == "invalid":
        result["whatTheyLearned"] = []
        result["buildGuess"] = {
            "title": "We couldn’t clearly analyze this image",
            "subtitle": result["whatWeFound"]["summary"]
        }

    return result


# =========================================================
# Gemini analysis
# =========================================================

def analyze_with_gemini(pil_img, age, image_hash):
    model = build_gemini_model()

    if not model:
        raise RuntimeError("GEMINI_API_KEY not found")

    prompt = build_troy_prompt(age, image_hash)

    response = model.generate_content([prompt, pil_img])

    text = getattr(response, "text", "")

    if not text:
        raise RuntimeError("Gemini returned empty response")

    return parse_json_response(text)


# =========================================================
# Groq fallback analysis
# =========================================================

def analyze_with_groq(image_data_url, age, image_hash):
    client = build_groq_client()

    if not client:
        raise RuntimeError("GROQ_API_KEY not found")

    prompt = build_troy_prompt(age, image_hash)

    completion = client.chat.completions.create(
        model=get_groq_vision_model(),
        messages=[
            {
                "role": "system",
                "content": "You are a careful, creative visual analysis assistant. Return valid JSON only."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ],
        temperature=0.75,
        top_p=0.95,
        max_completion_tokens=1500,
        response_format={
            "type": "json_object"
        }
    )

    text = completion.choices[0].message.content

    if not text:
        raise RuntimeError("Groq returned empty response")

    return parse_json_response(text)


# =========================================================
# Main fallback logic
# =========================================================

def analyze_image_with_fallback(pil_img, image_data_url, age, image_hash):
    errors = []

    # Gemini first
    try:
        parsed = analyze_with_gemini(pil_img, age, image_hash)
        result = normalize_analysis_response(parsed, image_hash)
        result["provider"] = "gemini"
        return result

    except Exception as e:
        error_text = str(e)
        print("Gemini failed:", error_text)
        errors.append(f"Gemini: {error_text}")

    # Groq fallback
    try:
        parsed = analyze_with_groq(image_data_url, age, image_hash)
        result = normalize_analysis_response(parsed, image_hash)
        result["provider"] = "groq"
        return result

    except Exception as e:
        error_text = str(e)
        print("Groq failed:", error_text)
        errors.append(f"Groq: {error_text}")

    raise RuntimeError("Both Gemini and Groq failed. " + " | ".join(errors))


def save_cache(cache_key, result):
    analysis_cache[cache_key] = result
    analysis_cache.move_to_end(cache_key)

    if len(analysis_cache) > MAX_CACHE_ITEMS:
        analysis_cache.popitem(last=False)


# =========================================================
# Routes
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Troy AI Analyzer backend is running",
        "server": "ok"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "gemini_key_loaded": bool(get_gemini_api_key()),
        "gemini_model": get_gemini_model(),
        "groq_key_loaded": bool(get_groq_api_key()),
        "groq_vision_model": get_groq_vision_model()
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        age = clean_text(request.form.get("age", ""))

        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        image_file = request.files["image"]

        if image_file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(image_file.filename):
            return jsonify({
                "error": "Invalid file type. Please upload PNG, JPG, JPEG, or WEBP."
            }), 400

        filename = secure_filename(image_file.filename)

        try:
            pil_img, image_data_url, image_hash = prepare_image_for_models(image_file)
        except Exception as e:
            return jsonify({
                "error": "Could not process image.",
                "details": str(e)
            }), 400

        cache_key = f"{image_hash}:{age}"

        if cache_key in analysis_cache:
            cached = analysis_cache[cache_key].copy()
            cached["cached"] = True
            analysis_cache.move_to_end(cache_key)
            return jsonify(cached), 200

        result = analyze_image_with_fallback(pil_img, image_data_url, age, image_hash)
        result["cached"] = False

        if os.environ.get("SHOW_DEBUG", "false").lower() == "true":
            result["debug"] = {
                "filename": filename,
                "image_hash": image_hash[:12],
                "gemini_model": get_gemini_model(),
                "groq_model": get_groq_vision_model(),
                "feedback_style": pick_feedback_style(image_hash)["name"]
            }

        save_cache(cache_key, result)
        sessions[result["session_id"]] = result

        # Keep sessions small
        if len(sessions) > 50:
            oldest_key = next(iter(sessions))
            sessions.pop(oldest_key, None)

        return jsonify(result), 200

    except Exception as e:
        error_text = str(e)
        print("Analyze error:", error_text)

        if is_rate_limit_error(error_text):
            return jsonify({
                "error": "AI usage limit reached right now. Please wait and try again."
            }), 429

        if is_invalid_key_error(error_text):
            return jsonify({
                "error": "API key issue. Check GEMINI_API_KEY and GROQ_API_KEY in Render."
            }), 403

        if is_temporary_error(error_text):
            return jsonify({
                "error": "AI service is temporarily unavailable. Please try again in a moment."
            }), 503

        return jsonify({
            "error": "Something went wrong",
            "details": error_text
        }), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json() or {}

        question = clean_text(data.get("question", ""))
        summary = clean_text(data.get("summary", ""))

        if not question:
            return jsonify({"error": "Question is required"}), 400

        client = build_groq_client()

        if not client:
            return jsonify({
                "answer": "Groq API key is missing, so follow-up chat is unavailable."
            }), 200

        prompt = f"""
You are helping a parent understand their child's Troy block build.

Build summary:
{summary}

Parent question:
{question}

Answer in a short, warm, creative but realistic way.
Use only the build details provided.
Do not invent hidden abilities or unseen parts.
Keep it to 3 to 5 short lines.
"""

        completion = client.chat.completions.create(
            model=get_groq_text_model(),
            messages=[
                {
                    "role": "system",
                    "content": "You are a warm parent-friendly assistant for Troy World."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.55,
            top_p=0.9,
            max_completion_tokens=300
        )

        answer = completion.choices[0].message.content

        return jsonify({
            "answer": clean_text(
                answer,
                "I’m unable to answer that right now. Please try again."
            )
        }), 200

    except Exception as e:
        print("Ask error:", str(e))

        return jsonify({
            "answer": "Live Q&A is temporarily unavailable right now. Please try again later."
        }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    )