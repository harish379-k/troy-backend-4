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

# Prevent huge uploads from killing Render memory
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 MB

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
CORS(app, origins=CORS_ORIGINS.split(","))

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Render-safe image size
MAX_BASE64_IMAGE_SIZE = 1_800_000
Image.MAX_IMAGE_PIXELS = 15_000_000

# Limited in-memory cache
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
            "temperature": 0.78,
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
    2. base64 data URL for Groq
    3. image hash for caching and feedback variation
    """

    img = Image.open(image_file.stream)
    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

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
# Creative variation helpers
# =========================================================

def pick_feedback_style(image_hash):
    styles = [
        {
            "name": "story-builder",
            "instruction": "Focus on the pretend-play story this build could become."
        },
        {
            "name": "designer",
            "instruction": "Focus on shape choices, arrangement, and design decisions."
        },
        {
            "name": "builder",
            "instruction": "Focus on supports, balance, levels, and how parts hold together."
        },
        {
            "name": "inventor",
            "instruction": "Focus on unusual combinations, hybrid ideas, and creative guessing."
        },
        {
            "name": "architect",
            "instruction": "Focus on rooms, floors, openings, height, layout, and spaces."
        },
        {
            "name": "movement-maker",
            "instruction": "Focus on motion, wheels, paths, travel, vehicles, or moving ideas."
        },
        {
            "name": "pattern-finder",
            "instruction": "Focus on repeated blocks, matching, spacing, and visual patterns."
        }
    ]

    seed_number = int(image_hash[:8], 16)
    return styles[seed_number % len(styles)]


def build_unique_hint(image_hash):
    hints = [
        "Use fresh wording for this image.",
        "Do not repeat generic feedback titles.",
        "Make the learning cards specific to this exact build.",
        "Let visible parts guide the learning feedback.",
        "Avoid basic titles like Creativity, Problem-Solving, or Spatial Awareness."
    ]

    seed_number = int(image_hash[8:16], 16)
    return hints[seed_number % len(hints)]


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
- Base every sentence only on visible details.
- Mention visible parts such as base, floors, levels, gaps, supports,
  repeated blocks, stacked sections, roof-like pieces, wheel-like parts,
  curved pieces, openings, paths, bridges, rooms, platforms, loose blocks,
  or upper/lower sections if visible.
- If the image is not a Troy/block build, mark it invalid.
- If unsure, use cautious phrases like "looks like", "could be", or "seems to".
- Keep the tone simple, warm, parent-friendly, and encouraging.
- Do not overclaim.
- Return JSON only.

BANNED LEARNING CARD TITLES:
Do not use these titles:
- Creativity
- Problem-Solving
- Problem Solving
- Spatial Awareness
- Spatial Thinking
- Imagination
- Motor Skills
- Fine Motor Skills
- Engineering
- STEM Learning
- Critical Thinking

Instead, use specific learning card titles based on the visible build, such as:
- Layer Planning
- Upper-Level Building
- Bridge Support
- Moving Base Idea
- Tiny Home Story
- Roof Shape Experiment
- Open-Space Design
- Block Pattern Play
- Creature-Making
- Careful Stacking
- Shape Combining
- Idea Mixing
- Build-and-Tell Practice
- Small-World Making
- Support Below, Space Above
- Vehicle Shape Thinking
- Room-Making
- Entrance Building
- Testing What Holds
- Above-Below Thinking

Each learning card must:
- be specific to this exact build
- mention a visible detail from the image
- avoid generic praise
- sound different from the other two cards

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
      "title": "specific learning skill title, not generic",
      "description": "specific explanation connected to visible details in this build",
      "color": "cream"
    }},
    {{
      "title": "specific learning skill title, not generic",
      "description": "specific explanation connected to visible details in this build",
      "color": "green"
    }},
    {{
      "title": "specific learning skill title, not generic",
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
# Specific feedback fallback logic
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
    context = build_context_text(build_guess, summary, noticed)
    main_detail = noticed[0] if noticed else "the visible block arrangement"

    card_pool = []

    # Multi-level / floor / layered builds
    if contains_any(context, ["level", "floor", "platform", "layer", "upper", "lower", "multi-level"]):
        card_pool.extend([
            {
                "title": "Upper-Level Building",
                "description": f"The child explored how one section can sit above another, especially around {main_detail}.",
                "color": "cream"
            },
            {
                "title": "Layer Planning",
                "description": "The child practiced making a build with lower and upper parts instead of one simple stack.",
                "color": "green"
            },
            {
                "title": "Support Below, Space Above",
                "description": "The build encourages the child to think about how bottom blocks can hold up higher sections.",
                "color": "blue"
            }
        ])

    # Vehicle / moving base builds
    if contains_any(context, ["wheel", "vehicle", "moving", "car", "base", "travel", "rolling"]):
        card_pool.extend([
            {
                "title": "Moving Base Idea",
                "description": "The child connected the bottom part of the build with the idea of movement or travel.",
                "color": "cream"
            },
            {
                "title": "Vehicle Shape Thinking",
                "description": "The child explored how a block base can become something that looks ready to move.",
                "color": "green"
            },
            {
                "title": "Parts Working Together",
                "description": "The child practiced combining a base and upper section into one complete build idea.",
                "color": "blue"
            }
        ])

    # House / room / roof builds
    if contains_any(context, ["house", "home", "room", "roof", "door", "window", "shelter"]):
        card_pool.extend([
            {
                "title": "Tiny Home Story",
                "description": "The child used blocks to suggest a small home-like space that can become part of a story.",
                "color": "cream"
            },
            {
                "title": "Roof Shape Experiment",
                "description": "The child explored how top pieces can make a build feel like a room, roof, or shelter.",
                "color": "green"
            },
            {
                "title": "Room-Making",
                "description": "The build helps the child think about inside and outside spaces using simple blocks.",
                "color": "blue"
            }
        ])

    # Bridge / gap / support builds
    if contains_any(context, ["bridge", "gap", "span", "across", "support", "beam"]):
        card_pool.extend([
            {
                "title": "Bridge Support",
                "description": "The child explored how blocks can stretch across a gap while still needing support.",
                "color": "cream"
            },
            {
                "title": "Across-and-Over Thinking",
                "description": "The build helps the child notice how one part can connect two separate sides.",
                "color": "green"
            },
            {
                "title": "Testing What Holds",
                "description": "The child can learn which blocks keep the bridge-like part steady and which parts wobble.",
                "color": "blue"
            }
        ])

    # Gate / arch / tunnel builds
    if contains_any(context, ["gate", "arch", "opening", "entrance", "tunnel", "curve"]):
        card_pool.extend([
            {
                "title": "Open-Space Design",
                "description": "The child explored how blocks can make an entrance, tunnel, or pass-through space.",
                "color": "cream"
            },
            {
                "title": "Entrance Building",
                "description": "The build invites the child to think about where something could go in or come out.",
                "color": "green"
            },
            {
                "title": "Curve and Shape Play",
                "description": "The child experimented with how curved or open shapes can change the build’s meaning.",
                "color": "blue"
            }
        ])

    # Pattern / repeated blocks
    if contains_any(context, ["repeat", "repeated", "same", "pattern", "symmetry", "line", "row"]):
        card_pool.extend([
            {
                "title": "Block Pattern Play",
                "description": "The child used repeated placement to make parts of the build feel organized.",
                "color": "cream"
            },
            {
                "title": "Matching and Repeating",
                "description": "The child practiced noticing which blocks look similar and how they can be placed together.",
                "color": "green"
            },
            {
                "title": "Visual Order",
                "description": "The repeated blocks help the child explore spacing, direction, and arrangement.",
                "color": "blue"
            }
        ])

    # Animal / creature builds
    if contains_any(context, ["animal", "creature", "head", "legs", "tail", "body", "neck"]):
        card_pool.extend([
            {
                "title": "Creature-Making",
                "description": "The child used simple block parts to suggest a body, head, legs, or creature-like shape.",
                "color": "cream"
            },
            {
                "title": "Character Building",
                "description": "The animal-like shape can become a pretend character in the child’s play story.",
                "color": "green"
            },
            {
                "title": "Body-Part Thinking",
                "description": "The child explored how separate blocks can stand for different parts of one living thing.",
                "color": "blue"
            }
        ])

    # Tall / stacking builds
    if contains_any(context, ["tower", "stack", "tall", "height", "vertical"]):
        card_pool.extend([
            {
                "title": "Careful Stacking",
                "description": f"The child practiced placing pieces upward while keeping the structure steady near {main_detail}.",
                "color": "cream"
            },
            {
                "title": "Height Control",
                "description": "The child explored how a build changes when blocks are placed higher and higher.",
                "color": "green"
            },
            {
                "title": "Steady Hands",
                "description": "The child practiced careful hand movement while adding blocks without knocking the build down.",
                "color": "blue"
            }
        ])

    # Hybrid / pretend builds
    if contains_any(context, ["hybrid", "combines", "combination", "moving house", "house-on-wheels", "machine", "pretend", "scene"]):
        card_pool.extend([
            {
                "title": "Idea Mixing",
                "description": "The child combined more than one idea into a single build instead of making only one simple object.",
                "color": "cream"
            },
            {
                "title": "Pretend-World Design",
                "description": "The build can become a small story world where different parts have different jobs.",
                "color": "green"
            },
            {
                "title": "Inventor Thinking",
                "description": "The child experimented with making something unusual by joining different block ideas together.",
                "color": "blue"
            }
        ])

    # Always available non-generic cards
    card_pool.extend([
        {
            "title": "Block Decision-Making",
            "description": f"The child made choices about where to place pieces, especially around {main_detail}.",
            "color": "cream"
        },
        {
            "title": "Shape Combining",
            "description": "The child explored how different block shapes can come together to create one bigger idea.",
            "color": "green"
        },
        {
            "title": "Build-and-Tell Practice",
            "description": "The structure gives the child something they can explain, rename, and turn into a story.",
            "color": "blue"
        },
        {
            "title": "Above-Below Thinking",
            "description": "The child practiced noticing which parts are above, below, beside, or connected to other parts.",
            "color": "cream"
        },
        {
            "title": "Small-World Making",
            "description": "The build can become a tiny play world with places, paths, rooms, or moving parts.",
            "color": "green"
        },
        {
            "title": "Build Revision",
            "description": "The child can look at the structure and decide what to add, remove, strengthen, or rename.",
            "color": "blue"
        }
    ])

    seed_number = int(image_hash[:10], 16)
    random.Random(seed_number).shuffle(card_pool)

    selected = []
    used_titles = set()

    for card in card_pool:
        key = card["title"].lower()

        if key in used_titles:
            continue

        selected.append(card)
        used_titles.add(key)

        if len(selected) == 3:
            break

    colors = ["cream", "green", "blue"]

    for index, card in enumerate(selected):
        card["color"] = colors[index]

    return selected


def is_weak_learning_card(card):
    title = clean_text(card.get("title")).lower()
    description = clean_text(card.get("description")).lower()

    if not title or not description:
        return True

    banned_titles = {
        "creativity",
        "problem-solving",
        "problem solving",
        "spatial awareness",
        "spatial thinking",
        "imagination",
        "motor skills",
        "fine motor skills",
        "engineering",
        "stem learning",
        "critical thinking"
    }

    if title in banned_titles:
        return True

    if len(description.split()) < 10:
        return True

    generic_phrases = [
        "showed creativity",
        "used creativity",
        "practiced problem-solving",
        "practiced problem solving",
        "demonstrated spatial awareness",
        "showed imagination",
        "developed motor skills",
        "improved problem solving",
        "learned engineering"
    ]

    if any(phrase in description for phrase in generic_phrases):
        return True

    return False


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
        temperature=0.78,
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

    try:
        parsed = analyze_with_gemini(pil_img, age, image_hash)
        result = normalize_analysis_response(parsed, image_hash)
        result["provider"] = "gemini"
        return result

    except Exception as e:
        error_text = str(e)
        print("Gemini failed:", error_text)
        errors.append(f"Gemini: {error_text}")

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