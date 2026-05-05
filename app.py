import os
import json
import uuid
import time
import base64
from io import BytesIO
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from groq import Groq


# =========================================================
# App setup
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
CORS(app, origins=CORS_ORIGINS.split(","))

sessions = {}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_BASE64_IMAGE_SIZE = 3_800_000


# =========================================================
# Groq config
# =========================================================

def get_groq_api_key():
    return (
        os.environ.get("GROQ_API_KEY", "").strip()
        or os.environ.get("RENDER_GROQ_KEY", "").strip()
    )


def get_vision_model_name():
    return os.environ.get(
        "GROQ_VISION_MODEL",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ).strip()


def get_text_model_name():
    return os.environ.get(
        "GROQ_TEXT_MODEL",
        "llama-3.3-70b-versatile"
    ).strip()


def build_groq_client():
    api_key = get_groq_api_key()

    if not api_key:
        return None

    return Groq(api_key=api_key)


print("Groq key:", "FOUND" if get_groq_api_key() else "NOT FOUND")
print("Groq vision model:", get_vision_model_name())
print("Groq text model:", get_text_model_name())


# =========================================================
# Basic helpers
# =========================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def clean_sentence(value, fallback=""):
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
        text = clean_sentence(item)
        key = text.lower()

        if text and key not in seen:
            seen.add(key)
            result.append(text)

    if not result:
        result = fallback

    return result[:limit]


def extract_json_block(text):
    if not text:
        return ""

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace + 1]

    return text


def parse_json_response(text):
    cleaned = extract_json_block(text)
    return json.loads(cleaned)


# =========================================================
# Error helpers
# =========================================================

def is_rate_limit_error(error_text):
    lower = error_text.lower()

    return (
        "429" in lower
        or "rate limit" in lower
        or "rate_limit" in lower
        or "too many requests" in lower
    )


def is_invalid_key_error(error_text):
    lower = error_text.lower()

    return (
        "401" in lower
        or "403" in lower
        or "invalid api key" in lower
        or "unauthorized" in lower
        or "forbidden" in lower
    )


def is_image_too_large_error(error_text):
    lower = error_text.lower()

    return (
        "413" in lower
        or "payload too large" in lower
        or "request too large" in lower
        or "image too large" in lower
    )


def is_temporary_error(error_text):
    lower = error_text.lower()

    return (
        "503" in lower
        or "502" in lower
        or "500" in lower
        or "service unavailable" in lower
        or "temporarily unavailable" in lower
    )


# =========================================================
# Image compression for Groq vision
# =========================================================

def encode_image_to_base64_jpeg(img, quality):
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    raw_bytes = buffer.getvalue()
    encoded = base64.b64encode(raw_bytes).decode("utf-8")
    return encoded


def image_file_to_data_url(image_file):
    img = Image.open(image_file.stream)
    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail((1400, 1400))

    for quality in [85, 75, 65, 55, 45]:
        encoded = encode_image_to_base64_jpeg(img, quality)

        if len(encoded.encode("utf-8")) <= MAX_BASE64_IMAGE_SIZE:
            return f"data:image/jpeg;base64,{encoded}"

    img.thumbnail((1000, 1000))

    for quality in [70, 60, 50, 40]:
        encoded = encode_image_to_base64_jpeg(img, quality)

        if len(encoded.encode("utf-8")) <= MAX_BASE64_IMAGE_SIZE:
            return f"data:image/jpeg;base64,{encoded}"

    raise ValueError(
        "Image is too large even after compression. Please upload a smaller image."
    )


# =========================================================
# Evidence + feedback logic
# =========================================================

def combined_evidence_text(visible_elements, extra_text=""):
    items = visible_elements or []
    return clean_sentence(" ".join(items) + " " + extra_text).lower()


def has_keyword(text, keywords):
    text = text.lower()
    return any(keyword in text for keyword in keywords)


def visible_phrase(visible_elements):
    if visible_elements:
        return visible_elements[0]
    return "the visible block arrangement"


def derive_category_from_evidence(raw_category, visible_elements):
    """
    Uses visibleElements to verify the category.
    This prevents Groq from saying 'car' or 'animal' with no evidence.
    """
    category = clean_sentence(raw_category, "abstract").lower()
    evidence = combined_evidence_text(visible_elements)

    proof_words = {
        "tower": [
            "tower", "stack", "stacked", "tall", "vertical", "height",
            "column", "upward", "above", "on top"
        ],
        "bridge": [
            "bridge", "span", "gap", "across", "support", "supports",
            "horizontal", "beam", "connects"
        ],
        "house": [
            "house", "home", "roof", "wall", "room", "door", "window",
            "triangle", "built space"
        ],
        "vehicle": [
            "vehicle", "car", "wheel", "wheels", "axle", "front", "back",
            "base", "body"
        ],
        "animal": [
            "animal", "head", "body", "leg", "legs", "neck", "tail",
            "face"
        ],
        "gate": [
            "gate", "opening", "entrance", "doorway", "arch", "curve"
        ],
        "castle": [
            "castle", "fort", "wall", "tower", "turret"
        ],
        "road": [
            "road", "path", "track", "route", "line", "trail"
        ]
    }

    allowed_categories = [
        "tower", "bridge", "house", "vehicle", "animal",
        "gate", "castle", "road", "abstract", "unclear", "unrelated"
    ]

    if category not in allowed_categories:
        category = "abstract"

    if category in proof_words:
        has_support = any(word in evidence for word in proof_words[category])

        if has_support:
            return category

        # If model gave a specific object but visible evidence does not support it,
        # make it safer instead of repeating a wrong guess.
        return "abstract"

    return category


def repair_weak_guess(build_category, title, subtitle, visible_elements, summary=""):
    """
    Keeps Groq's unique guess when it is reasonable.
    Only makes it safer if the guess is unsupported.
    """
    raw_category = clean_sentence(build_category, "abstract").lower()
    raw_title = clean_sentence(title)
    raw_subtitle = clean_sentence(subtitle)
    raw_summary = clean_sentence(summary)

    final_category = derive_category_from_evidence(raw_category, visible_elements)

    title_lower = raw_title.lower()

    object_words = [
        "car", "dog", "cat", "house", "castle", "bridge", "tower",
        "animal", "vehicle", "gate", "road"
    ]

    # If category became abstract, do not keep an overconfident object title.
    if final_category in ["abstract", "unclear"]:
        if raw_title and not any(word == title_lower for word in object_words):
            if "block" in title_lower or "structure" in title_lower or "build" in title_lower:
                final_title = raw_title
            else:
                final_title = f"{raw_title} block build"
        else:
            final_title = "Open-ended Troy block build"

        if raw_subtitle:
            final_subtitle = raw_subtitle
        elif visible_elements:
            final_subtitle = f"It uses visible details like {visible_elements[0]}."
        else:
            final_subtitle = "The child created an open-ended structure using Troy blocks."

        return {
            "category": final_category,
            "title": final_title,
            "subtitle": final_subtitle
        }

    # For supported categories, allow specific but cautious labels.
    safe_category_titles = {
        "tower": "Tower-like Troy block build",
        "bridge": "Bridge-like Troy block build",
        "house": "House-like Troy block build",
        "vehicle": "Vehicle-like Troy block build",
        "animal": "Animal-like Troy block build",
        "gate": "Gate-like Troy block build",
        "castle": "Castle-like Troy block build",
        "road": "Road or path-like Troy block layout",
        "unrelated": "Unclear image"
    }

    if raw_title and len(raw_title) >= 4:
        if raw_title.lower() in object_words:
            final_title = safe_category_titles.get(final_category, f"{raw_title} block build")
        elif "block" not in title_lower and "structure" not in title_lower and "build" not in title_lower:
            final_title = f"{raw_title} block build"
        else:
            final_title = raw_title
    else:
        final_title = safe_category_titles.get(final_category, "Troy block build")

    if raw_subtitle:
        final_subtitle = raw_subtitle
    elif visible_elements:
        final_subtitle = f"It seems to use details like {visible_elements[0]}."
    elif raw_summary:
        final_subtitle = raw_summary
    else:
        final_subtitle = "The child created a visible structure using Troy blocks."

    return {
        "category": final_category,
        "title": final_title,
        "subtitle": final_subtitle
    }


def build_learning_cards_from_evidence(category, visible_elements):
    """
    Backend-generated fallback cards.
    These are used only if Groq gives empty or weak learning cards.
    """
    evidence = combined_evidence_text(visible_elements)
    main_detail = visible_phrase(visible_elements)

    cards = []

    if has_keyword(evidence, ["stack", "stacked", "tall", "vertical", "height", "tower"]):
        cards.append({
            "title": "Stacking and Balance",
            "description": f"The child practiced placing blocks upward, especially around {main_detail}.",
            "color": "cream"
        })

    if has_keyword(evidence, ["wide base", "base", "support", "supports", "steady", "stability"]):
        cards.append({
            "title": "Stability",
            "description": "The child explored how a stronger base or support can help the build stay steady.",
            "color": "green"
        })

    if has_keyword(evidence, ["bridge", "span", "gap", "across", "horizontal", "beam", "connects"]):
        cards.append({
            "title": "Support and Span",
            "description": "The child explored how blocks can connect across a space or rest on supports.",
            "color": "cream"
        })

    if has_keyword(evidence, ["repeat", "repeated", "same", "pattern", "symmetry", "symmetrical"]):
        cards.append({
            "title": "Pattern Recognition",
            "description": "The child noticed how repeated blocks or similar placements can make a build more organized.",
            "color": "green"
        })

    if has_keyword(evidence, ["triangle", "roof", "curve", "curved", "arch", "different shapes"]):
        cards.append({
            "title": "Shape Matching",
            "description": "The child experimented with how different block shapes can fit into one structure.",
            "color": "blue"
        })

    if has_keyword(evidence, ["small block", "many pieces", "careful", "loose", "placed"]):
        cards.append({
            "title": "Fine Motor Control",
            "description": "Placing the pieces carefully helps the child practice hand control and focus.",
            "color": "blue"
        })

    if category == "tower":
        cards.append({
            "title": "Height Awareness",
            "description": "The child explored how a structure changes when pieces are placed higher.",
            "color": "cream"
        })

    if category == "bridge":
        cards.append({
            "title": "Cause and Effect",
            "description": "The child can test what happens when a support is moved, added, or removed.",
            "color": "blue"
        })

    if category == "house":
        cards.append({
            "title": "Structure Planning",
            "description": "The child practiced arranging blocks into parts that feel like a built space.",
            "color": "cream"
        })

    if category == "vehicle":
        cards.append({
            "title": "Part-to-Whole Thinking",
            "description": "The child used smaller pieces to represent one larger idea, like a vehicle form.",
            "color": "green"
        })

    if category == "animal":
        cards.append({
            "title": "Symbolic Thinking",
            "description": "The child used simple block shapes to represent something from real life.",
            "color": "cream"
        })

    if category == "gate":
        cards.append({
            "title": "Open Space Awareness",
            "description": "The child explored how blocks can create an entrance, opening, or arch-like space.",
            "color": "green"
        })

    if category == "road":
        cards.append({
            "title": "Sequencing",
            "description": "The child practiced placing blocks in an order to create a path or route.",
            "color": "cream"
        })

    if category in ["abstract", "unclear"]:
        cards.append({
            "title": "Spatial Thinking",
            "description": f"The child practiced deciding where each block should go, especially around {main_detail}.",
            "color": "cream"
        })
        cards.append({
            "title": "Creative Expression",
            "description": "The open-ended build allowed the child to turn an idea into a physical structure.",
            "color": "green"
        })

    default_cards = [
        {
            "title": "Block Placement",
            "description": f"The child practiced placing pieces carefully around {main_detail}.",
            "color": "cream"
        },
        {
            "title": "Balance and Stability",
            "description": "The child explored how blocks can stay steady when arranged carefully.",
            "color": "green"
        },
        {
            "title": "Creative Planning",
            "description": "The child turned an idea into a visible block structure.",
            "color": "blue"
        }
    ]

    cleaned = []
    used_titles = set()
    colors = ["cream", "green", "blue"]

    for card in cards + default_cards:
        title = clean_sentence(card.get("title"))
        description = clean_sentence(card.get("description"))
        color = clean_sentence(card.get("color", colors[len(cleaned) % 3])).lower()

        if not title or not description:
            continue

        if title.lower() in used_titles:
            continue

        if color not in colors:
            color = colors[len(cleaned) % 3]

        cleaned.append({
            "title": title,
            "description": description,
            "color": color
        })

        used_titles.add(title.lower())

        if len(cleaned) == 3:
            break

    for index, card in enumerate(cleaned):
        card["color"] = colors[index]

    return cleaned[:3]


def is_too_generic_card(card):
    title = clean_sentence(card.get("title")).lower()
    description = clean_sentence(card.get("description")).lower()

    if not title or not description:
        return True

    # Very short descriptions usually feel repeated/generic.
    if len(description.split()) < 8:
        return True

    weak_titles = [
        "creativity",
        "imagination",
        "problem solving",
        "engineering",
        "stem learning",
        "motor skills"
    ]

    # Allow these only if the description has real visible evidence.
    evidence_words = [
        "stack", "base", "support", "gap", "bridge", "roof", "triangle",
        "curve", "arch", "repeated", "horizontal", "vertical", "blocks",
        "pieces", "path", "opening"
    ]

    if title in weak_titles:
        return not any(word in description for word in evidence_words)

    return False


def choose_learning_cards(ai_cards, category, visible_elements):
    """
    Prefer Groq's unique cards.
    Use backend fallback only when Groq is empty or too generic.
    """
    backend_cards = build_learning_cards_from_evidence(category, visible_elements)

    if not isinstance(ai_cards, list) or len(ai_cards) < 3:
        return backend_cards

    cleaned = []
    allowed_colors = ["cream", "green", "blue"]

    for index, card in enumerate(ai_cards):
        if not isinstance(card, dict):
            continue

        title = clean_sentence(card.get("title"))
        description = clean_sentence(card.get("description"))
        color = clean_sentence(card.get("color", allowed_colors[index % 3])).lower()

        if not title or not description:
            continue

        if color not in allowed_colors:
            color = allowed_colors[index % 3]

        temp_card = {
            "title": title,
            "description": description,
            "color": color
        }

        if is_too_generic_card(temp_card):
            # Do not immediately reject all AI cards.
            # Just skip weak cards and keep useful ones.
            continue

        cleaned.append(temp_card)

    # If Groq gave at least 2 good cards, keep them and fill the rest with backend cards.
    if len(cleaned) >= 2:
        existing_titles = {card["title"].lower() for card in cleaned}

        for fallback in backend_cards:
            if len(cleaned) >= 3:
                break

            if fallback["title"].lower() not in existing_titles:
                cleaned.append(fallback)

    if len(cleaned) < 3:
        cleaned = backend_cards

    for i, card in enumerate(cleaned[:3]):
        card["color"] = allowed_colors[i]

    return cleaned[:3]


# =========================================================
# Invalid response
# =========================================================

def build_invalid_photo_response(reason, noticed=None, suggestions=None, ideas=None):
    session_id = str(uuid.uuid4())

    result = {
        "status": "success",
        "imageStatus": "invalid",
        "confidenceScore": 0,
        "analysisDetails": {
            "buildCategory": "unclear",
            "visibleElements": [],
            "blockCountEstimate": "unclear",
            "whyThisGuess": reason
        },
        "buildGuess": {
            "title": "We couldn’t clearly analyze this image",
            "subtitle": reason
        },
        "whatWeFound": {
            "title": "What we found",
            "summary": reason
        },
        "whatTheyLearned": [],
        "whatWeNoticed": noticed or [
            "The image does not clearly show enough visible Troy blocks.",
            "The structure may be blurry, cropped, too far away, or unrelated to Troy blocks.",
            "A clearer photo will help us give the right feedback."
        ],
        "suggestionsForParent": suggestions or [
            "Retake the photo with the full structure visible.",
            "Use better lighting and a cleaner background.",
            "Make sure the Troy block build is the main focus of the image."
        ],
        "nextBuildIdeas": ideas or [
            "Build a tall tower with a wide base.",
            "Build a small bridge using two supports.",
            "Build a simple house with a roof block."
        ],
        "session_id": session_id
    }

    sessions[session_id] = result
    return result


# =========================================================
# Groq prompt
# =========================================================

def build_troy_analysis_prompt(age):
    return f"""
You are Troy AI Analyzer for Troy World.

Your task:
Analyze one uploaded photo of a child's Troy wooden-block build.

Child age:
{age if age else "unknown"}

TROY BLOCK CONTEXT:
- Troy builds are made using wooden blocks.
- Pieces may include cubes, cuboids, long beams, planks, pillars, triangular roof pieces, curved/arch pieces, and connector-like wooden pieces.
- Children may build towers, bridges, houses, animals, vehicles, gates, castles, roads, pretend scenes, or abstract structures.

MOST IMPORTANT RULE:
Do not guess the child's intention randomly.
Only identify the build as a specific object if the visible structure strongly supports it.

GUESSING RULES:
- If it has stacked vertical pieces and height: call it a tower-like structure.
- If it has two supports with a horizontal piece across: call it a bridge-like structure.
- If it has walls, roof/triangle pieces, or room-like layout: call it a house-like structure.
- If it has wheels, axle-like parts, or clear vehicle shape: call it a vehicle-like structure.
- If it has body + legs/head/neck shape: call it an animal-like structure.
- If it has an entrance/opening/arch: call it a gate-like structure.
- If it is not clear, call it an abstract or open-ended block structure.
- Never confidently say car, dog, castle, or house unless there is strong visible evidence.
- Use cautious language like "looks like", "seems like", or "could be" when uncertain.

VISIBLE EVIDENCE RULES:
Every sentence must be based on visible details in the image.
Mention actual visible details such as:
- tall stack
- wide base
- repeated blocks
- horizontal beam
- vertical supports
- triangle/roof piece
- curved piece
- symmetry
- gaps
- loose blocks
- connected sections

UNIQUENESS RULE:
This analysis must feel different for each uploaded build.
Do not reuse the same buildGuess, learning card titles, or descriptions unless the visible build is actually very similar.
Use the specific visibleElements to make every learning card different.
Mention at least one unique visual detail from the photo in each learning description.

INVALID IMAGE RULES:
Mark imageStatus as "invalid" if:
- no Troy-style wooden blocks are visible
- image is mostly a person/selfie
- image is a screenshot/drawing
- image shows real buildings, furniture, toys, or random objects instead of Troy blocks
- image is too blurry/dark/cropped/far away to understand the build

WHAT THEY LEARNED RULES:
Each learning card must connect to something visible in the build.
Do not write generic learning cards.

Good learning card examples:
- If blocks are stacked: "Stacking and Balance"
- If there is a wide base: "Stability"
- If repeated blocks are used: "Pattern Recognition"
- If there are gaps/bridges: "Support and Span"
- If different shapes are combined: "Shape Matching"
- If the build is pretend/abstract: "Imagination and Storytelling"
- If the child used many pieces carefully: "Fine Motor Control"

Bad learning cards:
- "Creativity" alone is too generic.
- "Engineering skills" is too advanced.
- "Problem solving" without visible explanation is too generic.

CONFIDENCE RULE:
- confidenceScore must be 0 to 100.
- imageStatus must be "valid" only if visible wooden Troy-style blocks are clearly present.
- If confidenceScore is below 65, imageStatus must be "invalid".

Return valid JSON only.
No markdown.
No extra text.

Return exactly this JSON shape:

{{
  "status": "success",
  "imageStatus": "valid or invalid",
  "confidenceScore": 0,
  "analysisDetails": {{
    "buildCategory": "tower / bridge / house / vehicle / animal / gate / castle / road / abstract / unclear / unrelated",
    "visibleElements": [
      "specific visible detail 1",
      "specific visible detail 2",
      "specific visible detail 3"
    ],
    "blockCountEstimate": "rough estimate like 5-8, 10-15, 20+ or unclear",
    "whyThisGuess": "explain the guess using only visible details"
  }},
  "buildGuess": {{
    "title": "safe build interpretation",
    "subtitle": "one short sentence explaining what it looks like"
  }},
  "whatWeFound": {{
    "title": "What we found",
    "summary": "2 short sentences describing the actual visible structure"
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
    "specific observation 1 from the image",
    "specific observation 2 from the image",
    "specific observation 3 from the image"
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

For invalid images:
- Keep the same JSON shape.
- Set imageStatus to "invalid".
- Set confidenceScore below 65.
- Set whatTheyLearned to an empty list.
"""


# =========================================================
# Groq calls
# =========================================================

def call_groq_vision(client, image_data_url, age):
    prompt = build_troy_analysis_prompt(age)

    completion = client.chat.completions.create(
        model=get_vision_model_name(),
        messages=[
            {
                "role": "system",
                "content": "You are a careful visual analysis assistant. Return valid JSON only."
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
        temperature=0.55,
        top_p=0.95,
        max_completion_tokens=1400,
        response_format={
            "type": "json_object"
        }
    )

    content = completion.choices[0].message.content
    return parse_json_response(content)


def call_groq_vision_with_retry(client, image_data_url, age, max_retries=2):
    delay = 2

    for attempt in range(max_retries):
        try:
            return call_groq_vision(client, image_data_url, age)

        except Exception as e:
            error_text = str(e)
            print(f"Groq vision attempt {attempt + 1} failed:", error_text)

            if is_temporary_error(error_text) and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue

            raise


# =========================================================
# Routes
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Troy Groq backend is running",
        "server": "ok"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "groq_key_loaded": bool(get_groq_api_key()),
        "vision_model": get_vision_model_name(),
        "text_model": get_text_model_name()
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        age = clean_sentence(request.form.get("age", ""))

        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        image_file = request.files["image"]

        if image_file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(image_file.filename):
            return jsonify({
                "error": "Invalid file type. Please upload PNG, JPG, JPEG, or WEBP."
            }), 400

        client = build_groq_client()

        if not client:
            return jsonify({
                "error": "GROQ_API_KEY not found. Add it in Render environment variables."
            }), 500

        original_filename = secure_filename(image_file.filename)

        try:
            image_data_url = image_file_to_data_url(image_file)
        except Exception as e:
            return jsonify({
                "error": "Could not process image.",
                "details": str(e)
            }), 400

        parsed = call_groq_vision_with_retry(client, image_data_url, age)

        image_status = clean_sentence(parsed.get("imageStatus", "invalid")).lower()

        try:
            confidence_score = int(float(parsed.get("confidenceScore", 0)))
        except Exception:
            confidence_score = 0

        analysis_details = parsed.get("analysisDetails")
        if not isinstance(analysis_details, dict):
            analysis_details = {}

        raw_build_guess = parsed.get("buildGuess")
        if not isinstance(raw_build_guess, dict):
            raw_build_guess = {}

        what_found = parsed.get("whatWeFound")
        if not isinstance(what_found, dict):
            what_found = {}

        raw_build_category = clean_sentence(
            analysis_details.get("buildCategory"),
            "abstract"
        )

        visible_elements = ensure_list(
            analysis_details.get("visibleElements"),
            fallback=[],
            limit=3
        )

        raw_title = clean_sentence(raw_build_guess.get("title"))
        raw_subtitle = clean_sentence(raw_build_guess.get("subtitle"))
        raw_summary = clean_sentence(what_found.get("summary"))

        safe_guess = repair_weak_guess(
            build_category=raw_build_category,
            title=raw_title,
            subtitle=raw_subtitle,
            visible_elements=visible_elements,
            summary=raw_summary
        )

        final_category = safe_guess["category"]

        if image_status == "valid" and confidence_score >= 65:
            session_id = str(uuid.uuid4())

            noticed_fallback = visible_elements or [
                "The photo shows visible Troy-style wooden blocks arranged into a structure.",
                "The build has multiple pieces placed with intention.",
                "The child appears to be exploring shape, balance, and arrangement."
            ]

            if raw_summary:
                final_summary = raw_summary
            elif visible_elements:
                final_summary = (
                    f"This build has details like {', '.join(visible_elements[:2])}. "
                    "The child used the blocks to explore shape, placement, and structure."
                )
            else:
                final_summary = (
                    "This Troy build shows blocks arranged into a child-made structure. "
                    "The exact object is open-ended, but the placement still shows creative building."
                )

            result = {
                "status": "success",
                "imageStatus": "valid",
                "confidenceScore": confidence_score,
                "analysisDetails": {
                    "buildCategory": final_category,
                    "visibleElements": visible_elements,
                    "blockCountEstimate": clean_sentence(
                        analysis_details.get("blockCountEstimate"),
                        "unclear"
                    ),
                    "whyThisGuess": clean_sentence(
                        analysis_details.get("whyThisGuess"),
                        safe_guess["subtitle"]
                    )
                },
                "buildGuess": {
                    "title": safe_guess["title"],
                    "subtitle": safe_guess["subtitle"]
                },
                "whatWeFound": {
                    "title": "What we found",
                    "summary": final_summary
                },
                "whatTheyLearned": choose_learning_cards(
                    parsed.get("whatTheyLearned"),
                    category=final_category,
                    visible_elements=visible_elements
                ),
                "whatWeNoticed": ensure_list(
                    parsed.get("whatWeNoticed"),
                    fallback=noticed_fallback,
                    limit=3
                ),
                "suggestionsForParent": ensure_list(
                    parsed.get("suggestionsForParent"),
                    fallback=[
                        "Ask your child what each part of the structure represents.",
                        "Invite them to make one part stronger, taller, wider, or more detailed.",
                        "Take another photo after they improve the build and compare both versions."
                    ],
                    limit=3
                ),
                "nextBuildIdeas": ensure_list(
                    parsed.get("nextBuildIdeas"),
                    fallback=[
                        "Build a stronger version with a wider base.",
                        "Add one new feature, such as a roof, path, wheel, or support piece.",
                        "Try rebuilding the same idea using fewer blocks."
                    ],
                    limit=3
                ),
                "session_id": session_id
            }

            if os.environ.get("SHOW_DEBUG", "false").lower() == "true":
                result["debug"] = {
                    "filename": original_filename,
                    "model": get_vision_model_name(),
                    "rawCategory": raw_build_category,
                    "safeCategory": final_category,
                    "rawTitle": raw_title
                }

            sessions[session_id] = result
            return jsonify(result), 200

        invalid_reason = clean_sentence(
            raw_summary,
            "We couldn’t clearly analyze this image as a Troy block build."
        )

        result = build_invalid_photo_response(
            reason=invalid_reason,
            noticed=ensure_list(
                parsed.get("whatWeNoticed"),
                fallback=visible_elements or [
                    "The image may be unclear or unrelated to Troy blocks.",
                    "Too little of the build may be visible.",
                    "A clearer photo will help us analyze the build properly."
                ],
                limit=3
            ),
            suggestions=ensure_list(
                parsed.get("suggestionsForParent"),
                fallback=[
                    "Retake the photo with the full structure visible.",
                    "Use better lighting and a cleaner background.",
                    "Make sure the Troy block build is the main focus of the image."
                ],
                limit=3
            ),
            ideas=ensure_list(
                parsed.get("nextBuildIdeas"),
                fallback=[
                    "Build a tower with a wide base.",
                    "Build a bridge with two supports.",
                    "Build a small house with a roof block."
                ],
                limit=3
            )
        )

        result["confidenceScore"] = confidence_score
        result["analysisDetails"] = {
            "buildCategory": final_category or "unclear",
            "visibleElements": visible_elements,
            "blockCountEstimate": clean_sentence(
                analysis_details.get("blockCountEstimate"),
                "unclear"
            ),
            "whyThisGuess": clean_sentence(
                analysis_details.get("whyThisGuess"),
                invalid_reason
            )
        }

        result["buildGuess"] = {
            "title": "We couldn’t clearly analyze this image",
            "subtitle": invalid_reason
        }

        result["whatWeFound"] = {
            "title": "What we found",
            "summary": invalid_reason
        }

        return jsonify(result), 200

    except Exception as e:
        error_text = str(e)
        print("Analyze error:", error_text)

        if is_rate_limit_error(error_text):
            return jsonify({
                "error": "Groq usage limit reached right now. Please wait and try again."
            }), 429

        if is_invalid_key_error(error_text):
            return jsonify({
                "error": "Groq API key is invalid or missing. Add a fresh GROQ_API_KEY in Render environment variables."
            }), 403

        if is_image_too_large_error(error_text):
            return jsonify({
                "error": "The image is too large for Groq vision. Please upload a smaller or clearer compressed image."
            }), 413

        if is_temporary_error(error_text):
            return jsonify({
                "error": "Groq AI service is temporarily unavailable. Please try again in a moment."
            }), 503

        return jsonify({
            "error": "Something went wrong",
            "details": error_text
        }), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json() or {}

        question = clean_sentence(data.get("question", ""))
        summary = clean_sentence(data.get("summary", ""))

        if not question:
            return jsonify({"error": "Question is required"}), 400

        client = build_groq_client()

        if not client:
            return jsonify({
                "error": "GROQ_API_KEY not found. Add it in Render environment variables."
            }), 500

        prompt = f"""
You are helping a parent understand their child's Troy block build.

Build summary:
{summary}

Parent question:
{question}

Answer in a short, warm, simple way for a parent.
Use only the build details provided.
Do not invent hidden abilities or unseen parts.
Keep it to 3 to 5 short lines.
"""

        completion = client.chat.completions.create(
            model=get_text_model_name(),
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
            temperature=0.45,
            top_p=0.9,
            max_completion_tokens=350
        )

        answer = completion.choices[0].message.content

        return jsonify({
            "answer": clean_sentence(
                answer,
                "I’m unable to answer that right now. Please try again."
            )
        }), 200

    except Exception as e:
        error_text = str(e)
        print("Ask error:", error_text)

        if is_rate_limit_error(error_text):
            return jsonify({
                "answer": "Groq usage limit reached right now. Please wait and try again."
            }), 200

        if is_invalid_key_error(error_text):
            return jsonify({
                "answer": "The Groq API key is invalid or missing. Please update it in Render environment variables."
            }), 200

        if is_temporary_error(error_text):
            return jsonify({
                "answer": "Live AI Q&A is temporarily unavailable right now. Please try again later."
            }), 200

        return jsonify({
            "error": "Something went wrong",
            "details": error_text
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    )