import os
import re
import json
import uuid
import time
import base64
import hashlib
import random
from io import BytesIO
from pathlib import Path
from collections import OrderedDict
from datetime import datetime

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge


# =========================================================
# App setup
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)

# Old image-size setting
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 MB

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
CORS(app, origins=CORS_ORIGINS.split(","))

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Old image-size setting
MAX_BASE64_IMAGE_SIZE = 1_800_000
Image.MAX_IMAGE_PIXELS = 15_000_000

analysis_cache = OrderedDict()
MAX_CACHE_ITEMS = 12

sessions = {}

ip_usage = {}
UPLOAD_LIMIT_PER_IP_PER_DAY = int(os.environ.get("UPLOAD_LIMIT_PER_IP_PER_DAY", "25"))
UPLOAD_COOLDOWN_SECONDS = int(os.environ.get("UPLOAD_COOLDOWN_SECONDS", "10"))

# Set true only if you want extra accuracy.
# It uses more free API quota because Gemini may be called after Groq returns invalid.
SECOND_OPINION_ON_INVALID = os.environ.get("SECOND_OPINION_ON_INVALID", "false").lower() == "true"


# =========================================================
# Strictness settings
# =========================================================

# Very strict pattern matching.
BOOK_MATCH_THRESHOLD = 93

# Creative guess must still be confident.
VALID_CONFIDENCE_THRESHOLD = 78

# Below this, always invalid.
ABSOLUTE_MIN_CONFIDENCE = 70


# =========================================================
# Troy pattern reference library
# =========================================================

TROY_PATTERN_LIBRARY = [
    {
        "id": "rocking_chair",
        "name": "Rocking Chair",
        "category": "motion_structures",
        "visual_cues": [
            "chair-like seat",
            "slanted backrest",
            "curved rocker base",
            "small cylinders below the seat",
            "two curved rails touching the ground"
        ],
        "description": "A chair-like build with a seat, backrest, and curved rocker rails."
    },
    {
        "id": "earthquake_resistant",
        "name": "Earthquake Resistant Structure",
        "category": "motion_structures",
        "visual_cues": [
            "multi-floor building",
            "rectangular levels",
            "small top roof piece",
            "curved rocker-like base",
            "building on movable supports"
        ],
        "description": "A multi-level building placed on rocker-like or movable supports."
    },
    {
        "id": "troy_pendulum",
        "name": "Troy Pendulum",
        "category": "motion_structures",
        "visual_cues": [
            "two vertical supports",
            "horizontal beam",
            "curved pieces below",
            "triangle piece on top",
            "balanced pendulum-like setup"
        ],
        "description": "A pendulum-style structure with supports, top beam, and curved parts."
    },
    {
        "id": "newtons_pet",
        "name": "Newton's Pet",
        "category": "motion_structures",
        "visual_cues": [
            "low vehicle-like base",
            "long rectangular top",
            "cylinders underneath",
            "curved bottom pieces",
            "pet-like moving form"
        ],
        "description": "A low rolling pet-like structure using cylinders and curved pieces."
    },
    {
        "id": "charminar",
        "name": "Charminar",
        "category": "monuments",
        "visual_cues": [
            "square monument base",
            "four corner towers",
            "four minarets",
            "central opening",
            "symmetrical layout"
        ],
        "description": "A monument-like structure with four corner pillars and a central area."
    },
    {
        "id": "qutub_minar",
        "name": "Qutub Minar",
        "category": "monuments",
        "visual_cues": [
            "very tall narrow tower",
            "stacked vertical blocks",
            "wide base",
            "tapering tower",
            "monument tower"
        ],
        "description": "A tall tapering tower with a wider bottom and narrow upper part."
    },
    {
        "id": "shinto_arch",
        "name": "Shinto Arch",
        "category": "places_of_worship",
        "visual_cues": [
            "two tall pillars",
            "horizontal beam across top",
            "gate-like structure",
            "open space below",
            "angled top blocks"
        ],
        "description": "A gate or arch-like structure with two supports and a strong top beam."
    },
    {
        "id": "pickup_truck",
        "name": "Pickup Truck",
        "category": "vehicles",
        "visual_cues": [
            "vehicle-like shape",
            "cylinder wheels",
            "flat truck bed",
            "raised cabin",
            "long rectangular base"
        ],
        "description": "A pickup truck-like build with a long base, raised section, and wheels."
    },
    {
        "id": "highway",
        "name": "Highway",
        "category": "vehicles",
        "visual_cues": [
            "long road-like path",
            "ramp pieces",
            "bridge supports",
            "extended roadway",
            "sign-like vertical blocks"
        ],
        "description": "A long road or highway scene with ramps, supports, and roadway sections."
    },
    {
        "id": "india_gate",
        "name": "India Gate",
        "category": "monuments",
        "visual_cues": [
            "large arch opening",
            "two tall side pillars",
            "rectangular gateway",
            "stepped top",
            "monument gate"
        ],
        "description": "A monument-style gateway with a central arch and side supports."
    },
    {
        "id": "golden_gate_bridge",
        "name": "Golden Gate Bridge",
        "category": "monuments",
        "visual_cues": [
            "long bridge",
            "two vertical tower frames",
            "roadway stretching across",
            "repeated supports",
            "bridge-like span"
        ],
        "description": "A bridge-like build with tower frames and a long connecting road section."
    },
    {
        "id": "eiffel_tower",
        "name": "Eiffel Tower",
        "category": "monuments",
        "visual_cues": [
            "tall tower",
            "four angled legs",
            "narrow top",
            "wide bottom supports",
            "tapering tower"
        ],
        "description": "A tall tower with angled supports and a narrow upper section."
    },
    {
        "id": "mosque",
        "name": "Mosque",
        "category": "places_of_worship",
        "visual_cues": [
            "dome-like curved piece",
            "arched entrance",
            "sloped ramp",
            "small minarets",
            "religious building form"
        ],
        "description": "A mosque-style structure with an arch, dome-like feature, and entrance."
    },
    {
        "id": "gurudwara",
        "name": "Gurudwara",
        "category": "places_of_worship",
        "visual_cues": [
            "dome-like top",
            "arched opening",
            "tall side pillars",
            "sloped path",
            "place of worship structure"
        ],
        "description": "A Gurudwara-like place of worship with arch, dome-like feature, and side elements."
    },
    {
        "id": "greek_temple",
        "name": "Greek Temple",
        "category": "places_of_worship",
        "visual_cues": [
            "front columns",
            "temple roof",
            "rectangular platform",
            "stair-like front",
            "classical temple shape"
        ],
        "description": "A temple-like build with columns, roof, and front platform."
    },
    {
        "id": "taj_mahal",
        "name": "Taj Mahal",
        "category": "monuments",
        "visual_cues": [
            "central dome",
            "arched doorway",
            "four minarets",
            "symmetrical monument",
            "large central building"
        ],
        "description": "A Taj Mahal-like monument with a central dome, arch, and surrounding minarets."
    },
    {
        "id": "train",
        "name": "Train",
        "category": "vehicles",
        "visual_cues": [
            "long vehicle body",
            "cylinder chimney",
            "connected wagons",
            "curved roof pieces",
            "engine-like front"
        ],
        "description": "A train-like build with a long body, chimney, and wagon-like sections."
    },
    {
        "id": "ship",
        "name": "Ship",
        "category": "vehicles",
        "visual_cues": [
            "long boat-like hull",
            "cabin block",
            "cylinder chimney",
            "pointed or sloped front",
            "ship-like body"
        ],
        "description": "A ship-like build with a long hull, cabin, and chimney."
    },
    {
        "id": "temple",
        "name": "Temple",
        "category": "places_of_worship",
        "visual_cues": [
            "front ramp",
            "pillars",
            "roof structure",
            "temple-like body",
            "raised entrance"
        ],
        "description": "A temple-like structure with a front ramp, pillars, and roofed body."
    }
]


# =========================================================
# Environment config
# =========================================================

def get_gemini_api_key():
    return os.environ.get("GEMINI_API_KEY", "").strip()


def get_gemini_model():
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()


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


def get_provider_order():
    return os.environ.get("PROVIDER_ORDER", "groq_first").strip().lower()


print("Gemini key:", "FOUND" if get_gemini_api_key() else "NOT FOUND")
print("Gemini model:", get_gemini_model())
print("Groq key:", "FOUND" if get_groq_api_key() else "NOT FOUND")
print("Groq vision model:", get_groq_vision_model())
print("Provider order:", get_provider_order())
print("Second opinion on invalid:", SECOND_OPINION_ON_INVALID)


# =========================================================
# Error handling
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
        or "timeout" in lower
    )


def get_client_ip():
    forwarded = request.headers.get("X-Forwarded-For", "")

    if forwarded:
        return forwarded.split(",")[0].strip()

    return request.remote_addr or "unknown"


def check_rate_limit():
    ip = get_client_ip()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    now = time.time()

    record = ip_usage.get(ip)

    if not record or record.get("day") != today:
        ip_usage[ip] = {
            "day": today,
            "count": 0,
            "last_upload": 0
        }
        record = ip_usage[ip]

    if now - record["last_upload"] < UPLOAD_COOLDOWN_SECONDS:
        wait = int(UPLOAD_COOLDOWN_SECONDS - (now - record["last_upload"]))
        return False, f"Please wait {wait} seconds before uploading another image."

    if record["count"] >= UPLOAD_LIMIT_PER_IP_PER_DAY:
        return False, "Daily free upload limit reached for this user. Please try again tomorrow."

    record["count"] += 1
    record["last_upload"] = now

    return True, ""


def is_unclear_text(text):
    lower = clean_text(text).lower()

    unclear_phrases = [
        "not clear",
        "unclear",
        "couldn't clearly",
        "could not clearly",
        "cannot clearly",
        "can't clearly",
        "unable to clearly",
        "too blurry",
        "too dark",
        "not enough visible",
        "not enough detail",
        "not clearly visible",
        "couldn’t clearly",
        "hard to tell",
        "difficult to tell",
        "difficult to identify",
        "hard to identify",
        "partially visible",
        "cropped",
        "hidden",
        "too far",
        "low quality",
        "poor lighting",
        "not enough evidence",
        "insufficient evidence",
        "not enough information",
        "cannot determine",
        "can't determine",
        "unable to determine",
        "ambiguous",
        "vague",
        "not identifiable",
        "not recognizable"
    ]

    return any(phrase in lower for phrase in unclear_phrases)


def is_weak_guess_text(text):
    lower = clean_text(text).lower()

    weak_guess_phrases = [
        "maybe",
        "possibly",
        "perhaps",
        "it is hard to say",
        "hard to say",
        "not sure",
        "uncertain",
        "unknown",
        "cannot tell",
        "can't tell",
        "not enough evidence",
        "open-ended build",
        "open-ended troy block arrangement",
        "abstract structure",
        "generic structure",
        "random structure",
        "simple block structure",
        "uncertain build"
    ]

    return any(phrase in lower for phrase in weak_guess_phrases)


def remove_page_words(text):
    text = clean_text(text)

    for number in range(1, 31):
        text = text.replace(f"page {number}", "the pattern reference")
        text = text.replace(f"Page {number}", "the pattern reference")

    replacements = {
        "from page": "from the pattern reference",
        "on page": "in the pattern reference",
        "book page": "pattern reference",
        "page number": "pattern reference"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return clean_text(text)


def clean_build_title(title, fallback="Troy Block Creation"):
    text = remove_page_words(clean_text(title, fallback))

    bad_suffix_patterns = [
        r"\s+style\s+build$",
        r"\s+style\s+structure$",
        r"\s+style\s+model$",
        r"\s+style$",
        r"\s+troy\s+block\s+build$",
        r"\s+block\s+build$",
        r"\s+build$",
        r"\s+model$",
        r"\s+structure$"
    ]

    for pattern in bad_suffix_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        text = fallback

    return text


def normalize_name(value):
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


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
    img = Image.open(image_file.stream)
    img = ImageOps.exif_transpose(img)

    original_width, original_height = img.size

    if original_width < 300 or original_height < 300:
        raise ValueError("Image resolution is too low. Please upload a clearer photo.")

    if img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail((900, 900))

    for quality in [80, 70, 60, 50, 40]:
        raw_bytes, encoded = encode_image_to_base64_jpeg(img, quality)

        if len(encoded.encode("utf-8")) <= MAX_BASE64_IMAGE_SIZE:
            image_hash = hashlib.sha256(raw_bytes).hexdigest()
            data_url = f"data:image/jpeg;base64,{encoded}"
            return encoded, data_url, image_hash

    img.thumbnail((700, 700))

    for quality in [60, 50, 40, 35]:
        raw_bytes, encoded = encode_image_to_base64_jpeg(img, quality)

        if len(encoded.encode("utf-8")) <= MAX_BASE64_IMAGE_SIZE:
            image_hash = hashlib.sha256(raw_bytes).hexdigest()
            data_url = f"data:image/jpeg;base64,{encoded}"
            return encoded, data_url, image_hash

    raise ValueError(
        "Image is too large even after compression. Please upload a smaller image."
    )


# =========================================================
# Prompt
# =========================================================

def compact_pattern_library_text():
    lines = []

    for pattern in TROY_PATTERN_LIBRARY:
        cue_text = ", ".join(pattern["visual_cues"])
        lines.append(
            f'- id="{pattern["id"]}", name="{pattern["name"]}", category="{pattern["category"]}" — '
            f'{pattern["description"]} Visible cues: {cue_text}.'
        )

    return "\n".join(lines)


def build_troy_prompt(age, image_hash):
    pattern_library = compact_pattern_library_text()

    return f"""
You are Troy AI Analyzer.

You are analyzing one uploaded image of a child's Troy wooden-block build.

Child age:
{age if age else "unknown"}

Known Troy pattern reference list:
{pattern_library}

Your job:
1. First check whether the image clearly matches a known Troy pattern.
2. If it exactly matches, use the known pattern name exactly.
3. If it does not exactly match, analyze the build creatively using only visible evidence.
4. If the image is unclear or the evidence is weak, mark it invalid.

STRICT BOOK MATCH RULE:
If the uploaded build is the same as a known pattern:
- matchType must be "book_pattern"
- matchConfidence must be 93 or above
- matchedPattern.id must exactly match the known pattern id
- matchedPattern.name must exactly match the known pattern name
- buildGuess.title must exactly equal the known pattern name
- do not rename it
- do not add "Style Build"
- do not create a new creative name
- do not mention page numbers

A book match needs:
- same overall silhouette
- same main block arrangement
- at least 3 matching visible cues from the known pattern
- strong visible evidence

If it shares only one or two small features with a known pattern:
- do not call it a book pattern
- use creative_guess only if the build is still clearly visible

CREATIVE GUESS RULE:
If it does not match the book:
- matchType must be "creative_guess"
- matchedPattern must be null
- confidenceScore must be 78 or above
- buildGuess.title must be a short object name only
- no "Style Build"
- no "Build", "Model", or "Structure" suffix
- do not use vague names like "Abstract Structure" or "Open-ended Troy Block Build"
- give the best accurate guess based on visible features

INVALID RULE:
Mark imageStatus as "invalid" if:
- Troy blocks are not clearly visible
- the photo is blurry, dark, cropped, too far, partially hidden, or low quality
- the build takes up only a small part of the image
- there is not enough visible evidence for a strong guess
- you would need to say maybe, possibly, perhaps, hard to tell, unclear, or not sure
- fewer than 2 visible construction features can be identified

Visible construction features include:
base, supports, wheels/cylinders, levels/floors, bridge span, arch/opening, roof-like top, repeated blocks, symmetry, curved pieces, long body, cabin, tower, ramp, path, platform.

Important wording rules:
- Never mention page numbers.
- Do not say "page", "book page", or "from page".
- Do not use "Style Build".
- If the guess is Aeroplane, title should be exactly "Aeroplane".
- If the guess is Pickup Truck, title should be exactly "Pickup Truck".
- If the guess is Temple, title should be exactly "Temple".

Learning feedback rules:
- Every learning card must mention visible details.
- Do not use generic titles.
- Do not use: Creativity, Problem-Solving, Problem Solving, Spatial Awareness, Spatial Thinking, Imagination, Motor Skills, Fine Motor Skills, Engineering, STEM Learning, Critical Thinking.
- Use specific titles like: Layer Planning, Bridge Support, Moving Base Idea, Roof Shape Experiment, Open-Space Design, Block Pattern Play, Careful Stacking, Shape Combining, Build-and-Tell Practice, Support Below Space Above, Vehicle Shape Thinking, Room-Making, Testing What Holds.

Return JSON only in this exact shape:

{{
  "status": "success",
  "imageStatus": "valid or invalid",
  "confidenceScore": 0,
  "matchType": "book_pattern or creative_guess or invalid",
  "matchedPattern": {{
    "id": "pattern id or null",
    "name": "pattern name or null",
    "category": "pattern category or null",
    "matchConfidence": 0,
    "whyMatched": "short visual reason or null"
  }},
  "buildGuess": {{
    "title": "short exact title only",
    "subtitle": "short reason based only on visible details"
  }},
  "whatWeFound": {{
    "title": "What we found",
    "summary": "2 short sentences describing only visible details"
  }},
  "whatTheyLearned": [
    {{
      "title": "specific learning skill title",
      "description": "specific explanation connected to visible details",
      "color": "cream"
    }},
    {{
      "title": "specific learning skill title",
      "description": "specific explanation connected to visible details",
      "color": "green"
    }},
    {{
      "title": "specific learning skill title",
      "description": "specific explanation connected to visible details",
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

Invalid output rules:
- imageStatus must be "invalid"
- confidenceScore must be below 70
- matchType must be "invalid"
- matchedPattern must contain null values
- buildGuess.title must be exactly "We couldn’t clearly analyze this image"
- whatTheyLearned must be []
"""


# =========================================================
# Feedback fallback helpers
# =========================================================

def contains_any(text, words):
    text = text.lower()
    return any(word in text for word in words)


def build_context_text(build_guess, summary, noticed, matched_pattern=None):
    pattern_text = ""

    if matched_pattern and isinstance(matched_pattern, dict):
        pattern_text = " ".join([
            clean_text(matched_pattern.get("name", "")),
            clean_text(matched_pattern.get("whyMatched", "")),
            clean_text(matched_pattern.get("category", ""))
        ])

    return " ".join([
        clean_text(build_guess.get("title", "")),
        clean_text(build_guess.get("subtitle", "")),
        clean_text(summary),
        " ".join(noticed or []),
        pattern_text
    ]).lower()


def creative_fallback_cards(build_guess, summary, noticed, image_hash, matched_pattern=None):
    context = build_context_text(build_guess, summary, noticed, matched_pattern)
    main_detail = noticed[0] if noticed else "the visible block arrangement"

    card_pool = []

    if contains_any(context, ["level", "floor", "platform", "layer", "upper", "lower", "multi-level"]):
        card_pool.extend([
            {
                "title": "Layer Planning",
                "description": f"The child explored how one section can sit above another, especially around {main_detail}.",
                "color": "cream"
            },
            {
                "title": "Support Below Space Above",
                "description": "The child practiced thinking about how lower blocks can support upper parts.",
                "color": "green"
            }
        ])

    if contains_any(context, ["wheel", "vehicle", "moving", "car", "base", "travel", "rolling", "truck", "train", "ship", "aeroplane", "airplane", "plane"]):
        card_pool.extend([
            {
                "title": "Moving Base Idea",
                "description": "The child connected the block arrangement with the idea of movement or travel.",
                "color": "cream"
            },
            {
                "title": "Vehicle Shape Thinking",
                "description": "The child used visible parts like a base, body, or direction to suggest a moving object.",
                "color": "green"
            }
        ])

    if contains_any(context, ["house", "home", "room", "roof", "door", "window", "shelter"]):
        card_pool.extend([
            {
                "title": "Room-Making",
                "description": "The child used blocks to suggest an inside-outside space or small shelter.",
                "color": "cream"
            },
            {
                "title": "Roof Shape Experiment",
                "description": "The top section helps the child explore how a build can feel like a roof or room.",
                "color": "green"
            }
        ])

    if contains_any(context, ["bridge", "gap", "span", "across", "support", "beam", "highway"]):
        card_pool.extend([
            {
                "title": "Bridge Support",
                "description": "The child explored how blocks can stretch across a space while still needing support.",
                "color": "cream"
            },
            {
                "title": "Testing What Holds",
                "description": "The bridge-like arrangement helps the child notice which parts stay steady.",
                "color": "green"
            }
        ])

    if contains_any(context, ["gate", "arch", "opening", "entrance", "tunnel", "curve"]):
        card_pool.extend([
            {
                "title": "Open-Space Design",
                "description": "The child explored how blocks can create an entrance, opening, or pass-through space.",
                "color": "cream"
            },
            {
                "title": "Curve and Shape Play",
                "description": "The child experimented with how curved or open shapes change the build.",
                "color": "green"
            }
        ])

    if contains_any(context, ["repeat", "repeated", "same", "pattern", "symmetry", "line", "row", "monument"]):
        card_pool.extend([
            {
                "title": "Block Pattern Play",
                "description": "The child used repeated placement to make the build feel organized.",
                "color": "cream"
            },
            {
                "title": "Matching and Repeating",
                "description": "The child practiced noticing which blocks look similar and placing them together.",
                "color": "green"
            }
        ])

    if contains_any(context, ["tower", "stack", "tall", "height", "vertical", "qutub", "eiffel", "minar"]):
        card_pool.extend([
            {
                "title": "Careful Stacking",
                "description": f"The child practiced placing pieces upward while keeping the structure steady near {main_detail}.",
                "color": "cream"
            },
            {
                "title": "Height Control",
                "description": "The child explored how a build changes when blocks are placed higher.",
                "color": "green"
            }
        ])

    if matched_pattern and matched_pattern.get("name"):
        card_pool.append({
            "title": "Exact Pattern Recreation",
            "description": f"The child recreated the {matched_pattern.get('name')} pattern using matching visible shape and block arrangement.",
            "color": "cream"
        })

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

    return any(phrase in description for phrase in generic_phrases)


# =========================================================
# Response normalization
# =========================================================

def find_pattern_by_id(pattern_id):
    for pattern in TROY_PATTERN_LIBRARY:
        if pattern["id"] == pattern_id:
            return pattern

    return None


def find_pattern_by_name(pattern_name):
    wanted = normalize_name(pattern_name)

    if not wanted:
        return None

    for pattern in TROY_PATTERN_LIBRARY:
        if normalize_name(pattern["name"]) == wanted:
            return pattern

    return None


def normalize_matched_pattern(raw_matched_pattern, match_type):
    if not isinstance(raw_matched_pattern, dict):
        return None

    pattern_id = clean_text(raw_matched_pattern.get("id"))
    pattern_name = clean_text(raw_matched_pattern.get("name"))
    why_matched = clean_text(raw_matched_pattern.get("whyMatched"))

    try:
        match_confidence = int(float(raw_matched_pattern.get("matchConfidence", 0)))
    except Exception:
        match_confidence = 0

    library_pattern = find_pattern_by_id(pattern_id)

    if not library_pattern:
        library_pattern = find_pattern_by_name(pattern_name)

    if not library_pattern:
        return None

    if match_type != "book_pattern":
        return None

    if match_confidence < BOOK_MATCH_THRESHOLD:
        return None

    return {
        "id": library_pattern["id"],
        "name": library_pattern["name"],
        "category": library_pattern["category"],
        "matchConfidence": match_confidence,
        "whyMatched": remove_page_words(
            why_matched or "the visible structure matches the main shape and block arrangement"
        )
    }


def normalize_learning_cards(cards, build_guess, summary, noticed, image_hash, matched_pattern=None):
    allowed_colors = ["cream", "green", "blue"]
    cleaned = []

    if isinstance(cards, list):
        for index, card in enumerate(cards):
            if not isinstance(card, dict):
                continue

            title = clean_text(card.get("title"))
            description = remove_page_words(clean_text(card.get("description")))
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

    fallback_cards = creative_fallback_cards(
        build_guess,
        summary,
        noticed,
        image_hash,
        matched_pattern=matched_pattern
    )

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


def build_invalid_response(summary=None):
    reason = clean_text(
        summary,
        "The image is not clear enough to confidently analyze the Troy block build."
    )

    return {
        "status": "success",
        "imageStatus": "invalid",
        "confidenceScore": 0,
        "matchType": "invalid",
        "matchedPattern": None,
        "buildGuess": {
            "title": "We couldn’t clearly analyze this image",
            "subtitle": reason
        },
        "whatWeFound": {
            "title": "What we found",
            "summary": reason
        },
        "whatTheyLearned": [],
        "whatWeNoticed": [
            "The build is not clear enough in the photo.",
            "Some important parts may be blurry, cropped, too far away, or hidden.",
            "A clearer photo will help the analyzer give more accurate feedback."
        ],
        "suggestionsForParent": [
            "Retake the photo with the full build visible.",
            "Use better lighting and place the build on a plain surface.",
            "Take the photo closer, but make sure the entire structure is inside the frame."
        ],
        "nextBuildIdeas": [
            "Try taking one front-view photo of the same build.",
            "Ask your child to point out the main part of the build before retaking the photo.",
            "Try rebuilding the structure and taking a clearer photo."
        ],
        "session_id": str(uuid.uuid4())
    }


def normalize_analysis_response(parsed, image_hash):
    image_status = clean_text(parsed.get("imageStatus", "invalid")).lower()

    try:
        confidence = int(float(parsed.get("confidenceScore", 0)))
    except Exception:
        confidence = 0

    build_guess = safe_get_dict(parsed, "buildGuess")
    what_found = safe_get_dict(parsed, "whatWeFound")

    raw_title = clean_text(build_guess.get("title"))
    raw_subtitle = remove_page_words(clean_text(build_guess.get("subtitle")))
    raw_summary = remove_page_words(clean_text(what_found.get("summary")))

    combined_main_text = " ".join([raw_title, raw_subtitle, raw_summary])

    if (
        image_status != "valid"
        or confidence < ABSOLUTE_MIN_CONFIDENCE
        or is_unclear_text(combined_main_text)
        or is_weak_guess_text(combined_main_text)
    ):
        return build_invalid_response(raw_summary or raw_subtitle)

    raw_match_type = clean_text(parsed.get("matchType", "creative_guess")).lower()
    raw_matched_pattern = parsed.get("matchedPattern")

    final_match_type = "book_pattern" if raw_match_type == "book_pattern" else "creative_guess"
    matched_pattern = normalize_matched_pattern(raw_matched_pattern, final_match_type)

    if matched_pattern:
        final_match_type = "book_pattern"
    else:
        final_match_type = "creative_guess"

    if not matched_pattern and confidence < VALID_CONFIDENCE_THRESHOLD:
        return build_invalid_response(raw_summary or raw_subtitle)

    noticed = ensure_list(
        parsed.get("whatWeNoticed"),
        [
            "The build shows visible blocks arranged into a structure.",
            "The child used block placement to create a shape or idea.",
            "The structure has details that can be discussed with the child."
        ],
        limit=3
    )

    noticed = [remove_page_words(item) for item in noticed]

    if matched_pattern:
        normalized_build_guess = {
            "title": matched_pattern["name"],
            "subtitle": (
                f"This matches the {matched_pattern['name']} because "
                f"{matched_pattern['whyMatched']}."
            )
        }
    else:
        cleaned_title = clean_build_title(raw_title)

        if is_weak_guess_text(cleaned_title):
            return build_invalid_response(raw_summary or raw_subtitle)

        if len(cleaned_title.split()) > 6:
            return build_invalid_response(raw_summary or raw_subtitle)

        normalized_build_guess = {
            "title": cleaned_title,
            "subtitle": raw_subtitle or "The visible block arrangement supports this guess."
        }

    normalized_summary = raw_summary or "The image shows a child-made block structure with visible block placement."

    if is_unclear_text(normalized_summary) or is_weak_guess_text(normalized_summary):
        return build_invalid_response(normalized_summary)

    result = {
        "status": "success",
        "imageStatus": "valid",
        "confidenceScore": confidence,
        "matchType": final_match_type,
        "matchedPattern": matched_pattern,
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
            image_hash,
            matched_pattern=matched_pattern
        ),
        "whatWeNoticed": noticed,
        "suggestionsForParent": [
            remove_page_words(item) for item in ensure_list(
                parsed.get("suggestionsForParent"),
                [
                    "Ask your child what each part of the build represents.",
                    "Invite your child to add one new detail to the build.",
                    "Take another photo after your child improves or changes the structure."
                ],
                limit=3
            )
        ],
        "nextBuildIdeas": [
            remove_page_words(item) for item in ensure_list(
                parsed.get("nextBuildIdeas"),
                [
                    "Build a version with one extra level or section.",
                    "Add a path, door, bridge, or moving part.",
                    "Try rebuilding the same idea using fewer blocks."
                ],
                limit=3
            )
        ],
        "session_id": str(uuid.uuid4())
    }

    return result


# =========================================================
# AI provider calls using lightweight REST
# =========================================================

def analyze_with_gemini_rest(image_base64, age, image_hash):
    api_key = get_gemini_api_key()

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found")

    model = get_gemini_model()
    prompt = build_troy_prompt(age, image_hash)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.35,
            "topP": 0.8,
            "maxOutputTokens": 1500,
            "responseMimeType": "application/json"
        }
    }

    response = requests.post(url, json=payload, timeout=18)

    if response.status_code >= 400:
        raise RuntimeError(f"Gemini error {response.status_code}: {response.text[:800]}")

    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]

    return parse_json_response(text)


def analyze_with_groq_rest(image_data_url, age, image_hash):
    api_key = get_groq_api_key()

    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found")

    prompt = build_troy_prompt(age, image_hash)

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": get_groq_vision_model(),
        "messages": [
            {
                "role": "system",
                "content": "You are a strict but fair visual analyzer. Return valid JSON only."
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
        "temperature": 0.35,
        "top_p": 0.8,
        "max_completion_tokens": 1500,
        "response_format": {
            "type": "json_object"
        }
    }

    response = requests.post(url, headers=headers, json=payload, timeout=55)

    if response.status_code >= 400:
        raise RuntimeError(f"Groq error {response.status_code}: {response.text[:800]}")

    data = response.json()
    text = data["choices"][0]["message"]["content"]

    return parse_json_response(text)


# =========================================================
# Main fallback logic
# =========================================================

def analyze_image_with_fallback(image_base64, image_data_url, age, image_hash):
    errors = []
    provider_order = get_provider_order()

    if provider_order == "gemini_then_groq":
        providers = ["gemini", "groq"]
    else:
        providers = ["groq", "gemini"]

    best_invalid_result = None

    for provider in providers:
        if provider == "groq":
            try:
                print("Trying Groq...")
                parsed = analyze_with_groq_rest(image_data_url, age, image_hash)
                result = normalize_analysis_response(parsed, image_hash)
                result["provider"] = "groq"

                if result["imageStatus"] == "invalid" and SECOND_OPINION_ON_INVALID:
                    print("Groq returned invalid. Trying second opinion if available...")
                    best_invalid_result = result
                    continue

                print("Groq successful")
                return result

            except Exception as e:
                error_text = str(e)
                print("Groq failed:", error_text)
                errors.append(f"Groq: {error_text}")

        if provider == "gemini":
            if not get_gemini_api_key():
                print("Skipping Gemini because key is missing")
                continue

            try:
                print("Trying Gemini...")
                parsed = analyze_with_gemini_rest(image_base64, age, image_hash)
                result = normalize_analysis_response(parsed, image_hash)
                result["provider"] = "gemini"

                if result["imageStatus"] == "invalid" and SECOND_OPINION_ON_INVALID:
                    print("Gemini also returned invalid.")
                    if best_invalid_result is None:
                        best_invalid_result = result
                    continue

                print("Gemini successful")
                return result

            except Exception as e:
                error_text = str(e)
                print("Gemini failed:", error_text)
                errors.append(f"Gemini: {error_text}")

    if best_invalid_result:
        return best_invalid_result

    raise RuntimeError("All AI providers failed. " + " | ".join(errors))


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
        "groq_vision_model": get_groq_vision_model(),
        "provider_order": get_provider_order(),
        "second_opinion_on_invalid": SECOND_OPINION_ON_INVALID,
        "pattern_count": len(TROY_PATTERN_LIBRARY),
        "book_match_threshold": BOOK_MATCH_THRESHOLD,
        "valid_confidence_threshold": VALID_CONFIDENCE_THRESHOLD,
        "absolute_min_confidence": ABSOLUTE_MIN_CONFIDENCE,
        "cache_items": len(analysis_cache),
        "upload_limit_per_ip_per_day": UPLOAD_LIMIT_PER_IP_PER_DAY,
        "upload_cooldown_seconds": UPLOAD_COOLDOWN_SECONDS
    })


@app.route("/patterns", methods=["GET"])
def patterns():
    return jsonify({
        "count": len(TROY_PATTERN_LIBRARY),
        "patterns": TROY_PATTERN_LIBRARY
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        allowed, message = check_rate_limit()

        if not allowed:
            return jsonify({"error": message}), 429

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
            image_base64, image_data_url, image_hash = prepare_image_for_models(image_file)
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

        result = analyze_image_with_fallback(image_base64, image_data_url, age, image_hash)
        result["cached"] = False

        if os.environ.get("SHOW_DEBUG", "false").lower() == "true":
            result["debug"] = {
                "filename": filename,
                "image_hash": image_hash[:12],
                "gemini_model": get_gemini_model(),
                "groq_model": get_groq_vision_model(),
                "pattern_count": len(TROY_PATTERN_LIBRARY),
                "strict_mode": True
            }

        save_cache(cache_key, result)
        sessions[result["session_id"]] = result

        if len(sessions) > 30:
            oldest_key = next(iter(sessions))
            sessions.pop(oldest_key, None)

        return jsonify(result), 200

    except Exception as e:
        error_text = str(e)
        print("Analyze error:", error_text)

        if is_rate_limit_error(error_text):
            return jsonify({
                "error": "AI free usage limit reached right now. Please wait and try again."
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

        api_key = get_groq_api_key()

        if not api_key:
            return jsonify({
                "answer": "Follow-up chat is unavailable because the Groq key is missing."
            }), 200

        prompt = f"""
You are helping a parent understand their child's Troy block build.

Build summary:
{summary}

Parent question:
{question}

Answer in a short, warm, realistic way.
Use only the build details provided.
Do not invent hidden abilities or unseen parts.
Do not mention page numbers.
Do not use titles like Style Build.
Keep it to 3 to 5 short lines.
"""

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": get_groq_text_model(),
            "messages": [
                {
                    "role": "system",
                    "content": "You are a warm but strict parent-friendly assistant for Troy World."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.35,
            "top_p": 0.8,
            "max_completion_tokens": 250
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code >= 400:
            raise RuntimeError(response.text[:800])

        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        return jsonify({
            "answer": remove_page_words(
                clean_text(answer, "I’m unable to answer that right now. Please try again.")
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