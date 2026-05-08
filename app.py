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
from collections import OrderedDict, Counter
from datetime import datetime

import requests
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps, ImageDraw, ImageFont
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge


# =========================================================
# App setup
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"
ANALYTICS_FILE = DATA_DIR / "analytics.jsonl"

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 MB

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
CORS(app, origins=CORS_ORIGINS.split(","))

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

MAX_BASE64_IMAGE_SIZE = 1_800_000
Image.MAX_IMAGE_PIXELS = 15_000_000

analysis_cache = OrderedDict()
MAX_CACHE_ITEMS = 12

sessions = {}

ip_usage = {}
UPLOAD_LIMIT_PER_IP_PER_DAY = int(os.environ.get("UPLOAD_LIMIT_PER_IP_PER_DAY", "25"))
UPLOAD_COOLDOWN_SECONDS = int(os.environ.get("UPLOAD_COOLDOWN_SECONDS", "10"))

SECOND_OPINION_ON_INVALID = os.environ.get("SECOND_OPINION_ON_INVALID", "false").lower() == "true"


# =========================================================
# Strictness settings
# =========================================================

BOOK_MATCH_THRESHOLD = 94
VALID_CONFIDENCE_THRESHOLD = 72
ABSOLUTE_MIN_CONFIDENCE = 60


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


def get_admin_key():
    return os.environ.get("ADMIN_KEY", "").strip()


print("Gemini key:", "FOUND" if get_gemini_api_key() else "NOT FOUND")
print("Gemini model:", get_gemini_model())
print("Groq key:", "FOUND" if get_groq_api_key() else "NOT FOUND")
print("Groq vision model:", get_groq_vision_model())
print("Provider order:", get_provider_order())
print("Second opinion on invalid:", SECOND_OPINION_ON_INVALID)
print("Admin key:", "FOUND" if get_admin_key() else "NOT FOUND")


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


def append_jsonl(file_path, data):
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_jsonl(file_path, limit=None):
    if not file_path.exists():
        return []

    rows = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    if limit:
        return rows[-limit:]

    return rows


def require_admin():
    admin_key = get_admin_key()

    if not admin_key:
        return False

    provided_key = request.headers.get("X-Admin-Key", "").strip()
    return provided_key == admin_key


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


def clamp_score(value):
    try:
        score = int(float(value))
    except Exception:
        score = 1

    return max(1, min(5, score))


def meter_item(value, fallback_score=1, fallback_reason="Not enough visible evidence to score strongly."):
    if isinstance(value, dict):
        score = clamp_score(value.get("score", fallback_score))
        reason = clean_text(value.get("reason"), fallback_reason)
    else:
        score = clamp_score(value or fallback_score)
        reason = fallback_reason

    return {
        "score": score,
        "reason": reason
    }


def default_troy_thinking_meter(is_invalid=False):
    if is_invalid:
        reason = "The image needs to be clearer before this skill can be judged accurately."

        return {
            "symmetry": {"score": 1, "reason": reason},
            "creativity": {"score": 1, "reason": reason},
            "spatialSkills": {"score": 1, "reason": reason},
            "stability": {"score": 1, "reason": reason},
            "problemSolving": {"score": 1, "reason": reason},
            "focusAndDetail": {"score": 1, "reason": reason}
        }

    return {
        "symmetry": {
            "score": 3,
            "reason": "The build shows some organized placement, but symmetry depends on the visible left and right balance."
        },
        "creativity": {
            "score": 4,
            "reason": "The build appears to represent an idea rather than random stacking."
        },
        "spatialSkills": {
            "score": 3,
            "reason": "The child used visible space and block positioning to form a structure."
        },
        "stability": {
            "score": 3,
            "reason": "The build appears reasonably supported from the visible base and vertical parts."
        },
        "problemSolving": {
            "score": 3,
            "reason": "The child made placement choices to connect and support parts of the build."
        },
        "focusAndDetail": {
            "score": 3,
            "reason": "The build shows intentional placement, though more detail could make it stronger."
        }
    }


def normalize_troy_thinking_meter(raw_meter, is_invalid=False):
    if is_invalid:
        return default_troy_thinking_meter(is_invalid=True)

    if not isinstance(raw_meter, dict):
        return default_troy_thinking_meter(is_invalid=False)

    return {
        "symmetry": meter_item(
            raw_meter.get("symmetry"),
            fallback_score=3,
            fallback_reason="The build shows some balance, but perfect mirroring is not fully clear."
        ),
        "creativity": meter_item(
            raw_meter.get("creativity"),
            fallback_score=4,
            fallback_reason="The build shows an original idea or recognizable object."
        ),
        "spatialSkills": meter_item(
            raw_meter.get("spatialSkills"),
            fallback_score=3,
            fallback_reason="The child used visible block placement and space to form the build."
        ),
        "stability": meter_item(
            raw_meter.get("stability"),
            fallback_score=3,
            fallback_reason="The structure appears reasonably supported based on the visible base."
        ),
        "problemSolving": meter_item(
            raw_meter.get("problemSolving"),
            fallback_score=3,
            fallback_reason="The child solved simple placement and support challenges."
        ),
        "focusAndDetail": meter_item(
            raw_meter.get("focusAndDetail"),
            fallback_score=3,
            fallback_reason="The build shows intentional placement and some detail."
        )
    }


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


def clean_build_title(title, fallback="Troy Creation"):
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


def save_analysis_image(image_base64, image_hash):
    try:
        image_bytes = base64.b64decode(image_base64)
        image_path = IMAGES_DIR / f"{image_hash}.jpg"

        with open(image_path, "wb") as file:
            file.write(image_bytes)

        return image_hash

    except Exception as e:
        print("Could not save image:", str(e))
        return ""


def load_analysis_image(image_id):
    if not image_id:
        return None

    image_path = IMAGES_DIR / f"{image_id}.jpg"

    if not image_path.exists():
        return None

    try:
        return image_path.read_bytes()
    except Exception:
        return None


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
2. If it strongly matches a known pattern, use the known pattern name exactly.
3. If it does not strongly match the book, treat it as the child's own creative build.
4. For creative builds, give a fun, accurate, specific guess based on visible block features.
5. Score the child's build using the Troy Thinking Meter.
6. If the image is unclear or the build is not visible enough, mark it invalid.

STRICT BOOK MATCH RULE:
Use book_pattern only when the uploaded build is clearly the same as a known Troy pattern.

For book_pattern:
- matchType must be "book_pattern"
- matchConfidence must be 94 or above
- matchedPattern.id must exactly match the known pattern id
- matchedPattern.name must exactly match the known pattern name
- buildGuess.title must exactly equal the known pattern name
- do not rename it
- do not add "Style Build"
- do not create a new creative name
- do not mention page numbers
- matchedCues must contain at least 3 visible cues
- each cue must be directly visible in the uploaded photo

A book match needs:
- same overall silhouette
- same main block arrangement
- at least 3 matching visible cues from the known pattern
- strong visible evidence

If it shares only one or two small features with a known pattern:
- do not call it a book pattern
- use creative_guess

CREATIVE BUILD RULE:
If the child built something of their own:
- matchType must be "creative_guess"
- matchedPattern must be null
- confidenceScore should be 72 or above if the build is clear
- give the best object guess based on visible features
- the title must be short and natural
- examples: "Giraffe", "Rocket", "Crane", "Robot", "Animal House", "Tall Watchtower", "Playground Slide", "Castle", "Fort", "Boat"
- do not use dry titles like "Block Arrangement", "Troy Block Creation", "Abstract Structure", or "Open-ended Build"
- do not add "Style Build", "Build", "Model", or "Structure" at the end
- explain why you guessed it using visible details
- make the feedback feel personal to this exact creation

For animals or characters:
- mention visible body parts like head, neck, legs, body, tail, or face if visible
- learning cards should focus on body planning, character making, balance, storytelling, and shape mapping

For vehicles:
- mention base, wheels, body, front/back, cabin, or direction if visible

For buildings, castles, forts, or houses:
- mention floors, roof, entrance, pillars, levels, supports, towers, walls, base, or rooms if visible

For bridges/roads:
- mention span, path, gap, supports, ramps, or direction if visible

INVALID RULE:
Mark imageStatus as "invalid" only if:
- Troy blocks are not clearly visible
- the photo is blurry, dark, cropped, too far, partially hidden, or low quality
- the build takes up only a very small part of the image
- there is not enough visible evidence for any reasonable guess
- fewer than 2 visible construction features can be identified

Visible construction features include:
base, supports, wheels/cylinders, levels/floors, bridge span, arch/opening, roof-like top, repeated blocks, symmetry, curved pieces, long body, cabin, tower, ramp, path, platform, head, neck, legs, body, tail.

TROY THINKING METER:
Score each parameter from 1 to 5.
Return a short reason for each score.
Use only visible evidence from the uploaded build.

1. Symmetry:
"Look at the structure — are both sides balanced and mirrored? Rate 1-5 where 1 = completely random placement, 5 = both sides perfectly balanced."

2. Creativity:
"How imaginative and unique is this build? Does it represent something recognizable or show original thinking? Rate 1-5 where 1 = random stacking, 5 = highly creative recognizable structure."

3. Spatial Skills:
"How well are the blocks arranged in 3D space? Is there depth, layering, good use of space? Rate 1-5 where 1 = flat single layer, 5 = complex multi-level spatial arrangement."

4. Stability:
"Does the structure look physically stable and well-balanced? Rate 1-5 where 1 = looks like it would fall immediately, 5 = solid and well-engineered."

5. Problem Solving:
"Did the child solve structural challenges like bridging gaps, supporting weight, or creating height? Rate 1-5 where 1 = simple pile, 5 = clearly solved complex building challenges."

6. Focus & Detail:
"How complete and detailed is the build? Are edges neat, pieces aligned? Rate 1-5 where 1 = rough incomplete, 5 = neat detailed and complete."

Important scoring rules:
- Do not give all 5s unless the build is genuinely excellent.
- Do not give all 3s lazily.
- Creativity can be high even if symmetry is low.
- Stability can be high even if creativity is medium.
- Spatial Skills should be higher when there are layers, levels, height, depth, gaps, bridges, or 3D arrangement.
- Problem Solving should be higher when the child solved support, height, balancing, bridge, gap, or multi-part construction.
- Focus & Detail should be higher when the build looks complete, aligned, intentional, and neat.
- If imageStatus is invalid, all Troy Thinking Meter scores should be 1 with reasons saying the photo needs to be clearer.

Important wording rules:
- Never mention page numbers.
- Do not say "page", "book page", or "from page".
- Do not use "Style Build".
- If the guess is Giraffe, title should be exactly "Giraffe".
- If the guess is Aeroplane, title should be exactly "Aeroplane".
- If the guess is Pickup Truck, title should be exactly "Pickup Truck".
- If the guess is Temple, title should be exactly "Temple".
- If the guess is Castle, title should be exactly "Castle".

Learning feedback rules:
- Every learning card must mention visible details.
- Make the learning feedback specific to the child's creation.
- Do not give the same generic feedback repeatedly.
- Do not use generic titles.
- Do not use: Creativity, Problem-Solving, Problem Solving, Spatial Awareness, Spatial Thinking, Imagination, Motor Skills, Fine Motor Skills, Engineering, STEM Learning, Critical Thinking.
- Use specific titles like: Animal Body Planning, Long-Neck Balance, Character-Making, Shape Mapping, Castle Planning, Symmetry in Building, Layered Construction, Architectural Details, Bridge Support, Moving Base Idea, Roof Shape Experiment, Open-Space Design, Block Pattern Play, Careful Stacking, Shape Combining, Build-and-Tell Practice, Support Below Space Above, Vehicle Shape Thinking, Room-Making, Testing What Holds.

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
    "whyMatched": "short visual reason or null",
    "matchedCues": [
      "visible cue 1",
      "visible cue 2",
      "visible cue 3"
    ]
  }},
  "buildGuess": {{
    "title": "short exact title only",
    "subtitle": "short reason based only on visible details"
  }},
  "whatWeFound": {{
    "title": "What we found",
    "summary": "2 short sentences describing only visible details"
  }},
  "troyThinkingMeter": {{
    "symmetry": {{
      "score": 1,
      "reason": "short visible reason"
    }},
    "creativity": {{
      "score": 1,
      "reason": "short visible reason"
    }},
    "spatialSkills": {{
      "score": 1,
      "reason": "short visible reason"
    }},
    "stability": {{
      "score": 1,
      "reason": "short visible reason"
    }},
    "problemSolving": {{
      "score": 1,
      "reason": "short visible reason"
    }},
    "focusAndDetail": {{
      "score": 1,
      "reason": "short visible reason"
    }}
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
- confidenceScore must be below 60
- matchType must be "invalid"
- matchedPattern must contain null values
- buildGuess.title must be exactly "We couldn’t clearly analyze this image"
- whatTheyLearned must be []
- all troyThinkingMeter scores must be 1
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

    if contains_any(context, ["castle", "fort", "tower", "wall", "gate", "central tower", "side towers"]):
        card_pool.extend([
            {
                "title": "Castle Planning",
                "description": f"The child arranged blocks into a castle-like form, especially around {main_detail}.",
                "color": "cream"
            },
            {
                "title": "Symmetry in Building",
                "description": "The child used repeated or balanced parts to make the build feel organized and castle-like.",
                "color": "green"
            },
            {
                "title": "Architectural Details",
                "description": "The child used details such as towers, roofs, openings, or side parts to suggest a larger place.",
                "color": "blue"
            }
        ])

    if contains_any(context, ["animal", "giraffe", "neck", "legs", "head", "body", "tail", "creature", "dog", "cat", "horse", "dinosaur", "bird"]):
        card_pool.extend([
            {
                "title": "Animal Body Planning",
                "description": f"The child mapped simple blocks into animal-like parts, especially around {main_detail}.",
                "color": "cream"
            },
            {
                "title": "Long-Neck Balance",
                "description": "The child experimented with making a taller neck or upper section while keeping the body steady.",
                "color": "green"
            },
            {
                "title": "Character-Making",
                "description": "The build shows the child turning block shapes into a character that can be named and described.",
                "color": "blue"
            }
        ])

    if contains_any(context, ["level", "floor", "platform", "layer", "upper", "lower", "multi-level"]):
        card_pool.extend([
            {
                "title": "Layered Construction",
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
            "title": "Shape Mapping",
            "description": f"The child connected visible block positions to a meaningful form, especially around {main_detail}.",
            "color": "cream"
        },
        {
            "title": "Block Decision-Making",
            "description": f"The child made choices about where to place pieces, especially around {main_detail}.",
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

    matched_cues = ensure_list(
        raw_matched_pattern.get("matchedCues"),
        [],
        limit=5
    )

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

    if len(matched_cues) < 3:
        return None

    return {
        "id": library_pattern["id"],
        "name": library_pattern["name"],
        "category": library_pattern["category"],
        "matchConfidence": match_confidence,
        "whyMatched": remove_page_words(
            why_matched or "the visible structure matches the main shape and block arrangement"
        ),
        "matchedCues": [remove_page_words(cue) for cue in matched_cues]
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
        "troyThinkingMeter": default_troy_thinking_meter(is_invalid=True),
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
        "troyThinkingMeter": normalize_troy_thinking_meter(
            parsed.get("troyThinkingMeter"),
            is_invalid=False
        ),
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
# Analytics helpers
# =========================================================

def save_analysis_event(result, filename="", image_hash="", cached=False):
    meter = result.get("troyThinkingMeter") or {}

    event = {
        "created_at": datetime.utcnow().isoformat(),
        "session_id": result.get("session_id"),
        "image_status": result.get("imageStatus"),
        "match_type": result.get("matchType"),
        "guess": result.get("buildGuess", {}).get("title"),
        "confidence_score": result.get("confidenceScore"),
        "provider": result.get("provider"),
        "cached": cached,
        "filename": filename,
        "image_hash": image_hash[:12] if image_hash else "",
        "user_ip": get_client_ip(),
        "thinking_meter": {
            "symmetry": meter.get("symmetry", {}).get("score"),
            "creativity": meter.get("creativity", {}).get("score"),
            "spatialSkills": meter.get("spatialSkills", {}).get("score"),
            "stability": meter.get("stability", {}).get("score"),
            "problemSolving": meter.get("problemSolving", {}).get("score"),
            "focusAndDetail": meter.get("focusAndDetail", {}).get("score")
        }
    }

    append_jsonl(ANALYTICS_FILE, event)


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
            "temperature": 0.45,
            "topP": 0.85,
            "maxOutputTokens": 1900,
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
                "content": "You are a strict but creative visual analyzer. Return valid JSON only."
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
        "temperature": 0.45,
        "top_p": 0.85,
        "max_completion_tokens": 1900,
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
# Backend PNG feedback card generation
# =========================================================

def load_font(size, bold=False):
    font_paths = []

    if bold:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        ]
    else:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
        ]

    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue

    return ImageFont.load_default()


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), str(text), font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(draw, text, font, max_width):
    text = clean_text(text)

    if not text:
        return []

    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = f"{current} {word}".strip()
        width, _ = text_size(draw, test, font)

        if width <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def draw_wrapped_text(draw, x, y, text, font, fill, max_width, line_gap=5):
    lines = wrap_text(draw, text, font, max_width)

    _, line_height = text_size(draw, "Ag", font)
    line_height += line_gap

    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height

    return y


def measure_wrapped_text(draw, text, font, max_width, line_gap=5):
    lines = wrap_text(draw, text, font, max_width)

    if not lines:
        return 0

    _, line_height = text_size(draw, "Ag", font)
    return len(lines) * (line_height + line_gap)


def draw_section(draw, y, title, body=None, items=None, bg=(255, 247, 223), width=520):
    margin = 18
    padding = 14
    x1 = margin
    x2 = width - margin
    content_width = x2 - x1 - (padding * 2)

    title_font = load_font(14, bold=True)
    body_font = load_font(12, bold=False)

    title_height = measure_wrapped_text(draw, title, title_font, content_width)
    body_height = 0
    items_height = 0

    if body:
        body_height = measure_wrapped_text(draw, body, body_font, content_width)

    if items:
        for item in items:
            items_height += measure_wrapped_text(draw, f"• {item}", body_font, content_width) + 5

    section_height = padding + title_height + 8 + body_height + items_height + padding

    draw.rounded_rectangle(
        [x1, y, x2, y + section_height],
        radius=14,
        fill=bg,
        outline=(234, 223, 202),
        width=1
    )

    cy = y + padding
    cy = draw_wrapped_text(draw, x1 + padding, cy, title, title_font, (47, 58, 47), content_width)
    cy += 8

    if body:
        cy = draw_wrapped_text(draw, x1 + padding, cy, body, body_font, (75, 85, 99), content_width)

    if items:
        for item in items:
            cy = draw_wrapped_text(draw, x1 + padding, cy, f"• {item}", body_font, (75, 85, 99), content_width)
            cy += 5

    return y + section_height + 12


def draw_learning_section(draw, y, cards, width=520):
    if not cards:
        return y

    margin = 18
    padding = 14
    x1 = margin
    x2 = width - margin
    content_width = x2 - x1 - (padding * 2)

    title_font = load_font(14, bold=True)
    card_title_font = load_font(12, bold=True)
    body_font = load_font(11, bold=False)

    section_height = padding + measure_wrapped_text(draw, "🧠 What they learned", title_font, content_width) + 10

    card_heights = []

    for index, card in enumerate(cards):
        card_title = f"{index + 1}. {clean_text(card.get('title'))}"
        card_body = clean_text(card.get("description"))

        h = 12
        h += measure_wrapped_text(draw, card_title, card_title_font, content_width - 20)
        h += 5
        h += measure_wrapped_text(draw, card_body, body_font, content_width - 20)
        h += 12

        card_heights.append(h)
        section_height += h + 8

    section_height += padding

    draw.rounded_rectangle(
        [x1, y, x2, y + section_height],
        radius=14,
        fill=(234, 247, 237),
        outline=(201, 229, 210),
        width=1
    )

    cy = y + padding
    cy = draw_wrapped_text(draw, x1 + padding, cy, "🧠 What they learned", title_font, (47, 58, 47), content_width)
    cy += 10

    for index, card in enumerate(cards):
        card_title = f"{index + 1}. {clean_text(card.get('title'))}"
        card_body = clean_text(card.get("description"))
        h = card_heights[index]

        draw.rounded_rectangle(
            [x1 + padding, cy, x2 - padding, cy + h],
            radius=12,
            fill=(255, 255, 255),
            outline=(234, 223, 202),
            width=1
        )

        inner_y = cy + 10
        inner_y = draw_wrapped_text(
            draw,
            x1 + padding + 10,
            inner_y,
            card_title,
            card_title_font,
            (47, 58, 47),
            content_width - 20
        )

        inner_y += 5

        draw_wrapped_text(
            draw,
            x1 + padding + 10,
            inner_y,
            card_body,
            body_font,
            (75, 85, 99),
            content_width - 20
        )

        cy += h + 8

    return y + section_height + 12


def draw_troy_meter_section(draw, y, meter, width=520):
    if not isinstance(meter, dict):
        return y

    labels = [
        ("Symmetry", "symmetry"),
        ("Creativity", "creativity"),
        ("Spatial Skills", "spatialSkills"),
        ("Stability", "stability"),
        ("Problem Solving", "problemSolving"),
        ("Focus & Detail", "focusAndDetail")
    ]

    margin = 18
    padding = 14
    x1 = margin
    x2 = width - margin
    content_width = x2 - x1 - (padding * 2)

    title_font = load_font(14, bold=True)
    label_font = load_font(12, bold=True)
    body_font = load_font(10, bold=False)

    section_height = padding + measure_wrapped_text(draw, "📊 Troy Thinking Meter", title_font, content_width) + 12

    row_heights = []

    for label, key in labels:
        item = meter.get(key, {})
        reason = clean_text(item.get("reason"), "")
        h = 28 + measure_wrapped_text(draw, reason, body_font, content_width - 20)
        row_heights.append(h)
        section_height += h + 8

    section_height += padding

    draw.rounded_rectangle(
        [x1, y, x2, y + section_height],
        radius=14,
        fill=(255, 247, 223),
        outline=(234, 223, 202),
        width=1
    )

    cy = y + padding
    cy = draw_wrapped_text(draw, x1 + padding, cy, "📊 Troy Thinking Meter", title_font, (47, 58, 47), content_width)
    cy += 12

    for index, (label, key) in enumerate(labels):
        item = meter.get(key, {})
        score = clamp_score(item.get("score", 1))
        reason = clean_text(item.get("reason"), "")

        row_y = cy
        row_h = row_heights[index]

        draw.rounded_rectangle(
            [x1 + padding, row_y, x2 - padding, row_y + row_h],
            radius=12,
            fill=(255, 255, 255),
            outline=(234, 223, 202),
            width=1
        )

        tx = x1 + padding + 10
        ty = row_y + 8

        draw.text((tx, ty), f"{label}: {score}/5", fill=(47, 58, 47), font=label_font)

        dots_x = x2 - padding - 95
        dot_y = ty + 4

        for dot in range(5):
            fill = (47, 107, 79) if dot < score else (220, 220, 220)
            draw.ellipse(
                [dots_x + dot * 18, dot_y, dots_x + dot * 18 + 10, dot_y + 10],
                fill=fill
            )

        ty += 22

        draw_wrapped_text(
            draw,
            tx,
            ty,
            reason,
            body_font,
            (75, 85, 99),
            content_width - 20
        )

        cy += row_h + 8

    return y + section_height + 12


def create_feedback_card_png(analysis, image_bytes=None):
    width = 520
    max_height = 10000

    bg_color = (255, 250, 240)
    canvas = Image.new("RGB", (width, max_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(28, bold=True)
    subtitle_font = load_font(14, bold=False)
    small_font = load_font(12, bold=True)

    margin = 18
    y = 16

    if image_bytes:
        try:
            build_img = Image.open(BytesIO(image_bytes))
            build_img = ImageOps.exif_transpose(build_img)

            if build_img.mode != "RGB":
                build_img = build_img.convert("RGB")

            build_img.thumbnail((width - margin * 2, 260))

            img_x = (width - build_img.width) // 2

            mask = Image.new("L", build_img.size, 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rounded_rectangle(
                [0, 0, build_img.width, build_img.height],
                radius=14,
                fill=255
            )

            canvas.paste(build_img, (img_x, y), mask)
            y += build_img.height + 16

        except Exception as e:
            print("Could not add image to feedback card:", str(e))

    match_type = clean_text(analysis.get("matchType"))
    badge_text = "Creative Analysis"

    if match_type == "book_pattern":
        badge_text = "Pattern Matched"
    elif match_type == "invalid":
        badge_text = "Needs Clearer Photo"

    badge_w, badge_h = text_size(draw, badge_text, small_font)

    draw.rounded_rectangle(
        [margin, y, margin + badge_w + 20, y + badge_h + 12],
        radius=999,
        fill=(255, 247, 223),
        outline=(234, 223, 202),
        width=1
    )

    draw.text((margin + 10, y + 6), badge_text, fill=(154, 106, 33), font=small_font)
    y += badge_h + 22

    build_guess = safe_get_dict(analysis, "buildGuess")
    title = clean_text(build_guess.get("title"), "Troy Build")
    subtitle = clean_text(build_guess.get("subtitle"), "")

    y = draw_wrapped_text(draw, margin, y, title, title_font, (154, 106, 33), width - margin * 2)
    y += 8

    y = draw_wrapped_text(draw, margin, y, subtitle, subtitle_font, (75, 85, 99), width - margin * 2)
    y += 14

    what_found = safe_get_dict(analysis, "whatWeFound")
    summary = clean_text(what_found.get("summary"))

    y = draw_section(
        draw,
        y,
        "🔍 What we found",
        body=summary,
        bg=(255, 247, 223),
        width=width
    )

    y = draw_troy_meter_section(
        draw,
        y,
        analysis.get("troyThinkingMeter") or {},
        width=width
    )

    matched_pattern = analysis.get("matchedPattern")

    if match_type == "book_pattern" and isinstance(matched_pattern, dict):
        matched_name = clean_text(matched_pattern.get("name"))
        strength = clean_text(matched_pattern.get("matchConfidence"))
        cues = ensure_list(matched_pattern.get("matchedCues"), [], limit=5)

        body = f"Matched Pattern: {matched_name}\nMatch Strength: {strength}%"

        y = draw_section(
            draw,
            y,
            "📘 Pattern Match Details",
            body=body,
            items=cues,
            bg=(234, 247, 237),
            width=width
        )

    y = draw_learning_section(
        draw,
        y,
        analysis.get("whatTheyLearned") or [],
        width=width
    )

    y = draw_section(
        draw,
        y,
        "👀 What we noticed",
        items=ensure_list(analysis.get("whatWeNoticed"), [], limit=5),
        bg=(255, 247, 223),
        width=width
    )

    y = draw_section(
        draw,
        y,
        "💡 Suggestions for parent",
        items=ensure_list(analysis.get("suggestionsForParent"), [], limit=5),
        bg=(234, 243, 255),
        width=width
    )

    y = draw_section(
        draw,
        y,
        "➡️ Next build ideas",
        items=ensure_list(analysis.get("nextBuildIdeas"), [], limit=5),
        bg=(251, 235, 255),
        width=width
    )

    footer_font = load_font(13, bold=True)
    draw.text((margin, y + 4), "Troy AI Analyzer", fill=(154, 106, 33), font=footer_font)
    y += 36

    final = canvas.crop((0, 0, width, min(y, max_height)))

    buffer = BytesIO()
    final.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)

    return buffer


def decode_data_url(data_url):
    if not data_url:
        return None

    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]

        return base64.b64decode(data_url)

    except Exception:
        return None


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
        "admin_key_loaded": bool(get_admin_key()),
        "pattern_count": len(TROY_PATTERN_LIBRARY),
        "book_match_threshold": BOOK_MATCH_THRESHOLD,
        "valid_confidence_threshold": VALID_CONFIDENCE_THRESHOLD,
        "absolute_min_confidence": ABSOLUTE_MIN_CONFIDENCE,
        "cache_items": len(analysis_cache),
        "upload_limit_per_ip_per_day": UPLOAD_LIMIT_PER_IP_PER_DAY,
        "upload_cooldown_seconds": UPLOAD_COOLDOWN_SECONDS,
        "download_card_enabled": True,
        "troy_thinking_meter_enabled": True
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

        image_id = save_analysis_image(image_base64, image_hash)

        cache_key = f"{image_hash}:{age}"

        if cache_key in analysis_cache:
            cached = analysis_cache[cache_key].copy()
            cached["cached"] = True
            cached["image_id"] = image_id or cached.get("image_id", "")
            analysis_cache.move_to_end(cache_key)

            save_analysis_event(
                cached,
                filename=filename,
                image_hash=image_hash,
                cached=True
            )

            sessions[cached["session_id"]] = cached

            return jsonify(cached), 200

        result = analyze_image_with_fallback(image_base64, image_data_url, age, image_hash)
        result["cached"] = False
        result["image_id"] = image_id

        save_analysis_event(
            result,
            filename=filename,
            image_hash=image_hash,
            cached=False
        )

        if os.environ.get("SHOW_DEBUG", "false").lower() == "true":
            result["debug"] = {
                "filename": filename,
                "image_hash": image_hash[:12],
                "gemini_model": get_gemini_model(),
                "groq_model": get_groq_vision_model(),
                "pattern_count": len(TROY_PATTERN_LIBRARY),
                "strict_book_matching": True,
                "creative_mode": True,
                "troy_thinking_meter": True,
                "download_card_url": f"/download-card/{result['session_id']}"
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


@app.route("/download-card/<session_id>", methods=["GET"])
def download_card_by_session(session_id):
    try:
        result = sessions.get(session_id)

        if not result:
            return jsonify({
                "error": "Session not found. Please analyze the image again."
            }), 404

        image_bytes = load_analysis_image(result.get("image_id"))
        card_buffer = create_feedback_card_png(result, image_bytes=image_bytes)

        title = clean_build_title(result.get("buildGuess", {}).get("title", "troy-feedback"))
        safe_title = secure_filename(title.lower().replace(" ", "-")) or "troy-feedback"

        return send_file(
            card_buffer,
            mimetype="image/png",
            as_attachment=True,
            download_name=f"{safe_title}-feedback-card.png"
        )

    except Exception as e:
        print("Download card error:", str(e))

        return jsonify({
            "error": "Could not generate feedback card.",
            "details": str(e)
        }), 500


@app.route("/download-card", methods=["POST"])
def download_card_from_payload():
    try:
        image_bytes = None
        analysis = None

        if request.content_type and "multipart/form-data" in request.content_type:
            analysis_json = request.form.get("analysis_json", "")

            if not analysis_json:
                return jsonify({"error": "analysis_json is required."}), 400

            analysis = json.loads(analysis_json)

            if "image" in request.files:
                image_file = request.files["image"]
                image_bytes = image_file.read()

        else:
            data = request.get_json() or {}
            analysis = data.get("analysis") or data
            image_bytes = decode_data_url(data.get("image_data_url") or data.get("imageDataUrl"))

        if not isinstance(analysis, dict):
            return jsonify({"error": "Valid analysis JSON is required."}), 400

        if "troyThinkingMeter" not in analysis:
            analysis["troyThinkingMeter"] = default_troy_thinking_meter(
                is_invalid=analysis.get("imageStatus") == "invalid"
            )

        card_buffer = create_feedback_card_png(analysis, image_bytes=image_bytes)

        title = clean_build_title(analysis.get("buildGuess", {}).get("title", "troy-feedback"))
        safe_title = secure_filename(title.lower().replace(" ", "-")) or "troy-feedback"

        return send_file(
            card_buffer,
            mimetype="image/png",
            as_attachment=True,
            download_name=f"{safe_title}-feedback-card.png"
        )

    except Exception as e:
        print("Download card from payload error:", str(e))

        return jsonify({
            "error": "Could not generate feedback card.",
            "details": str(e)
        }), 500


@app.route("/feedback", methods=["POST"])
def save_feedback():
    try:
        data = request.get_json() or {}

        feedback_item = {
            "created_at": datetime.utcnow().isoformat(),
            "session_id": clean_text(data.get("session_id")),
            "rating": clean_text(data.get("rating")),
            "actual_build": clean_text(data.get("actualBuild")),
            "ai_guess": clean_text(data.get("aiGuess")),
            "match_type": clean_text(data.get("matchType")),
            "confidence_score": data.get("confidenceScore", 0),
            "provider": clean_text(data.get("provider")),
            "troy_thinking_meter": data.get("troyThinkingMeter", {}),
            "user_ip": get_client_ip()
        }

        append_jsonl(FEEDBACK_FILE, feedback_item)

        return jsonify({
            "status": "saved",
            "message": "Feedback saved successfully."
        }), 200

    except Exception as e:
        print("Feedback error:", str(e))

        return jsonify({
            "error": "Could not save feedback."
        }), 500


@app.route("/admin/stats", methods=["GET"])
def admin_stats():
    try:
        if not require_admin():
            return jsonify({"error": "Unauthorized"}), 401

        analyses = read_jsonl(ANALYTICS_FILE)
        feedbacks = read_jsonl(FEEDBACK_FILE)

        total_uploads = len(analyses)
        valid_uploads = sum(1 for item in analyses if item.get("image_status") == "valid")
        invalid_uploads = sum(1 for item in analyses if item.get("image_status") == "invalid")

        book_matches = sum(1 for item in analyses if item.get("match_type") == "book_pattern")
        creative_guesses = sum(1 for item in analyses if item.get("match_type") == "creative_guess")

        provider_counts = Counter(item.get("provider") or "unknown" for item in analyses)
        guess_counts = Counter(item.get("guess") or "Unknown" for item in analyses)

        correct_feedback = sum(1 for item in feedbacks if item.get("rating") == "correct")
        wrong_feedback = sum(1 for item in feedbacks if item.get("rating") == "wrong")

        meter_keys = [
            "symmetry",
            "creativity",
            "spatialSkills",
            "stability",
            "problemSolving",
            "focusAndDetail"
        ]

        meter_totals = {key: 0 for key in meter_keys}
        meter_counts = {key: 0 for key in meter_keys}

        for item in analyses:
            thinking_meter = item.get("thinking_meter") or {}

            for key in meter_keys:
                score = thinking_meter.get(key)

                if isinstance(score, (int, float)):
                    meter_totals[key] += score
                    meter_counts[key] += 1

        meter_averages = {}

        for key in meter_keys:
            if meter_counts[key] > 0:
                meter_averages[key] = round(meter_totals[key] / meter_counts[key], 2)
            else:
                meter_averages[key] = 0

        return jsonify({
            "total_uploads": total_uploads,
            "valid_uploads": valid_uploads,
            "invalid_uploads": invalid_uploads,
            "book_matches": book_matches,
            "creative_guesses": creative_guesses,
            "provider_counts": dict(provider_counts),
            "top_guesses": guess_counts.most_common(10),
            "thinking_meter_averages": meter_averages,
            "feedback": {
                "total": len(feedbacks),
                "correct": correct_feedback,
                "wrong": wrong_feedback
            },
            "recent_uploads": analyses[-20:][::-1],
            "recent_feedback": feedbacks[-20:][::-1]
        }), 200

    except Exception as e:
        print("Admin stats error:", str(e))

        return jsonify({
            "error": "Could not load admin stats."
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