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

MAX_BASE64_IMAGE_SIZE = 1_800_000
Image.MAX_IMAGE_PIXELS = 15_000_000

analysis_cache = OrderedDict()
MAX_CACHE_ITEMS = 30

sessions = {}


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
        "description": "A chair with a seat, backrest, and curved rocker rails at the bottom."
    },
    {
        "id": "earthquake_resistant",
        "name": "Earthquake Resistant Structure",
        "category": "motion_structures",
        "visual_cues": [
            "multi-floor building",
            "rectangular levels",
            "small top tower or roof piece",
            "curved rocker-like base pieces",
            "building sitting on movable supports"
        ],
        "description": "A multi-level building placed on rocker-like supports."
    },
    {
        "id": "troy_pendulum",
        "name": "Troy Pendulum",
        "category": "motion_structures",
        "visual_cues": [
            "two vertical supports",
            "horizontal beam across the top",
            "curved pieces below",
            "triangle piece on top",
            "balanced pendulum-like setup"
        ],
        "description": "A pendulum-style structure with supports, a top beam, curved parts, and a central triangular piece."
    },
    {
        "id": "newtons_pet",
        "name": "Newton's Pet",
        "category": "motion_structures",
        "visual_cues": [
            "low vehicle-like base",
            "long rectangular block on top",
            "cylinders underneath",
            "curved pieces at the bottom",
            "animal or pet-like moving form"
        ],
        "description": "A low rolling pet-like build using cylinders and curved base pieces."
    },
    {
        "id": "charminar",
        "name": "Charminar",
        "category": "monuments",
        "visual_cues": [
            "square monument base",
            "four corner towers",
            "four pillars or minarets",
            "central opening",
            "symmetrical monument layout"
        ],
        "description": "A monument-like build with four corner pillars or minarets and a central structure."
    },
    {
        "id": "qutub_minar",
        "name": "Qutub Minar",
        "category": "monuments",
        "visual_cues": [
            "very tall narrow tower",
            "stacked vertical blocks",
            "wide base",
            "tapering tower shape",
            "monument tower form"
        ],
        "description": "A tall tapering tower structure with stacked blocks and a wider base."
    },
    {
        "id": "shinto_arch",
        "name": "Shinto Arch",
        "category": "places_of_worship",
        "visual_cues": [
            "two tall pillars",
            "horizontal beam across top",
            "gate-like structure",
            "angled blocks near the top",
            "open space below the beam"
        ],
        "description": "A gate or arch structure with two vertical supports and a strong horizontal top beam."
    },
    {
        "id": "pickup_truck",
        "name": "Pickup Truck",
        "category": "vehicles",
        "visual_cues": [
            "vehicle-like shape",
            "four cylinder wheels",
            "flat truck bed",
            "raised cabin or block section",
            "long rectangular base"
        ],
        "description": "A pickup truck build with a long base, raised cabin or bed, and cylinder wheels."
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
        "description": "A long highway or roadway scene with ramps, supports, and road sections."
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
            "monument gate shape"
        ],
        "description": "A large monument-style gateway with a central arch and tall side supports."
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
        "description": "A long bridge structure with tower-like frames and a stretched road section."
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
        "description": "A tall Eiffel Tower-like structure with angled supports and a narrow upper section."
    },
    {
        "id": "mosque",
        "name": "Mosque",
        "category": "places_of_worship",
        "visual_cues": [
            "dome-like curved piece",
            "arched entrance",
            "sloped ramp",
            "small towers or minarets",
            "religious building form"
        ],
        "description": "A mosque-style structure with an arch, dome-like curved piece, and entrance."
    },
    {
        "id": "gurudwara",
        "name": "Gurudwara",
        "category": "places_of_worship",
        "visual_cues": [
            "dome-like top",
            "arched opening",
            "tall side pillars",
            "sloped path or ramp",
            "place of worship structure"
        ],
        "description": "A Gurudwara-style place of worship with an arched body, dome-like feature, and tall side elements."
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
        "description": "A Greek temple-like structure with columns, a roof, and a front platform or stairs."
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
        "description": "A train build with a long body, chimney cylinder, and wagon-like sections."
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
        "description": "A ship or boat-like build with a long hull, cabin, and cylinder chimney."
    },
    {
        "id": "temple",
        "name": "Temple",
        "category": "places_of_worship",
        "visual_cues": [
            "stair or ramp front",
            "pillars",
            "roof structure",
            "temple-like body",
            "raised entrance"
        ],
        "description": "A temple-like build with a front ramp or stairs, pillars, and a roofed structure."
    }
]

BOOK_MATCH_THRESHOLD = 78


# =========================================================
# API config
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


def build_gemini_model():
    api_key = get_gemini_api_key()

    if not api_key:
        return None

    genai.configure(api_key=api_key)

    return genai.GenerativeModel(
        get_gemini_model(),
        generation_config={
            "temperature": 0.72,
            "top_p": 0.95,
            "max_output_tokens": 1600
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

    raise ValueError("Image is too large even after compression. Please upload a smaller image.")


# =========================================================
# Feedback variation
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
# Prompt builders
# =========================================================

def compact_pattern_library_text():
    lines = []

    for pattern in TROY_PATTERN_LIBRARY:
        cue_text = ", ".join(pattern["visual_cues"])
        lines.append(
            f'- {pattern["name"]} ({pattern["category"]}) — '
            f'{pattern["description"]} Visible cues: {cue_text}.'
        )

    return "\n".join(lines)


def build_troy_prompt(age, image_hash):
    style = pick_feedback_style(image_hash)
    unique_hint = build_unique_hint(image_hash)
    pattern_library = compact_pattern_library_text()

    return f"""
You are Troy AI Analyzer.

You are analyzing one uploaded image of a child's Troy wooden-block build.

Child age:
{age if age else "unknown"}

You have a Troy pattern book reference list.

Your job has TWO MODES:

MODE 1: Pattern matching
If the child's uploaded build clearly matches one known Troy pattern, return matchType as "book_pattern".
Only do this when the match is visually strong.

MODE 2: Creative open-ended analysis
If the uploaded build does not clearly match any known pattern, return matchType as "creative_guess".
Do not force random builds into the pattern list.

Known Troy patterns:
{pattern_library}

Important display rule:
- Never mention page numbers in buildGuess, whatWeFound, whatTheyLearned, whatWeNoticed, suggestionsForParent, or nextBuildIdeas.
- Do not say "page", "book page", or "from page".
- If a known pattern matches, simply say it looks like the named pattern.
- If no strong match exists, do normal creative analysis.

Pattern matching rules:
- Use the known pattern only when the uploaded build has the same main structure and visible cues.
- A random tower should not become Qutub Minar unless it has a tapering monument-like tower shape.
- A random multi-level block build should not become Earthquake Resistant Structure unless it has a multi-floor building form with rocker or movable base cues.
- A random vehicle should not become Pickup Truck unless it has a long vehicle base and wheel or cylinder cues.
- A random arch should not become India Gate unless it has a monument gateway form with side pillars and a central arch.
- If similarity is weak or uncertain, use creative_guess.
- If matchConfidence is below 78, use creative_guess.
- If matchType is book_pattern, include pattern name and why it matches.
- If matchType is creative_guess, matchedPattern must be null.

Feedback style for this image:
{style["name"]} — {style["instruction"]}

Uniqueness instruction:
{unique_hint}

General analysis rules:
- Look at the whole image first.
- Give a creative but realistic guess.
- Do not force labels like tower, house, bridge, or car.
- If it looks like a hybrid idea, describe the hybrid naturally.
- If it has floors or sections going upward, do not automatically call it a tower.
- It may be a multi-level building, layered structure, raised house, platform scene, parking-garage-like build, lookout station, or pretend-play setup.
- Base every sentence only on visible details.
- Mention visible parts such as base, floors, levels, gaps, supports, repeated blocks, stacked sections, roof-like pieces, wheel-like parts, curved pieces, openings, paths, bridges, rooms, platforms, loose blocks, or upper/lower sections if visible.
- If the image is not a Troy/block build, mark it invalid.
- If unsure, use cautious phrases like "looks like", "could be", or "seems to".
- Keep the tone simple, warm, parent-friendly, and encouraging.
- Return JSON only.

BANNED LEARNING CARD TITLES:
Do not use:
Creativity, Problem-Solving, Problem Solving, Spatial Awareness, Spatial Thinking, Imagination, Motor Skills, Fine Motor Skills, Engineering, STEM Learning, Critical Thinking.

Use specific learning card titles like:
Layer Planning, Upper-Level Building, Bridge Support, Moving Base Idea, Tiny Home Story, Roof Shape Experiment, Open-Space Design, Block Pattern Play, Creature-Making, Careful Stacking, Shape Combining, Idea Mixing, Build-and-Tell Practice, Small-World Making, Support Below Space Above, Vehicle Shape Thinking, Room-Making, Entrance Building, Testing What Holds, Above-Below Thinking.

Return this exact JSON shape:

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

Invalid image rules:
- imageStatus must be "invalid"
- matchType must be "invalid"
- confidenceScore must be below 65
- matchedPattern must contain null values
- whatTheyLearned must be []
"""


# =========================================================
# Specific fallback feedback logic
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

    if contains_any(context, ["rocking", "rocker", "curved rail", "chair"]):
        card_pool.extend([
            {
                "title": "Motion Design",
                "description": "The child explored how curved base pieces can make a build feel like it could rock or move.",
                "color": "cream"
            },
            {
                "title": "Seat-and-Back Planning",
                "description": "The build encourages thinking about how a seat and backrest can work together.",
                "color": "green"
            }
        ])

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
                "title": "Support Below Space Above",
                "description": "The build encourages the child to think about how bottom blocks can hold up higher sections.",
                "color": "blue"
            }
        ])

    if contains_any(context, ["wheel", "vehicle", "moving", "car", "base", "travel", "rolling", "truck", "train", "ship"]):
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

    if contains_any(context, ["bridge", "gap", "span", "across", "support", "beam", "highway"]):
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

    if contains_any(context, ["gate", "arch", "opening", "entrance", "tunnel", "curve", "india gate", "shinto"]):
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

    if contains_any(context, ["repeat", "repeated", "same", "pattern", "symmetry", "line", "row", "monument"]):
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

    if contains_any(context, ["tower", "stack", "tall", "height", "vertical", "qutub", "eiffel", "minar"]):
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

    if matched_pattern and matched_pattern.get("name"):
        card_pool.append({
            "title": "Pattern Matching",
            "description": f"The child recreated parts of the {matched_pattern.get('name')} pattern while making choices about placement and shape.",
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


def normalize_matched_pattern(raw_matched_pattern, match_type):
    if not isinstance(raw_matched_pattern, dict):
        return None

    pattern_id = clean_text(raw_matched_pattern.get("id"))
    pattern_name = clean_text(raw_matched_pattern.get("name"))
    category = clean_text(raw_matched_pattern.get("category"))
    why_matched = clean_text(raw_matched_pattern.get("whyMatched"))

    try:
        match_confidence = int(float(raw_matched_pattern.get("matchConfidence", 0)))
    except Exception:
        match_confidence = 0

    library_pattern = find_pattern_by_id(pattern_id)

    if library_pattern:
        pattern_name = library_pattern["name"]
        category = library_pattern["category"]

    if match_type != "book_pattern" or match_confidence < BOOK_MATCH_THRESHOLD:
        return None

    if not pattern_name:
        return None

    return {
        "id": pattern_id or None,
        "name": pattern_name,
        "category": category or None,
        "matchConfidence": match_confidence,
        "whyMatched": why_matched or "the visible structure matches the main shape and block arrangement"
    }


def normalize_learning_cards(cards, build_guess, summary, noticed, image_hash, matched_pattern=None):
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


def remove_page_words(text):
    """
    Extra safety: prevents page/book-page phrases from reaching frontend.
    """
    text = clean_text(text)

    blocked_phrases = [
        "from page",
        "on page",
        "book page",
        "page number",
        "page 1",
        "page 2",
        "page 3",
        "page 4",
        "page 5",
        "page 6",
        "page 7",
        "page 8",
        "page 9",
        "page 10",
        "page 11",
        "page 12",
        "page 13",
        "page 14",
        "page 15",
        "page 16",
        "page 17",
        "page 18",
        "page 19",
        "page 20"
    ]

    lowered = text.lower()

    if any(phrase in lowered for phrase in blocked_phrases):
        text = (
            text.replace("from page", "from the pattern reference")
            .replace("on page", "in the pattern reference")
            .replace("book page", "pattern reference")
            .replace("page number", "pattern reference")
        )

        for number in range(1, 21):
            text = text.replace(f"page {number}", "the pattern reference")
            text = text.replace(f"Page {number}", "the pattern reference")

    return clean_text(text)


def normalize_analysis_response(parsed, image_hash):
    image_status = clean_text(parsed.get("imageStatus", "invalid")).lower()

    try:
        confidence = int(float(parsed.get("confidenceScore", 0)))
    except Exception:
        confidence = 0

    raw_match_type = clean_text(parsed.get("matchType", "creative_guess")).lower()
    raw_matched_pattern = parsed.get("matchedPattern")

    if image_status == "invalid":
        final_match_type = "invalid"
    elif raw_match_type == "book_pattern":
        final_match_type = "book_pattern"
    else:
        final_match_type = "creative_guess"

    matched_pattern = normalize_matched_pattern(raw_matched_pattern, final_match_type)

    if not matched_pattern and final_match_type == "book_pattern":
        final_match_type = "creative_guess"

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

    noticed = [remove_page_words(item) for item in noticed]

    if matched_pattern:
        default_title = f"{matched_pattern['name']} Style Build"
        default_subtitle = (
            f"This looks like the {matched_pattern['name']} pattern because "
            f"{matched_pattern['whyMatched']}."
        )
    else:
        default_title = "Open-ended Troy block build"
        default_subtitle = "The child created a visible structure using blocks."

    normalized_build_guess = {
        "title": remove_page_words(clean_text(build_guess.get("title"), default_title)),
        "subtitle": remove_page_words(clean_text(build_guess.get("subtitle"), default_subtitle))
    }

    if matched_pattern:
        normalized_build_guess["title"] = f"{matched_pattern['name']} Style Build"
        normalized_build_guess["subtitle"] = (
            f"This looks like the {matched_pattern['name']} pattern because "
            f"{matched_pattern['whyMatched']}."
        )

    normalized_summary = remove_page_words(
        clean_text(
            what_found.get("summary"),
            "The image shows a child-made block structure with visible block placement."
        )
    )

    result = {
        "status": "success",
        "imageStatus": "valid" if image_status == "valid" and confidence >= 65 else "invalid",
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

    if result["imageStatus"] == "invalid":
        result["matchType"] = "invalid"
        result["matchedPattern"] = None
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
        temperature=0.72,
        top_p=0.95,
        max_completion_tokens=1600,
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
        print("Trying Gemini first...")
        parsed = analyze_with_gemini(pil_img, age, image_hash)
        result = normalize_analysis_response(parsed, image_hash)
        result["provider"] = "gemini"
        print("Gemini analysis successful")
        return result

    except Exception as e:
        error_text = str(e)
        print("Gemini failed:", error_text)
        errors.append(f"Gemini: {error_text}")

    try:
        print("Trying Groq fallback...")
        parsed = analyze_with_groq(image_data_url, age, image_hash)
        result = normalize_analysis_response(parsed, image_hash)
        result["provider"] = "groq"
        print("Groq fallback successful")
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
        "groq_vision_model": get_groq_vision_model(),
        "pattern_count": len(TROY_PATTERN_LIBRARY),
        "book_match_threshold": BOOK_MATCH_THRESHOLD
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
                "feedback_style": pick_feedback_style(image_hash)["name"],
                "pattern_count": len(TROY_PATTERN_LIBRARY)
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
Do not mention page numbers.
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