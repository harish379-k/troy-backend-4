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
# Evidence + creative category logic
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


def detect_features(evidence):
    """
    Detects visible-supported features.
    This allows creative hybrid guesses but prevents random guesses.
    """

    multi_level_words = [
        "floor", "floors", "level", "levels", "storey", "storeys",
        "story", "stories", "platform", "platforms", "layer", "layers",
        "terrace", "balcony", "upper level", "lower level", "multi-level",
        "multiple levels", "stacked floors", "horizontal layers",
        "sections above", "rooms", "room-like", "building-like",
        "second floor", "two floors", "three floors"
    ]

    tower_words = [
        "tower", "narrow", "single column", "column", "pillar",
        "vertical stack", "tall stack", "stacked upward", "one above another"
    ]

    bridge_words = [
        "bridge", "span", "gap", "across", "support", "supports",
        "horizontal beam", "beam across", "connects two sides",
        "two supports", "raised path"
    ]

    house_words = [
        "house", "home", "roof", "wall", "walls", "room", "door",
        "window", "triangle roof", "built space", "cabin", "shelter",
        "roof-like", "house-like"
    ]

    vehicle_words = [
        "vehicle", "car", "wheel", "wheels", "axle", "front", "back",
        "moving", "base with wheels", "rolling", "truck", "bus", "car-like",
        "vehicle-like"
    ]

    animal_words = [
        "animal", "head", "body", "leg", "legs", "neck", "tail", "face",
        "creature", "animal-like"
    ]

    gate_words = [
        "gate", "opening", "entrance", "doorway", "arch", "curve",
        "archway"
    ]

    castle_words = [
        "castle", "fort", "wall", "turret", "battlement", "castle-like"
    ]

    road_words = [
        "road", "path", "track", "route", "line", "trail", "way"
    ]

    scene_words = [
        "scene", "play area", "pretend", "story", "village", "town",
        "setup", "arrangement"
    ]

    return {
        "multi_level": has_keyword(evidence, multi_level_words),
        "tower": has_keyword(evidence, tower_words),
        "bridge": has_keyword(evidence, bridge_words),
        "house": has_keyword(evidence, house_words),
        "vehicle": has_keyword(evidence, vehicle_words),
        "animal": has_keyword(evidence, animal_words),
        "gate": has_keyword(evidence, gate_words),
        "castle": has_keyword(evidence, castle_words),
        "road": has_keyword(evidence, road_words),
        "scene": has_keyword(evidence, scene_words),
    }


def derive_category_from_evidence(raw_category, visible_elements, raw_title="", raw_subtitle="", raw_summary=""):
    """
    Strict + creative classifier.

    Fixes:
    - A build with floors going up is multi_level, not automatically tower.
    - A car with a house on top becomes house_vehicle.
    - Hybrid builds are allowed only when multiple visible features are detected.
    """
    category = clean_sentence(raw_category, "abstract").lower()

    evidence = combined_evidence_text(
        visible_elements,
        extra_text=f"{raw_category} {raw_title} {raw_subtitle} {raw_summary}"
    )

    features = detect_features(evidence)

    allowed_categories = [
        "house_vehicle",
        "multi_level_house",
        "bridge_house",
        "castle_gate",
        "multi_level",
        "tower",
        "bridge",
        "house",
        "vehicle",
        "animal",
        "gate",
        "castle",
        "road",
        "scene",
        "abstract",
        "unclear",
        "unrelated"
    ]

    if category not in allowed_categories:
        category = "abstract"

    # Highest priority: supported hybrid builds.
    if features["house"] and features["vehicle"]:
        return "house_vehicle"

    if features["multi_level"] and features["house"]:
        return "multi_level_house"

    if features["bridge"] and features["house"]:
        return "bridge_house"

    if features["castle"] and features["gate"]:
        return "castle_gate"

    # Then more specific single categories.
    if features["multi_level"]:
        return "multi_level"

    if features["bridge"]:
        return "bridge"

    if features["house"]:
        return "house"

    if features["vehicle"]:
        return "vehicle"

    if features["animal"]:
        return "animal"

    if features["gate"]:
        return "gate"

    if features["castle"]:
        return "castle"

    if features["road"]:
        return "road"

    if features["scene"]:
        return "scene"

    # Tower only when evidence is clearly a narrow vertical stack.
    # Height alone should not become tower.
    if features["tower"]:
        return "tower"

    if category == "tower":
        return "abstract"

    return category


def repair_weak_guess(build_category, title, subtitle, visible_elements, summary=""):
    """
    Keeps Groq's creative guess when it is supported.
    Corrects unsupported guesses.
    """
    raw_category = clean_sentence(build_category, "abstract").lower()
    raw_title = clean_sentence(title)
    raw_subtitle = clean_sentence(subtitle)
    raw_summary = clean_sentence(summary)

    final_category = derive_category_from_evidence(
        raw_category,
        visible_elements,
        raw_title=raw_title,
        raw_subtitle=raw_subtitle,
        raw_summary=raw_summary
    )

    title_lower = raw_title.lower()

    safe_category_titles = {
        "house_vehicle": "House-on-wheels Troy build",
        "multi_level_house": "Multi-level house-like Troy build",
        "bridge_house": "Bridge-house Troy build",
        "castle_gate": "Castle-gate Troy build",
        "multi_level": "Multi-level Troy block structure",
        "tower": "Tower-like Troy block build",
        "bridge": "Bridge-like Troy block build",
        "house": "House-like Troy block build",
        "vehicle": "Vehicle-like Troy block build",
        "animal": "Animal-like Troy block build",
        "gate": "Gate-like Troy block build",
        "castle": "Castle-like Troy block build",
        "road": "Road or path-like Troy block layout",
        "scene": "Pretend-play Troy block scene",
        "abstract": "Open-ended Troy block build",
        "unclear": "Open-ended Troy block build",
        "unrelated": "Unclear image"
    }

    safe_category_subtitles = {
        "house_vehicle": "The build seems to combine a vehicle-like base with a house or room-like top section.",
        "multi_level_house": "The build appears to have house-like parts arranged across more than one level.",
        "bridge_house": "The build seems to combine a bridge-like connection with a small built space.",
        "castle_gate": "The build appears to include a gate or entrance with castle-like sections.",
        "multi_level": "The build appears to have floors, layers, or sections placed above each other.",
        "tower": "The build appears to be mainly a narrow upward stack.",
        "bridge": "The build appears to use supports and a connecting section.",
        "house": "The build appears to have parts that could represent a built space.",
        "vehicle": "The build could represent a vehicle if the child intended it that way.",
        "animal": "The build could represent an animal if the child intended it that way.",
        "gate": "The build appears to include an opening or entrance-like shape.",
        "castle": "The build appears to have castle-like parts such as walls, sections, or height.",
        "road": "The build appears to arrange blocks like a path or route.",
        "scene": "The build looks like a small pretend-play setup made from blocks.",
        "abstract": "The child created an open-ended structure using block placement and shape.",
        "unclear": "The build is visible, but the exact object is not clear from the image.",
        "unrelated": "The image does not clearly show a Troy block build."
    }

    # For supported hybrid categories, prefer strong creative titles.
    if final_category in ["house_vehicle", "multi_level_house", "bridge_house", "castle_gate"]:
        final_title = safe_category_titles[final_category]
        final_subtitle = safe_category_subtitles[final_category]

        return {
            "category": final_category,
            "title": final_title,
            "subtitle": final_subtitle
        }

    object_words = [
        "car", "dog", "cat", "house", "castle", "bridge", "tower",
        "animal", "vehicle", "gate", "road", "building"
    ]

    # Keep Groq's unique title if it is useful and not overconfident.
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
        final_subtitle = f"It seems to use visible details like {visible_elements[0]}."
    elif raw_summary:
        final_subtitle = raw_summary
    else:
        final_subtitle = safe_category_subtitles.get(
            final_category,
            "The child created a visible structure using Troy blocks."
        )

    return {
        "category": final_category,
        "title": final_title,
        "subtitle": final_subtitle
    }


# =========================================================
# Learning card logic
# =========================================================

def build_learning_cards_from_evidence(category, visible_elements):
    evidence = combined_evidence_text(visible_elements)
    main_detail = visible_phrase(visible_elements)

    cards = []

    if category == "house_vehicle":
        cards.extend([
            {
                "title": "Combining Ideas",
                "description": "The child combined two ideas: a moving base and a house-like top section.",
                "color": "cream"
            },
            {
                "title": "Part-to-Whole Design",
                "description": "The child explored how different parts can work together as one bigger build.",
                "color": "green"
            },
            {
                "title": "Pretend-Play Storytelling",
                "description": "A house-on-wheels idea can become a story about travel, homes, or movement.",
                "color": "blue"
            }
        ])

    if category == "multi_level_house":
        cards.extend([
            {
                "title": "Layered Building",
                "description": "The child explored how one floor or section can sit above another.",
                "color": "cream"
            },
            {
                "title": "Support Planning",
                "description": "The build shows thinking about how lower parts support upper levels.",
                "color": "green"
            },
            {
                "title": "Structure Organization",
                "description": "The child practiced arranging a building into separate levels or spaces.",
                "color": "blue"
            }
        ])

    if category == "bridge_house":
        cards.extend([
            {
                "title": "Connection Design",
                "description": "The child explored how a bridge-like section can connect parts of a structure.",
                "color": "cream"
            },
            {
                "title": "Support and Balance",
                "description": "The build encourages thinking about how raised parts stay steady.",
                "color": "green"
            },
            {
                "title": "Creative Combination",
                "description": "The child combined a pathway idea with a built-space idea.",
                "color": "blue"
            }
        ])

    if category == "castle_gate":
        cards.extend([
            {
                "title": "Entrance Design",
                "description": "The child explored how blocks can create an opening or gate-like space.",
                "color": "cream"
            },
            {
                "title": "Pretend Structure",
                "description": "Castle-like parts can support storytelling about forts, gates, or protected spaces.",
                "color": "green"
            },
            {
                "title": "Shape Arrangement",
                "description": "The child practiced placing blocks to make a recognizable entrance or section.",
                "color": "blue"
            }
        ])

    if category == "multi_level":
        cards.extend([
            {
                "title": "Layered Building",
                "description": "The child explored how one level can be placed above another.",
                "color": "cream"
            },
            {
                "title": "Vertical Planning",
                "description": "The build shows thinking about how lower parts support upper floors or sections.",
                "color": "green"
            },
            {
                "title": "Structure Organization",
                "description": "The child practiced arranging the build into different levels instead of one simple stack.",
                "color": "blue"
            }
        ])

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

    if category == "scene":
        cards.append({
            "title": "Story Building",
            "description": "The child used blocks to create a small pretend-play setup.",
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

    evidence_words = [
        "stack", "base", "support", "gap", "bridge", "roof", "triangle",
        "curve", "arch", "repeated", "horizontal", "vertical", "blocks",
        "pieces", "path", "opening", "floor", "level", "wheel", "house",
        "vehicle", "room", "platform"
    ]

    if title in weak_titles:
        return not any(word in description for word in evidence_words)

    return False


def choose_learning_cards(ai_cards, category, visible_elements):
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
            continue

        cleaned.append(temp_card)

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
- Pieces may include cubes, cuboids, long beams, planks, pillars, triangular roof pieces, curved/arch pieces, connector-like pieces, and wheel-like/base pieces if visible.
- Children may build towers, bridges, houses, animals, vehicles, gates, castles, roads, pretend scenes, hybrid builds, or abstract structures.

MOST IMPORTANT RULE:
Be strict, but creative.
Do not guess randomly.
Only make a creative guess when the visible structure supports it.

CREATIVE HYBRID GUESSING RULES:
- If the build has vehicle-like parts plus house-like parts, call it a house-on-wheels, mobile-home, camper, or moving-house style build.
- If the build has floors, platforms, layers, or sections above each other, call it a multi-level structure, not a tower.
- If the build has house-like parts across more than one level, call it a multi-level house-like build.
- If the build has bridge-like supports plus a room/roof section, call it a bridge-house or raised-house style build.
- If the build has a gate/opening plus castle-like walls or sections, call it a castle-gate style build.
- If multiple object ideas are visible, explain the combination instead of reducing it to just "tower".
- Do not invent wheels, roof, floors, animals, or doors unless they are visible.

NORMAL GUESSING RULES:
- If it is mainly one narrow vertical stack or column: call it a tower-like structure.
- If it has two supports with a horizontal piece across: call it a bridge-like structure.
- If it has walls, roof/triangle pieces, or room-like layout: call it a house-like structure.
- If it has wheels, axle-like parts, or clear vehicle shape: call it a vehicle-like structure.
- If it has body + legs/head/neck shape: call it an animal-like structure.
- If it has an entrance/opening/arch: call it a gate-like structure.
- If it is not clear, call it an abstract or open-ended block structure.
- Use cautious language like "looks like", "seems like", or "could be" when uncertain.

VISIBLE EVIDENCE RULES:
Every sentence must be based on visible details in the image.
Mention actual visible details such as:
- floors
- levels
- platforms
- layered sections
- upper and lower parts
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
- wheel-like parts
- house-like top
- vehicle-like base

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
- If the build has multiple levels: "Layered Building"
- If the build combines a vehicle and a house: "Combining Ideas"
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
    "buildCategory": "house_vehicle / multi_level_house / bridge_house / castle_gate / multi_level / tower / bridge / house / vehicle / animal / gate / castle / road / scene / abstract / unclear / unrelated",
    "visibleElements": [
      "specific visible detail 1",
      "specific visible detail 2",
      "specific visible detail 3"
    ],
    "blockCountEstimate": "rough estimate like 5-8, 10-15, 20+ or unclear",
    "whyThisGuess": "explain the guess using only visible details"
  }},
  "buildGuess": {{
    "title": "safe but creative build interpretation",
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
        temperature=0.65,
        top_p=0.95,
        max_completion_tokens=1600,
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