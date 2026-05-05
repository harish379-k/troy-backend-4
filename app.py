import os
import json
import uuid
import time
import base64
import hashlib
from io import BytesIO
from pathlib import Path

import requests
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

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_BASE64_IMAGE_SIZE = 3_800_000

# Simple memory cache to avoid re-analyzing same image while server is running
analysis_cache = {}
sessions = {}


# =========================================================
# API config
# =========================================================

def get_openrouter_api_key():
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def get_openrouter_model():
    return os.environ.get(
        "OPENROUTER_MODEL",
        "meta-llama/llama-4-scout:free"
    ).strip()


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


def build_groq_client():
    key = get_groq_api_key()

    if not key:
        return None

    return Groq(api_key=key)


print("OpenRouter key:", "FOUND" if get_openrouter_api_key() else "NOT FOUND")
print("OpenRouter model:", get_openrouter_model())
print("Groq key:", "FOUND" if get_groq_api_key() else "NOT FOUND")
print("Groq vision model:", get_groq_vision_model())


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


def safe_get_dict(data, key):
    value = data.get(key)

    if isinstance(value, dict):
        return value

    return {}


def is_rate_limit_error(error_text):
    lower = error_text.lower()
    return (
        "429" in lower
        or "rate limit" in lower
        or "too many requests" in lower
    )


def is_invalid_key_error(error_text):
    lower = error_text.lower()
    return (
        "401" in lower
        or "403" in lower
        or "unauthorized" in lower
        or "forbidden" in lower
        or "invalid api key" in lower
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

def image_to_data_url_and_hash(image_file):
    """
    Converts upload into compressed JPEG base64 data URL.
    Also returns hash so repeated same uploads can use cached result.
    """
    img = Image.open(image_file.stream)
    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail((1400, 1400))

    final_bytes = None

    for quality in [85, 75, 65, 55, 45]:
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        raw = buffer.getvalue()

        encoded = base64.b64encode(raw).decode("utf-8")

        if len(encoded.encode("utf-8")) <= MAX_BASE64_IMAGE_SIZE:
            final_bytes = raw
            image_data_url = f"data:image/jpeg;base64,{encoded}"
            image_hash = hashlib.sha256(raw).hexdigest()
            return image_data_url, image_hash

    img.thumbnail((1000, 1000))

    for quality in [70, 60, 50, 40]:
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        raw = buffer.getvalue()

        encoded = base64.b64encode(raw).decode("utf-8")

        if len(encoded.encode("utf-8")) <= MAX_BASE64_IMAGE_SIZE:
            final_bytes = raw
            image_data_url = f"data:image/jpeg;base64,{encoded}"
            image_hash = hashlib.sha256(raw).hexdigest()
            return image_data_url, image_hash

    raise ValueError("Image is too large even after compression. Please upload a smaller image.")


# =========================================================
# Natural analysis prompt
# =========================================================

def build_simple_troy_prompt(age):
    return f"""
You are Troy AI Analyzer.

You are analyzing one uploaded image of a child's Troy wooden-block build.

Child age:
{age if age else "unknown"}

Goal:
Give feedback similar to how a careful human teacher would look at the photo.

Important:
- Look at the whole image first.
- Give a creative but realistic guess about what the child may have built.
- Do not force labels like tower, house, bridge, or car.
- If it looks like a hybrid idea, describe the hybrid. Example: moving house, house-on-wheels, bridge-house, layered building, castle gate, pretend-play scene, animal vehicle, abstract machine, etc.
- If it has floors or sections going upward, do not automatically call it a tower. It may be a multi-level building, layered structure, raised house, or pretend-play scene.
- Base every statement only on visible details.
- Mention visible parts such as base, floors, levels, blocks, gaps, supports, roof-like pieces, wheel-like pieces, curved pieces, repeated pieces, openings, or loose blocks.
- If the image is not a Troy/block build, mark it invalid.
- If you are unsure, use cautious words like "looks like", "could be", or "seems to".
- Keep the tone simple, warm, and parent-friendly.
- Do not overclaim. Avoid big words like engineering mastery.
- Return JSON only.

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
    "summary": "2 short sentences describing the visible build"
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
# Response cleanup
# =========================================================

def normalize_analysis_response(parsed):
    """
    Keeps AI's natural answer, but guarantees frontend-safe structure.
    """
    image_status = clean_text(parsed.get("imageStatus", "invalid")).lower()

    try:
        confidence = int(float(parsed.get("confidenceScore", 0)))
    except Exception:
        confidence = 0

    build_guess = safe_get_dict(parsed, "buildGuess")
    what_found = safe_get_dict(parsed, "whatWeFound")

    normalized = {
        "status": "success",
        "imageStatus": "valid" if image_status == "valid" and confidence >= 65 else "invalid",
        "confidenceScore": confidence,
        "buildGuess": {
            "title": clean_text(
                build_guess.get("title"),
                "Open-ended Troy block build"
            ),
            "subtitle": clean_text(
                build_guess.get("subtitle"),
                "The child created a visible structure using blocks."
            )
        },
        "whatWeFound": {
            "title": "What we found",
            "summary": clean_text(
                what_found.get("summary"),
                "The image shows a child-made block structure with visible block placement."
            )
        },
        "whatTheyLearned": normalize_learning_cards(
            parsed.get("whatTheyLearned")
        ),
        "whatWeNoticed": ensure_list(
            parsed.get("whatWeNoticed"),
            [
                "The build shows visible blocks arranged into a structure.",
                "The child used block placement to create a shape or idea.",
                "The structure has details that can be discussed with the child."
            ],
            limit=3
        ),
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

    if normalized["imageStatus"] == "invalid":
        normalized["whatTheyLearned"] = []
        normalized["buildGuess"] = {
            "title": "We couldn’t clearly analyze this image",
            "subtitle": normalized["whatWeFound"]["summary"]
        }

    return normalized


def normalize_learning_cards(cards):
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

            cleaned.append({
                "title": title,
                "description": description,
                "color": color
            })

    fallback = [
        {
            "title": "Block Placement",
            "description": "The child practiced deciding where each block should go.",
            "color": "cream"
        },
        {
            "title": "Balance and Structure",
            "description": "The child explored how blocks can stay steady together.",
            "color": "green"
        },
        {
            "title": "Creative Thinking",
            "description": "The child turned an idea into a physical build.",
            "color": "blue"
        }
    ]

    for item in fallback:
        if len(cleaned) >= 3:
            break

        cleaned.append(item)

    for i, card in enumerate(cleaned[:3]):
        card["color"] = allowed_colors[i]

    return cleaned[:3]


# =========================================================
# OpenRouter analysis
# =========================================================

def analyze_with_openrouter(image_data_url, age):
    api_key = get_openrouter_api_key()

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found")

    prompt = build_simple_troy_prompt(age)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("APP_URL", "https://troy-world-demo"),
        "X-Title": os.environ.get("APP_NAME", "Troy AI Analyzer")
    }

    payload = {
        "model": get_openrouter_model(),
        "messages": [
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
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 1600,
        "response_format": {
            "type": "json_object"
        }
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code >= 400:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    return parse_json_response(content)


# =========================================================
# Groq fallback analysis
# =========================================================

def analyze_with_groq(image_data_url, age):
    client = build_groq_client()

    if not client:
        raise RuntimeError("GROQ_API_KEY not found")

    prompt = build_simple_troy_prompt(age)

    completion = client.chat.completions.create(
        model=get_groq_vision_model(),
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
        temperature=0.7,
        top_p=0.95,
        max_completion_tokens=1600,
        response_format={
            "type": "json_object"
        }
    )

    content = completion.choices[0].message.content
    return parse_json_response(content)


def analyze_image_with_fallback(image_data_url, age):
    """
    First try OpenRouter.
    If it fails, try Groq.
    """
    errors = []

    try:
        parsed = analyze_with_openrouter(image_data_url, age)
        result = normalize_analysis_response(parsed)
        result["provider"] = "openrouter"
        return result

    except Exception as e:
        error_text = str(e)
        print("OpenRouter failed:", error_text)
        errors.append(f"OpenRouter: {error_text}")

    try:
        parsed = analyze_with_groq(image_data_url, age)
        result = normalize_analysis_response(parsed)
        result["provider"] = "groq"
        return result

    except Exception as e:
        error_text = str(e)
        print("Groq failed:", error_text)
        errors.append(f"Groq: {error_text}")

    raise RuntimeError("Both OpenRouter and Groq failed. " + " | ".join(errors))


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
        "openrouter_key_loaded": bool(get_openrouter_api_key()),
        "openrouter_model": get_openrouter_model(),
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
            image_data_url, image_hash = image_to_data_url_and_hash(image_file)
        except Exception as e:
            return jsonify({
                "error": "Could not process image.",
                "details": str(e)
            }), 400

        cache_key = f"{image_hash}:{age}"

        if cache_key in analysis_cache:
            cached = analysis_cache[cache_key].copy()
            cached["cached"] = True
            return jsonify(cached), 200

        result = analyze_image_with_fallback(image_data_url, age)

        result["cached"] = False

        if os.environ.get("SHOW_DEBUG", "false").lower() == "true":
            result["debug"] = {
                "filename": filename,
                "image_hash": image_hash[:12],
                "openrouter_model": get_openrouter_model(),
                "groq_vision_model": get_groq_vision_model()
            }

        analysis_cache[cache_key] = result
        sessions[result["session_id"]] = result

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
                "error": "API key issue. Check OPENROUTER_API_KEY and GROQ_API_KEY in Render."
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

Answer in a short, warm, simple way.
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
            temperature=0.45,
            top_p=0.9,
            max_completion_tokens=350
        )

        answer = completion.choices[0].message.content

        return jsonify({
            "answer": clean_text(answer, "I’m unable to answer that right now. Please try again.")
        }), 200

    except Exception as e:
        error_text = str(e)
        print("Ask error:", error_text)

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