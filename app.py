# app.py
from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from functools import wraps
from http import HTTPStatus
from io import BytesIO
from typing import Any
import uuid

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

try:
    from groq import Groq
except ImportError as e:
    raise SystemExit("groq package not found. Run: pip install groq") from e

try:
    from PIL import Image, ImageEnhance
except ImportError as e:
    raise SystemExit("Pillow not found. Run: pip install Pillow") from e

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("troy_analyzer")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
CORS(app, origins=["https://troy-frontend-alpha.vercel.app", "http://localhost:3000"])

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

ALLOWED_MIME_TYPES  = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_IMAGE_DIMENSION = 2048
GROQ_MODEL          = "meta-llama/llama-4-scout-17b-16e-instruct"

sessions = {}

SYSTEM_PROMPT = """You are an expert visual analyst and child development specialist for Troy wooden block sets.

Your job is to look at an image and determine:
1. Whether it shows a Troy wooden block build
2. If yes — what the build actually looks like, based purely on its physical shape
3. Provide structured developmental feedback

---

## WHAT TROY WOODEN BLOCKS LOOK LIKE

Troy wooden blocks are:
- Solid wood — matte or satin finish, NOT shiny plastic, NOT foam, NOT cardboard
- Simple geometric shapes: cube, rectangular prism, cylinder, arch, triangular prism, semicircle, cone
- Sized for small children (roughly 5-15 cm per block)
- Colors: natural wood tan/beige, red, blue, yellow, green — flat solid colors only
- No studs, no connectors, no printed text or logos
- Edges slightly rounded for child safety
- Look heavy and solid, NOT hollow or transparent

## WHAT IS NOT A TROY BUILD

Classify as invalid if you see:
- LEGO or Duplo (circular studs on top)
- Mega Bloks (large hollow plastic)
- Magnetic tiles (flat, translucent, plastic frames)
- Foam blocks (soft-looking, letters/numbers on them)
- Cardboard boxes
- Random household objects
- People, animals, food, or scenery with no blocks present
- A single loose block not part of any build

---

## STEP 1 — CLASSIFY

"valid" → 2 or more Troy wooden blocks clearly and deliberately arranged into a structure.
"invalid" → Blurry, too dark, too cropped, only one block, not Troy blocks, or unrelated image.

---

## STEP 2 — SELF CHECK

Answer these internally before writing output:
1. Can I see at least 2 blocks clearly?
2. Do the blocks look like solid matte wood?
3. Are shapes simple geometric solids with no studs or prints?
4. Is this a deliberate build — not just scattered blocks?
5. Am I at least 85% confident this is a Troy wooden block build?

If ANY answer is NO → imageStatus must be "invalid".

---

## STEP 3 — DEEP VISUAL ANALYSIS (only if valid)

SILHOUETTE: What is the overall outline?
- Tall and thin? Wide and low? Any bumps, long neck, four legs, pointed top?

STRUCTURE: How are blocks arranged?
- Any blocks sticking out like legs, arms, or wings?
- Any narrow neck connecting two wider sections?
- Any small block on top like a head?
- Any gap or opening like a bridge?

RESEMBLANCE: What does this most closely look like?
- Animals: giraffe, dog, snake, bird, dinosaur, elephant
- Buildings: house, castle, lighthouse, tower, barn
- Vehicles: car, train, rocket, boat
- Other: bridge, gate, table, chair

---

## STEP 4 — NAMING

buildGuess title must match STEP 3:
- TALL NARROW stack on wider base = giraffe, lighthouse, rocket, tower
- FOUR OUTWARD PROTRUSIONS = dog, horse, table
- WIDE BASE with pointed top = house, barn, castle
- GAP in middle = bridge, gate, arch
- LONG HORIZONTAL body = train, snake, crocodile
- NEVER name something that contradicts your visual observations

---

## OUTPUT FORMAT

Return ONLY valid JSON. No markdown. No prose outside JSON.

IF VALID TROY BUILD:
{
  "imageStatus": "valid",
  "buildGuess": {
    "title": "short creative name based on actual shape (max 6 words)",
    "subtitle": "one sentence describing what the child likely built based on the shape"
  },
  "whatWeFound": {
    "title": "What we found",
    "summary": "1-2 sentences describing the actual structure and arrangement you see"
  },
  "whatTheyLearned": [
    {
      "title": "skill name",
      "description": "short specific description of what this build shows developmentally",
      "color": "cream"
    },
    {
      "title": "skill name",
      "description": "short specific description of what this build shows developmentally",
      "color": "green"
    },
    {
      "title": "skill name",
      "description": "short specific description of what this build shows developmentally",
      "color": "blue"
    }
  ],
  "whatWeNoticed": [],
  "suggestionsForParent": [
    "specific suggestion referencing the actual build",
    "specific suggestion referencing the actual build",
    "specific suggestion referencing the actual build"
  ],
  "nextBuildIdeas": [
    "short idea 1",
    "short idea 2",
    "short idea 3"
  ]
}

IF INVALID / UNCLEAR / NOT TROY:
{
  "imageStatus": "invalid",
  "buildGuess": {
    "title": "We couldn't clearly analyze this image",
    "subtitle": "Please try again with a clearer photo of your Troy block build."
  },
  "whatWeFound": {
    "title": "What we found",
    "summary": "The image does not clearly show a Troy wooden block build."
  },
  "whatTheyLearned": [],
  "whatWeNoticed": [
    "point 1",
    "point 2",
    "point 3"
  ],
  "suggestionsForParent": [
    "Retake the photo with the full structure visible",
    "Use better lighting and a cleaner background",
    "Make sure the Troy block build is the main focus"
  ],
  "nextBuildIdeas": [
    "Build a tower",
    "Build a bridge",
    "Build a small house"
  ]
}

ABSOLUTE RULES:
- Always complete STEP 3 before deciding buildGuess title
- buildGuess title must match what you actually see
- Never name something that contradicts your visual observations
- Output JSON only — no markdown, no explanation outside JSON
- Never output "Creative Troy block build" as a title — always name based on actual shape"""


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------
def _auto_rotate(img: Image.Image) -> Image.Image:
    try:
        from PIL import ExifTags
        exif = img._getexif()
        if exif:
            for tag, val in exif.items():
                if ExifTags.TAGS.get(tag) == "Orientation":
                    rotations = {3: 180, 6: 270, 8: 90}
                    if val in rotations:
                        img = img.rotate(rotations[val], expand=True)
                    break
    except Exception:
        pass
    return img


def _enhance_image(img: Image.Image) -> Image.Image:
    try:
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = ImageEnhance.Sharpness(img).enhance(1.25)
        img = ImageEnhance.Color(img).enhance(1.1)
    except Exception:
        pass
    return img


def preprocess_image(raw_bytes: bytes, mime_type: str) -> tuple[str, str]:
    img = Image.open(BytesIO(raw_bytes))
    img = _auto_rotate(img)

    min_dim = min(img.width, img.height)
    if min_dim < 512:
        scale = 512 / min_dim
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    max_dim = max(img.width, img.height)
    if max_dim > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max_dim
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    img = _enhance_image(img)

    if img.mode in ("RGBA", "P", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return encoded, "image/jpeg"


# ---------------------------------------------------------------------------
# Groq vision call
# ---------------------------------------------------------------------------
def call_groq_vision(b64_image: str, media_type: str) -> dict[str, Any]:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.0,
        max_tokens=1200,
        top_p=1.0,
        stream=False,
        messages=[
            {
                "role": "system",
                "content": "You are a precise visual analyst. Output only valid raw JSON. No markdown, no prose outside the JSON.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_image}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    },
                ],
            },
        ],
    )

    raw_text = response.choices[0].message.content.strip()
    logger.info("Raw Groq response: %s", raw_text[:500])

    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        raw_text = parts[1] if len(parts) > 1 else raw_text
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    start = raw_text.find("{")
    end   = raw_text.rfind("}") + 1
    if start != -1 and end > start:
        raw_text = raw_text[start:end]

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned non-JSON output: {raw_text[:200]}") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_list(value, fallback=None):
    if isinstance(value, list):
        result = [str(x).strip() for x in value if str(x).strip()]
        return result if result else (fallback or [])
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return fallback or []


def ensure_learning_cards(cards):
    if not isinstance(cards, list):
        return []
    cleaned = []
    allowed_colors = {"cream", "green", "blue"}
    for card in cards:
        if not isinstance(card, dict):
            continue
        title       = str(card.get("title", "")).strip()
        description = str(card.get("description", "")).strip()
        color       = str(card.get("color", "cream")).strip().lower()
        if not title or not description:
            continue
        if color not in allowed_colors:
            color = "cream"
        cleaned.append({"title": title, "description": description, "color": color})
    return cleaned[:3]


def error_response(message: str, status: int) -> tuple[Response, int]:
    return jsonify({"error": message}), status


def require_image_upload(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "image" not in request.files:
            return error_response("No image field in request.", HTTPStatus.BAD_REQUEST)
        file = request.files["image"]
        if file.filename == "":
            return error_response("Empty filename.", HTTPStatus.BAD_REQUEST)
        mime_type = file.content_type or ""
        if mime_type not in ALLOWED_MIME_TYPES:
            return error_response(
                f"Unsupported image type '{mime_type}'.",
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )
        return f(*args, image_bytes=file.read(), mime_type=mime_type, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Troy backend is running", "server": "ok"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": GROQ_MODEL})


@app.route("/analyze", methods=["POST"])
@require_image_upload
def analyze(image_bytes: bytes, mime_type: str) -> tuple[Response, int]:
    age = request.form.get("age", "").strip()

    try:
        b64_image, processed_mime = preprocess_image(image_bytes, mime_type)
    except Exception as exc:
        logger.exception("Image preprocessing failed")
        return error_response(f"Could not process image: {exc}", HTTPStatus.BAD_REQUEST)

    try:
        parsed = call_groq_vision(b64_image, processed_mime)
    except ValueError as exc:
        logger.error("Groq returned unparseable output: %s", exc)
        return error_response(
            "The vision model returned an unexpected response. Please try again.",
            HTTPStatus.BAD_GATEWAY,
        )
    except Exception as exc:
        logger.exception("Groq API call failed")
        error_msg = str(exc)
        if "429" in error_msg or "rate_limit" in error_msg.lower():
            return error_response(
                "Too many requests. Please wait a moment and try again.",
                HTTPStatus.TOO_MANY_REQUESTS,
            )
        return error_response(f"Vision API error: {exc}", HTTPStatus.BAD_GATEWAY)

    image_status = str(parsed.get("imageStatus", "invalid")).strip().lower()
    session_id   = str(uuid.uuid4())

    if image_status == "valid":
        cards = ensure_learning_cards(parsed.get("whatTheyLearned"))
        if len(cards) < 3:
            return jsonify({
                "status": "success",
                "imageStatus": "invalid",
                "buildGuess": {
                    "title": "We couldn't clearly analyze this image",
                    "subtitle": "Please try again with a clearer Troy blocks photo."
                },
                "whatWeFound": {
                    "title": "What we found",
                    "summary": "We could not confidently analyze this image."
                },
                "whatTheyLearned": [],
                "whatWeNoticed": [
                    "The image does not clearly show enough visible Troy blocks",
                    "A clearer photo will help us give better feedback",
                    "Make sure all blocks are visible and in focus"
                ],
                "suggestionsForParent": [
                    "Retake the photo with the full structure visible",
                    "Use better lighting and a cleaner background",
                    "Make sure the Troy block build is the main focus"
                ],
                "nextBuildIdeas": ["Build a tower", "Build a bridge", "Build a small house"],
                "session_id": session_id
            }), 200

        result = {
            "status":      "success",
            "imageStatus": "valid",
            "buildGuess": {
                "title":    str(parsed.get("buildGuess", {}).get("title", "")).strip() or "A Creative Block Build",
                "subtitle": str(parsed.get("buildGuess", {}).get("subtitle", "")).strip()
            },
            "whatWeFound": {
                "title":   "What we found",
                "summary": str(parsed.get("whatWeFound", {}).get("summary", "")).strip()
            },
            "whatTheyLearned": cards,
            "whatWeNoticed":   [],
            "suggestionsForParent": ensure_list(
                parsed.get("suggestionsForParent"),
                ["Ask your child to explain what they built",
                 "Encourage them to rebuild it taller or wider",
                 "Try making a stronger version together"]
            ),
            "nextBuildIdeas": ensure_list(
                parsed.get("nextBuildIdeas"),
                ["Build a bridge", "Build a tower", "Build a small castle"]
            ),
            "session_id": session_id
        }

        logger.info("Valid build: %s", result["buildGuess"]["title"])
        sessions[session_id] = result
        return jsonify(result), 200

    # invalid path
    invalid_reason = str(parsed.get("whatWeFound", {}).get("summary", "")).strip()
    if not invalid_reason:
        invalid_reason = "We couldn't clearly analyze this image."

    result = {
        "status":      "success",
        "imageStatus": "invalid",
        "buildGuess": {
            "title":    str(parsed.get("buildGuess", {}).get("title", "We couldn't clearly analyze this image")).strip(),
            "subtitle": str(parsed.get("buildGuess", {}).get("subtitle", invalid_reason)).strip()
        },
        "whatWeFound": {
            "title":   "What we found",
            "summary": invalid_reason
        },
        "whatTheyLearned": [],
        "whatWeNoticed": ensure_list(
            parsed.get("whatWeNoticed"),
            ["The image may be unclear or not related to Troy blocks",
             "Too little of the build may be visible",
             "A clearer image will help us analyze properly"]
        ),
        "suggestionsForParent": ensure_list(
            parsed.get("suggestionsForParent"),
            ["Retake the photo with the full structure visible",
             "Use better lighting and a cleaner background",
             "Make sure the Troy block build is the main focus"]
        ),
        "nextBuildIdeas": ensure_list(
            parsed.get("nextBuildIdeas"),
            ["Build a tower", "Build a bridge", "Build a small house"]
        ),
        "session_id": session_id
    }

    sessions[session_id] = result
    return jsonify(result), 200


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data     = request.get_json()
        question = str(data.get("question", "")).strip()
        summary  = str(data.get("summary", "")).strip()

        if not question:
            return jsonify({"error": "Question is required"}), 400

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": "You are helping a parent understand their child's Troy wooden block build. Answer warmly and simply in 3-5 short lines."
                },
                {
                    "role": "user",
                    "content": f"Build summary:\n{summary}\n\nParent question:\n{question}"
                }
            ]
        )

        return jsonify({"answer": response.choices[0].message.content.strip()}), 200

    except Exception as exc:
        error_text = str(exc)
        logger.error("Ask error: %s", error_text)
        if "429" in error_text or "rate_limit" in error_text.lower():
            return jsonify({"answer": "AI usage limit reached. Please wait a minute and try again."}), 200
        return jsonify({"error": "Something went wrong", "details": error_text}), 500


if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)