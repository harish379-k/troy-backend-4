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

from flask import Flask, jsonify, request, Response

try:
    from groq import Groq
except ImportError as e:
    raise SystemExit("groq package not found. Run: pip install groq") from e

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("Pillow package not found. Run: pip install Pillow") from e

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("troy_analyzer")

# ---------------------------------------------------------------------------
# App & client
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALLOWED_MIME_TYPES   = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_IMAGE_DIMENSION  = 2048
GROQ_MODEL           = "meta-llama/llama-4-maverick-17b-128e-instruct"

VALID_CLASSIFICATIONS = {"valid_troy_build", "unclear_image", "non_troy_image"}
VALID_CATEGORIES      = {"tower", "bridge", "house", "vehicle", "abstract", "enclosure", "animal", "other"}
VALID_COMPLEXITY      = {"simple", "medium", "complex"}
VALID_STABILITY       = {"stable", "somewhat_stable", "precarious"}

# ---------------------------------------------------------------------------
# Prompt — maximum accuracy, shape-aware, anti-hallucination
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert visual analyst and child development specialist for Troy wooden block sets.

Your job is to look at an image and determine:
1. Whether it shows a Troy wooden block build
2. If yes — what the build actually looks like and resembles, based purely on its physical shape

---

## WHAT TROY WOODEN BLOCKS LOOK LIKE

Troy wooden blocks are:
- Solid wood — matte or satin finish, NOT shiny plastic, NOT foam, NOT cardboard
- Simple geometric shapes: cube, rectangular prism, cylinder, arch, triangular prism, semicircle, cone
- Sized for small children (roughly 5–15 cm per block)
- Colors: natural wood tan/beige, red, blue, yellow, green — flat solid colors only
- No studs, no connectors, no printed text or logos
- Edges slightly rounded for child safety
- Look heavy and solid, NOT hollow or transparent

## WHAT IS NOT A TROY BUILD

Classify as non_troy_image if you see:
- LEGO or Duplo (circular studs on top)
- Mega Bloks (large hollow plastic)
- Magnetic tiles (flat, translucent, plastic frames)
- Foam blocks (soft-looking, letters/numbers on them)
- Cardboard boxes
- K'NEX, Lincoln Logs, or connector-based toys
- Drawings or illustrations
- Random household objects
- People, animals, food, or scenery with no blocks present
- A single loose block not part of any build

---

## STEP 1 — CLASSIFY

Look at the entire image. Assign exactly one classification:

"valid_troy_build"
→ 2 or more Troy wooden blocks clearly and deliberately arranged into a structure. Material and shapes must be unambiguous.

"unclear_image"
→ Blurry, too dark, too cropped, only one block visible, or you cannot clearly confirm the blocks are Troy wooden. When in doubt, use this.

"non_troy_image"
→ Clearly shows something other than Troy wooden blocks.

---

## STEP 2 — SELF CHECK

Before writing any output, answer these internally:
1. Can I see at least 2 blocks clearly?
2. Do the blocks look like solid matte wood (not plastic or foam)?
3. Are the shapes simple geometric solids with no studs, connectors, or prints?
4. Is this a deliberate build — not just scattered blocks?
5. Am I at least 85% confident this is a Troy wooden block build?

If ANY answer is NO → use "unclear_image" or "non_troy_image". Never force a valid classification when uncertain.

---

## STEP 3 — VISUAL DESCRIPTION (critical — do this before naming)

If classification is "valid_troy_build", you MUST do this before deciding the build name:

Describe ONLY what you literally see in the image:
- What is the overall silhouette or outline of the build? (tall and thin? wide and low? has a bump on top? has legs?)
- How are the blocks physically arranged? (stacked vertically, spread horizontally, arch on top, blocks sticking out to the sides?)
- Are there any protruding parts, neck-like sections, leg-like sections, or roof-like sections?
- What does the overall shape remind you of when you look at it as a whole?

This raw visual description drives everything else. You are NOT allowed to guess the name before completing this.

---

## STEP 4 — SHAPE-BASED NAMING

Using ONLY your visual description from STEP 3, decide:

build_name
→ A short imaginative name (max 6 words) based on what the overall silhouette and shape actually looks like.
→ Common build shapes children make: tower, castle, house, bridge, car, train, rocket, robot, giraffe, dog, dinosaur, gate, wall, throne, barn, lighthouse, crane, boat.
→ If the build has a tall narrow section on top of a wide base → could be a giraffe, lighthouse, rocket, or tower.
→ If it has 4 leg-like protrusions and a body → could be a dog, horse, or table.
→ If it has a wide flat base with an arch → could be a bridge or gate.
→ NEVER name it something that contradicts your visual description. If your description says "tall narrow neck on a wide base" — do NOT call it a chair.

build_category
→ Exactly one of: "tower", "bridge", "house", "vehicle", "abstract", "enclosure", "animal", "other"
→ "animal" is now a valid category — use it when the build resembles a living creature.

---

## STEP 5 — FULL ANALYSIS

Provide all of the following:

block_count
→ Integer. Count every visible block including partially hidden ones you can clearly infer.

identified_shapes
→ List every distinct block shape TYPE you can see. Only include shapes you can actually confirm.
→ Choose from: "cube", "rectangular_prism", "cylinder", "arch", "triangular_prism", "semicircle", "cone", "square", "other"

complexity_level
→ Exactly one of:
  "simple"  → 1–4 blocks
  "medium"  → 5–10 blocks
  "complex" → 11+ blocks

stability_rating
→ Exactly one of:
  "stable"           → Wide base, low center of gravity, well-aligned
  "somewhat_stable"  → Moderate balance, some risk of toppling
  "precarious"       → Narrow base, tall, or visibly unbalanced

color_observations
→ List only the colors you can actually see on the blocks.
→ Example: ["natural wood", "red", "blue"]

spatial_observations
→ 1–2 sentences describing the literal physical arrangement — where blocks are placed relative to each other, what sticks out, what the silhouette looks like. Be precise and factual.

visual_reasoning
→ 1 sentence explaining WHY you named it what you did, based on the shape you saw.
→ Example: "Named 'Tall Giraffe Tower' because the build has a wide rectangular base with a long narrow vertical stack rising from one end, resembling a giraffe's body and neck."

developmental_feedback
→ 2–3 warm, encouraging sentences for the child. Reference specific shapes and arrangements you can actually see. Never generic — always tied to what's visible.

suggestions
→ Exactly 2 concrete, age-appropriate suggestions for extending the build. Reference the current structure.

---

## OUTPUT FORMAT

Single raw JSON object only. No prose, no markdown, no explanation outside the JSON.

If "valid_troy_build":
{
  "classification": "valid_troy_build",
  "build_name": "...",
  "block_count": <integer>,
  "identified_shapes": ["..."],
  "build_category": "...",
  "complexity_level": "...",
  "stability_rating": "...",
  "color_observations": ["..."],
  "spatial_observations": "...",
  "visual_reasoning": "...",
  "developmental_feedback": "...",
  "suggestions": ["...", "..."]
}

If "unclear_image":
{
  "classification": "unclear_image",
  "message": "The photo is too unclear to analyze. Please try again with better lighting, move back slightly so all the blocks are in frame, and make sure the image is in focus."
}

If "non_troy_image":
{
  "classification": "non_troy_image",
  "message": "This doesn't appear to be a Troy wooden block build. Please upload a photo of a structure built with Troy wooden blocks."
}

---

ABSOLUTE RULES:
- Complete STEP 3 visual description internally before deciding the build_name
- The build_name must be consistent with spatial_observations and visual_reasoning
- Never name something that contradicts what you described seeing
- Never output anything outside the JSON
- Never invent details you cannot see
- Your JSON must be valid — no trailing commas, no comments"""

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
@dataclass
class ValidBuildResponse:
    classification: str
    build_name: str
    block_count: int
    identified_shapes: list[str]
    build_category: str
    complexity_level: str
    stability_rating: str
    color_observations: list[str]
    spatial_observations: str
    visual_reasoning: str
    developmental_feedback: str
    suggestions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification":        self.classification,
            "build_name":            self.build_name,
            "block_count":           self.block_count,
            "identified_shapes":     self.identified_shapes,
            "build_category":        self.build_category,
            "complexity_level":      self.complexity_level,
            "stability_rating":      self.stability_rating,
            "color_observations":    self.color_observations,
            "spatial_observations":  self.spatial_observations,
            "visual_reasoning":      self.visual_reasoning,
            "developmental_feedback": self.developmental_feedback,
            "suggestions":           self.suggestions,
        }


@dataclass
class GatedResponse:
    classification: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"classification": self.classification, "message": self.message}


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
        from PIL import ImageEnhance
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = ImageEnhance.Sharpness(img).enhance(1.25)
        img = ImageEnhance.Color(img).enhance(1.1)
    except Exception:
        pass
    return img


def preprocess_image(raw_bytes: bytes, mime_type: str) -> tuple[str, str]:
    img = Image.open(BytesIO(raw_bytes))
    img = _auto_rotate(img)

    # Upscale very small images
    min_dim = min(img.width, img.height)
    if min_dim < 512:
        scale = 512 / min_dim
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # Downscale very large images
    max_dim = max(img.width, img.height)
    if max_dim > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max_dim
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    img = _enhance_image(img)

    # Normalize to RGB
    if img.mode in ("RGBA", "P", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(
            img,
            mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None
        )
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
                "content": (
                    "You are a precise visual analyst. "
                    "You always describe what you literally see before drawing conclusions. "
                    "You output only valid raw JSON — no markdown, no prose outside the JSON."
                ),
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

    # Strip markdown fences if present
    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        raw_text = parts[1] if len(parts) > 1 else raw_text
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    # Extract JSON object if there's stray text around it
    start = raw_text.find("{")
    end = raw_text.rfind("}") + 1
    if start != -1 and end > start:
        raw_text = raw_text[start:end]

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned non-JSON output: {raw_text[:200]}") from exc


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def validate_and_build(raw: dict[str, Any]) -> ValidBuildResponse | GatedResponse:
    classification = raw.get("classification")

    if classification not in VALID_CLASSIFICATIONS:
        raise ValueError(f"Invalid classification '{classification}'.")

    if classification in ("unclear_image", "non_troy_image"):
        message = raw.get("message", "")
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"Missing 'message' for '{classification}'.")
        return GatedResponse(classification=classification, message=message.strip())

    # valid_troy_build
    required_fields = [
        "build_name", "block_count", "identified_shapes",
        "build_category", "complexity_level", "stability_rating",
        "color_observations", "spatial_observations", "visual_reasoning",
        "developmental_feedback", "suggestions",
    ]
    missing = [f for f in required_fields if f not in raw]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    build_name = str(raw["build_name"]).strip()[:80]

    try:
        block_count = int(raw["block_count"])
        if block_count < 1:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"'block_count' must be a positive integer, got: {raw['block_count']!r}")

    identified_shapes = raw["identified_shapes"]
    if not isinstance(identified_shapes, list) or not identified_shapes:
        raise ValueError("'identified_shapes' must be a non-empty list.")
    identified_shapes = [str(s).lower().strip() for s in identified_shapes]

    build_category = str(raw["build_category"]).lower().strip()
    if build_category not in VALID_CATEGORIES:
        build_category = "other"

    complexity_level = str(raw["complexity_level"]).lower().strip()
    if complexity_level not in VALID_COMPLEXITY:
        raise ValueError(f"'complexity_level' must be one of {VALID_COMPLEXITY}.")

    stability_rating = str(raw["stability_rating"]).lower().strip()
    if stability_rating not in VALID_STABILITY:
        raise ValueError(f"'stability_rating' must be one of {VALID_STABILITY}.")

    color_observations = raw.get("color_observations", [])
    if not isinstance(color_observations, list):
        color_observations = []
    color_observations = [str(c).strip() for c in color_observations if str(c).strip()]

    spatial_observations  = str(raw["spatial_observations"]).strip()
    visual_reasoning      = str(raw["visual_reasoning"]).strip()
    developmental_feedback = str(raw["developmental_feedback"]).strip()
    if not developmental_feedback:
        raise ValueError("'developmental_feedback' must not be empty.")

    suggestions = raw["suggestions"]
    if not isinstance(suggestions, list):
        raise ValueError("'suggestions' must be a list.")
    suggestions = [str(s).strip() for s in suggestions if str(s).strip()][:2]

    return ValidBuildResponse(
        classification="valid_troy_build",
        build_name=build_name,
        block_count=block_count,
        identified_shapes=identified_shapes,
        build_category=build_category,
        complexity_level=complexity_level,
        stability_rating=stability_rating,
        color_observations=color_observations,
        spatial_observations=spatial_observations,
        visual_reasoning=visual_reasoning,
        developmental_feedback=developmental_feedback,
        suggestions=suggestions,
    )


# ---------------------------------------------------------------------------
# Request guard decorator
# ---------------------------------------------------------------------------
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
                f"Unsupported image type '{mime_type}'. "
                f"Accepted: {', '.join(sorted(ALLOWED_MIME_TYPES))}",
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )
        return f(*args, image_bytes=file.read(), mime_type=mime_type, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------
def error_response(message: str, status: int) -> tuple[Response, int]:
    return jsonify({"error": message}), status


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health() -> tuple[Response, int]:
    return jsonify({"status": "ok", "model": GROQ_MODEL}), HTTPStatus.OK


@app.route("/analyze", methods=["POST"])
@require_image_upload
def analyze(image_bytes: bytes, mime_type: str) -> tuple[Response, int]:
    try:
        b64_image, processed_mime = preprocess_image(image_bytes, mime_type)
    except Exception as exc:
        logger.exception("Image preprocessing failed")
        return error_response(f"Could not process image: {exc}", HTTPStatus.BAD_REQUEST)

    try:
        raw_output = call_groq_vision(b64_image, processed_mime)
    except ValueError as exc:
        logger.error("Groq returned unparseable output: %s", exc)
        return error_response(
            "The vision model returned an unexpected response. Please try again.",
            HTTPStatus.BAD_GATEWAY,
        )
    except Exception as exc:
        logger.exception("Groq API call failed")
        return error_response(f"Vision API error: {exc}", HTTPStatus.BAD_GATEWAY)

    try:
        result = validate_and_build(raw_output)
    except ValueError as exc:
        logger.error("Schema validation failed: %s | raw: %s", exc, raw_output)
        return error_response(
            "The vision model returned a structurally invalid response. Please try again.",
            HTTPStatus.BAD_GATEWAY,
        )

    logger.info(
        "Classification: %s | Build: %s",
        result.classification,
        getattr(result, "build_name", "n/a"),
    )
    return jsonify(result.to_dict()), HTTPStatus.OK


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)