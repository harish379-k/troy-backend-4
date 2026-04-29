import os
import json
import uuid
import time
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai

# -----------------------------
# App setup
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "heic", "heif"}

app = Flask(__name__)
CORS(app, origins="*")

sessions = {}


# -----------------------------
# Gemini config helpers
# -----------------------------
def get_api_key():
    hex_key = os.environ.get("RENDER_GEMINI_KEY_HEX", "").strip()
    if hex_key:
        try:
            return bytes.fromhex(hex_key).decode("utf-8").strip()
        except Exception as e:
            print("Failed to decode RENDER_GEMINI_KEY_HEX:", e)

    return (
        os.environ.get("RENDER_GEMINI_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )


def get_model_name():
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()


def build_model():
    api_key = get_api_key()
    model_name = get_model_name()

    if not api_key:
        return None, api_key, model_name

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name,
        generation_config={
            "temperature": 0.2
        }
    )
    return model, api_key, model_name


print("Loaded Gemini key:", "FOUND" if get_api_key() else "NOT FOUND")
print("Model:", get_model_name())


# -----------------------------
# Utility helpers
# -----------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_json_block(text: str) -> str:
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


def parse_json_response(text: str):
    cleaned = extract_json_block(text)
    return json.loads(cleaned)


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

        title = str(card.get("title", "")).strip()
        description = str(card.get("description", "")).strip()
        color = str(card.get("color", "cream")).strip().lower()

        if not title or not description:
            continue

        if color not in allowed_colors:
            color = "cream"

        cleaned.append({
            "title": title,
            "description": description,
            "color": color
        })

    return cleaned[:3]


def is_quota_error(error_text: str) -> bool:
    lower = error_text.lower()
    return (
        "429" in error_text
        or "resource_exhausted" in lower
        or "quota" in lower
        or "free_tier_requests" in lower
    )


def is_temporary_error(error_text: str) -> bool:
    lower = error_text.lower()
    return "503" in error_text or "unavailable" in lower


def is_leaked_key_error(error_text: str) -> bool:
    lower = error_text.lower()
    return (
        "403" in error_text
        and (
            "leaked" in lower
            or "reported as leaked" in lower
            or "api_key_service_blocked" in lower
        )
    )


def is_invalid_key_error(error_text: str) -> bool:
    lower = error_text.lower()
    return (
        "api_key_invalid" in lower
        or "api key not valid" in lower
        or "api_key_service_blocked" in lower
    )


def call_gemini(model, parts):
    return model.generate_content(parts)


# -----------------------------
# Response builders
# -----------------------------
def build_invalid_photo_response(age, reason, noticed=None, suggestions=None, ideas=None):
    session_id = str(uuid.uuid4())

    result = {
        "status": "success",
        "imageStatus": "invalid",
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
            "The image does not clearly show enough visible Troy blocks",
            "The structure may be blurry, cropped, too far away, or unrelated to Troy blocks",
            "A clearer photo will help us give the right feedback"
        ],
        "suggestionsForParent": suggestions or [
            "Retake the photo with the full structure visible",
            "Use better lighting and a cleaner background",
            "Make sure the Troy block build is the main focus of the image"
        ],
        "nextBuildIdeas": ideas or [
            "Build a tower",
            "Build a bridge",
            "Build a small house"
        ],
        "session_id": session_id
    }

    sessions[session_id] = result
    return result


def build_service_retry_response():
    return jsonify({
        "error": "The AI analysis is temporarily unavailable. Please try again in a moment."
    }), 503


# -----------------------------
# Strict image classification
# -----------------------------
def classify_troy_image(model, img):
    prompt = """
You are checking whether an image can be analyzed as a Troy wooden blocks build.

Return ONLY valid JSON in exactly this format:
{
  "isTroyBuild": true,
  "isImageClear": true,
  "enoughBlocksVisible": true,
  "visibleBlockCountEstimate": 10,
  "confidence": 0.92,
  "reason": "short reason"
}

Rules:
- isTroyBuild = true only if the image clearly shows a Troy wooden block build or structure
- If the image is random, unrelated, mostly a person, mostly background, toys other than Troy blocks, or not a wooden block build, set isTroyBuild = false
- isImageClear = false if the image is blurry, too dark, too cropped, too far away, or unclear
- enoughBlocksVisible = false if there are too few visible blocks to analyze meaningfully
- visibleBlockCountEstimate should be an integer estimate
- confidence must be between 0 and 1
- reason should be short and practical
- Output JSON only
"""
    response = call_gemini(model, [prompt, img])
    return parse_json_response(response.text)


# -----------------------------
# Main analysis
# -----------------------------
def analyze_troy_build(model, img, age):
    prompt = f"""
You are analyzing a child's building made with Troy wooden blocks.

The child age is: {age if age else "unknown"}.

Return ONLY valid JSON in exactly this format:
{{
  "status": "success",
  "imageStatus": "valid",
  "buildGuess": {{
    "title": "short guessed build name",
    "subtitle": "one short sentence describing what the child likely built"
  }},
  "whatWeFound": {{
    "title": "What we found",
    "summary": "1 to 2 short sentences"
  }},
  "whatTheyLearned": [
    {{
      "title": "skill name",
      "description": "short description",
      "color": "cream"
    }},
    {{
      "title": "skill name",
      "description": "short description",
      "color": "green"
    }},
    {{
      "title": "skill name",
      "description": "short description",
      "color": "blue"
    }}
  ],
  "whatWeNoticed": [],
  "suggestionsForParent": [
    "short suggestion 1",
    "short suggestion 2",
    "short suggestion 3"
  ],
  "nextBuildIdeas": [
    "short idea 1",
    "short idea 2",
    "short idea 3"
  ]
}}

Rules:
- Guess the structure from what is visible
- Keep the language simple and parent-friendly
- Do not use long paragraphs
- Do not include markdown
- Do not include any extra explanation outside JSON
"""
    response = call_gemini(model, [prompt, img])
    return parse_json_response(response.text)


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Troy backend is running",
        "server": "ok"
    })


@app.route("/health", methods=["GET"])
def health():
    raw_key = get_api_key()
    raw_hex = os.environ.get("RENDER_GEMINI_KEY_HEX", "")
    return jsonify({
        "status": "ok",
        "env_has_render_gemini_key_hex": "RENDER_GEMINI_KEY_HEX" in os.environ,
        "env_has_render_gemini_key": "RENDER_GEMINI_KEY" in os.environ,
        "env_has_gemini_key": "GEMINI_API_KEY" in os.environ,
        "env_has_gemini_model": "GEMINI_MODEL" in os.environ,
        "hex_length": len(raw_hex),
        "gemini_key_loaded": bool(raw_key),
        "key_length": len(raw_key),
        "key_preview": (raw_key[:6] + "...") if raw_key else "NONE",
        "model": get_model_name()
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    age = request.form.get("age", "").strip()

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        image_file = request.files["image"]

        if image_file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(image_file.filename):
            return jsonify({"error": "Invalid file type. Use png, jpg, jpeg, webp, heic, or heif"}), 400

        model, current_api_key, model_name = build_model()

        if not current_api_key:
            return jsonify({"error": "GEMINI_API_KEY not found"}), 500

        file_ext = image_file.filename.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4()}.{file_ext}"
        filepath = UPLOAD_FOLDER / filename
        image_file.save(str(filepath))

        try:
            img = Image.open(filepath)
        except Exception:
            return jsonify({
                "error": "Could not open this image. Please try JPG, PNG, or WEBP."
            }), 400

        # First: strict validation
        classification = classify_troy_image(model, img)

        is_troy = bool(classification.get("isTroyBuild", False))
        is_clear = bool(classification.get("isImageClear", False))
        enough_blocks = bool(classification.get("enoughBlocksVisible", False))
        visible_count = int(classification.get("visibleBlockCountEstimate", 0) or 0)
        confidence = float(classification.get("confidence", 0) or 0)
        reason = str(classification.get("reason", "")).strip() or "We could not clearly analyze this image."

        if not is_troy:
            return jsonify(build_invalid_photo_response(
                age,
                "This image does not appear to show a Troy blocks build.",
                noticed=[
                    "The image does not look like a Troy wooden blocks structure",
                    "The main subject may be unrelated to Troy blocks",
                    "Please upload a clear Troy block build image"
                ],
                suggestions=[
                    "Upload a photo where the Troy block structure is clearly visible",
                    "Make sure the build is the main subject of the image",
                    "Try again with a proper Troy blocks construction photo"
                ]
            )), 200

        if not is_clear:
            return jsonify(build_invalid_photo_response(
                age,
                "The image is too unclear to analyze properly. Please try again with a clearer photo.",
                noticed=[
                    "The image appears blurry, cropped, dark, or unclear",
                    "The structure is not visible enough for proper analysis",
                    "We need a clearer view of the Troy build"
                ],
                suggestions=[
                    "Retake the photo with better lighting",
                    "Make sure the whole structure is visible",
                    "Move closer and keep the image steady"
                ]
            )), 200

        if not enough_blocks or visible_count < 4 or confidence < 0.65:
            return jsonify(build_invalid_photo_response(
                age,
                "Not enough Troy blocks are visible to analyze this build confidently.",
                noticed=[
                    "Too few blocks are visible in the image",
                    "The build may be too small, too cropped, or partially hidden",
                    "We need a fuller view of the structure"
                ],
                suggestions=[
                    "Retake the photo showing more of the build",
                    "Capture the full structure from a little farther back",
                    "Make sure the block arrangement is fully visible"
                ]
            )), 200

        # Second: full analysis only if valid
        parsed = analyze_troy_build(model, img, age)

        cards = ensure_learning_cards(parsed.get("whatTheyLearned"))
        if len(cards) < 3:
            return jsonify(build_invalid_photo_response(
                age,
                "We could not confidently analyze this photo. Please try again with a clearer Troy block build image."
            )), 200

        session_id = str(uuid.uuid4())

        result = {
            "status": "success",
            "imageStatus": "valid",
            "buildGuess": {
                "title": str(parsed.get("buildGuess", {}).get("title", "Creative Troy block build")).strip(),
                "subtitle": str(parsed.get("buildGuess", {}).get("subtitle", "Your little one created a thoughtful Troy block structure!")).strip()
            },
            "whatWeFound": {
                "title": "What we found",
                "summary": str(parsed.get("whatWeFound", {}).get("summary", "This looks like a meaningful Troy block build.")).strip()
            },
            "whatTheyLearned": cards,
            "whatWeNoticed": ensure_list(parsed.get("whatWeNoticed"), []),
            "suggestionsForParent": ensure_list(
                parsed.get("suggestionsForParent"),
                [
                    "Ask your child to explain what they built",
                    "Encourage them to rebuild it taller or wider",
                    "Try making a stronger version together"
                ]
            ),
            "nextBuildIdeas": ensure_list(
                parsed.get("nextBuildIdeas"),
                [
                    "Build a bridge",
                    "Build a tower",
                    "Build a small castle"
                ]
            ),
            "session_id": session_id
        }

        sessions[session_id] = result
        return jsonify(result), 200

    except Exception as e:
        error_text = str(e)
        print("Analyze error:", error_text)

        if is_leaked_key_error(error_text):
            return jsonify({
                "error": "Gemini API key was blocked as leaked. Create a new key and keep it only in Render environment variables."
            }), 403

        if is_invalid_key_error(error_text):
            return jsonify({
                "error": "Gemini API key is invalid. Add a fresh key in Render environment variables."
            }), 403

        if is_quota_error(error_text) or is_temporary_error(error_text):
            return build_service_retry_response()

        return jsonify({
            "error": "Something went wrong",
            "details": error_text
        }), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = str(data.get("question", "")).strip()
        summary = str(data.get("summary", "")).strip()

        if not question:
            return jsonify({"error": "Question is required"}), 400

        model, current_api_key, model_name = build_model()

        if not current_api_key:
            return jsonify({"error": "GEMINI_API_KEY not found"}), 500

        prompt = f"""
You are helping a parent understand their child's Troy block build.

Build summary:
{summary}

Parent question:
{question}

Answer in a short, warm, simple way for a parent.
Keep it to 3 to 5 short lines.
"""

        response = model.generate_content(prompt)

        return jsonify({
            "answer": response.text.strip()
        }), 200

    except Exception as e:
        error_text = str(e)
        print("Ask error:", error_text)

        if is_leaked_key_error(error_text):
            return jsonify({
                "error": "Gemini API key was blocked as leaked. Create a new key and keep it only in Render environment variables."
            }), 403

        if is_invalid_key_error(error_text):
            return jsonify({
                "error": "Gemini API key is invalid. Add a fresh key in Render environment variables."
            }), 403

        if is_quota_error(error_text) or is_temporary_error(error_text):
            return jsonify({
                "answer": "Live AI Q&A is temporarily unavailable right now. Please try again later."
            }), 200

        return jsonify({
            "error": "Something went wrong",
            "details": error_text
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)