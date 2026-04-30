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


def call_gemini_with_retry(model, parts, max_retries=2):
    delay = 2

    for attempt in range(max_retries):
        try:
            return model.generate_content(parts)
        except Exception as e:
            error_text = str(e)
            print(f"Gemini attempt {attempt + 1} failed:", error_text)

            if is_temporary_error(error_text) and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue

            raise


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


def build_rate_limit_response():
    return jsonify({
        "error": "AI usage limit reached right now. Please wait a minute and try again."
    }), 429


def build_service_retry_response():
    return jsonify({
        "error": "The AI analysis is temporarily unavailable. Please try again in a moment."
    }), 503


# -----------------------------
# Single-call image analysis
# -----------------------------
def analyze_troy_image_once(model, img, age):
    prompt = f"""
You are analyzing whether an uploaded image is a Troy wooden blocks build.

The child age is: {age if age else "unknown"}.

You must decide whether the image is:
1. a clear Troy blocks build that can be analyzed
2. an unclear / blurry / cropped / too-dark image
3. not a Troy blocks image or unrelated image

Return ONLY valid JSON in exactly one of these formats.

IF THE IMAGE IS A CLEAR TROY BLOCKS BUILD:
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

IF THE IMAGE IS UNCLEAR / BLURRY / CROPPED / TOO FEW BLOCKS VISIBLE:
{{
  "status": "success",
  "imageStatus": "invalid",
  "buildGuess": {{
    "title": "We couldn’t clearly analyze this image",
    "subtitle": "The image is too unclear to analyze properly. Please try again with a clearer photo."
  }},
  "whatWeFound": {{
    "title": "What we found",
    "summary": "The image appears blurry, cropped, dark, or does not show enough of the build clearly."
  }},
  "whatTheyLearned": [],
  "whatWeNoticed": [
    "short point 1",
    "short point 2",
    "short point 3"
  ],
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

IF THE IMAGE IS NOT A TROY BLOCKS IMAGE / UNRELATED:
{{
  "status": "success",
  "imageStatus": "invalid",
  "buildGuess": {{
    "title": "This doesn’t look like a Troy blocks build",
    "subtitle": "We could not identify a Troy wooden blocks structure in this image."
  }},
  "whatWeFound": {{
    "title": "What we found",
    "summary": "This image does not appear to show a Troy blocks construction."
  }},
  "whatTheyLearned": [],
  "whatWeNoticed": [
    "short point 1",
    "short point 2",
    "short point 3"
  ],
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

Important rules:
- Be strict
- Do NOT mark the image as valid unless it is clearly a Troy blocks structure and visible enough to analyze
- If the image is random, mostly a person, mostly background, unrelated object, or not a clear wooden block build, mark it invalid
- If the image is unclear, mark it invalid
- Output JSON only
- No markdown
- No explanation outside JSON
"""

    response = call_gemini_with_retry(model, [prompt, img])
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

        parsed = analyze_troy_image_once(model, img, age)

        image_status = str(parsed.get("imageStatus", "invalid")).strip().lower()

        if image_status == "valid":
            cards = ensure_learning_cards(parsed.get("whatTheyLearned"))
            if len(cards) < 3:
                return jsonify(build_invalid_photo_response(
                    age,
                    "We could not confidently analyze this image. Please try again with a clearer Troy blocks photo."
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
                "whatWeNoticed": [],
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

        # invalid path
        invalid_reason = str(parsed.get("whatWeFound", {}).get("summary", "")).strip()
        if not invalid_reason:
            invalid_reason = "We couldn’t clearly analyze this image."

        result = build_invalid_photo_response(
            age,
            invalid_reason,
            noticed=ensure_list(
                parsed.get("whatWeNoticed"),
                [
                    "The image may be unclear or not related to Troy blocks",
                    "Too little of the build may be visible",
                    "A clearer image will help us analyze properly"
                ]
            ),
            suggestions=ensure_list(
                parsed.get("suggestionsForParent"),
                [
                    "Retake the photo with the full structure visible",
                    "Use better lighting and a cleaner background",
                    "Make sure the Troy block build is the main focus of the image"
                ]
            ),
            ideas=ensure_list(
                parsed.get("nextBuildIdeas"),
                [
                    "Build a tower",
                    "Build a bridge",
                    "Build a small house"
                ]
            )
        )

        result["buildGuess"] = {
            "title": str(parsed.get("buildGuess", {}).get("title", "We couldn’t clearly analyze this image")).strip(),
            "subtitle": str(parsed.get("buildGuess", {}).get("subtitle", invalid_reason)).strip()
        }

        result["whatWeFound"] = {
            "title": "What we found",
            "summary": invalid_reason
        }

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

        if is_quota_error(error_text):
            return build_rate_limit_response()

        if is_temporary_error(error_text):
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

        if is_quota_error(error_text):
            return jsonify({
                "answer": "AI usage limit reached right now. Please wait a minute and try again."
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
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)