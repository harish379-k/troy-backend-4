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

# Temporary in-memory session storage
sessions = {}


# -----------------------------
# Gemini config helpers
# -----------------------------
def get_api_key():
    return (
        os.environ.get("AIzaSyDnPgWAr07vZ84orhgaLGoJq140XGTrJr0", "").strip()
        or os.environ.get("AIzaSyDnPgWAr07vZ84orhgaLGoJq140XGTrJr0", "").strip()
    )


def get_model_name():
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()


def build_model():
    api_key = get_api_key()
    model_name = get_model_name()

    if not api_key:
        return None, api_key, model_name

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    return model, api_key, model_name


print("Loaded Gemini key:", "FOUND" if get_api_key() else "NOT FOUND")
print("Model:", get_model_name())


# -----------------------------
# Helpers
# -----------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return text


def ensure_list(value, fallback=None):
    if isinstance(value, list):
        cleaned = [str(x).strip() for x in value if str(x).strip()]
        if cleaned:
            return cleaned
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return fallback or []


def ensure_learning_cards(cards):
    if not isinstance(cards, list):
        return []

    cleaned = []
    allowed_colors = {"cream", "green", "blue", "amber", "mint"}

    for card in cards:
        if not isinstance(card, dict):
            continue

        title = str(card.get("title", "")).strip()
        description = str(card.get("description", "")).strip()
        color = str(card.get("color", card.get("theme", "cream"))).strip().lower()

        if not title or not description:
            continue

        if color not in allowed_colors:
            color = "cream"

        if color == "amber":
            color = "cream"
        if color == "mint":
            color = "green"

        cleaned.append({
            "title": title,
            "description": description,
            "color": color
        })

    return cleaned


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


def call_gemini_with_retry(model, prompt, img):
    try:
        return model.generate_content([prompt, img])
    except Exception as e:
        error_text = str(e)

        if is_temporary_error(error_text):
            print("Gemini temporarily unavailable. Retrying in 3 seconds...")
            time.sleep(3)
            return model.generate_content([prompt, img])

        raise


def build_invalid_photo_response(age, reason="We could not clearly identify a Troy block creation in this image."):
    session_id = str(uuid.uuid4())

    result = {
        "status": "success",
        "imageStatus": "invalid",
        "buildGuess": {
            "title": "Not enough Troy blocks detected",
            "subtitle": "We could not clearly identify a Troy block creation in this image."
        },
        "whatWeFound": {
            "title": "What we found",
            "summary": reason
        },
        "whatTheyLearned": [],
        "whatWeNoticed": [
            "Not enough visible Troy blocks were detected",
            "The structure may be too far, unclear, cropped, or unrelated to Troy blocks",
            "A clearer photo will help us give better feedback"
        ],
        "suggestionsForParent": [
            "Retake the photo with the full structure visible",
            "Use a cleaner background if possible",
            "Try again with better lighting"
        ],
        "nextBuildIdeas": [
            "Build a tower",
            "Build a bridge",
            "Build a small house"
        ],
        "session_id": session_id
    }

    sessions[session_id] = {
        "age": age,
        "imageStatus": "invalid",
        "buildGuess": result["buildGuess"],
        "whatWeFound": result["whatWeFound"],
        "whatTheyLearned": result["whatTheyLearned"],
        "whatWeNoticed": result["whatWeNoticed"],
        "suggestionsForParent": result["suggestionsForParent"],
        "nextBuildIdeas": result["nextBuildIdeas"]
    }

    return result


def build_fallback_valid_response(age, note_text="Live AI feedback is temporarily unavailable, so a fallback result was used."):
    session_id = str(uuid.uuid4())

    result = {
        "status": "success",
        "imageStatus": "valid",
        "buildGuess": {
            "title": "Creative Troy block build",
            "subtitle": "Your little one just built a thoughtful Troy structure and explored how pieces work together!"
        },
        "whatWeFound": {
            "title": "What we found",
            "summary": "This looks like a meaningful Troy block structure with balance, imagination, and careful placement."
        },
        "whatTheyLearned": [
            {
                "title": "Balance & Gravity",
                "description": "They explored how support underneath helps the structure stay stable.",
                "color": "cream"
            },
            {
                "title": "Spatial Awareness",
                "description": "They practiced understanding shape, size, and placement in space.",
                "color": "green"
            },
            {
                "title": "Patience & Focus",
                "description": "Careful placement helped build concentration and control.",
                "color": "blue"
            }
        ],
        "whatWeNoticed": [],
        "suggestionsForParent": [
            "Ask your child to explain what they built",
            "Encourage them to rebuild it taller or wider",
            "Try making a stronger version together"
        ],
        "nextBuildIdeas": [
            "Build a bridge",
            "Build a tower",
            "Build a small castle"
        ],
        "note": note_text,
        "session_id": session_id
    }

    sessions[session_id] = {
        "age": age,
        "imageStatus": "valid",
        "buildGuess": result["buildGuess"],
        "whatWeFound": result["whatWeFound"],
        "whatTheyLearned": result["whatTheyLearned"],
        "whatWeNoticed": result["whatWeNoticed"],
        "suggestionsForParent": result["suggestionsForParent"],
        "nextBuildIdeas": result["nextBuildIdeas"]
    }

    return result


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
    model_name = get_model_name()

    return jsonify({
        "status": "ok",
        "env_has_render_gemini_key": "RENDER_GEMINI_KEY" in os.environ,
        "env_has_gemini_key": "GEMINI_API_KEY" in os.environ,
        "env_has_gemini_model": "GEMINI_MODEL" in os.environ,
        "gemini_key_loaded": bool(raw_key),
        "key_length": len(raw_key),
        "key_preview": (raw_key[:6] + "...") if raw_key else "NONE",
        "model": model_name
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        image_file = request.files["image"]
        age = request.form.get("age", "").strip()

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

        prompt = f"""
You are analyzing a child's building made with Troy wooden blocks.

The child age is: {age if age else "unknown"}.

IMPORTANT:
- Output must be simple, warm, readable, and exciting for parents.
- Do not produce long paragraphs.
- Do not include a celebration section.
- Do not include any ask/chat section.
- First try to GUESS what the child built.
- If the image is random, unclear, unrelated, or does not show enough visible Troy blocks, mark it as invalid.
- If invalid, clearly say there are not enough Troy blocks visible for analysis.
- Never overclaim. If unsure, say "looks like" or "appears to be".

Return ONLY valid JSON with EXACTLY one of these 2 structures:

1) If image is VALID and enough Troy blocks are visible:
{{
  "status": "success",
  "imageStatus": "valid",
  "buildGuess": {{
    "title": "short guessed build name",
    "subtitle": "one short sentence for parents about what the child likely built"
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

2) If image is INVALID / random / unclear / unrelated / not enough Troy blocks:
{{
  "status": "success",
  "imageStatus": "invalid",
  "buildGuess": {{
    "title": "Not enough Troy blocks detected",
    "subtitle": "We could not clearly identify a Troy block creation in this image."
  }},
  "whatWeFound": {{
    "title": "What we found",
    "summary": "The photo does not show enough visible Troy blocks for a proper analysis."
  }},
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

Rules:
- Output only JSON
- No markdown
- No explanation outside JSON
- Keep language parent-friendly
"""

        response = call_gemini_with_retry(model, prompt, img)

        raw_text = response.text.strip()
        cleaned = clean_json_response(raw_text)
        print("Gemini raw output:", cleaned)

        try:
            parsed = json.loads(cleaned)
        except Exception:
            return jsonify(
                build_fallback_valid_response(
                    age,
                    "AI response could not be read properly, so a fallback result was used."
                )
            ), 200

        if parsed.get("imageStatus") == "invalid":
            reason = (
                parsed.get("whatWeFound", {}).get("summary")
                or "We could not clearly identify enough Troy blocks in the photo."
            )
            return jsonify(build_invalid_photo_response(age, reason)), 200

        if parsed.get("imageStatus") == "valid":
            cards = ensure_learning_cards(parsed.get("whatTheyLearned"))
            if len(cards) < 3:
                return jsonify(
                    build_fallback_valid_response(
                        age,
                        "AI returned incomplete learning cards, so a fallback result was used."
                    )
                ), 200

            session_id = str(uuid.uuid4())

            result = {
                "status": "success",
                "imageStatus": "valid",
                "buildGuess": {
                    "title": str(parsed.get("buildGuess", {}).get("title", "Creative Troy block build")).strip(),
                    "subtitle": str(parsed.get("buildGuess", {}).get("subtitle", "Your little one just built a creative Troy structure!")).strip()
                },
                "whatWeFound": {
                    "title": "What we found",
                    "summary": str(parsed.get("whatWeFound", {}).get("summary", "This looks like a thoughtful Troy block structure.")).strip()
                },
                "whatTheyLearned": cards[:3],
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

            sessions[session_id] = {
                "age": age,
                "imageStatus": "valid",
                "buildGuess": result["buildGuess"],
                "whatWeFound": result["whatWeFound"],
                "whatTheyLearned": result["whatTheyLearned"],
                "whatWeNoticed": result["whatWeNoticed"],
                "suggestionsForParent": result["suggestionsForParent"],
                "nextBuildIdeas": result["nextBuildIdeas"]
            }

            return jsonify(result), 200

        return jsonify(build_invalid_photo_response(age)), 200

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
            return jsonify(
                build_fallback_valid_response(
                    request.form.get("age", "").strip(),
                    "Live AI feedback is temporarily unavailable because the Gemini quota has been reached, so a fallback result was used."
                )
            ), 200

        if is_temporary_error(error_text):
            return jsonify(
                build_fallback_valid_response(
                    request.form.get("age", "").strip(),
                    "Live AI feedback is temporarily unavailable, so a fallback result was used."
                )
            ), 200

        return jsonify({
            "error": "Something went wrong",
            "details": error_text
        }), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        summary = data.get("summary", "").strip()

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
Keep it to 3-5 lines max.
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