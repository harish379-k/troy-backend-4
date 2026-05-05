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

# Groq docs mention a 4MB limit for base64 encoded image requests.
# Keep it slightly below 4MB for safety.
MAX_BASE64_IMAGE_SIZE = 3_800_000


# =========================================================
# Groq helpers
# =========================================================

def get_groq_api_key():
    """
    Use GROQ_API_KEY in Render environment variables.
    Do not hardcode the API key in this file.
    """
    return (
        os.environ.get("GROQ_API_KEY", "").strip()
        or os.environ.get("RENDER_GROQ_KEY", "").strip()
    )


def get_vision_model_name():
    """
    Groq vision-capable model.
    You can override this in Render using GROQ_VISION_MODEL.
    """
    return os.environ.get(
        "GROQ_VISION_MODEL",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ).strip()


def get_text_model_name():
    """
    Used for /ask route.
    You can override this in Render using GROQ_TEXT_MODEL.
    """
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
# General helpers
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


def ensure_learning_cards(cards, fallback_cards=None):
    allowed_colors = ["cream", "green", "blue"]
    cleaned_cards = []

    if isinstance(cards, list):
        for index, card in enumerate(cards):
            if not isinstance(card, dict):
                continue

            title = clean_sentence(card.get("title"))
            description = clean_sentence(card.get("description"))
            color = clean_sentence(card.get("color", allowed_colors[index % 3])).lower()

            if not title or not description:
                continue

            if color not in allowed_colors:
                color = allowed_colors[index % 3]

            cleaned_cards.append({
                "title": title,
                "description": description,
                "color": color
            })

    if len(cleaned_cards) < 3 and fallback_cards:
        for card in fallback_cards:
            if len(cleaned_cards) >= 3:
                break

            if isinstance(card, dict):
                cleaned_cards.append(card)

    default_cards = [
        {
            "title": "Spatial Thinking",
            "description": "The child practiced placing blocks in relation to each other.",
            "color": "cream"
        },
        {
            "title": "Balance and Stability",
            "description": "The child explored how the structure stays standing.",
            "color": "green"
        },
        {
            "title": "Creative Planning",
            "description": "The child turned an idea into a visible block structure.",
            "color": "blue"
        }
    ]

    for card in default_cards:
        if len(cleaned_cards) >= 3:
            break
        cleaned_cards.append(card)

    return cleaned_cards[:3]


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
# Image preparation for Groq vision
# =========================================================

def encode_image_to_base64_jpeg(img, quality):
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    raw_bytes = buffer.getvalue()
    encoded = base64.b64encode(raw_bytes).decode("utf-8")
    return encoded


def image_file_to_data_url(image_file):
    """
    Converts uploaded image into a compressed base64 data URL.

    Groq vision accepts image_url input.
    For local uploads, we pass the image as:
    data:image/jpeg;base64,<base64>
    """
    img = Image.open(image_file.stream)
    img = ImageOps.exif_transpose(img)

    if img.mode not in ("RGB",):
        img = img.convert("RGB")

    # First resize
    img.thumbnail((1400, 1400))

    qualities = [85, 75, 65, 55, 45]

    for quality in qualities:
        encoded = encode_image_to_base64_jpeg(img, quality)

        if len(encoded.encode("utf-8")) <= MAX_BASE64_IMAGE_SIZE:
            return f"data:image/jpeg;base64,{encoded}"

    # If still too large, resize more aggressively
    img.thumbnail((1000, 1000))

    for quality in [70, 60, 50, 40]:
        encoded = encode_image_to_base64_jpeg(img, quality)

        if len(encoded.encode("utf-8")) <= MAX_BASE64_IMAGE_SIZE:
            return f"data:image/jpeg;base64,{encoded}"

    raise ValueError("Image is too large even after compression. Please upload a smaller image.")


# =========================================================
# Fallback response helpers
# =========================================================

def fallback_cards_from_category(category):
    category = clean_sentence(category).lower()

    if "bridge" in category:
        return [
            {
                "title": "Support and Span",
                "description": "The child explored how blocks can stretch across a gap.",
                "color": "cream"
            },
            {
                "title": "Balance",
                "description": "The build uses side supports to keep the bridge steady.",
                "color": "green"
            },
            {
                "title": "Problem Solving",
                "description": "The child tested how pieces connect to make a pathway.",
                "color": "blue"
            }
        ]

    if "tower" in category:
        return [
            {
                "title": "Vertical Building",
                "description": "The child practiced stacking blocks upward with control.",
                "color": "cream"
            },
            {
                "title": "Stability",
                "description": "The child explored how taller structures need stronger support.",
                "color": "green"
            },
            {
                "title": "Patience",
                "description": "Stacking pieces carefully helps develop focus and hand control.",
                "color": "blue"
            }
        ]

    if "animal" in category:
        return [
            {
                "title": "Symbolic Thinking",
                "description": "The child used simple blocks to represent a living thing.",
                "color": "cream"
            },
            {
                "title": "Body Parts",
                "description": "The build shows early thinking about parts such as body, legs, head, or neck.",
                "color": "green"
            },
            {
                "title": "Imagination",
                "description": "The child connected block shapes to a real-world idea.",
                "color": "blue"
            }
        ]

    if "house" in category or "castle" in category or "building" in category:
        return [
            {
                "title": "Structure Design",
                "description": "The child created a build with organized parts like base, walls, or roof-like shapes.",
                "color": "cream"
            },
            {
                "title": "Shape Recognition",
                "description": "The child used different block shapes to form a recognizable structure.",
                "color": "green"
            },
            {
                "title": "Planning",
                "description": "The build shows an idea being arranged into visible parts.",
                "color": "blue"
            }
        ]

    return [
        {
            "title": "Spatial Thinking",
            "description": "The child practiced arranging pieces in space.",
            "color": "cream"
        },
        {
            "title": "Balance and Stability",
            "description": "The child explored how blocks can stay steady together.",
            "color": "green"
        },
        {
            "title": "Creative Expression",
            "description": "The child used wooden pieces to express an idea through building.",
            "color": "blue"
        }
    ]


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
# Groq vision prompt
# =========================================================

def build_troy_analysis_prompt(age):
    return f"""
You are Troy AI Analyzer for Troy World.

Your job:
Analyze a child's physical Troy-block build from one uploaded photo.

Child age:
{age if age else "unknown"}

TROY BLOCK CONTEXT:
- Troy builds are made from light/natural wooden blocks.
- Common Troy pieces may include cubes, rectangular blocks, long beams, planks, pillars, triangular roof-like pieces, curved/arch pieces, and small connector-like wooden pieces.
- Children may build animals, towers, bridges, houses, gates, castles, roads, vehicles, abstract structures, or pretend scenes.

VERY IMPORTANT ACCURACY RULES:
1. Do not give generic feedback.
2. Every sentence must be based on visible evidence in the image.
3. Mention concrete visual details such as:
   - tall stack
   - wide base
   - bridge span
   - roof or triangle piece
   - repeated blocks
   - symmetry
   - gaps
   - curved pieces
   - loose pieces
   - horizontal/vertical arrangement
4. If you are not sure what the build is, call it an abstract structure instead of inventing a wrong object.
5. Reject selfies, people-only photos, screenshots, drawings, real buildings, random objects, or images without visible Troy-style wooden blocks.
6. Reject blurry, dark, cropped, or far-away photos where the structure cannot be understood.
7. Keep the tone warm, simple, and parent-friendly.
8. Do not mention hidden learning claims that cannot be inferred from the build.
9. Return valid JSON only. No markdown. No extra text.

CONFIDENCE RULE:
- imageStatus must be "valid" only when visible Troy-style wooden blocks occupy a meaningful part of the image and the structure can be described.
- confidenceScore must be from 0 to 100.
- If confidenceScore is below 65, imageStatus must be "invalid".

Return exactly this JSON shape:

{{
  "status": "success",
  "imageStatus": "valid or invalid",
  "confidenceScore": 0,
  "analysisDetails": {{
    "buildCategory": "animal / tower / bridge / house / gate / castle / road / vehicle / abstract / unclear / unrelated",
    "visibleElements": [
      "specific visible detail 1",
      "specific visible detail 2",
      "specific visible detail 3"
    ],
    "blockCountEstimate": "rough estimate like 5-8, 10-15, 20+ or unclear",
    "whyThisGuess": "one short sentence explaining the guess using visible details"
  }},
  "buildGuess": {{
    "title": "specific guessed build name",
    "subtitle": "one short sentence describing what the child likely built"
  }},
  "whatWeFound": {{
    "title": "What we found",
    "summary": "2 short sentences that mention the actual visible structure and parts"
  }},
  "whatTheyLearned": [
    {{
      "title": "specific skill name",
      "description": "specific explanation connected to this exact build",
      "color": "cream"
    }},
    {{
      "title": "specific skill name",
      "description": "specific explanation connected to this exact build",
      "color": "green"
    }},
    {{
      "title": "specific skill name",
      "description": "specific explanation connected to this exact build",
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
- Explain whether the image is unclear, unrelated, or not a Troy block build.
"""


def call_groq_vision(client, image_data_url, age):
    prompt = build_troy_analysis_prompt(age)

    completion = client.chat.completions.create(
        model=get_vision_model_name(),
        messages=[
            {
                "role": "system",
                "content": "You are a careful image analysis assistant. Always return valid JSON only."
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
        temperature=0.45,
        top_p=0.9,
        max_completion_tokens=1400,
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

        build_category = clean_sentence(
            analysis_details.get("buildCategory"),
            "abstract"
        )

        visible_elements = ensure_list(
            analysis_details.get("visibleElements"),
            fallback=[],
            limit=3
        )

        if image_status == "valid" and confidence_score >= 65:
            session_id = str(uuid.uuid4())

            fallback_cards = fallback_cards_from_category(build_category)

            noticed_fallback = visible_elements or [
                "The photo shows visible Troy-style wooden blocks arranged into a structure.",
                "The build has multiple pieces placed with intention.",
                "The child appears to be exploring shape, balance, and arrangement."
            ]

            result = {
                "status": "success",
                "imageStatus": "valid",
                "confidenceScore": confidence_score,
                "analysisDetails": {
                    "buildCategory": build_category,
                    "visibleElements": visible_elements,
                    "blockCountEstimate": clean_sentence(
                        analysis_details.get("blockCountEstimate"),
                        "unclear"
                    ),
                    "whyThisGuess": clean_sentence(
                        analysis_details.get("whyThisGuess"),
                        "The guess is based on the visible block arrangement."
                    )
                },
                "buildGuess": {
                    "title": clean_sentence(
                        parsed.get("buildGuess", {}).get("title")
                        if isinstance(parsed.get("buildGuess"), dict)
                        else "",
                        "Troy block structure"
                    ),
                    "subtitle": clean_sentence(
                        parsed.get("buildGuess", {}).get("subtitle")
                        if isinstance(parsed.get("buildGuess"), dict)
                        else "",
                        "The child created a visible structure using Troy blocks."
                    )
                },
                "whatWeFound": {
                    "title": "What we found",
                    "summary": clean_sentence(
                        parsed.get("whatWeFound", {}).get("summary")
                        if isinstance(parsed.get("whatWeFound"), dict)
                        else "",
                        "This image shows a Troy block structure with visible wooden pieces."
                    )
                },
                "whatTheyLearned": ensure_learning_cards(
                    parsed.get("whatTheyLearned"),
                    fallback_cards=fallback_cards
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
                "session_id": session_id,
                "debug": {
                    "filename": original_filename,
                    "model": get_vision_model_name()
                } if os.environ.get("SHOW_DEBUG", "false").lower() == "true" else None
            }

            if result["debug"] is None:
                result.pop("debug", None)

            sessions[session_id] = result

            return jsonify(result), 200

        invalid_reason = clean_sentence(
            parsed.get("whatWeFound", {}).get("summary")
            if isinstance(parsed.get("whatWeFound"), dict)
            else "",
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
            "buildCategory": build_category or "unclear",
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

        if isinstance(parsed.get("buildGuess"), dict):
            result["buildGuess"] = {
                "title": clean_sentence(
                    parsed.get("buildGuess", {}).get("title"),
                    "We couldn’t clearly analyze this image"
                ),
                "subtitle": clean_sentence(
                    parsed.get("buildGuess", {}).get("subtitle"),
                    invalid_reason
                )
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
            temperature=0.4,
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