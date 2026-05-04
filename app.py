import os
import json
import uuid
import time
from pathlib import Path
import base64
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageEnhance
from groq import Groq

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

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


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
        cleaned.append({"title": title, "description": description, "color": color})
    return cleaned[:3]


def ensure_thinking_meter(meter):
    if not isinstance(meter, dict):
        return {"creativity": 3, "problemSolving": 3, "spatialSkills": 3, "focus": 3}

    def clamp(val, default=3):
        try:
            v = int(val)
            return max(1, min(6, v))
        except Exception:
            return default

    return {
        "creativity":     clamp(meter.get("creativity", 3)),
        "problemSolving": clamp(meter.get("problemSolving", 3)),
        "spatialSkills":  clamp(meter.get("spatialSkills", 3)),
        "focus":          clamp(meter.get("focus", 3)),
    }


def is_rate_limit_error(error_text: str) -> bool:
    lower = error_text.lower()
    return "429" in error_text or "rate_limit" in lower or "too many" in lower


def is_temporary_error(error_text: str) -> bool:
    lower = error_text.lower()
    return "503" in error_text or "unavailable" in lower


# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(filepath) -> tuple[str, str]:
    img = Image.open(filepath)

    # Auto rotate
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

    # Upscale small images
    min_dim = min(img.width, img.height)
    if min_dim < 512:
        scale = 512 / min_dim
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    # Downscale large images
    max_dim = max(img.width, img.height)
    if max_dim > 2048:
        scale = 2048 / max_dim
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    # Enhance for better model perception
    try:
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = ImageEnhance.Sharpness(img).enhance(1.25)
        img = ImageEnhance.Color(img).enhance(1.1)
    except Exception:
        pass

    # Normalize to RGB
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


# -----------------------------
# Response builders
# -----------------------------
def build_invalid_photo_response(age, reason, noticed=None, suggestions=None, ideas=None):
    session_id = str(uuid.uuid4())
    result = {
        "status": "success",
        "imageStatus": "invalid",
        "buildGuess": {
            "title": "We couldn't clearly analyze this image",
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
        "troyThinkingMeter": {
            "creativity": 0,
            "problemSolving": 0,
            "spatialSkills": 0,
            "focus": 0
        },
        "session_id": session_id
    }
    sessions[session_id] = result
    return result


# -----------------------------
# Groq vision call
# -----------------------------
def analyze_troy_image_once(b64_image: str, media_type: str, age: str) -> dict:
    prompt = f"""
You are an expert visual analyst and child development specialist for Troy wooden block sets.
The child's age is: {age if age else "unknown"}.

You are analyzing ONE SPECIFIC IMAGE right now.
Every single part of your response MUST be based entirely on what you actually see
in THIS specific image — the exact shapes, colors, block count, arrangement, and
overall silhouette of THIS particular build.
NEVER give generic or template responses. NEVER repeat the same feedback for different builds.

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

Mark imageStatus as "invalid" immediately if you see:
- LEGO or Duplo (circular studs on top)
- Mega Bloks (large hollow plastic)
- Magnetic tiles (flat, translucent, plastic frames)
- Foam blocks (soft-looking, letters/numbers printed on them)
- Cardboard boxes or packaging
- K'NEX, Lincoln Logs, or any connector-based system
- Drawings or illustrations of blocks
- Random household objects stacked together
- People, animals, food, or scenery with no blocks present
- A single loose block not part of any build

---

## STEP 1 — CLASSIFY THIS IMAGE

Look at this image carefully and decide:
1. Is this a clear Troy blocks build? (imageStatus: "valid")
2. Is it unclear / blurry / too dark / too cropped? (imageStatus: "invalid")
3. Is it not Troy blocks at all? (imageStatus: "invalid")

---

## STEP 2 — SELF CHECK

Answer these internally before writing anything:
1. Can I see at least 2 Troy blocks clearly in THIS image?
2. Do the blocks look like solid matte wood — not plastic or foam?
3. Are the shapes simple geometric solids with no studs or printed logos?
4. Is this a deliberate build — not just loose scattered blocks?
5. Am I at least 85% confident this is a Troy wooden block build?

If ANY answer is NO → imageStatus must be "invalid".

---

## STEP 3 — DEEP VISUAL ANALYSIS (only if valid)

Look at THIS specific image very carefully and answer all of these internally:

EXACT SILHOUETTE OF THIS BUILD:
- What is the precise overall outline of this specific structure?
- Is it tall and thin like a tower or lighthouse?
- Does it have a long narrow section rising from one end of a wider base — like a giraffe neck?
- Does it have four leg-like protrusions extending outward from a central body?
- Does it have a wide flat base with a pointed or domed top like a house?
- Does it have a gap or opening in the middle like a bridge or arch?
- Does it have a long horizontal body like a train or snake?
- What is the actual height vs width ratio?

EXACT STRUCTURE OF THIS BUILD:
- Count or estimate the exact number of blocks visible in THIS image
- Describe precisely where each block or group of blocks sits
- Which blocks are on the bottom layer? What shapes are they?
- Which blocks are stacked on top? What shapes?
- Are any blocks sticking out to the sides — forming legs, wings, or arms?
- Is there a narrow connecting section between two wider sections?
- Is there a small block on top that looks like a head?
- What specific colors can you see on the blocks?
- Are the blocks aligned symmetrically or asymmetrically?

WHAT THIS BUILD RESEMBLES:
- Based ONLY on what you literally see in THIS image, what does this build most closely look like?
- Do not guess — describe what the actual shape tells you
- Consider all possibilities:
  Animals: giraffe, dog, cat, horse, elephant, dinosaur, snake, bird, crocodile, rabbit
  Buildings: house, castle, lighthouse, tower, barn, church, pyramid
  Vehicles: car, train, rocket, boat, truck, crane
  Structures: bridge, arch, gate, wall, table, chair, throne, enclosure

---

## STEP 4 — NAME THIS SPECIFIC BUILD

Based ONLY on your STEP 3 analysis of THIS image:
- TALL NARROW stack rising from one side of a wider base → giraffe, lighthouse, rocket
- FOUR OUTWARD PROTRUSIONS from a body → dog, horse, table, spider, crab
- WIDE BASE with pointed or domed top → house, castle, barn, pyramid
- GAP or ARCH opening in the middle → bridge, gate, arch
- LONG HORIZONTAL body with protrusions → train, snake, crocodile, car
- SYMMETRIC TALL STACK → tower, skyscraper, lighthouse
- ENCLOSED RECTANGULAR SHAPE → garage, barn, enclosure, room
- The title MUST match your STEP 3 silhouette description exactly
- NEVER use "Creative Troy block build" — always name based on actual shape

---

## STEP 5 — TROY THINKING METER

Rate THIS specific build out of 6 for each metric.
Base scores ONLY on what you actually see in THIS image:

creativity (1-6): How imaginative and original is THIS specific arrangement?
- 1-2: Very basic stack of same blocks
- 3-4: Some variety in shapes or arrangement
- 5-6: Highly original, unexpected use of shapes, clearly represents something

problemSolving (1-6): How complex and engineered is THIS specific structure?
- 1-2: 1-3 blocks simply stacked
- 3-4: 4-8 blocks with some planning
- 5-6: 9+ blocks, complex layering, structural thinking visible

spatialSkills (1-6): How well are blocks arranged and balanced in THIS build?
- 1-2: Blocks placed randomly, poor balance
- 3-4: Some intentional placement, moderate balance
- 5-6: Precise placement, excellent balance, clear spatial awareness

focus (1-6): How complete and detailed does THIS build appear?
- 1-2: Looks unfinished or very minimal
- 3-4: Reasonably complete, some detail
- 5-6: Highly detailed, clearly complete, deliberate finishing touches

---

## OUTPUT FORMAT

Return ONLY valid raw JSON. No markdown. No explanation outside the JSON.
Every field must be 100% specific to THIS build — never generic or reusable.

IF VALID TROY BUILD:
{{
  "status": "success",
  "imageStatus": "valid",
  "buildGuess": {{
    "title": "specific name based on THIS build's actual silhouette (max 6 words)",
    "subtitle": "one sentence specific to THIS build — describe what makes it look like what you named it"
  }},
  "whatWeFound": {{
    "title": "What we found",
    "summary": "2 sentences specific to THIS build — describe the actual shapes, colors, and arrangement you see"
  }},
  "whatTheyLearned": [
    {{
      "title": "specific skill shown in THIS build",
      "description": "specific to THIS build's actual shapes and arrangement — mention specific blocks or colors you see — relevant to age {age if age else 'this child'}",
      "color": "cream"
    }},
    {{
      "title": "specific skill shown in THIS build",
      "description": "specific to THIS build's actual shapes and arrangement — mention specific blocks or colors you see — relevant to age {age if age else 'this child'}",
      "color": "green"
    }},
    {{
      "title": "specific skill shown in THIS build",
      "description": "specific to THIS build's actual shapes and arrangement — mention specific blocks or colors you see — relevant to age {age if age else 'this child'}",
      "color": "blue"
    }}
  ],
  "whatWeNoticed": [],
  "suggestionsForParent": [
    "specific suggestion referencing THIS build's actual shapes and what the child did",
    "specific suggestion referencing THIS build's actual shapes and what the child did",
    "specific suggestion referencing THIS build's actual shapes and what the child did"
  ],
  "nextBuildIdeas": [
    "idea that naturally extends THIS specific build",
    "idea based on what THIS child clearly enjoys building",
    "challenge idea based on THIS build's complexity level"
  ],
  "troyThinkingMeter": {{
    "creativity": <integer 1-6 based on THIS build>,
    "problemSolving": <integer 1-6 based on THIS build>,
    "spatialSkills": <integer 1-6 based on THIS build>,
    "focus": <integer 1-6 based on THIS build>
  }}
}}

IF INVALID / UNCLEAR / NOT TROY:
{{
  "status": "success",
  "imageStatus": "invalid",
  "buildGuess": {{
    "title": "We couldn't clearly analyze this image",
    "subtitle": "Please try again with a clearer photo of your Troy block build."
  }},
  "whatWeFound": {{
    "title": "What we found",
    "summary": "The image does not clearly show a Troy wooden block build."
  }},
  "whatTheyLearned": [],
  "whatWeNoticed": [
    "specific observation about THIS image",
    "specific observation about THIS image",
    "specific observation about THIS image"
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
  ],
  "troyThinkingMeter": {{
    "creativity": 0,
    "problemSolving": 0,
    "spatialSkills": 0,
    "focus": 0
  }}
}}

ABSOLUTE RULES:
- Every field must be unique and specific to THIS image — never copy-paste or reuse responses
- buildGuess title must exactly match THIS build's actual silhouette from STEP 3
- whatWeFound must describe what you literally see — specific shapes, colors, block count
- whatTheyLearned descriptions must mention specific visible elements of THIS build
- suggestionsForParent must reference THIS build's specific shapes and arrangement
- troyThinkingMeter scores must reflect THIS build's actual complexity and quality
- Two different builds must ALWAYS produce completely different responses
- NEVER output "Creative Troy block build" as a title
- Output JSON only — no markdown, no explanation outside JSON
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.0,
        max_tokens=1500,
        top_p=1.0,
        stream=False,
        messages=[
            {
                "role": "system",
                "content": "You are a precise visual analyst. You analyze each image individually and always output unique, specific responses based on what you actually see. Output only valid raw JSON — no markdown, no prose outside the JSON."
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
                        "text": prompt,
                    },
                ],
            },
        ],
    )

    raw_text = response.choices[0].message.content.strip()
    print("Raw Groq response:", raw_text[:500])
    return parse_json_response(raw_text)


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Troy backend is running", "server": "ok"})


@app.route("/health", methods=["GET"])
def health():
    api_key = os.environ.get("GROQ_API_KEY", "")
    return jsonify({
        "status": "ok",
        "model": GROQ_MODEL,
        "groq_key_loaded": bool(api_key),
        "key_preview": (api_key[:6] + "...") if api_key else "NONE"
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

        if not os.environ.get("GROQ_API_KEY"):
            return jsonify({"error": "GROQ_API_KEY not found"}), 500

        file_ext = image_file.filename.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4()}.{file_ext}"
        filepath = UPLOAD_FOLDER / filename
        image_file.save(str(filepath))

        try:
            b64_image, media_type = preprocess_image(filepath)
        except Exception:
            return jsonify({
                "error": "Could not open this image. Please try JPG, PNG, or WEBP."
            }), 400

        parsed = analyze_troy_image_once(b64_image, media_type, age)
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
                    "title": str(parsed.get("buildGuess", {}).get("title", "A Block Build")).strip(),
                    "subtitle": str(parsed.get("buildGuess", {}).get("subtitle", "")).strip()
                },
                "whatWeFound": {
                    "title": "What we found",
                    "summary": str(parsed.get("whatWeFound", {}).get("summary", "")).strip()
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
                    ["Build a bridge", "Build a tower", "Build a small castle"]
                ),
                "troyThinkingMeter": ensure_thinking_meter(
                    parsed.get("troyThinkingMeter")
                ),
                "session_id": session_id
            }
            sessions[session_id] = result
            return jsonify(result), 200

        # invalid path
        invalid_reason = str(parsed.get("whatWeFound", {}).get("summary", "")).strip()
        if not invalid_reason:
            invalid_reason = "We couldn't clearly analyze this image."

        result = build_invalid_photo_response(
            age,
            invalid_reason,
            noticed=ensure_list(parsed.get("whatWeNoticed"), [
                "The image may be unclear or not related to Troy blocks",
                "Too little of the build may be visible",
                "A clearer image will help us analyze properly"
            ]),
            suggestions=ensure_list(parsed.get("suggestionsForParent"), [
                "Retake the photo with the full structure visible",
                "Use better lighting and a cleaner background",
                "Make sure the Troy block build is the main focus of the image"
            ]),
            ideas=ensure_list(parsed.get("nextBuildIdeas"), [
                "Build a tower", "Build a bridge", "Build a small house"
            ])
        )

        result["buildGuess"] = {
            "title": str(parsed.get("buildGuess", {}).get("title", "We couldn't clearly analyze this image")).strip(),
            "subtitle": str(parsed.get("buildGuess", {}).get("subtitle", invalid_reason)).strip()
        }
        result["whatWeFound"] = {"title": "What we found", "summary": invalid_reason}
        return jsonify(result), 200

    except Exception as e:
        error_text = str(e)
        print("Analyze error:", error_text)

        if is_rate_limit_error(error_text):
            return jsonify({"error": "AI usage limit reached. Please wait a minute and try again."}), 429

        if is_temporary_error(error_text):
            return jsonify({"error": "The AI is temporarily unavailable. Please try again in a moment."}), 503

        return jsonify({"error": "Something went wrong", "details": error_text}), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = str(data.get("question", "")).strip()
        summary = str(data.get("summary", "")).strip()

        if not question:
            return jsonify({"error": "Question is required"}), 400

        if not os.environ.get("GROQ_API_KEY"):
            return jsonify({"error": "GROQ_API_KEY not found"}), 500

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": "You are helping a parent understand their child's Troy wooden block build. Answer warmly, simply, and specifically in 3-5 short lines. Always reference the specific build described."
                },
                {
                    "role": "user",
                    "content": f"Build summary:\n{summary}\n\nParent question:\n{question}"
                }
            ]
        )

        return jsonify({"answer": response.choices[0].message.content.strip()}), 200

    except Exception as e:
        error_text = str(e)
        print("Ask error:", error_text)

        if is_rate_limit_error(error_text):
            return jsonify({"answer": "AI usage limit reached. Please wait a minute and try again."}), 200

        if is_temporary_error(error_text):
            return jsonify({"answer": "Live AI Q&A is temporarily unavailable. Please try again later."}), 200

        return jsonify({"error": "Something went wrong", "details": error_text}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)