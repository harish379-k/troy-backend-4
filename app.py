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


def ensure_star_rating(value):
    try:
        v = int(value)
        return max(1, min(5, v))
    except Exception:
        return 3


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

    min_dim = min(img.width, img.height)
    if min_dim < 512:
        scale = 512 / min_dim
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    max_dim = max(img.width, img.height)
    if max_dim > 2048:
        scale = 2048 / max_dim
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    try:
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = ImageEnhance.Sharpness(img).enhance(1.25)
        img = ImageEnhance.Color(img).enhance(1.1)
    except Exception:
        pass

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
        "starRating": 0,
        "session_id": session_id
    }
    sessions[session_id] = result
    return result


# -----------------------------
# Groq vision call
# -----------------------------
def analyze_troy_image_once(b64_image: str, media_type: str, age: str) -> dict:
    prompt = f"""
You are a playful, imaginative child development specialist and expert visual analyst
for Troy wooden block sets. You have a gift for seeing the world through a child's eyes.

The child's age is: {age if age else "unknown"}.

You are looking at ONE specific image right now.
Your entire response must be based on what you LITERALLY see in this image.
Think like a child looking at this build — not like an adult.

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

Mark imageStatus as "invalid" if you see:
- LEGO or Duplo (circular studs on top)
- Mega Bloks (large hollow plastic)
- Magnetic tiles (flat translucent plastic frames)
- Foam blocks (soft-looking with letters or numbers)
- Cardboard boxes
- Random household objects
- People, animals, food, or scenery with no blocks
- A single loose block not part of any build

---

## PHASE 1 — RAW VISUAL DESCRIPTION (do this before anything else)

Look at the image and write out internally, in plain descriptive language, exactly what
physical shape you see. Do NOT use block names yet. Describe it like you are describing
a sculpture to someone who cannot see it:

- What is the overall silhouette from left to right and top to bottom?
- Where is the tallest point? Is it centered or off to one side?
- Are there parts that stick outward — left, right, forward?
- Are there parts that are thin and narrow connecting to wider parts?
- Is there a small piece on top of a taller section?
- Is there an empty space or gap inside the structure?
- Is it wide and flat or tall and narrow?
- Is it symmetric or lopsided?
- What colors are visible and where?
- How many blocks roughly — 1 to 5, 6 to 10, 11 to 20, or 20 plus?

Write this description completely before moving to Phase 2.

---

## PHASE 2 — WHAT DOES IT LOOK LIKE?

Now read your Phase 1 description and ask:
"If a child made this — what were they trying to make?"

Children build things like:
- Animals with long necks (giraffe, dinosaur, dragon, snake, flamingo, camel)
- Animals with four legs (dog, cat, horse, cow, elephant, lion, spider, crab)
- Animals with wings (bird, butterfly, plane, helicopter)
- Tall structures (rocket, lighthouse, skyscraper, wizard tower, beanstalk)
- Houses and castles (cottage, palace, igloo, treehouse, fortress, pyramid)
- Vehicles (racing car, fire truck, submarine, spaceship, bulldozer, steamroller)
- Fantasy things (magic wand, crown, throne, treasure chest, volcano)
- People and robots (robot, knight, person, superhero)
- Nature (mountain, bridge, cave, waterfall, island)
- Everyday things (table, chair, sofa, TV, bed, bookshelf, swimming pool)

Do NOT default to the most common or obvious name.
Pick the name that best matches the SPECIFIC silhouette you described in Phase 1.

Key silhouette clues:
- Tall narrow column on one side of a wide base = giraffe, lighthouse, flamingo, rocket
- Four protrusions from a flat central body = dog, spider, table, crab, horse
- Wide base narrowing upward to a point = pyramid, mountain, volcano, wizard hat
- Arch or gap in the middle = bridge, rainbow, cave entrance, goal post
- Long horizontal shape with small bumps on top = train, car, caterpillar, crocodile
- Two tall columns with something across the top = gate, goalpost, doorway
- Tall symmetric stack = tower, rocket, lighthouse, skyscraper
- Wide flat rectangle with small things on top = bed, swimming pool, football field
- Circular or semicircular arrangement = cave, igloo, crown, swimming pool

---

## PHASE 3 — CREATIVE NAMING

Now give the build a fun, specific, imaginative name that a child would love.
Do NOT use boring generic names like "Tower" or "House" or "Bridge".
Use descriptive exciting names like:
- "Zigzag Dragon with Red Spikes"
- "Lopsided Rocket Ship"
- "Sleeping Elephant"
- "Wonky Rainbow Bridge"
- "Giant Robot with One Arm"
- "Tiny Castle with a Secret Door"
- "Super Tall Giraffe"
- "Racing Car with No Wheels Yet"
- "Sleeping Crocodile"
- "Purple Volcano"

The name must match what you described in Phase 1 and Phase 2.
If the silhouette has a long neck — the name must reference that.
If it is lopsided — say so in the name. Children are proud of their unique builds.

---

## PHASE 4 — UNIQUE BUILD-SPECIFIC FEEDBACK

Now write feedback that could ONLY apply to THIS specific build.
Reference the actual colors, shapes, and arrangement you see.

For whatWeFound — describe what you literally see:
- Mention specific colors visible
- Mention specific shapes (arch, cylinder, cube etc)
- Mention how they are arranged
- Example: "We spotted a tall stack of red and yellow rectangular blocks rising up on the left side, with a wide blue arch at the bottom and a small cube balanced right on top."

For whatTheyLearned — tie each skill to something visible:
- Creativity: reference a specific surprising or imaginative choice you see
- Spatial Skills: reference how they balanced or arranged the specific blocks
- Problem Solving: reference a specific structural decision visible in the build
- Each description must be impossible to copy-paste to a different build

For suggestionsForParent — make them specific:
- Reference the actual build name and shapes
- Example: "Ask your child why they put the arch at the bottom of their dragon — did it need a big mouth?"

For nextBuildIdeas — extend THIS specific build:
- Example: "Add four small cubes as legs to turn this into a proper giraffe"
- Example: "Try adding a triangular prism on top as a nose"

---

## PHASE 5 — THINKING METER AND STAR RATING

troyThinkingMeter — score THIS specific build out of 6:

creativity (1-6):
- 1: Single block type stacked straight
- 2: Two block types, basic arrangement
- 3: Mix of shapes, some variety
- 4: Clear creative intent, recognizable as something
- 5: Multiple shape types used imaginatively
- 6: Highly original, unexpected combination, clearly represents a complex idea

problemSolving (1-6):
- 1: 1-3 blocks
- 2: 4-5 blocks simply stacked
- 3: 6-8 blocks with some structure
- 4: 9-12 blocks with clear planning
- 5: 13-16 blocks, multiple layers
- 6: 17+ blocks or complex interlocking structure

spatialSkills (1-6):
- 1: Random placement, falling over
- 2: Some alignment, unstable looking
- 3: Mostly aligned, moderate balance
- 4: Good balance, intentional symmetry or asymmetry
- 5: Precise placement, clear spatial awareness
- 6: Excellent balance, sophisticated spatial arrangement

focus (1-6):
- 1: 1-2 blocks, looks abandoned
- 2: Minimal effort, clearly unfinished
- 3: Basic complete shape
- 4: Detail added beyond minimum
- 5: Clearly finished with deliberate details
- 6: Highly detailed, complete, every block intentional

starRating (1-5) — overall quality:
- 1 star: 1-3 blocks, very simple
- 2 stars: 4-6 blocks, basic but deliberate
- 3 stars: 7-10 blocks or creative structure
- 4 stars: 11-15 blocks or impressive design
- 5 stars: 16+ blocks or exceptional creativity

---

## OUTPUT FORMAT

Return ONLY valid raw JSON. No markdown. No text outside the JSON.

IF VALID TROY BUILD:
{{
  "status": "success",
  "imageStatus": "valid",
  "buildGuess": {{
    "title": "fun creative specific name from Phase 3 (max 8 words)",
    "subtitle": "one sentence describing specifically why it looks like what you named it"
  }},
  "whatWeFound": {{
    "title": "What we found",
    "summary": "2 sentences describing specific colors, shapes, and arrangement visible in THIS image"
  }},
  "whatTheyLearned": [
    {{
      "title": "Creativity",
      "description": "specific creative choice visible in THIS build — mention actual shapes or colors — for age {age if age else 'this child'}",
      "color": "cream"
    }},
    {{
      "title": "Spatial Skills",
      "description": "specific spatial arrangement visible in THIS build — mention actual block positions — for age {age if age else 'this child'}",
      "color": "green"
    }},
    {{
      "title": "Problem Solving",
      "description": "specific structural decision visible in THIS build — mention actual shapes used — for age {age if age else 'this child'}",
      "color": "blue"
    }}
  ],
  "whatWeNoticed": [],
  "suggestionsForParent": [
    "specific suggestion referencing THIS build's name and actual shapes",
    "specific question to ask the child about THIS specific build",
    "specific extension idea for THIS build referencing actual blocks visible"
  ],
  "nextBuildIdeas": [
    "specific idea extending THIS build — reference actual shapes visible",
    "specific idea based on what THIS child clearly enjoys",
    "fun challenge based on THIS build's complexity and style"
  ],
  "troyThinkingMeter": {{
    "creativity": <1-6>,
    "problemSolving": <1-6>,
    "spatialSkills": <1-6>,
    "focus": <1-6>
  }},
  "starRating": <1-5>
}}

IF INVALID:
{{
  "status": "success",
  "imageStatus": "invalid",
  "buildGuess": {{
    "title": "We couldn't clearly see the build",
    "subtitle": "Please try again with a clearer photo showing all the blocks."
  }},
  "whatWeFound": {{
    "title": "What we found",
    "summary": "The image does not clearly show a Troy wooden block build."
  }},
  "whatTheyLearned": [],
  "whatWeNoticed": [
    "observation about THIS specific image",
    "observation about THIS specific image",
    "observation about THIS specific image"
  ],
  "suggestionsForParent": [
    "Move back so the whole build is in frame",
    "Try better lighting so the blocks are clearly visible",
    "Make sure the build is the main focus of the photo"
  ],
  "nextBuildIdeas": ["Build a tower", "Build a bridge", "Build a small house"],
  "troyThinkingMeter": {{"creativity": 0, "problemSolving": 0, "spatialSkills": 0, "focus": 0}},
  "starRating": 0
}}

ABSOLUTE RULES:
- Complete Phase 1 raw visual description BEFORE naming anything
- The name must match the Phase 1 silhouette description exactly
- Every description must reference something specifically visible in THIS image
- No two builds should ever get the same feedback
- Never use boring names like "Tower" "House" "Bridge" alone — always add a descriptive word
- Never copy-paste feedback from one build to another
- Output JSON only
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
                "content": "You are a playful visual analyst who sees the world through a child's eyes. You describe exactly what you see before naming anything. You always output unique specific responses. Output only valid raw JSON — no markdown, no prose outside JSON."
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
                "starRating": ensure_star_rating(
                    parsed.get("starRating", 3)
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