import os
import json
import base64
import time
from tqdm import tqdm
import re

from openai import OpenAI

# =========================
# CONFIG
# =========================
DATASET_ROOT = "data/leaf_11"
OUTPUT_JSON = "results/leaf_labels_vlm.json"

MODEL = "gpt-4o"  
MAX_RETRIES = 3
SLEEP_TIME = 1  # seconds between requests

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# LABEL SPACE
# =========================
MARGIN = ['entire', 'indented', 'lobed', 'serrate', 'serrulate', 'undulate']
SHAPE = ['elliptical', 'lanceolate', 'oblong', 'obovate', 'ovate']
TEXTURE = ['glossy', 'leathery', 'smooth', 'rough']

# =========================
# PROMPT
# =========================
def build_prompt():
    return f"""
            Classify the leaf image.

            Return ONLY a JSON object. No text, no explanation.

            Valid labels:

            margin: {MARGIN}
            shape: {SHAPE}
            texture: {TEXTURE}

            Format:
            {{
            "margin": "...",
            "shape": "...",
            "texture": "..."
            }}
            """

# =========================
# IMAGE ENCODING
# =========================
def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_json(text):
    try:
        return json.loads(text)
    except:
        pass

    # intentar extraer bloque JSON
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None

# =========================
# CALL GPT VLM
# =========================
def classify_image(image_path):
    base64_image = encode_image_base64(image_path)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise and consistent visual classifier. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": build_prompt()},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content

            parsed = extract_json(content)

            if parsed is not None:
                return parsed
            if parsed is None:
                print("RAW OUTPUT:")
                print(content)

        except Exception as e:
            print(f"Retry {attempt+1} failed for {image_path}: {e}")

        time.sleep(2)

    return None

# =========================
# DATASET LOADER
# =========================
def load_dataset(root):
    data = []
    idx = 0

    VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    for species in os.listdir(root):
        species_path = os.path.join(root, species)

        if not os.path.isdir(species_path):
            continue

        for img_name in os.listdir(species_path):
            if not img_name.lower().endswith(VALID_EXTENSIONS):
                continue

            img_path = os.path.join(species_path, img_name)

            data.append({
                "id": f"leaf_{idx:06d}",
                "image": img_path,
                "species": species
            })

            idx += 1

    return data

# =========================
# VALIDATION
# =========================
def validate_output(output):
    if output is None:
        return False

    return (
        output.get("margin") in MARGIN and
        output.get("shape") in SHAPE and
        output.get("texture") in TEXTURE
    )

# =========================
# MAIN LOOP
# =========================
dataset = load_dataset(DATASET_ROOT)

print(f"Dataset size: {len(dataset)}")

results = []

for item in tqdm(dataset):
    result = classify_image(item["image"])

    if not validate_output(result):
        print(f"Invalid output for {item['image']}: {result}")
        continue

    results.append({
        "id": item["id"],
        "image": item["image"],
        "species": item["species"],
        "margin": result["margin"],
        "shape": result["shape"],
        "texture": result["texture"]
    })

    time.sleep(SLEEP_TIME)

# =========================
# SAVE
# =========================
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=4)

print(f"Saved to {OUTPUT_JSON}")