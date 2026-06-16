import os
import json
from PIL import Image
from tqdm import tqdm

import torch
import clip

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"   # or ViT-L/14
TEMPERATURE = 0.5         
THRESHOLD = None          # None or e.g. 0.4

DATASET_ROOT = "data/leaf_11"   
OUTPUT_JSON = "results/annotated_dataset.json"

# =========================
# PROMPT ENSEMBLING
# =========================
PROMPTS = {
    "margin": {
        "entire": [
            "a leaf with smooth edges",
            "a plant leaf with entire margin",
            "a leaf with no serrations"
        ],
        "serrate": [
            "a leaf with serrated edges",
            "a plant leaf with jagged margin",
            "a leaf with saw-toothed edges"
        ],
        "serrulate": [
            "a leaf with fine serrations",
            "a plant leaf with small jagged edges"
        ],
        "lobed": [
            "a leaf with lobes",
            "a plant leaf with deep rounded divisions"
        ],
        "undulate": [
            "a leaf with wavy edges",
            "a plant leaf with undulating margin"
        ],
        "indented": [
            "a leaf with indented edges",
            "a plant leaf with inward curves on the margin"
        ]
    },
    "shape": {
        "ovate": [
            "an ovate leaf",
            "a leaf shaped like an egg, wider at the base"
        ],
        "lanceolate": [
            "a lance-shaped leaf",
            "a long narrow leaf tapering to a point"
        ],
        "elliptical": [
            "an elliptical leaf",
            "a leaf shaped like an ellipse"
        ],
        "oblong": [
            "an oblong leaf",
            "a leaf longer than wide with parallel sides"
        ],
        "obovate": [
            "an obovate leaf",
            "a leaf wider at the top than the base"
        ]
    },
    "texture": {
        "smooth": [
            "a smooth leaf surface",
            "a leaf with no roughness"
        ],
        "rough": [
            "a rough leaf surface",
            "a leaf with a coarse texture"
        ],
        "glossy": [
            "a glossy shiny leaf",
            "a leaf with reflective surface"
        ],
        "leathery": [
            "a thick leathery leaf",
            "a stiff durable leaf surface"
        ]
    }
}

# =========================
# LOAD MODEL
# =========================
print("Loading CLIP...")
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
model.eval()
print("CLIP loaded!")

# =========================
# PROMPT ENCODING
# =========================
def encode_prompt_ensemble(prompts_dict):
    class_embeddings = {}

    for cls, prompts in prompts_dict.items():
        tokens = clip.tokenize(prompts).to(DEVICE)

        with torch.no_grad():
            embeddings = model.encode_text(tokens)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

        mean_embedding = embeddings.mean(dim=0)
        mean_embedding /= mean_embedding.norm()

        class_embeddings[cls] = mean_embedding

    return class_embeddings


def build_feature_matrix(class_embeddings):
    class_names = list(class_embeddings.keys())
    features = torch.stack([class_embeddings[c] for c in class_names])
    return class_names, features


print("Encoding prompts...")

margin_names, margin_features = build_feature_matrix(
    encode_prompt_ensemble(PROMPTS["margin"])
)

shape_names, shape_features = build_feature_matrix(
    encode_prompt_ensemble(PROMPTS["shape"])
)

texture_names, texture_features = build_feature_matrix(
    encode_prompt_ensemble(PROMPTS["texture"])
)

# =========================
# IMAGE ENCODING
# =========================
def encode_image(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features


# =========================
# CLASSIFICATION
# =========================
def classify(image_features, class_features, class_names):
    logits = image_features @ class_features.T
    probs = torch.softmax(logits / TEMPERATURE, dim=-1).cpu().numpy()[0]

    idx = probs.argmax()
    conf = float(probs[idx])
    pred = class_names[idx]

    if THRESHOLD is not None and conf < THRESHOLD:
        return "uncertain", conf

    return pred, conf


# =========================
# DATASET ITERATOR
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


dataset = load_dataset(DATASET_ROOT)

print(f"Dataset size: {len(dataset)}")

# =========================
# ANNOTATION LOOP
# =========================
results = []

for item in tqdm(dataset):
    try:
        image_features = encode_image(item["image"])

        margin_pred, margin_conf = classify(
            image_features, margin_features, margin_names
        )

        shape_pred, shape_conf = classify(
            image_features, shape_features, shape_names
        )

        texture_pred, texture_conf = classify(
            image_features, texture_features, texture_names
        )

        results.append({
            "id": item["id"],
            "image": item["image"],
            "species": item["species"],
            "margin": margin_pred,
            "margin_conf": margin_conf,
            "shape": shape_pred,
            "shape_conf": shape_conf,
            "texture": texture_pred,
            "texture_conf": texture_conf
        })

    except Exception as e:
        print(f"Error with {item['image']}: {e}")

# =========================
# SAVE OUTPUT
# =========================
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=4)

print(f"Saved to {OUTPUT_JSON}")