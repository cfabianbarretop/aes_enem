# ==========================================
# CLIP DATASET ANNOTATION PIPELINE (OPTIMIZED)
# ==========================================

import os
import json
from tqdm import tqdm
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# =========================
# CONFIG
# =========================

DATASET_PATH = "data/leaf_11"
OUTPUT_FILE = "results/leaf_labels_clip.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
CONF_THRESHOLD = None  # set None to disable filtering

MODEL_NAME = "openai/clip-vit-base-patch32"

# =========================
# LOAD MODEL
# =========================

print("Loading CLIP...")

processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

print("CLIP loaded!")

# =========================
# CLASSES
# =========================

MARGIN_CLASSES = ["entire", "indented", "lobed", "serrate", "serrulate", "undulate"]
SHAPE_CLASSES = ["elliptical", "lanceolate", "oblong", "obovate", "ovate"]
TEXTURE_CLASSES = ["glossy", "leathery", "smooth", "rough"]

# =========================
# PROMPT ENSEMBLING
# =========================

def build_prompts():

    margin_templates = [
        "a close-up photo of a leaf with {} margin",
        "a botanical image of a leaf showing {} edges",
        "the margin of the leaf is {}"
    ]

    shape_templates = [
        "a leaf with {} shape",
        "the leaf is {} in shape",
        "a botanical illustration of a {} leaf"
    ]

    texture_templates = [
        "a leaf with {} texture",
        "the surface of the leaf is {}",
        "a close-up showing {} leaf surface"
    ]

    def expand(classes, templates):
        return [t.format(c) for c in classes for t in templates]

    return (
        expand(MARGIN_CLASSES, margin_templates),
        expand(SHAPE_CLASSES, shape_templates),
        expand(TEXTURE_CLASSES, texture_templates),
        len(margin_templates),
        len(shape_templates),
        len(texture_templates),
    )

(
    MARGIN_PROMPTS,
    SHAPE_PROMPTS,
    TEXTURE_PROMPTS,
    N_MARGIN_TEMPLATES,
    N_SHAPE_TEMPLATES,
    N_TEXTURE_TEMPLATES
) = build_prompts()

# =========================
# TEXT EMBEDDINGS
# =========================

@torch.no_grad()
def compute_text_embeddings(prompts):
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    text_features = model.get_text_features(**inputs)
    return text_features / text_features.norm(dim=-1, keepdim=True)

print("Computing text embeddings...")

margin_text_features = compute_text_embeddings(MARGIN_PROMPTS)
shape_text_features = compute_text_embeddings(SHAPE_PROMPTS)
texture_text_features = compute_text_embeddings(TEXTURE_PROMPTS)

print("Text embeddings ready!")

# =========================
# IMAGE LOADER
# =========================

def load_images(paths):
    return [Image.open(p).convert("RGB") for p in paths]

# =========================
# CLASSIFICATION
# =========================

@torch.no_grad()
def classify(images, text_features, class_names, n_templates):

    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    logits = image_features @ text_features.T

    logits = logits.view(len(images), len(class_names), n_templates)
    logits = logits.mean(dim=2)

    probs = logits.softmax(dim=1)

    preds = probs.argmax(dim=1)
    confs = probs.max(dim=1).values

    labels = [class_names[i] for i in preds]

    return labels, confs.tolist()

# =========================
# MAIN PIPELINE
# =========================

def process_dataset():

    results = []
    img_id = 0

    for class_name in sorted(os.listdir(DATASET_PATH)):
        class_path = os.path.join(DATASET_PATH, class_name)

        if not os.path.isdir(class_path):
            continue

        image_files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc=class_name):

            batch_files = image_files[i:i+BATCH_SIZE]
            batch_paths = [os.path.join(class_path, f) for f in batch_files]

            try:
                images = load_images(batch_paths)

                margin, margin_conf = classify(
                    images, margin_text_features, MARGIN_CLASSES, N_MARGIN_TEMPLATES
                )

                shape, shape_conf = classify(
                    images, shape_text_features, SHAPE_CLASSES, N_SHAPE_TEMPLATES
                )

                texture, texture_conf = classify(
                    images, texture_text_features, TEXTURE_CLASSES, N_TEXTURE_TEMPLATES
                )

                for path, m, mc, s, sc, t, tc in zip(
                    batch_paths, margin, margin_conf,
                    shape, shape_conf,
                    texture, texture_conf
                ):

                    # Optional filtering
                    if CONF_THRESHOLD is not None:
                        if mc < CONF_THRESHOLD:
                            m = "uncertain"
                        if sc < CONF_THRESHOLD:
                            s = "uncertain"
                        if tc < CONF_THRESHOLD:
                            t = "uncertain"

                    results.append({
                        "id": f"leaf_{img_id:06d}",
                        "image": path,
                        "species": class_name,
                        "margin": m,
                        "margin_conf": mc,
                        "shape": s,
                        "shape_conf": sc,
                        "texture": t,
                        "texture_conf": tc
                    })

                    img_id += 1

            except Exception as e:
                print(f"Batch error: {e}")

    return results

# =========================
# SAVE
# =========================

def save_results(results):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

# =========================
# RUN
# =========================

if __name__ == "__main__":
    print("Processing dataset with CLIP...")
    results = process_dataset()
    save_results(results)
    print(f"Saved results to {OUTPUT_FILE}")