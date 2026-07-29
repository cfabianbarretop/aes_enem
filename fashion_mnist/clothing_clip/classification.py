from transformers import CLIPProcessor, CLIPModel
from torchvision.datasets import FashionMNIST
from tqdm import tqdm
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_MNIST_FASHION_PATH = "data"        # Original dataset path
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, "../..", DATA_MNIST_FASHION_PATH))
print(data_dir)
print(device)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.to(device)
model.eval()

labels = [
    "a piece of clothing called T-shirt",
    "a piece of clothing called trouser",
    "a piece of clothing called pullover",
    "a piece of clothing called dress",
    "a piece of clothing called coat",
    "a piece of clothing called sandal",
    "a piece of clothing called shirt",
    "a piece of clothing called sneaker",
    "a piece of clothing called bag",
    "a piece of clothing called ankle boot"
]

dataset = FashionMNIST(
    root=data_dir,
    train=False,
    download=True
)
correct = 0
iter = tqdm(dataset, total=len(dataset), desc="CLIP")
for image, gt in iter:

    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)

    prediction = probs.argmax(dim=1).item()

    if prediction == gt:
        correct += 1
    accuracy = 100 * (correct / len(dataset))
    iter.set_postfix(Accuracy = f"{accuracy:.2f}%")
# print(f"Ground Truth : {gt} - {labels[gt]}")
# print(f"Prediction   : {prediction} - {labels[prediction]}")
# print("Probabilities:")
# for label, p in zip(labels, probs[0]):
    # print(f"{label:15s}: {p.item():.4f}")

# 62% test y 62,72% train with prompt => labels = [
#     "a T-shirt",
#     "a trouser",
#     "a pullover",
#     "a dress",
#     "a coat",
#     "a sandal",
#     "a shirt",
#     "a sneaker",
#     "a bag",
#     "an ankle boot"
# ]

# Melhora 62,86 in testing with prompt => labels = [
#     "a piece of clothing called T-shirt",
#     "a piece of clothing called trouser",
#     "a piece of clothing called pullover",
#     "a piece of clothing called dress",
#     "a piece of clothing called coat",
#     "a piece of clothing called sandal",
#     "a piece of clothing called shirt",
#     "a piece of clothing called sneaker",
#     "a piece of clothing called bag",
#     "a piece of clothing called ankle boot"
# ]