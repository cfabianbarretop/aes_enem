from torchvision.datasets import FashionMNIST
from tqdm import tqdm
import torch
import os
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_MNIST_FASHION_PATH = "data"        # Original dataset path
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, "../..", DATA_MNIST_FASHION_PATH))
print(data_dir)
print(device)

# model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
# tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

model.to(device)
# model.eval()

# labels = [
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

# labels = [
#     "T-shirt",
#     "Trouser",
#     "Pullover",
#     "Dress",
#     "Coat",
#     "Sandal",
#     "Shirt",
#     "Sneaker",
#     "Bag",
#     "Ankle boot"
# ]

labels = [
    "a grayscale image of a single T-shirt or top, centered on a black background",
    "a grayscale image of a single pair of trousers, centered on a black background",
    "a grayscale image of a single pullover sweater, centered on a black background",
    "a grayscale image of a single dress, centered on a black background",
    "a grayscale image of a single coat, centered on a black background",
    "a grayscale image of a single sandal, centered on a black background",
    "a grayscale image of a single shirt, centered on a black background",
    "a grayscale image of a single sneaker, centered on a black background",
    "a grayscale image of a single bag, centered on a black background",
    "a grayscale image of a single ankle boot, centered on a black background"
]

dataset = FashionMNIST(
    root=data_dir,
    train=False,
    download=True
)
correct = 0
iter = tqdm(dataset, total=len(dataset), desc="OPEN_CLIP")
text = tokenizer(labels).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for image, gt in iter:

        image = preprocess(image).unsqueeze(0).to(device)
        
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        probs = image_features @ text_features.T

        prediction = probs.argmax(dim=-1).item()

        if prediction == gt:
            correct += 1
        accuracy = 100 * (correct / len(dataset))
        iter.set_postfix(Accuracy = f"{accuracy:.2f}%")

# There is an accuracy the 77,10 in testing and 77,58 in training % whit promp => labels = [
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

# There is an accuracy the 73,23 % in testing whit promp => labels = [
#     "T-shirt",
#     "Trouser",
#     "Pullover",
#     "Dress",
#     "Coat",
#     "Sandal",
#     "Shirt",
#     "Sneaker",
#     "Bag",
#     "Ankle boot"
# ]

# There is an accuracy the 80,07 % in testing whit promp => labels = [
#     "a grayscale image of a single T-shirt or top, centered on a black background",
#     "a grayscale image of a single pair of trousers, centered on a black background",
#     "a grayscale image of a single pullover sweater, centered on a black background",
#     "a grayscale image of a single dress, centered on a black background",
#     "a grayscale image of a single coat, centered on a black background",
#     "a grayscale image of a single sandal, centered on a black background",
#     "a grayscale image of a single shirt, centered on a black background",
#     "a grayscale image of a single sneaker, centered on a black background",
#     "a grayscale image of a single bag, centered on a black background",
#     "a grayscale image of a single ankle boot, centered on a black background"
# ]