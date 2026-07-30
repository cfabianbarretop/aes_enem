from torchvision.datasets import FashionMNIST
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_MNIST_FASHION_PATH = "data"  # Original dataset path
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, "../..", DATA_MNIST_FASHION_PATH))
print(data_dir)
print(device)

# ==============================================
# Confusion Matrix
# ==============================================


def conf_matrix(file_path, file_name, data):
    last_record = data
    ground_truth = last_record["ground_truth"]
    output = last_record["output"]
    # Crear matriz de confusión
    labels = np.unique(np.concatenate([ground_truth, output]))
    cm = confusion_matrix(ground_truth, output, labels=labels)
    # Mostrar con etiquetas personalizadas
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation="vertical")
    # Añadir título
    plt.title("Confusion Matrix - Clasification")
    # Guardar como imagen
    # plt.savefig(f"{file_path}/{FILE_RESULT_MATRIX}_{file_name}.png", dpi=300)
    plt.show()
    plt.close()


model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

model.to(device)


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
    "a grayscale image of a single ankle boot, centered on a black background",
]

dataset = FashionMNIST(root=data_dir, train=False, download=True)
correct = 0
iter = tqdm(dataset, total=len(dataset), desc="OPEN_CLIP")
text = tokenizer(labels).to(device)
ground_truth = []
output = []

with torch.no_grad():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for image, gt in iter:

        image = preprocess(image).unsqueeze(0).to(device)

        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        probs = image_features @ text_features.T

        prediction = probs.argmax(dim=-1).item()
        ground_truth.append(gt)
        output.append(prediction)

        if prediction == gt:
            correct += 1
        accuracy = 100 * (correct / len(dataset))
        iter.set_postfix(Accuracy=f"{accuracy:.2f}%")
data = {"ground_truth": ground_truth, "output": output}
conf_matrix("","",data=data)
