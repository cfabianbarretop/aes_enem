from torchvision.datasets import FashionMNIST
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import open_clip
import csv

# ==============================================
# Configuration
# ==============================================
device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_MNIST_FASHION_PATH = "data"  # Original dataset path
DATA_RESULT_PATH = "result/dataset"  # Result dataset path
FILE_RESULT_LABELING = "result_label"  # Name file result and probabilities
FILE_RESULT_MATRIX = "result_matrix"  # Name file result matrix
TRAINING = True  # Identified if it is traing or testing
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, "../../..", DATA_MNIST_FASHION_PATH))
result_dir = os.path.join(base_dir, DATA_RESULT_PATH)
print("ROOT_PATH:", data_dir)
print("RESULT_PATH:", result_dir)
print("Device: ", device)


# ==============================================
# Save results
# ==============================================
def save_label(file_path, file_name, data):
    name_file = f"{file_path}/{FILE_RESULT_LABELING}_{file_name}.csv"
    with open(name_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "id",
                "ground_truth",
                "output",
                "probability",
            ]
        )
        for row in data:
            writer.writerow(
                [
                    row["no"],
                    row["ground_truth"],
                    row["output"],
                    row["probability"],
                ]
            )


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
    # Save as imagen
    plt.savefig(f"{file_path}/{FILE_RESULT_MATRIX}_{file_name}.png", dpi=300)
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

name_type = "test"
if TRAINING:
    name_type = "train"

dataset = FashionMNIST(root=data_dir, train=TRAINING, download=True)
correct = 0
iter = tqdm(dataset, total=len(dataset))
text = tokenizer(labels).to(device)
ground_truth = []
output = []
probabilities = []
dataset_clip = []

with torch.no_grad():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for idx, (image, gt) in enumerate(iter):

        # if idx == 80:
        #     break

        image_input = preprocess(image).unsqueeze(0).to(device)

        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        prediction = probs.argmax(dim=-1).item()
        ground_truth.append(gt)
        output.append(prediction)
        sample = {
            "id": idx,
            "image": image,
            "ground_truth": gt,
            "output": prediction,
            "probability": probs.squeeze(0).cpu(),
        }
        probabilities.append(
            {"no": idx, "ground_truth": gt, "output": prediction, "probability": probs.squeeze(0).cpu().tolist()}
        )
        dataset_clip.append(sample)
        if prediction == gt:
            correct += 1
        accuracy = 100 * (correct / len(dataset))
        iter.set_description(
            f"[OPEN_CLIP] Accuracy: {correct}/{len(dataset)} ({accuracy:.2f}%)"
        )
torch.save(dataset_clip, os.path.join(result_dir, f"fashion_clip_{name_type}.pt"))
data = {"ground_truth": ground_truth, "output": output}
conf_matrix(result_dir, name_type, data=data)
save_label(result_dir, name_type, data=probabilities)
