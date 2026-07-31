import torch
import torch.nn as nn
import open_clip
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_MNIST_FASHION_PATH = "data"  # Original dataset path
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, "../../..", DATA_MNIST_FASHION_PATH))

class OpenCLIPLinearProbe(nn.Module):

    def __init__(
        self,
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        num_classes=10,
        freeze_backbone=True,
    ):
        super().__init__()

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
        )

        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        embedding_dim = self.clip_model.visual.output_dim
        print(f"Embedding dimension: {embedding_dim}")

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes,
        )

    def forward(self, images):
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classifier(features)
        return logits

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

model = OpenCLIPLinearProbe().to(device)

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)

transform = model.preprocess

train_dataset = FashionMNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transform,
)

test_dataset = FashionMNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=1e-3,
)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    num_items = len(train_loader.dataset)
    running_loss = 0
    correct = 0
    total = 0
    ground_truth =[]
    output = []
    iter = tqdm(train_loader, total=len(train_loader))
    for images, labels in iter:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        _, predicted = logits.max(1)
        ground_truth.extend(labels.cpu())
        output.extend(predicted.cpu())
        correct += predicted.eq(labels).sum().item()
        perc = 100.0 * correct / num_items
        iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f} Accuracy: {correct}/{num_items} ({perc:.2f}%)")
data = {"ground_truth": ground_truth, "output": output}
conf_matrix("","",data=data)
model.eval()
correct = 0
total = 0
ground_truth =[]
output = []
with torch.no_grad():
    num_items = len(test_loader.dataset)
    iter = tqdm(test_loader, total=len(test_loader))
    for images, labels in iter:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        _, predicted = logits.max(1)
        ground_truth.extend(labels.cpu())
        output.extend(predicted.cpu())
        correct += predicted.eq(labels).sum().item()
        perc = 100.0 * correct / num_items
        iter.set_description(f"[Testing] Loss: {loss.item():.4f} Accuracy: {correct}/{num_items} ({perc:.2f}%)")
data = {"ground_truth": ground_truth, "output": output}
conf_matrix("","",data=data)