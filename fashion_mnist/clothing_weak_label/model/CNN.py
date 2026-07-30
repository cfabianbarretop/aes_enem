from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_MNIST_FASHION_PATH = "data"  # Original dataset path
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, "../../..", DATA_MNIST_FASHION_PATH))
print(data_dir)
print(device)

# ==============================================
# Dataset MNIST Fashion
# ==============================================
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
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
    # Guardar como imagen
    # plt.savefig(f"{file_path}/{FILE_RESULT_MATRIX}_{file_name}.png", dpi=300)
    plt.show()
    plt.close()

# ==============================================
# Modelo Neural
# ==============================================
class MNISTFashionNet(nn.Module):
    def __init__(self):
        super(MNISTFashionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# ==============================================
# Entrenamiento y Test
# ==============================================
model = MNISTFashionNet().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

num_epochs = 10
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
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        ground_truth.extend(labels.cpu())
        output.extend(predicted.cpu())
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        perc = 100.0 * correct / num_items
        iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f} Accuracy: {correct}/{num_items} ({perc:.2f}%)")
    train_acc = 100 * correct / total
    print(
        f"Epoch {epoch+1}: "
        f"Loss={running_loss/len(train_loader):.4f} "
        f"Acc={train_acc:.2f}%"
    )
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
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = outputs.max(1)
        ground_truth.extend(labels.cpu())
        output.extend(predicted.cpu())
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        perc = 100.0 * correct / num_items
        iter.set_description(f"[Test] Loss: {loss.item():.4f} Accuracy: {correct}/{num_items} ({perc:.2f}%)")
test_acc = 100 * correct / total
print(f"Accuracy de prueba: {test_acc:.2f}%")
data = {"ground_truth": ground_truth, "output": output}
conf_matrix("","",data=data)
