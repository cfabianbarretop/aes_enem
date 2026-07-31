import torch
import torch.nn as nn
import open_clip
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

from typing import *
from tqdm import tqdm
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==============================================
# CONFIG
# ==============================================
device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_MNIST_FASHION_PATH = "data"  # Original dataset path
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, "../../..", DATA_MNIST_FASHION_PATH))

UPPER = {0, 2, 4, 6}
LOWER = {1}
SHOES = {5, 7, 9}

# ==============================================
# Dataset MNIST Fashion
# ==============================================
# mnist_img_transform = transforms.Compose(
#     [transforms.ToTensor()]
# )
mnist_img_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)


class MNISTFashionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        cache_file: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        # Si existe cache, cargar directamente
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.index_map, self.mnist_dataset = pickle.load(f)
            return

        # Contains a MNIST dataset
        self.mnist_dataset = datasets.FashionMNIST(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        targets = np.array(self.mnist_dataset.targets)

        # Filtrar índices
        upper_idx = np.where(np.isin(targets, list(UPPER)))[0]
        lower_idx = np.where(np.isin(targets, list(LOWER)))[0]
        shoes_idx = np.where(np.isin(targets, list(SHOES)))[0]
        all_idx = np.arange(len(targets))

        # Número máximo de válidos
        n_valid = min(len(upper_idx), len(lower_idx), len(shoes_idx))

        # Selección sin repetición
        upper_sel = np.random.choice(upper_idx, n_valid, replace=False)
        lower_sel = np.random.choice(lower_idx, n_valid, replace=False)
        shoes_sel = np.random.choice(shoes_idx, n_valid, replace=False)

        # Guardar válidos
        valid_outfits = [(u, l, s) for u, l, s in zip(upper_sel, lower_sel, shoes_sel)]

        # Generar inválidos balanceados
        invalid_outfits = []
        used = set(upper_sel) | set(lower_sel) | set(shoes_sel)
        while len(invalid_outfits) < n_valid:
            trio = np.random.choice(all_idx, 3, replace=False)
            d1, d2, d3 = [targets[i] for i in trio]
            if valid_outfit(d1, d2, d3) == 0 and not any(i in used for i in trio):
                invalid_outfits.append(tuple(trio))
                used.update(trio)

        # self.index_map = list(range(len(self.mnist_dataset)))
        # random.shuffle(self.index_map)
        self.index_map = [(u, l, s) for (u, l, s) in valid_outfits] + invalid_outfits
        random.shuffle(self.index_map)

        # Guardar en cache
        with open(cache_file, "wb") as f:
            pickle.dump((self.index_map, self.mnist_dataset), f)

    def __len__(self):
        # return len(self.mnist_dataset) // 3
        return len(self.index_map)

    def __getitem__(self, idx):
        # Get three data points
        i1, i2, i3 = self.index_map[idx]
        img1, d1 = self.mnist_dataset[i1]
        img2, d2 = self.mnist_dataset[i2]
        img3, d3 = self.mnist_dataset[i3]

        return (
            img1,
            img2,
            img3,
            d1,
            d2,
            d3,
            d1 + d2 + d3,
            valid_outfit(d1, d2, d3),
        )

    @staticmethod
    def collate_fn(batch):
        img1 = torch.stack([item[0] for item in batch])
        img2 = torch.stack([item[1] for item in batch])
        img3 = torch.stack([item[2] for item in batch])
        digit1 = torch.stack([torch.tensor(item[3]).long() for item in batch])
        digit2 = torch.stack([torch.tensor(item[4]).long() for item in batch])
        digit3 = torch.stack([torch.tensor(item[5]).long() for item in batch])
        sum_3 = torch.stack([torch.tensor(item[6]).long() for item in batch])
        label = torch.stack([torch.tensor(item[7]).long() for item in batch])
        return ((img1, img2, img3), (digit1, digit2, digit3), (sum_3, label))


# ==============================================
# Funcion verifica outfit
# ==============================================
def valid_outfit(digit1, digit2, digit3):
    upper = digit1 in UPPER
    lower = digit2 in LOWER
    shoes = digit3 in SHOES
    return int(upper and lower and shoes)


# ==============================================
# Funcion data loader
# ==============================================
def mnist_fashion_loader(data_dir, batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        MNISTFashionDataset(
            data_dir,
            train=True,
            download=True,
            transform=mnist_img_transform,
            cache_file="3_fashion_outfits_train.pkl",
        ),
        collate_fn=MNISTFashionDataset.collate_fn,
        batch_size=batch_size_train,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        MNISTFashionDataset(
            data_dir,
            train=False,
            download=True,
            transform=mnist_img_transform,
            cache_file="3_fashion_outfits_test.pkl",
        ),
        collate_fn=MNISTFashionDataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=True,
    )

    return train_loader, test_loader

class OpenCLIPLinearProbe(nn.Module):

    def __init__(
        self,
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        num_classes=10,
        freeze_backbone=False,
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
        # with torch.no_grad():
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

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam([
    {
        "params": model.clip_model.parameters(),
        "lr": 1e-5,      # Muy pequeño
    },
    {
        "params": model.classifier.parameters(),
        "lr": 1e-3,      # Más grande
    },
])

train_loader, test_loader = mnist_fashion_loader(data_dir, 32, 64)
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    num_items = 3 * len(train_loader.dataset)
    running_loss = 0
    correct = 0
    acc_1 = 0
    acc_2 = 0
    acc_3 = 0
    ground_truth =[]
    output = []
    iter = tqdm(train_loader, total=len(train_loader))
    for images, digits, labels in iter:
        a_imgs, b_imgs, c_imgs = images
        a_digit, b_digit, c_digit = digits
        a_imgs = a_imgs.to(device)
        b_imgs = b_imgs.to(device)
        c_imgs = c_imgs.to(device)
        a_digit = a_digit.to(device)
        b_digit = b_digit.to(device)
        c_digit = c_digit.to(device)
        optimizer.zero_grad()
        output1 = model(a_imgs)
        output2 = model(b_imgs)
        output3 = model(c_imgs)
        loss1 = criterion(output1, a_digit)
        loss2 = criterion(output2, b_digit)
        loss3 = criterion(output3, c_digit)
        loss = (loss1 + loss2 + loss3) / 3
        loss.backward()
        optimizer.step()
        _, predicted1 = output1.max(1)
        _, predicted2 = output2.max(1)
        _, predicted3 = output3.max(1)
        ground_truth.extend(a_digit.cpu())
        output.extend(predicted1.cpu())
        acc_1 += predicted1.eq(a_digit).sum().item()
        correct += predicted1.eq(a_digit).sum().item()
        ground_truth.extend(b_digit.cpu())
        output.extend(predicted2.cpu())
        acc_2 += predicted2.eq(b_digit).sum().item()
        correct += predicted2.eq(b_digit).sum().item()
        ground_truth.extend(c_digit.cpu())
        output.extend(predicted3.cpu())
        acc_3 += predicted3.eq(c_digit).sum().item()
        correct += predicted3.eq(c_digit).sum().item()
        perc = 100.0 * correct / num_items
        perc_1 = 100.0 * acc_1 / len(train_loader.dataset)
        perc_2 = 100.0 * acc_2 / len(train_loader.dataset)
        perc_3 = 100.0 * acc_3 / len(train_loader.dataset)
        iter.set_description(
            f"[Train Epoch {epoch}] Loss: {loss.item():.4f} Accuracy: {correct}/{num_items} ({perc:.2f}%) => ({perc_1:.2f}%), ({perc_2:.2f}%), ({perc_3:.2f}%)"
        )
data = {"ground_truth": ground_truth, "output": output}
conf_matrix("","",data=data)
model.eval()
correct = 0
acc_1 = 0
acc_2 = 0
acc_3 = 0
ground_truth =[]
output = []
with torch.no_grad():
    num_items = 3 * len(test_loader.dataset)
    iter = tqdm(test_loader, total=len(test_loader))
    for images, digits, labels in iter:
        a_imgs, b_imgs, c_imgs = images
        a_digit, b_digit, c_digit = digits
        a_imgs = a_imgs.to(device)
        b_imgs = b_imgs.to(device)
        c_imgs = c_imgs.to(device)
        a_digit = a_digit.to(device)
        b_digit = b_digit.to(device)
        c_digit = c_digit.to(device)
        output1 = model(a_imgs)
        output2 = model(b_imgs)
        output3 = model(c_imgs)
        loss1 = criterion(output1, a_digit)
        loss2 = criterion(output2, b_digit)
        loss3 = criterion(output3, c_digit)
        loss = (loss1 + loss2 + loss3) / 3
        _, predicted1 = output1.max(1)
        _, predicted2 = output2.max(1)
        _, predicted3 = output3.max(1)
        ground_truth.extend(a_digit.cpu())
        output.extend(predicted1.cpu())
        acc_1 += predicted1.eq(a_digit).sum().item()
        correct += predicted1.eq(a_digit).sum().item()
        ground_truth.extend(b_digit.cpu())
        output.extend(predicted2.cpu())
        acc_2 += predicted2.eq(b_digit).sum().item()
        correct += predicted2.eq(b_digit).sum().item()
        ground_truth.extend(c_digit.cpu())
        output.extend(predicted3.cpu())
        acc_3 += predicted3.eq(c_digit).sum().item()
        correct += predicted3.eq(c_digit).sum().item()
        perc = 100.0 * correct / num_items
        perc_1 = 100.0 * acc_1 / len(test_loader.dataset)
        perc_2 = 100.0 * acc_2 / len(test_loader.dataset)
        perc_3 = 100.0 * acc_3 / len(test_loader.dataset)
        iter.set_description(
            f"[Test] Loss: {loss.item():.4f} Accuracy: {correct}/{num_items} ({perc:.2f}%) => ({perc_1:.2f}%), ({perc_2:.2f}%), ({perc_3:.2f}%)"
        )
data = {"ground_truth": ground_truth, "output": output}
conf_matrix("","",data=data)