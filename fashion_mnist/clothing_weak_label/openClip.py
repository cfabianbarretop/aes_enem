from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
import open_clip

# ==============================================
# CONFIG
# ==============================================
device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_MNIST_FASHION_PATH = "data"  # Original dataset path
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, "../../..", DATA_MNIST_FASHION_PATH))
print(data_dir)
print(device)

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

train_loader, test_loader = mnist_fashion_loader(data_dir, 64, 64)
correct = 0
acc_1 = 0
acc_2 = 0
acc_3 = 0
dataset = test_loader
iter = tqdm(dataset, total=len(dataset))
text = tokenizer(labels).to(device)
ground_truth = []
output = []
num_items = len(dataset.dataset)
with torch.no_grad():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for images, digits, labels in iter:
        a_imgs, b_imgs, c_imgs = images
        a_digit, b_digit, c_digit = digits

        image = torch.cat([a_imgs, b_imgs, c_imgs], dim=0).to(device)

        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        prediction = probs.argmax(dim=1).cpu()

        bs = a_imgs.size(0)

        pred_a = prediction[:bs]
        pred_b = prediction[bs : 2 * bs]
        pred_c = prediction[2 * bs :]

        ground_truth.extend(a_digit.numpy())
        ground_truth.extend(b_digit.numpy())
        ground_truth.extend(c_digit.numpy())
        output.extend(pred_a.numpy())
        output.extend(pred_b.numpy())
        output.extend(pred_c.numpy())

        correct += pred_a.eq(a_digit).sum().item()
        acc_1 += pred_a.eq(a_digit).sum().item()
        correct += pred_b.eq(b_digit).sum().item()
        acc_2 += pred_b.eq(b_digit).sum().item()
        correct += pred_c.eq(c_digit).sum().item()
        acc_3 += pred_c.eq(c_digit).sum().item()
        accuracy = 100 * (correct / (3 * num_items))
        accuracy1 = 100 * (acc_1 / num_items)
        accuracy2 = 100 * (acc_2 / num_items)
        accuracy3 = 100 * (acc_3 / num_items)
        iter.set_description(
            f"[OPEN_CLIP] Accuracy: {correct}/{3 * num_items} ({accuracy:.2f}%) => ({accuracy1:.2f}%), ({accuracy2:.2f}%), ({accuracy3:.2f}%)"
        )
data = {"ground_truth": ground_truth, "output": output}
conf_matrix("", "", data=data)
