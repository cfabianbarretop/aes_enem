import os
import random
import csv
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from argparse import ArgumentParser

from graphs import  main_graph
from distribution import main_distribution

import scallopy

import kagglehub

# ==============================================
# CONFIG
# ==============================================
DATA_PATH = "data"
DATA_RESULT_PATH = "result"              # Result data path
METRIC_RESULT_PATH = "result_metric"     # Name file result
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

# Download latest version
data_path = kagglehub.dataset_download("aladdinss/license-plate-digits-classification-dataset")

DATA_PATH = data_path + "/CNN letter Dataset"


# ==============================================
# Dataset
# ==============================================

transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor()
])

class LicensePlateDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_samples=10000):
        self.root_dir = root_dir
        self.transform = transform
        self.num_samples = num_samples
        
        # Carpetas de dígitos y letras
        self.digits = [str(i) for i in range(10)]
        self.letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]

        # Diccionario de mapeo
        self.classes = self.digits + self.letters
        self.char_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

        # Mapear imágenes
        self.data_map = {}
        for cls in self.classes:
            folder = os.path.join(root_dir, cls)
            if os.path.exists(folder) and os.path.isdir(folder):
                files = os.listdir(folder)
                if files:
                    self.data_map[cls] = [os.path.join(folder, f) for f in files]
                else:
                    print(f"Carpeta {folder} está vacía, se omite.")
            else:
                print(f"Carpeta {folder} no encontrada, se omite.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        while True:  # asegura que siempre tengamos 6 imágenes
            d1, d2, d3 = random.choices(self.digits, k=3)
            l1, l2, l3 = random.choices(self.letters, k=3)

            chars = [d1, d2, d3, l1, l2, l3]
            imgs = []

            valid = True
            for char in chars:
                if char not in self.data_map or not self.data_map[char]:
                    valid = False
                    break

                img_path = random.choice(self.data_map[char])
                img = Image.open(img_path).convert("RGB")
                img = img.resize((64, 64))
                imgs.append(img)

            if valid:
                break  # salimos cuando tenemos todo correcto

        # Concatenar horizontalmente
        plate_img = Image.new("RGB", (64 * 6, 64))
        for i, im in enumerate(imgs):
            plate_img.paste(im, (i * 64, 0))

        if self.transform:
            plate_img = self.transform(plate_img)

        # -------------------------
        # LABELS POR CONCEPTO
        # -------------------------
        d1_idx = int(d1)
        d2_idx = int(d2)
        d3_idx = int(d3)

        l1_idx = self.char_to_idx[l1]
        l2_idx = self.char_to_idx[l2]
        l3_idx = self.char_to_idx[l3]

        # -------------------------
        # LABEL FINAL y
        # -------------------------
        y = 1 if d1_idx % 2 == 0 else 0

        # Convertir a tensores
        concepts = torch.tensor([d1_idx, d2_idx, d3_idx, l1_idx, l2_idx, l3_idx], dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return plate_img, concepts, y

def show_sample(loader):
    images, concepts, y = next(iter(loader))

    plt.imshow(images[0].permute(1, 2, 0))
    plt.axis("off")
    plt.show()

    print("Conceptos (d1,d2,d3,l1,l2,l3):", concepts[0].tolist())
    print("y:", y[0].item())


def dataloader(root_dir, batch_size_train=32, batch_size_test=32, num_samples=10000):
    dataset = LicensePlateDataset(root_dir, transform=transform, num_samples=num_samples)
    
    # Split train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    print(f"Train loader: {len(train_loader.dataset)} samples")
    print(f"Test loader: {len(test_loader.dataset)} samples")

    return train_loader, test_loader

# ==============================================
# Neural Model
# ==============================================
class CharNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 → 32

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 → 16
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # return F.softmax(x, dim=1)
        return x
    
# ==============================================
# Logic Model
# ==============================================
class LicensePlateNet(nn.Module):
    def __init__(self, provenance="difftopkproofs", k=3):
        super().__init__()

        self.digit_net = CharNet(num_classes=10)
        self.letter_net = CharNet(num_classes=26)

        self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)

        self.scl_ctx.add_relation("digit_1", int, input_mapping=list(range(10)))

        self.scl_ctx.add_rule("valid(1) :- digit_1(d), d % 2 == 0")
        self.scl_ctx.add_rule("valid(0) :- digit_1(d), d % 2 == 1")

        self.valid = self.scl_ctx.forward_function("valid", output_mapping=[(0,), (1,)])

    def split_plate(self, x):
        # x: (B, 3, 64, 384)
        return torch.split(x, 64, dim=3)

    def forward(self, x):
        chars = self.split_plate(x)

        d1 = self.digit_net(chars[0])
        d2 = self.digit_net(chars[1])
        d3 = self.digit_net(chars[2])

        l1 = self.letter_net(chars[3])
        l2 = self.letter_net(chars[4])
        l3 = self.letter_net(chars[5])

        d1_probs = F.softmax(d1, dim=1)
        d2_probs = F.softmax(d2, dim=1)
        d3_probs = F.softmax(d3, dim=1)

        l1_probs = F.softmax(l1, dim=1)
        l2_probs = F.softmax(l2, dim=1)
        l3_probs = F.softmax(l3, dim=1)

        y_pred = self.valid(digit_1=d1_probs)

        return {
            "digits": (d1_probs, d2_probs, d3_probs),
            "letters": (l1_probs, l2_probs, l3_probs),
            "y": y_pred
        }


# ==============================================
# Training and Testing
# ==============================================
def bce_loss(output, ground_truth):
    gt = F.one_hot(ground_truth, num_classes=output.shape[1]).float()
    gt = gt.to(output.device)
    return F.binary_cross_entropy_with_logits(output, gt)

# def bce_loss(output, ground_truth):
#   (_, dim) = output.shape
#   gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
#   return F.binary_cross_entropy(output, gt)

def save_metrics(file_path, file_name, metric):
    name_file = f"{file_path}/{METRIC_RESULT_PATH}_{file_name}.csv"
    with open(name_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch","loss",
                         "accY", "accC", 
                         "acc_C1", "acc_C2", "acc_C3", "acc_C4", "acc_C5", "acc_C6"])
        for row in metric:
            writer.writerow([
                row["epoch"],
                row["loss"],
                row["accY"],
                row["accC"],
                row["acc_C1"],
                row["acc_C2"],
                row["acc_C3"],
                row["acc_C4"],
                row["acc_C5"],
                row["acc_C6"]
            ])

class Trainer():
    def __init__(self, result_dir, train_loader, test_loader, learning_rate, loss, k, provenance):
        self.network = LicensePlateNet(provenance, k).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.result_dir = result_dir
        self.metrics_train = []
        self.metrics_test = []
        
        if loss == "bce":
            self.loss = bce_loss
        else:
            raise Exception(f"Unknown loss function `{loss}`")

    def train_epoch(self, epoch):
        self.network.train()
        correct = 0
        num_items = len(self.train_loader.dataset)

        iter = tqdm(self.train_loader, total=len(self.train_loader))

        # Ground truth
        g = [[] for _ in range(6)]  # d1,d2,d3,l1,l2,l3

        # Predicted concepts
        c = [[] for _ in range(6)]  # c1,c2,c3,c4,c5,c6

        y_true = []
        y_pred = []

        for (imgs, concepts, target) in iter:
            imgs = imgs.to(device)
            concepts = concepts.to(device)
            target = target.to(device)

            self.optimizer.zero_grad()

            output = self.network(imgs)

            d1, d2, d3 = output["digits"]
            l1, l2, l3 = output["letters"]

            y_out = output["y"]  # (B,2)

            # -------------------------
            # Predictions
            # -------------------------
            preds_digits = [d1, d2, d3]
            preds_letters = [l1, l2, l3]

            all_preds = preds_digits + preds_letters

            for i in range(6):
                _, pred = all_preds[i].max(dim=1)
                c[i].extend(pred.cpu().tolist())
                g[i].extend(concepts[:, i].cpu().tolist())

            # -------------------------
            # Y
            # -------------------------
            _, pred_y = y_out.max(dim=1)

            y_pred.extend(pred_y.cpu().tolist())
            y_true.extend(target.cpu().tolist())

            loss = self.loss(y_out, target)

            correct += pred_y.eq(target).sum().item()

            loss.backward()
            self.optimizer.step()

            perc = 100. * correct / num_items
            iter.set_description(f"[Train {epoch}] Loss: {loss.item():.4f} AccY: {perc:.2f}%")

        # -------------------------
        # Concept Metrics
        # -------------------------
        acc_ci = []
        for i in range(6):
            acc = 100.0 * sum(a == b for a, b in zip(g[i], c[i])) / len(g[i])
            acc_ci.append(acc)

        # todos correctos
        correct_all = sum(
            all(g[i][j] == c[i][j] for i in range(6))
            for j in range(len(g[0]))
        )
        accC = 100.0 * correct_all / len(g[0])

        # -------------------------
        # Save metrics
        # -------------------------
        self.metrics_train.append({
            "epoch": epoch,
            "loss": loss.item(),
            "accY": 100.0 * correct / num_items,
            "accC": accC,
            "acc_C1": acc_ci[0],
            "acc_C2": acc_ci[1],
            "acc_C3": acc_ci[2],
            "acc_C4": acc_ci[3],
            "acc_C5": acc_ci[4],
            "acc_C6": acc_ci[5]
        })

    def test(self, epoch):
        self.network.eval()
        num_items = len(self.test_loader.dataset)

        test_loss = 0
        correct = 0

        g = [[] for _ in range(6)]
        c = [[] for _ in range(6)]

        y_true = []
        y_pred = []

        with torch.no_grad():
            iter = tqdm(self.test_loader, total=len(self.test_loader))

            for (imgs, concepts, target) in iter:
                imgs = imgs.to(device)
                concepts = concepts.to(device)
                target = target.to(device)

                output = self.network(imgs)

                d1, d2, d3 = output["digits"]
                l1, l2, l3 = output["letters"]
                y_out = output["y"]

                all_preds = [d1, d2, d3, l1, l2, l3]

                for i in range(6):
                    _, pred = all_preds[i].max(dim=1)
                    c[i].extend(pred.cpu().tolist())
                    g[i].extend(concepts[:, i].cpu().tolist())

                _, pred_y = y_out.max(dim=1)

                y_pred.extend(pred_y.cpu().tolist())
                y_true.extend(target.cpu().tolist())

                test_loss += self.loss(y_out, target).item()
                correct += pred_y.eq(target).sum().item()

                perc = 100. * correct / num_items
                iter.set_description(f"[Test {epoch}] Loss: {test_loss:.4f} AccY: {perc:.2f}%")

        # -------------------------
        # Concept Metrics
        # -------------------------
        acc_ci = []
        for i in range(6):
            acc = 100.0 * sum(a == b for a, b in zip(g[i], c[i])) / len(g[i])
            acc_ci.append(acc)

        correct_all = sum(
            all(g[i][j] == c[i][j] for i in range(6))
            for j in range(len(g[0]))
        )
        accC = 100.0 * correct_all / len(g[0])

        self.metrics_test.append({
            "epoch": epoch,
            "loss": test_loss,
            "accY": 100.0 * correct / num_items,
            "accC": accC,
            "acc_C1": acc_ci[0],
            "acc_C2": acc_ci[1],
            "acc_C3": acc_ci[2],
            "acc_C4": acc_ci[3],
            "acc_C5": acc_ci[4],
            "acc_C6": acc_ci[5]
        })

    def train(self, n_epochs):
        self.test(0)
        for epoch in range(1, n_epochs + 1):
            print("-----------> EPOCH: ",epoch)
            self.train_epoch(epoch)
            self.test(epoch)
        save_metrics(self.result_dir, "train", self.metrics_train)
        save_metrics(self.result_dir, "test", self.metrics_test)

if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("license_plate")
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--batch-size-train", type=int, default=64)
    parser.add_argument("--batch-size-test", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--loss-fn", type=str, default="bce")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--provenance", type=str, default="difftopkproofs")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    # Parameters
    n_epochs = args.n_epochs
    batch_size_train = args.batch_size_train
    batch_size_test = args.batch_size_test
    learning_rate = args.learning_rate
    loss_fn = args.loss_fn
    k = args.top_k
    provenance = args.provenance
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, "..", DATA_PATH))
    result_dir = os.path.join(base_dir, DATA_RESULT_PATH)
    train_file = f"{data_dir}/{DATA_PATH}/mnist_addition_level_VI.pt"
    print("PATH data -> ", data_dir)

    # Dataloaders
    train_loader, test_loader = dataloader(DATA_PATH, batch_size_train, batch_size_test, num_samples=10000)
    # Create trainer and train
    trainer = Trainer(result_dir, train_loader, test_loader, learning_rate, loss_fn, k, provenance)
    trainer.train(n_epochs)
    # main_graph("train")
    # main_distribution(train_loader, test_loader)

    # show_sample(train_loader)
    # show_sample(test_loader)







