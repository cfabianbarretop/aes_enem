import os
import random
import torch
import torchvision
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scallopy
import math
import csv

from typing import *
from argparse import ArgumentParser
from distribution import main_distribution
from graphs import main_graph
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm

# ==============================================
# CONFIG
# ==============================================
DATA_MNIST_FASHION_PATH = "data"  # Original dataset path
DATA_RESULT_PATH = "result"  # Result data path
FILE_RESUL_METRIC = "result_metric"  # Name file result
device = "cuda" if torch.cuda.is_available() else "cpu"
cy = {0: 988, 1: 12}
UPPER = {0, 2, 4, 6}
LOWER = {1}
SHOES = {5, 7, 9}
print("Device: ", device)


# ==============================================
# Dataset
# ==============================================
mnist_img_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)


class MNISTFashionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        # Contains a MNIST dataset
        self.mnist_dataset = torchvision.datasets.FashionMNIST(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        if not train:
            targets = self.mnist_dataset.targets.numpy()
            indices = np.arange(len(targets))
            selected_indices, _ = train_test_split(
                indices, train_size=9000, stratify=targets, random_state=42
            )
            self.mnist_dataset = Subset(self.mnist_dataset, selected_indices)

        self.index_map = list(range(len(self.mnist_dataset)))
        random.shuffle(self.index_map)

    def __len__(self):
        return len(self.mnist_dataset) // 3

    def __getitem__(self, idx):
        # Get three data points
        i = idx * 3
        img1, digit1 = self.mnist_dataset[self.index_map[i]]
        img2, digit2 = self.mnist_dataset[self.index_map[i + 1]]
        img3, digit3 = self.mnist_dataset[self.index_map[i + 2]]

        # Each data has two images and the GT is the sum of two digits
        return (
            img1,
            img2,
            img3,
            digit1,
            digit2,
            digit3,
            digit1 + digit2 + digit3,
            valid_outfit(digit1, digit2, digit3),
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
    upper, lower, shoes = 0, 0, 0
    for d in [digit1, digit2, digit3]:
        if d in UPPER:
            upper += 1
        elif d in LOWER:
            lower += 1
        elif d in SHOES:
            shoes += 1
    return int(upper == 1 and lower == 1 and shoes == 1)
    # upper = digit1 in UPPER
    # lower = digit2 in LOWER
    # shoes = digit3 in SHOES
    # return int(upper and lower and shoes)


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
        ),
        collate_fn=MNISTFashionDataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=True,
    )

    return train_loader, test_loader


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
# Modelo Lógico
# ==============================================
class MNISTFashionLogic(nn.Module):
    def __init__(self, provenance, k):
        super(MNISTFashionLogic, self).__init__()

        # MNIST Digit Recognition Network
        self.mnist_one_net = MNISTFashionNet()
        # self.mnist_two_net = MNISTFashionNet()
        # self.mnist_three_net = MNISTFashionNet()

        # Scallop Context
        self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
        self.scl_ctx.add_relation("digit_1", int, input_mapping=list(range(10)))
        self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(10)))
        self.scl_ctx.add_relation("digit_3", int, input_mapping=list(range(10)))
        self.scl_ctx.add_relation("upper", int)
        self.scl_ctx.add_relation("lower", int)
        self.scl_ctx.add_relation("shoe", int)

        self.scl_ctx.add_facts(
            "upper",
            [
                (torch.tensor(1.0, device=device), (0,)),
                (torch.tensor(1.0, device=device), (2,)),
                (torch.tensor(1.0, device=device), (4,)),
                (torch.tensor(1.0, device=device), (6,)),
            ],
        )
        self.scl_ctx.add_facts("lower", [(torch.tensor(1.0, device=device), (1,))])
        self.scl_ctx.add_facts(
            "shoe",
            [
                (torch.tensor(1.0, device=device), (5,)),
                (torch.tensor(1.0, device=device), (7,)),
                (torch.tensor(1.0, device=device), (9,)),
            ],
        )

        self.scl_ctx.add_rule("wear(X) :- digit_1(X)")
        self.scl_ctx.add_rule("wear(X) :- digit_2(X)")
        self.scl_ctx.add_rule("wear(X) :- digit_3(X)")

        self.scl_ctx.add_rule("has_upper(X) :- wear(X), upper(X)")
        self.scl_ctx.add_rule("has_lower(X) :- wear(X), lower(X)")
        self.scl_ctx.add_rule("has_shoe(X) :- wear(X), shoe(X)")
        # self.scl_ctx.add_rule("has_upper(X) :- digit_1(X), upper(X)")
        # self.scl_ctx.add_rule("has_lower(X) :- digit_2(X), lower(X)")
        # self.scl_ctx.add_rule("has_shoe(X) :- digit_3(X), shoe(X)")

        self.scl_ctx.add_rule("valid() :- has_upper(U), has_lower(L), has_shoe(S)")

        self.valid = self.scl_ctx.forward_function("valid", output_mapping=[()])

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        a_imgs, b_imgs, c_imgs = x

        # First recognize the two digits
        a_distrs = self.mnist_one_net(a_imgs)  # Tensor 64 x 10
        b_distrs = self.mnist_one_net(b_imgs)  # Tensor 64 x 10
        c_distrs = self.mnist_one_net(c_imgs)  # Tensor 64 x 10

        # Then execute the reasoning module; the result is a size 19 tensor
        return (
            a_distrs,
            b_distrs,
            c_distrs,
            self.valid(digit_1=a_distrs, digit_2=b_distrs, digit_3=c_distrs),
        )  # Tensor 64 x 19


# ==============================================
# Guardar resultados
# ==============================================
def save_metrics(file_path, file_name, metric):
    name_file = f"{file_path}/{FILE_RESUL_METRIC}_{file_name}.csv"
    with open(name_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "loss",
                "acc",
                "GAcc",
                "gt",
                "rs",
                "RSR",
                "RSRw",
                "prob_model",
                "prob_mod_no",
            ]
        )
        for row in metric:
            writer.writerow(
                [
                    row["epoch"],
                    row["loss"],
                    row["acc"],
                    row["GAcc"],
                    row["gt"],
                    row["rs"],
                    row["RSR"],
                    row["RSRw"],
                    row["prob_model"],
                    row["prob_mod_no"],
                ]
            )


# ==============================================
# Metricas
# ==============================================
def metrics(g1, g2, g3, y, c1, pc1, c2, pc2, c3, pc3, p):
    pred_tuples = list(zip(c1, c2, c3, p))
    gt_tuples = list(zip(g1, g2, g3, y))
    cont = 0
    cont_gt = 0
    sum_ars = 0
    sum_gt = 0
    sum_model = 0
    count = 0
    count_with_0 = 0
    count_without_1 = 0
    for i, (pred, gt) in enumerate(zip(pred_tuples, gt_tuples)):
        if pred[3] == 1:
            count += 1
        if pred != gt:
            if pred[3] == gt[3]:
                peso = cy.get(pred[3], 0)
                sum_ars += math.log(1 / peso)
                p_c1 = pc1[i]
                p_c2 = pc2[i]
                p_c3 = pc3[i]
                sum_model += (1 - (p_c1 * p_c2 * p_c3)) * math.log(1 / peso)
                if gt[3] == 1:
                    count_without_1 += 1
                    # print(f"Error en índice {i}: pred={pred}, gt={gt}")
                else:
                    count_with_0 += 1
                # print(f"Error en índice {i}: pred={pred}, gt={gt}")
                cont += 1
        else:
            peso = cy.get(pred[3], 0)
            sum_gt += math.log(1 / peso)
            p_c1 = pc1[i]
            p_c2 = pc2[i]
            p_c3 = pc3[i]
            sum_model += (1 - (p_c1 * p_c2 * p_c3)) * math.log(1 / peso)
            cont_gt += 1
            if gt[3] == 1:
                print(f"Correcto ----> {i}: pred={pred}, gt={gt}")

    print(f"\tTotal de valores errados: {cont}")
    print(f"\t                       1: {count_without_1}")
    print(f"\t                       0: {count_with_0}")
    print(f"\tTotal de valores verdaderos: {cont_gt}")
    print(f"\tTotal de valores acertados: {cont + cont_gt}")
    print(f"\tTotal del dataset original con vestimenta correcta: {count}")
    return (
        cont_gt,
        cont,
        cont / (cont + cont_gt),
        sum_ars / (sum_ars + sum_gt),
        sum_model,
        sum_model / (sum_ars + sum_gt),
    )


# ==============================================
# Calculo de error
# ==============================================
def bce_loss(output, ground_truth):
    _, dim = output.shape
    gt = torch.stack(
        [
            torch.tensor([1.0 if i == t else 0.0 for i in range(dim)])
            for t in ground_truth
        ]
    )
    return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
    eps = 1e-8
    return F.nll_loss(torch.log(output + eps), ground_truth)

def aal_loss(output, ground_truth, alpha=31):
    # batch_size = output.shape[0]
    # loss = torch.tensor(0.0, device=output.device)
    # for b, i in enumerate(ground_truth):
    #     y = i.item()
    #     p = torch.log(output[b, y].clamp(min=1e-8))
    #     weight = cy.get(y, 1)
    #     w = torch.log(torch.tensor(1 + (alpha / weight), device=output.device))
    #     w = w / torch.log(torch.tensor(1 + alpha, device=output.device))
    #     w = w.detach()
    #     loss += -p * w 
    # return loss / batch_size
    # output = output.view(-1)
    # ground_truth = ground_truth.float().view(-1)
    batch_size = output.shape[0]
    loss = torch.tensor(0.0, device=output.device)

    for b in range(batch_size):
        y = int(ground_truth[b].item())

        # Evitar log(0)
        p = output[b].clamp(min=1e-8, max=1-1e-8)

        if y == 1:
            log_prob = torch.log(p)
        else:
            log_prob = torch.log(1.0 - p)

        weight = cy[y]

        w = torch.log(torch.tensor(1 + alpha / weight, device=output.device))
        w = w / torch.log(torch.tensor(1 + alpha, device=output.device))
        w = w.detach()

        loss += -w * log_prob

    return loss / batch_size


# ==============================================
# Entrenamiento y Test
# ==============================================
class Trainer:
    def __init__(
        self, result_dir, train_loader, test_loader, learning_rate, loss, k, provenance
    ):
        self.network = MNISTFashionLogic(provenance, k).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.result_dir = result_dir
        self.metrics_train = []
        self.metrics_test = []

        if loss == "nll":
            self.loss = nll_loss
        elif loss == "bce":
            self.loss = bce_loss
        elif loss == "ll":
            self.loss = nn.BCELoss()
        elif loss == "aal":
            self.loss = aal_loss
        else:
            raise Exception(f"Unknown loss function `{loss}`")

    def train_epoch(self, epoch):
        self.network.train()
        correct = 0
        iter = tqdm(self.train_loader, total=len(self.train_loader))
        num_items = len(self.train_loader.dataset)
        c1, c2, c3 = [], [], []
        pc1, pc2, pc3 = [], [], []
        g1, g2, g3 = [], [], []
        y, p, pb = [], [], []
        for images, digits, labels in iter:
            a_imgs, b_imgs, c_imgs = images
            a_digit, b_digit, c_digit = digits
            a_imgs = a_imgs.to(device)
            b_imgs = b_imgs.to(device)
            c_imgs = c_imgs.to(device)
            images = (a_imgs, b_imgs, c_imgs)
            sum_3, target = labels
            self.optimizer.zero_grad()
            a_distrs, b_distrs, c_distrs, output = self.network(images)
            output = output.cpu()
            g1.extend(a_digit.tolist())
            g2.extend(b_digit.tolist())
            g3.extend(c_digit.tolist())
            t_pc1, t_c1 = a_distrs.max(dim=1)
            t_pc2, t_c2 = b_distrs.max(dim=1)
            t_pc3, t_c3 = c_distrs.max(dim=1)
            t_pb, t_p = output.max(dim=1)
            t_p = (t_pb >= 0.5).int()
            c1.extend(t_c1.tolist())
            c2.extend(t_c2.tolist())
            c3.extend(t_c3.tolist())
            pc1.extend(t_pc1.tolist())
            pc2.extend(t_pc2.tolist())
            pc3.extend(t_pc3.tolist())
            p.extend(t_p.tolist())
            pb.extend(t_pb.tolist())
            y.extend(target.tolist())
            loss = self.loss(output.squeeze(1), target.float())
            # pred = output.data.max(1, keepdim=True)[1]
            pred = t_p
            correct += pred.eq(target.data.view_as(pred)).sum()
            perc = 100.0 * correct / num_items
            loss.backward()
            self.optimizer.step()
            iter.set_description(
                f"[Train Epoch {epoch}] Loss: {loss.item():.4f} Accuracy: {correct}/{num_items} ({perc:.2f}%)"
            )
        gt, rs, rsr, rsrw, prob_model, prob_mod_no = metrics(
            g1, g2, g3, y, c1, pc1, c2, pc2, c3, pc3, p
        )
        correct_concepts = (
            sum(gt == pred for gt, pred in zip(g1, c1))
            + sum(gt == pred for gt, pred in zip(g2, c2))
            + sum(gt == pred for gt, pred in zip(g3, c3))
        )
        total_concepts = 3 * len(g1)
        gacc = 100.0 * correct_concepts / total_concepts
        self.metrics_train.append(
            {
                "epoch": epoch,
                "loss": loss.item(),
                "acc": perc.item(),
                "GAcc": gacc,
                "gt": gt,
                "rs": rs,
                "RSR": rsr,
                "RSRw": rsrw,
                "prob_model": prob_model,
                "prob_mod_no": prob_mod_no,
            }
        )

    def test(self, epoch):
        self.network.eval()
        num_items = len(self.test_loader.dataset)
        test_loss = 0
        correct = 0
        c1, c2, c3 = [], [], []
        pc1, pc2, pc3 = [], [], []
        g1, g2, g3 = [], [], []
        y, p, pb = [], [], []
        with torch.no_grad():
            iter = tqdm(self.test_loader, total=len(self.test_loader))
            for images, digits, labels in iter:
                a_imgs, b_imgs, c_imgs = images
                a_digit, b_digit, c_digit = digits
                a_imgs = a_imgs.to(device)
                b_imgs = b_imgs.to(device)
                c_imgs = c_imgs.to(device)
                images = (a_imgs, b_imgs, c_imgs)
                sum_3, target = labels
                a_distrs, b_distrs, c_distrs, output = self.network(images)
                output = output.cpu()
                g1.extend(a_digit.tolist())
                g2.extend(b_digit.tolist())
                g3.extend(c_digit.tolist())
                t_pc1, t_c1 = a_distrs.max(dim=1)
                t_pc2, t_c2 = b_distrs.max(dim=1)
                t_pc3, t_c3 = c_distrs.max(dim=1)
                t_pb, t_p = output.max(dim=1)
                t_p = (t_pb >= 0.5).int()
                c1.extend(t_c1.tolist())
                c2.extend(t_c2.tolist())
                c3.extend(t_c3.tolist())
                pc1.extend(t_pc1.tolist())
                pc2.extend(t_pc2.tolist())
                pc3.extend(t_pc3.tolist())
                p.extend(t_p.tolist())
                pb.extend(t_pb.tolist())
                y.extend(target.tolist())
                test_loss = self.loss(output.squeeze(1), target.float())
                # test_loss += self.loss(output.squeeze(1), target.float()).item()
                # pred = output.data.max(1, keepdim=True)[1]
                pred = t_p
                correct += pred.eq(target.data.view_as(pred)).sum()
                perc = 100.0 * correct / num_items
                iter.set_description(
                    f"[Test Epoch {epoch}] Total loss: {test_loss.item():.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)"
                )
            gt, rs, rsr, rsrw, prob_model, prob_mod_no = metrics(
                g1, g2, g3, y, c1, pc1, c2, pc2, c3, pc3, p
            )
            correct_concepts = (
                sum(gt == pred for gt, pred in zip(g1, c1))
                + sum(gt == pred for gt, pred in zip(g2, c2))
                + sum(gt == pred for gt, pred in zip(g3, c3))
            )
            total_concepts = 3 * len(g1)
            gacc = 100.0 * correct_concepts / total_concepts
            self.metrics_test.append(
                {
                    "epoch": epoch,
                    "loss": test_loss.item(),
                    "acc": perc.item(),
                    "GAcc": gacc,
                    "gt": gt,
                    "rs": rs,
                    "RSR": rsr,
                    "RSRw": rsrw,
                    "prob_model": prob_model,
                    "prob_mod_no": prob_mod_no,
                }
            )

    def train(self, n_epochs):
        self.test(0)
        for epoch in range(1, n_epochs + 1):
            print("-----------> EPOCH: ", epoch)
            self.train_epoch(epoch)
            self.test(epoch)
        save_metrics(self.result_dir, "train", self.metrics_train)
        save_metrics(self.result_dir, "test", self.metrics_test)


# ==============================================
# MAIN
# ==============================================
if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("mnist_fashion")
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--batch-size-train", type=int, default=64)
    parser.add_argument("--batch-size-test", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--loss-fn", type=str, default="aal")
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

    # Obtiene el directorio donde está este archivo.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Une el directorio de base_dir con la carpeta "data"
    data_dir = os.path.abspath(os.path.join(base_dir, "..", DATA_MNIST_FASHION_PATH))
    result_dir = os.path.join(base_dir, DATA_RESULT_PATH)
    print("PATH data -> ", data_dir)
    train_loader, test_loader = mnist_fashion_loader(
        data_dir, batch_size_train, batch_size_test
    )
    # Create trainer and train
    trainer = Trainer(
        result_dir, train_loader, test_loader, learning_rate, loss_fn, k, provenance
    )
    # trainer.train(n_epochs)
    main_graph("train")
    # main_distribution(train_loader, test_loader)
