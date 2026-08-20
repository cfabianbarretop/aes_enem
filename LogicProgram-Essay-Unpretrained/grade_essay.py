import os
import json
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
import pickle
import open_clip

from typing import *
from argparse import ArgumentParser
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from graphs import main_graph

# ==============================================
# CONFIG
# ==============================================
DATA_MNIST_FASHION_PATH = "data"  # Original dataset path
DATA_RESULT_PATH = "result"  # Result data path
FILE_RESULT_METRIC = "result_metric"  # Name file result metrics
FILE_RESULT_MATRIX = "result_matrix"  # Name file result matrix
OUTPUT_FILE_NAME = "weak_labels_LLM-JBCS.json"
TOKENIZER_NAME = f"neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
cy = {0: 1, 1: 1, 2: 6, 3: 5, 4: 3, 5: 1}
print("Device: ", device)


def get_ground_thun(result_dir, id_dataset):
    OUTPUT_FILE = f"{result_dir}/{OUTPUT_FILE_NAME}"
    gt_syntax, gt_mistake = 0, 0
    pb_syntax = [0, 0, 0, 0, 0]
    pb_mistake = [0, 0, 0, 0]

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        weak_labels = json.load(f)
    for item in weak_labels:
        if item["id"] == id_dataset:
            pb_syntax = item["one_hot"]["estrutura_sintatica"]
            pb_mistake = item["one_hot"]["desvios"]
            gt_syntax = item["weak_label"]["estrutura_sintatica"]["score"]
            gt_mistake = item["weak_label"]["desvios"]["score"]
    return gt_syntax, gt_mistake, pb_syntax, pb_mistake


# ==============================================
# Dataset
# ==============================================
class MNISTFashionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        download: bool = False,
    ):
        # Contains a MNIST dataset
        self.split_name = split
        self.dir_result = root
        if split == "train":
            self.essays = load_dataset(
                "igorcs/LLM-JBCS", cache_dir="tmp/aes_enem", trust_remote_code=True
            )["train"]
            # self.essays = self._normalizar(self.essays)
        elif split == "test":
            self.essays = load_dataset(
                "igorcs/LLM-JBCS", cache_dir="tmp/aes_enem", trust_remote_code=True
            )["test"]
            # self.essays = self._normalizar(self.essays)
        # elif split in ["test-grade-suba", "test-sub-suba"]:
        #     self.essays = load_dataset("igorcs/C1-A", trust_remote_code=True)["test"]
        # elif split in ["test-grade-subb", "test-sub-subb"]:
        #     self.essays = load_dataset("igorcs/C1-B", trust_remote_code=True)["test"]
        # elif split == "resumido-api":
        #     self.essays = load_dataset("igorcs/Sabia3ExtractorC1")["train"]
        else:
            self.essays = self.essays = load_dataset(
                "igorcs/LLM-JBCS", cache_dir="tmp/aes_enem", trust_remote_code=True
            )["validation"]
            # self.essays = self._normalizar(self.essays)

    def _normalizar(self, ds):
        df = ds.to_pandas()
        lista_dic = []
        for idx, row in df.iterrows():
            identificacao = f"{row['id']}-{row['id_prompt']}"
            for j in row["justificativa"][:]:
                dic = {}
                dic["id"] = identificacao
                # dic['justificativa'] = " ".join(row['justificativa'])
                dic["justificativa"] = j
                dic["label"] = row["label"]
                lista_dic.append(dic)
        return Dataset.from_list(lista_dic)

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, idx):
        tokenized_text = tokenizer(
            self.essays[idx]["essay_text"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        # print( self.essays[idx]['justificativa'][0] )
        if self.split_name.startswith("test-sub-"):
            label = (int(self.essays[idx]["syntax"]), int(self.essays[idx]["mistakes"]))
        else:
            if isinstance(self.essays[idx]["label"], str):
                label = eval(self.essays[idx]["label"]) // 40
            else:
                label = self.essays[idx]["label"] // 40
        id_dataset = f"{self.essays[idx]['id']}-{self.essays[idx]['id_prompt']}"
        gt_syntax, gt_mistake, pb_syntax, pb_mistake = get_ground_thun(
            self.dir_result, id_dataset
        )
        # Each data has two images and the GT is the sum of two digits
        return (
            tokenized_text,
            label,
            gt_syntax,
            gt_mistake,
            pb_syntax,
            pb_mistake,
        )  # (a_img, b_img, a_digit + b_digit)

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([item[0]["input_ids"][0] for item in batch])
        token_type = torch.stack([item[0]["token_type_ids"][0] for item in batch])
        attention_mask = torch.stack([item[0]["attention_mask"][0] for item in batch])
        digits = torch.stack([torch.tensor(item[1]).long() for item in batch])
        gt_syntax = torch.tensor([item[2] for item in batch], dtype=torch.long)
        gt_mistake = torch.tensor([item[3] for item in batch], dtype=torch.long)
        prob_syntax = torch.stack([torch.tensor(item[4]) for item in batch])
        prob_mistake = torch.stack([torch.tensor(item[5]) for item in batch])
        return (
            (input_ids, token_type, attention_mask),
            digits,
            (gt_syntax, gt_mistake),
            (prob_syntax, prob_mistake),
        )


# ==============================================
# Funcion data loader
# ==============================================
def mnist_fashion_loader(data_dir, batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        MNISTFashionDataset(
            data_dir,
            split="train",
            download=True,
        ),
        collate_fn=MNISTFashionDataset.collate_fn,
        batch_size=batch_size_train,
        shuffle=False,
    )

    test_loader = torch.utils.data.DataLoader(
        MNISTFashionDataset(
            data_dir,
            split="test",
            download=True,
        ),
        collate_fn=MNISTFashionDataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=False,
    )

    validation_loader = torch.utils.data.DataLoader(
        MNISTFashionDataset(
            data_dir,
            split="validation",
            download=True,
        ),
        collate_fn=MNISTFashionDataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=False,
    )

    return train_loader, validation_loader, test_loader


# ==============================================
# Modelo Neural
# ==============================================
class MNISTFashionModel(nn.Module):
    def __init__(self):
        super(MNISTFashionModel, self).__init__()
        self.sintaxe = AutoModelForSequenceClassification.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            cache_dir="/tmp/aes_enem2",
            num_labels=5,
        )

        self.desvios = AutoModelForSequenceClassification.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            cache_dir="/tmp/aes_enem2",
            num_labels=4,
        )

    def forward(self, x):
        output1 = self.sintaxe(
            input_ids=x[0].to(device),
            token_type_ids=x[1].to(device),
            attention_mask=x[2].to(device),
        )
        output2 = self.desvios(
            input_ids=x[0].to(device),
            token_type_ids=x[1].to(device),
            attention_mask=x[2].to(device),
        )

        return (F.softmax(output1.logits, dim=1), F.softmax(output2.logits, dim=1))


# ==============================================
# Modelo Lógico
# ==============================================
class MNISTFashionLogic(nn.Module):
    def __init__(self, provenance, k):
        super(MNISTFashionLogic, self).__init__()

        # MNIST Digit Recognition Network
        self.mnist_net = MNISTFashionModel()

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        texto = x
        # First recognize the two digits
        resposta_a, resposta_b = self.mnist_net(texto)  # Tensor 64 x 10

        sum_2 = []

        # sum_2(0) :- digit_1(0)
        p0 = resposta_a[:, 0]

        # sum_2(1) :- resposta_a(1), digit_2(0)
        p1 = resposta_a[:, 1] * resposta_b[:, 0]

        # sum_2(2)
        p2 = (
            resposta_a[:, 1] * resposta_b[:, 1:].sum(dim=1)
            + resposta_a[:, 2:].sum(dim=1) * resposta_b[:, 0]
        )

        # sum_2(3)
        p3 = (
            resposta_a[:, 2] * resposta_b[:, 1:].sum(dim=1)
            + resposta_a[:, 3:].sum(dim=1) * resposta_b[:, 1]
        )

        # sum_2(4)
        p4 = (
            resposta_a[:, 3] * resposta_b[:, 2:].sum(dim=1)
            + resposta_a[:, 4] * resposta_b[:, 2]
        )

        # sum_2(5)
        p5 = resposta_a[:, 4] * resposta_b[:, 3]

        sum_2 = torch.stack([p0, p1, p2, p3, p4, p5], dim=1)

        # Then execute the reasoning module; the result is a size 19 tensor
        return (
            resposta_a,
            resposta_b,
            sum_2,
        )


# ==============================================
# Guardar resultados
# ==============================================
def save_metrics(file_path, file_name, metric):
    name_file = f"{file_path}/{FILE_RESULT_METRIC}_{file_name}.csv"
    with open(name_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "loss",
                "acc",
                "f1_macro",
                "f1_weighted",
                "GAcc",
                "acc_C1",
                "acc_C2",
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
                    row["f1_macro"],
                    row["f1_weighted"],
                    row["GAcc"],
                    row["acc_C1"],
                    row["acc_C2"],
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
def metrics(g1, g2, y, c1, pc1, c2, pc2, p):
    pred_tuples = list(zip(c1, c2, p))
    gt_tuples = list(zip(g1, g2, y))
    cont = 0
    cont_gt = 0
    sum_ars = 0
    sum_gt = 0
    sum_model = 0
    for i, (pred, gt) in enumerate(zip(pred_tuples, gt_tuples)):
        if pred != gt:
            if pred[2] == gt[2]:
                peso = cy.get(pred[2], 0)
                sum_ars += math.log(1 / peso)
                p_c1 = pc1[i]
                p_c2 = pc2[i]
                sum_model += (1 - (p_c1 * p_c2)) * math.log(1 / peso)
                cont += 1
        else:
            peso = cy.get(pred[2], 0)
            sum_gt += math.log(1 / peso)
            p_c1 = pc1[i]
            p_c2 = pc2[i]
            sum_model += (1 - (p_c1 * p_c2)) * math.log(1 / peso)
            cont_gt += 1
            # print(f"Correcto ----> {i}: pred={pred}, gt={gt}")

    print(f"\tTotal de valores errados: {cont}")
    print(f"\tTotal de valores verdaderos: {cont_gt}")
    print(f"\tTotal de valores acertados: {cont + cont_gt}")
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


def con_loss(output, target, gt1, gt2, pred1, pred2):
    loss = torch.tensor(0.0, device=output.device)
    loss_concep = torch.tensor(0.0, device=output.device)
    entropy = torch.tensor(0.0, device=output.device)

    lambda_label = torch.tensor(1.0, device=output.device)
    lambda_concept = torch.tensor(0.1, device=output.device)
    lambda_entropy = torch.tensor(0.001, device=output.device)

    loss_label = bce_loss(output, target)

    loss1 = -(gt1 * torch.log(pred1 + 1e-8)).sum(dim=1).mean()
    loss2 = -(gt2 * torch.log(pred2 + 1e-8)).sum(dim=1).mean()

    entropy1 = -(pred1 * torch.log(pred1 + 1e-8)).sum(dim=1).mean()
    entropy2 = -(pred2 * torch.log(pred2 + 1e-8)).sum(dim=1).mean()

    loss_concep = loss1 + loss2
    entropy = entropy1 + entropy2

    loss = (
        lambda_concept * loss_concep
        + lambda_entropy * entropy
        + lambda_label * loss_label
    )
    # return loss
    return loss_label


# ==============================================
# Confusion Matrix
# ==============================================


def conf_matrix(file_path, file_name, data):
    last_record = data[-1]
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
    plt.savefig(f"{file_path}/{FILE_RESULT_MATRIX}_{file_name}.png", dpi=300)
    plt.close()


# ==============================================
# Entrenamiento y Test
# ==============================================
class Trainer:
    def __init__(
        self,
        result_dir,
        train_loader,
        validation_loader,
        test_loader,
        learning_rate,
        loss,
        k,
        provenance,
    ):
        self.network = MNISTFashionLogic(provenance, k).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.result_dir = result_dir
        self.metrics_train = []
        self.metrics_test = []

        if loss == "nll":
            self.loss = nll_loss
        elif loss == "bce":
            self.loss = bce_loss
        elif loss == "ll":
            self.loss = con_loss
        else:
            raise Exception(f"Unknown loss function `{loss}`")

    def train_epoch(self, epoch):
        self.network.train()
        correct = 0
        iter = tqdm(self.train_loader, total=len(self.train_loader))
        num_items = len(self.train_loader.dataset)
        c1, c2 = [], []
        pc1, pc2 = [], []
        g1, g2 = [], []
        y, p, pb = [], [], []
        for data, target, (gt_syntax, gt_mistake), (p_syntax, p_mistake) in iter:
            p_syntax = p_syntax.to(device)
            p_mistake = p_mistake.to(device)
            self.optimizer.zero_grad()
            a_distrs, b_distrs, output = self.network(data)
            output = output.cpu()
            g1.extend(gt_syntax.tolist())
            g2.extend(gt_mistake.tolist())
            t_pc1, t_c1 = a_distrs.max(dim=1)
            t_pc2, t_c2 = b_distrs.max(dim=1)
            t_pb, t_p = output.max(dim=1)
            c1.extend(t_c1.tolist())
            c2.extend(t_c2.tolist())
            pc1.extend(t_pc1.tolist())
            pc2.extend(t_pc2.tolist())
            p.extend(t_p.tolist())
            pb.extend(t_pb.tolist())
            y.extend(target.tolist())
            loss = self.loss(output, target, p_syntax, p_mistake, a_distrs, b_distrs)
            pred = t_p
            correct += pred.eq(target.view_as(pred)).sum().item()
            perc = 100.0 * correct / num_items
            loss.backward()
            self.optimizer.step()
            iter.set_description(
                f"[Train Epoch {epoch}] Loss: {loss.item():.4f} Accuracy: {correct}/{num_items} ({perc:.2f}%)"
            )
        gt, rs, rsr, rsrw, prob_model, prob_mod_no = metrics(
            g1, g2, y, c1, pc1, c2, pc2, p
        )
        correct_concepts = sum(
            (a == b) and (c == d) for a, b, c, d in zip(g1, c1, g2, c2)
        )
        gacc = 100.0 * correct_concepts / len(g1)

        f1_macro = f1_score(y, p, average="macro")
        f1_weighted = f1_score(y, p, average="weighted")

        acc_c1 = 100.0 * sum(a == b for a, b in zip(g1, c1)) / len(g1)
        acc_c2 = 100.0 * sum(a == b for a, b in zip(g2, c2)) / len(g2)

        self.metrics_train.append(
            {
                "epoch": epoch,
                "loss": loss.item(),
                "acc": perc,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "GAcc": gacc,
                "acc_C1": acc_c1,
                "acc_C2": acc_c2,
                "gt": gt,
                "rs": rs,
                "RSR": rsr,
                "RSRw": rsrw,
                "prob_model": prob_model,
                "prob_mod_no": prob_mod_no,
                "ground_truth": y,
                "output": p,
            }
        )

    def test(self, epoch):
        self.network.eval()
        num_items = len(self.test_loader.dataset)
        test_loss = 0
        correct = 0
        c1, c2 = [], []
        pc1, pc2 = [], []
        g1, g2 = [], []
        y, p, pb = [], [], []
        with torch.no_grad():
            iter = tqdm(self.test_loader, total=len(self.test_loader))
            for data, target, (gt_syntax, gt_mistake), (p_syntax, p_mistake) in iter:
                p_syntax = p_syntax.to(device)
                p_mistake = p_mistake.to(device)
                a_distrs, b_distrs, output = self.network(data)
                output = output.cpu()
                g1.extend(gt_syntax.tolist())
                g2.extend(gt_mistake.tolist())
                t_pc1, t_c1 = a_distrs.max(dim=1)
                t_pc2, t_c2 = b_distrs.max(dim=1)
                t_pb, t_p = output.max(dim=1)
                c1.extend(t_c1.tolist())
                c2.extend(t_c2.tolist())
                pc1.extend(t_pc1.tolist())
                pc2.extend(t_pc2.tolist())
                p.extend(t_p.tolist())
                pb.extend(t_pb.tolist())
                y.extend(target.tolist())
                test_loss = self.loss(
                    output, target, p_syntax, p_mistake, a_distrs, b_distrs
                )
                pred = t_p
                correct += pred.eq(target.view_as(pred)).sum().item()
                perc = 100.0 * correct / num_items
                iter.set_description(
                    f"[Test Epoch {epoch}] Total loss: {test_loss.item():.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)"
                )
            gt, rs, rsr, rsrw, prob_model, prob_mod_no = metrics(
                g1, g2, y, c1, pc1, c2, pc2, p
            )
            correct_concepts = sum(
                (a == b) and (c == d) for a, b, c, d in zip(g1, c1, g2, c2)
            )
            gacc = 100.0 * correct_concepts / len(g1)

            f1_macro = f1_score(y, p, average="macro")

            f1_weighted = f1_score(y, p, average="weighted")

            acc_c1 = 100.0 * sum(a == b for a, b in zip(g1, c1)) / len(g1)
            acc_c2 = 100.0 * sum(a == b for a, b in zip(g2, c2)) / len(g2)
            self.metrics_test.append(
                {
                    "epoch": epoch,
                    "loss": test_loss.item(),
                    "acc": perc,
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted,
                    "GAcc": gacc,
                    "acc_C1": acc_c1,
                    "acc_C2": acc_c2,
                    "gt": gt,
                    "rs": rs,
                    "RSR": rsr,
                    "RSRw": rsrw,
                    "prob_model": prob_model,
                    "prob_mod_no": prob_mod_no,
                    "ground_truth": y,
                    "output": p,
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
        conf_matrix(self.result_dir, "train", self.metrics_train)
        conf_matrix(self.result_dir, "test", self.metrics_test)


# ==============================================
# MAIN
# ==============================================
if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("mnist_fashion")
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--batch-size-train", type=int, default=1)
    parser.add_argument("--batch-size-test", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.000001)
    parser.add_argument("--loss-fn", type=str, default="ll")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--provenance", type=str, default="diffaddmultprob")
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
    data_dir = os.path.abspath(os.path.join(base_dir, "../..", DATA_MNIST_FASHION_PATH))
    result_dir = os.path.join(base_dir, DATA_RESULT_PATH)
    print("PATH data -> ", data_dir)
    # train_loader, validation_loader, test_loader = mnist_fashion_loader(
    #     result_dir, batch_size_train, batch_size_test
    # )
    # Create trainer and train
    # trainer = Trainer(
    #     result_dir,
    #     train_loader,
    #     validation_loader,
    #     test_loader,
    #     learning_rate,
    #     loss_fn,
    #     k,
    #     provenance,
    # )
    # trainer.train(n_epochs)
    main_graph("train", DATA_RESULT_PATH)
    # main_graph("test", DATA_RESULT_PATH)
    # main_distribution(train_loader, test_loader)
