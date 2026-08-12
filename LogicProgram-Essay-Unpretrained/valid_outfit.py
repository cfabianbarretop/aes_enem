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
import pickle
import open_clip

from typing import *
from argparse import ArgumentParser
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

# ==============================================
# CONFIG
# ==============================================
DATA_MNIST_FASHION_PATH = "data"  # Original dataset path
DATA_RESULT_PATH = "result"  # Result data path
FILE_RESULT_METRIC = "result_metric"  # Name file result metrics
FILE_RESULT_MATRIX = "result_matrix"  # Name file result matrix
TOKENIZER_NAME = f"neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
cy = {0: 988, 1: 12}
print("Device: ", device)


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
        # Each data has two images and the GT is the sum of two digits
        return (tokenized_text, label)  # (a_img, b_img, a_digit + b_digit)

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([item[0]["input_ids"][0] for item in batch])
        token_type = torch.stack([item[0]["token_type_ids"][0] for item in batch])
        attention_mask = torch.stack([item[0]["attention_mask"][0] for item in batch])
        digits = torch.stack([torch.tensor(item[1]).long() for item in batch])
        return ((input_ids, token_type, attention_mask), digits)


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

        # Scallop Context
        self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
        self.scl_ctx.add_relation("digit_1", int, input_mapping=list(range(5)))
        self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(4)))
        # self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(10)))
        self.scl_ctx.add_rule("sum_2(0) :- digit_1(0)")
        self.scl_ctx.add_rule("sum_2(1) :- digit_1(1), digit_2(0)")
        # soma2
        self.scl_ctx.add_rule("sum_2(2) :- digit_1(1), digit_2(b), b>=1")
        self.scl_ctx.add_rule("sum_2(2) :- digit_1(a), digit_2(0), a>=2")
        # self.scl_ctx.add_rule("sum_2(2) :- digit_1(a), digit_2(0), a>=2")
        # soma3
        self.scl_ctx.add_rule("sum_2(3) :- digit_1(2), digit_2(b), b>=1")
        self.scl_ctx.add_rule("sum_2(3) :- digit_1(a), digit_2(1), a>=3")
        # self.scl_ctx.add_rule("sum_2(3) :- digit_1(a), digit_2(1), a>=2")
        # soma4
        self.scl_ctx.add_rule("sum_2(4) :- digit_1(3), digit_2(b), b>=2")
        self.scl_ctx.add_rule("sum_2(4) :- digit_1(a), digit_2(2), a>=4")
        # self.scl_ctx.add_rule("sum_2(4) :- digit_1(a), digit_2(2), a>=3")
        # soma5
        self.scl_ctx.add_rule("sum_2(5) :- digit_1(4), digit_2(3)")
        # The `sum_2` logical reasoning module
        self.sum_2 = self.scl_ctx.forward_function(
            "sum_2", output_mapping=[(i,) for i in range(6)]
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        texto = x
        # First recognize the two digits
        resposta_a, resposta_b = self.mnist_net(texto)  # Tensor 64 x 10

        # b_distrs = self.mnist_net(b_imgs) # Tensor 64 x 10

        # Then execute the reasoning module; the result is a size 19 tensor
        return (
            resposta_a,
            resposta_b,
            self.sum_2(digit_1=resposta_a, digit_2=resposta_b),
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
                "GAcc",
                "acc_C1",
                "acc_C2",
                "acc_C3",
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
                    row["acc_C1"],
                    row["acc_C2"],
                    row["acc_C3"],
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
            # if gt[3] == 1:
            # print(f"Correcto ----> {i}: pred={pred}, gt={gt}")

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
    batch_size = output.shape[0]
    loss = torch.tensor(0.0, device=output.device)

    for b in range(batch_size):
        y = int(ground_truth[b].item())

        # Evitar log(0)
        p = output[b].clamp(min=1e-8, max=1 - 1e-8)

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
        self, result_dir, train_loader, validation_loader, test_loader, learning_rate, loss, k, provenance
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
        for (data, target) in iter:
            self.optimizer.zero_grad()
            a_distrs, b_distrs, output = self.network(data)
            output = output.cpu()
            # g1.extend(a_digit.tolist())
            # g2.extend(b_digit.tolist())
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
            loss = self.loss(output, target)
            pred = t_p
            correct += pred.eq(target.view_as(pred)).sum().item()
            perc = 100.0 * correct / num_items
            loss.backward()
            self.optimizer.step()
            iter.set_description(
                f"[Train Epoch {epoch}] Loss: {loss.item():.4f} Accuracy: {correct}/{num_items} ({perc:.2f}%)"
            )
        # gt, rs, rsr, rsrw, prob_model, prob_mod_no = metrics(
        #     g1, g2, g3, y, c1, pc1, c2, pc2, c3, pc3, p
        # )
        # correct_concepts = sum(
        #     (a == b) and (c == d) and (e == f)
        #     for a, b, c, d, e, f in zip(g1, c1, g2, c2, g3, c3)
        # )
        # gacc = 100.0 * correct_concepts / len(g1)

        # acc_c1 = 100.0 * sum(a == b for a, b in zip(g1, c1)) / len(g1)
        # acc_c2 = 100.0 * sum(a == b for a, b in zip(g2, c2)) / len(g2)
        # acc_c3 = 100.0 * sum(a == b for a, b in zip(g3, c3)) / len(g3)
        # self.metrics_train.append(
        #     {
        #         "epoch": epoch,
        #         "loss": loss.item(),
        #         "acc": perc,
        #         "GAcc": gacc,
        #         "acc_C1": acc_c1,
        #         "acc_C2": acc_c2,
        #         "acc_C3": acc_c3,
        #         "gt": gt,
        #         "rs": rs,
        #         "RSR": rsr,
        #         "RSRw": rsrw,
        #         "prob_model": prob_model,
        #         "prob_mod_no": prob_mod_no,
        #         "ground_truth": y,
        #         "output": p,
        #     }
        # )

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
            for (data, target) in iter:
                a_distrs, b_distrs, output = self.network(data)
                output = output.cpu()
                # g1.extend(a_digit.tolist())
                # g2.extend(b_digit.tolist())
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
                test_loss = self.loss(output, target)
                pred = t_p
                correct += pred.eq(target.view_as(pred)).sum().item()
                perc = 100.0 * correct / num_items
                iter.set_description(
                    f"[Test Epoch {epoch}] Total loss: {test_loss.item():.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)"
                )
            # gt, rs, rsr, rsrw, prob_model, prob_mod_no = metrics(
            #     g1, g2, g3, y, c1, pc1, c2, pc2, c3, pc3, p
            # )
            # correct_concepts = sum(
            #     (a == b) and (c == d) and (e == f)
            #     for a, b, c, d, e, f in zip(g1, c1, g2, c2, g3, c3)
            # )
            # gacc = 100.0 * correct_concepts / len(g1)

            # acc_c1 = 100.0 * sum(a == b for a, b in zip(g1, c1)) / len(g1)
            # acc_c2 = 100.0 * sum(a == b for a, b in zip(g2, c2)) / len(g2)
            # acc_c3 = 100.0 * sum(a == b for a, b in zip(g3, c3)) / len(g3)
            # self.metrics_test.append(
            #     {
            #         "epoch": epoch,
            #         "loss": test_loss.item(),
            #         "acc": perc,
            #         "GAcc": gacc,
            #         "acc_C1": acc_c1,
            #         "acc_C2": acc_c2,
            #         "acc_C3": acc_c3,
            #         "gt": gt,
            #         "rs": rs,
            #         "RSR": rsr,
            #         "RSRw": rsrw,
            #         "prob_model": prob_model,
            #         "prob_mod_no": prob_mod_no,
            #         "ground_truth": y,
            #         "output": p,
            #     }
            # )

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
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size-train", type=int, default=1)
    parser.add_argument("--batch-size-test", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.000001)
    parser.add_argument("--loss-fn", type=str, default="bce")
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
    train_loader, validation_loader, test_loader = mnist_fashion_loader(
        data_dir, batch_size_train, batch_size_test
    )
    # Create trainer and train
    trainer = Trainer(
        result_dir, train_loader, validation_loader, test_loader, learning_rate, loss_fn, k, provenance
    )
    trainer.train(n_epochs)
    # main_graph("train", DATA_RESULT_PATH)
    # main_graph("test", DATA_RESULT_PATH)
    # main_distribution(train_loader, test_loader)
