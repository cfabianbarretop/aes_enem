import json
import os
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    cohen_kappa_score
)

# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================
DATA_RESULT_PATH = "result"
OUTPUT_FILE_NAME = "weak_labels_C1A.json"
GRAPH_SYNTAX_NAME = "syntax_matrix.png"
GRAPH_MISTAKE_NAME = "mistake_matrix.png"

base_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(base_dir, DATA_RESULT_PATH)


OUTPUT_FILE = f"{result_dir}/{OUTPUT_FILE_NAME}"

# ============================================
# Cargar dataset original
# ============================================

dataset = load_dataset(
    "igorcs/C1-A",
    cache_dir="tmp/aes_enem",
    trust_remote_code=True
)["train"]


# ============================================
# Cargar weak labels
# ============================================

with open(
    OUTPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    weak_labels = json.load(f)

syntax_true = []
syntax_pred = []

mistakes_true = []
mistakes_pred = []


for row, weak in zip(dataset, weak_labels):

    # --------------------------
    # Ground truth
    # --------------------------

    syntax_true.append(row["syntax"])
    mistakes_true.append(row["mistakes"])


    # --------------------------
    # Weak labels del LLM
    # --------------------------

    syntax_pred.append(
        weak["weak_label"]["estrutura_sintatica"]["score"]
    )

    mistakes_pred.append(
        weak["weak_label"]["desvios"]["score"]
    )

syntax_accuracy = accuracy_score(
    syntax_true,
    syntax_pred
)

mistakes_accuracy = accuracy_score(
    mistakes_true,
    mistakes_pred
)

print(
    f"Syntax accuracy: {100*syntax_accuracy:.2f}%"
)

print(
    f"Mistakes accuracy: {100*mistakes_accuracy:.2f}%"
)

syntax_mae = mean_absolute_error(
    syntax_true,
    syntax_pred
)

mistakes_mae = mean_absolute_error(
    mistakes_true,
    mistakes_pred
)

print(
    f"Syntax MAE: {syntax_mae:.4f}"
)

print(
    f"Mistakes MAE: {mistakes_mae:.4f}"
)

syntax_qwk = cohen_kappa_score(
    syntax_true,
    syntax_pred,
    weights="quadratic"
)

mistakes_qwk = cohen_kappa_score(
    mistakes_true,
    mistakes_pred,
    weights="quadratic"
)

print(
    f"Syntax QWK: {syntax_qwk:.4f}"
)

print(
    f"Mistakes QWK: {mistakes_qwk:.4f}"
)

cm_syntax = confusion_matrix(
    syntax_true,
    syntax_pred,
    labels=[0, 1, 2, 3, 4, 5]
)

plt.figure(figsize=(7, 6))

plt.imshow(cm_syntax)

plt.title("Estrutura Sintática")
plt.xlabel("Weak Label — LLM")
plt.ylabel("Label original — INEP")

plt.xticks(range(6), range(6))
plt.yticks(range(6), range(6))

plt.colorbar(label="Quantidade")

for i in range(6):
    for j in range(6):
        plt.text(
            j,
            i,
            cm_syntax[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()

cm_mistakes = confusion_matrix(
    mistakes_true,
    mistakes_pred,
    labels=[0, 1, 2, 3, 4]
)

plt.figure(figsize=(7, 6))

plt.imshow(cm_mistakes)

plt.title("Desvios")
plt.xlabel("Weak Label — LLM")
plt.ylabel("Label original — INEP")

plt.xticks(range(5), range(5))
plt.yticks(range(5), range(5))

plt.colorbar(label="Quantidade")

for i in range(5):
    for j in range(5):
        plt.text(
            j,
            i,
            cm_mistakes[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 6))

plt.scatter(
    syntax_true,
    syntax_pred,
    alpha=0.7
)

plt.plot(
    [0, 5],
    [0, 5],
    linestyle="--"
)

plt.xlabel("Label original — syntax")
plt.ylabel("Weak Label — LLM")
plt.title("Syntax: INEP vs LLM")

plt.xticks(range(6))
plt.yticks(range(6))

plt.grid(True)

plt.tight_layout()
plt.show()