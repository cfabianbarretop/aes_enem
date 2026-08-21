import os
import json
import pandas as pd
from collections import Counter
from datasets import load_dataset

# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================
DATA_RESULT_PATH = "result"
INPUT_FILE_NAME = "weak_labels_QWEN.json"
OUTPUT_FILE_NAME = "erros_concepts_dataset_qwen.csv"

base_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(base_dir, DATA_RESULT_PATH)

INPUT_FILE = f"{result_dir}/{INPUT_FILE_NAME}"
OUTPUT_FILE = f"{result_dir}/{OUTPUT_FILE_NAME}"


# ============================================================
# 2. DADOS
# ============================================================
dataset = load_dataset(
    "igorcs/LLM-JBCS", cache_dir="tmp/aes_enem", trust_remote_code=True
)["train"]

df = dataset.to_pandas()

print(df[["sintaxe", "desvios", "label"]].head())


#     return None
def aplicar_reglas(syntax, mistake):
    resultados = []

    # soma 0
    if syntax == 0:
        resultados.append(0)

    # soma 1
    if syntax == 1 and mistake == 0:
        resultados.append(1)

    # soma 2
    if syntax == 1 and mistake >= 1:
        resultados.append(2)

    if syntax >= 2 and mistake == 0:
        resultados.append(2)

    # soma 3
    if syntax == 2 and mistake >= 1:
        resultados.append(3)

    if syntax >= 3 and mistake == 1:
        resultados.append(3)

    # soma 4
    if syntax == 3 and mistake >= 2:
        resultados.append(4)

    if syntax >= 4 and mistake == 2:
        resultados.append(4)

    # soma 5
    if syntax == 4 and mistake == 3:
        resultados.append(5)

    return resultados


def get_ground_thun(id_dataset, tipo):
    gt_syntax, gt_mistake = 0, 0
    valor_numerico = -1
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        weak_labels = json.load(f)
    for item in weak_labels:
        if item["id"] == id_dataset:
            weak_label = item["weak_label"]
            gt_syntax = weak_label["estrutura_sintatica"]
            gt_mistake = weak_label["desvios"]
    if tipo == "sintaxe":
        valor_numerico = gt_syntax
    elif tipo == "desvios":
        valor_numerico = gt_mistake
    return valor_numerico

df["nota"] = df.apply(
    lambda row: (row["label"] // 40),
    axis=1,
)

df["syntax"] = df.apply(
    lambda row: get_ground_thun(f"{row['id']}-{row['id_prompt']}", "sintaxe"),
    axis=1,
)

df["mistakes"] = df.apply(
    lambda row: get_ground_thun(f"{row['id']}-{row['id_prompt']}", "desvios"),
    axis=1,
)

df["nota_regra"] = df.apply(
    lambda row: aplicar_reglas(row["syntax"], row["mistakes"]),
    axis=1,
)

df["coincide"] = df.apply(lambda row: row["nota"] in row["nota_regra"], axis=1)

print(df["coincide"].value_counts())
accuracy = df["coincide"].mean()
print(f"Coincidência: {accuracy:.2%}")

erros = df[df["coincide"] == False]

# print(
# erros[
#     ["id", "id_prompt", "reference", "syntax", "mistakes", "nota", "nota_regra"]
# ]
# )

erros_csv = erros[
    ["id", "id_prompt", "reference", "syntax", "mistakes", "nota", "nota_regra"]
]

erros_csv.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)
