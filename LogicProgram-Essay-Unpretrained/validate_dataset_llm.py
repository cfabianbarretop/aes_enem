import os
import pandas as pd
from collections import Counter
from datasets import load_dataset

# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================
DATA_RESULT_PATH = "result"
OUTPUT_FILE_NAME = "erros_concepts_dataset_llm.csv"

base_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(base_dir, DATA_RESULT_PATH)


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


map_sintaxe = {"ruim": 0, "deficitária": 1, "regular": 2, "boa": 3, "excelente": 4}

map_desvios = {
    "muitos": 0,
    "alguns": 1,
    "poucos": 2,
    "menos de dois": 3,
}


def get_valor_numerico(valor, tipo):
    valor_numerico = -1
    valor_mas_repetido = Counter(valor).most_common(1)[0][0]
    if tipo == "sintaxe":
        valor_numerico = map_sintaxe[valor_mas_repetido]
    elif tipo == "desvios":
        valor_numerico = map_desvios[valor_mas_repetido]
    return valor_numerico


df["nota"] = df.apply(
    lambda row: (row["label"] // 40),
    axis=1,
)

df["syntax"] = df.apply(
    lambda row: get_valor_numerico(row["sintaxe"], "sintaxe"),
    axis=1,
)

df["mistakes"] = df.apply(
    lambda row: get_valor_numerico(row["desvios"], "desvios"),
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
