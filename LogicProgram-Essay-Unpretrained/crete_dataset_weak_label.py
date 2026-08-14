import json
import os
from pathlib import Path
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================

MODEL = "gpt-5.5"
DATASET_NAME = "igorcs/LLM-JBCS"
OUTPUT_FILE_NAME = "weak_labels_LLM-JBCS.json"
API_KEY_SECRET = "OPEN_IA"  # Api key
API_KEY_SECRET_DIR = "api_key.json"  # Api key data path
PROMPT_PATH = "prompt/competencia1.txt"
DATA_RESULT_PATH = "result"

base_dir = os.path.dirname(os.path.abspath(__file__))
prompt_dir = os.path.join(base_dir, PROMPT_PATH)
result_dir = os.path.join(base_dir, DATA_RESULT_PATH)

# ============================================================
# 2. PROMPT
# ============================================================

SYSTEM_PROMPT = Path(prompt_dir).read_text(encoding="utf-8")

# ============================================================
# 3. JSON SCHEMA
# ============================================================

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "estrutura_sintatica": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 4},
                "evidencias": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "trecho": {"type": "string"},
                            "tipo": {"type": "string"},
                            "explicacao": {"type": "string"},
                        },
                        "required": ["trecho", "tipo", "explicacao"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["score", "evidencias"],
            "additionalProperties": False,
        },
        "desvios": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 3},
                "evidencias": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "trecho": {"type": "string"},
                            "tipo": {"type": "string"},
                            "explicacao": {"type": "string"},
                        },
                        "required": ["trecho", "tipo", "explicacao"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["score", "evidencias"],
            "additionalProperties": False,
        },
    },
    "required": ["estrutura_sintatica", "desvios"],
    "additionalProperties": False,
}


# ============================================================
# 4. FUNÇÃO PARA OBTENER A KEY API OPEN AI GPT
# ============================================================
def getApiKey(name_api_key):
    with open(API_KEY_SECRET_DIR, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config[name_api_key]


client = OpenAI(api_key=getApiKey(API_KEY_SECRET))

# ============================================================
# 5. FUNÇÃO PARA GERAR UMA WEAK LABEL
# ============================================================


def generate_weak_label(essay):
    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analise a seguinte redação.\n\n"
                    "REDAÇÃO:\n"
                    "--------------------\n"
                    f"{essay}\n"
                    "--------------------"
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "weak_concept_labels",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
    )

    return json.loads(response.output_text)


# ============================================================
# 6. CARREGAR DATASET
# ============================================================

dataset = load_dataset(DATASET_NAME, cache_dir="tmp/aes_enem", trust_remote_code=True)[
    "train"
]

# ============================================================
#  7. SALVAR OS RESULTADOS
# ============================================================


def load_results(output_file):
    """
    Carrega resultados existentes.
    Se o arquivo não existir, retorna lista vazia.
    """

    if not os.path.exists(output_file):
        return []

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)

    except json.JSONDecodeError:
        print("O arquivo JSON está corrompido.")
        print("Iniciando com resultados vazios.")
        return []


def save_results(results, output_file):
    """
    Salva os resultados imediatamente.
    """

    # cria o diretório caso não exista
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # arquivo temporário
    temp_file = output_file + ".tmp"

    with open(temp_file, "w", encoding="utf-8") as f:

        json.dump(results, f, ensure_ascii=False, indent=4)

        # garante que os dados sejam escritos
        f.flush()
        os.fsync(f.fileno())

    # substitui o arquivo antigo somente depois
    # que o novo foi completamente escrito
    os.replace(temp_file, output_file)


# ============================================================
# 8. TRABALHA COMO ESSAY
# ============================================================
def one_hot(score, num_classes):
    vector = [0] * num_classes
    vector[score] = 1
    return vector


def generate_one_hot(result):
    return {
        "estrutura_sintatica": one_hot(result["estrutura_sintatica"]["score"], 5),
        "desvios": one_hot(result["desvios"]["score"], 4),
    }


# results = []

OUTPUT_FILE = os.path.join(result_dir, OUTPUT_FILE_NAME)

results = load_results(OUTPUT_FILE)

print(f"Resultados já salvos: {len(results)}")

processed_ids = {item["id"] for item in results}

print(f"IDs já processados: {len(processed_ids)}")

iter = tqdm(dataset, total=len(dataset))
for i, row in enumerate(iter):
    id_essay = f"{row['id']}-{row['id_prompt']}"

    # ============================================
    # Já processado?
    # ============================================
    if id_essay in processed_ids:
        continue
    essay = row["essay_text"]

    try:
        weak_label = generate_weak_label(essay)

        result = {
            "id": id_essay,
            "weak_label": weak_label,
            "one_hot": generate_one_hot(weak_label),
        }
        results.append(result)

        # adiciona ao conjunto
        processed_ids.add(id_essay)

        save_results(results, OUTPUT_FILE)

    except Exception as e:

        print(f"Erro no ensaio {id_essay}: {e}")
        continue

    iter.set_description(f"[Weak label: Total {i+1}/{len(dataset)}]")
