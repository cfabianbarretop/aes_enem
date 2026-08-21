from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
import json
import os
from pathlib import Path
from tqdm import tqdm

# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================
DATASET_NAME = "igorcs/LLM-JBCS"
DATA_RESULT_PATH = "result"
PROMPT_PATH = "prompt/competencia1.txt"
base_dir = os.path.dirname(os.path.abspath(__file__))
prompt_dir = os.path.join(base_dir, PROMPT_PATH)
result_dir = os.path.join(base_dir, DATA_RESULT_PATH)
OUTPUT_FILE_NAME = "weak_labels_QWEN.json"

print("CUDA disponible:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")

# ============================================================
# 2. PROMPT
# ============================================================

SYSTEM_PROMPT = Path(prompt_dir).read_text(encoding="utf-8")


# ============================================================
# 2. MODELO
# ============================================================
model_name = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

print("Modelo cargado correctamente")
print("Device map: ", model.hf_device_map)

# ============================================================
# 5. FUNÇÃO PARA GERAR UMA WEAK LABEL
# ============================================================
def generate_weak_label(essay):

    user_prompt = (
        "Analise a seguinte redação.\n\n"
        "REDAÇÃO:\n"
        "--------------------\n"
        f"{essay}\n"
        "--------------------"
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    # Formata a conversa no formato esperado pelo Qwen3
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=False
        )

    # Pegamos somente o que o modelo gerou
    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    response_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


    print("RESPOSTA DO MODELO:")
    print(response_text)

    return response_text


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
        }
        results.append(result)

        # adiciona ao conjunto
        processed_ids.add(id_essay)

        save_results(results, OUTPUT_FILE)

    except Exception as e:

        print(f"Erro no ensaio {id_essay}: {e}")
        continue

    iter.set_description(f"[Weak label: Total {i+1}/{len(dataset)}]")