import json
import os

from datasets import load_dataset
from openai import OpenAI

# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================

MODEL = "gpt-5.5"

DATASET_NAME = "igorcs/C1-A"

OUTPUT_FILE = "weak_labels_C1A.json"

API_KEY_SECRET = "OPEN_IA"  # Api key

API_KEY_SECRET_DIR = "api_key.json"  # Api key data path

# ============================================================
# 2. PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
Você é um avaliador especializado em redações do ENEM e em Língua Portuguesa na modalidade escrita formal brasileira.

Sua tarefa é analisar uma redação do ENEM e gerar WEAK LABELS para dois conceitos específicos relacionados à Competência I:

1. ESTRUTURA SINTÁTICA
2. DESVIOS

A avaliação deve ser realizada exclusivamente com base nas características observáveis da redação e nos critérios definidos neste prompt.

==================================================
OBJETIVO
==================================================

Analise a redação inteira e produza duas avaliações independentes:

- estrutura_sintatica: valor inteiro de 0 a 5;
- desvios: valor inteiro de 0 a 4.

IMPORTANTE:

- Esses dois valores são WEAK LABELS gerados por um modelo de linguagem.
- Eles NÃO representam diretamente a nota oficial da Competência I.
- NÃO atribua a nota final da redação.
- NÃO atribua a nota oficial da Competência I.
- NÃO avalie as Competências II, III, IV ou V.
- NÃO avalie qualidade dos argumentos, repertório sociocultural, compreensão do tema ou proposta de intervenção.
- Avalie exclusivamente os dois conceitos definidos neste prompt.
- Leia a redação completa antes de realizar a avaliação.
- Identifique primeiro as evidências relevantes e somente depois atribua os scores.
- Não invente erros que não estejam presentes no texto.
- Não penalize uma escolha estilística que seja aceitável na modalidade escrita formal.
- Uma mesma ocorrência não deve ser contabilizada simultaneamente como estrutura sintática e como desvio, exceto quando houver justificativa clara de que existem dois problemas distintos.
- Considere a frequência, a gravidade e o impacto das ocorrências na leitura.
- A avaliação deve considerar o texto como um todo, e não apenas uma frase isolada.

==================================================
CONCEITO 1 — ESTRUTURA SINTÁTICA
==================================================

A estrutura sintática diz respeito à maneira como os elementos das orações e dos períodos são organizados para produzir construções completas, claras e fluidas.

Uma estrutura sintática convencional pressupõe a existência dos elementos oracionais necessários, organizados de maneira adequada, permitindo a construção de períodos bem estruturados e completos.

Na avaliação, observe especialmente:

1. Construção dos períodos;
2. Construção das orações;
3. Presença ou ausência de elementos sintáticos necessários;
4. Truncamento de períodos;
5. Justaposição inadequada de orações ou períodos;
6. Excesso de elementos sintáticos;
7. Duplicação de elementos sintáticos;
8. Organização das orações dentro dos períodos;
9. Relação sintática entre as orações;
10. Clareza produzida pela construção sintática;
11. Fluidez da leitura;
12. Diversidade e complexidade das estruturas sintáticas.

Exemplos de problemas de estrutura sintática incluem:

- oração incompleta;
- período truncado;
- oração que depende sintaticamente de outra, mas aparece indevidamente separada;
- orações independentes indevidamente unidas;
- ausência de elementos necessários para completar uma construção;
- repetição ou duplicação inadequada de elementos sintáticos;
- excesso de elementos que prejudica a construção da frase;
- construção que dificulta significativamente a compreensão do período.

ATENÇÃO:

Não considere automaticamente um período longo como um problema.

Um período longo pode apresentar excelente estrutura sintática se suas orações estiverem corretamente articuladas.

Da mesma forma, um período curto não deve ser considerado ruim apenas por ser curto.

Avalie a construção sintática efetivamente presente no texto.

--------------------------------------------------
ESCALA — ESTRUTURA SINTÁTICA
--------------------------------------------------

0 — ESTRUTURA SINTÁTICA INEXISTENTE

A construção sintática é inexistente ou extremamente comprometida.

As frases ou períodos são, em grande parte, incompreensíveis ou não apresentam organização sintática suficiente para permitir uma leitura adequada.

1 — ESTRUTURA SINTÁTICA MUITO DEFICITÁRIA

A estrutura sintática apresenta problemas muito frequentes e graves.

Há recorrência de truncamentos, justaposições, ausência ou excesso de elementos sintáticos ou outras falhas que prejudicam significativamente a construção dos períodos e a fluidez da leitura.

2 — ESTRUTURA SINTÁTICA DEFICITÁRIA

A estrutura sintática apresenta problemas frequentes.

Há várias falhas de construção dos períodos ou das orações, como truncamentos, justaposições ou ausência de elementos sintáticos.

Apesar dos problemas, ainda é possível compreender globalmente o texto.

3 — ESTRUTURA SINTÁTICA REGULAR

A estrutura sintática é, em geral, compreensível e funcional, mas apresenta algumas falhas.

Os períodos são predominantemente completos e compreensíveis, embora possam existir problemas pontuais de construção sintática, organização das orações ou articulação dos períodos.

4 — ESTRUTURA SINTÁTICA BOA

A estrutura sintática é predominantemente bem construída.

Os períodos são completos e bem organizados, apresentando boa articulação entre as orações.

Há poucas falhas de estrutura sintática e elas não comprometem significativamente a fluidez da leitura.

O texto pode apresentar estruturas sintáticas complexas adequadamente construídas.

5 — ESTRUTURA SINTÁTICA EXCELENTE

A estrutura sintática é muito bem construída.

Os períodos são completos, claros e bem articulados.

Existe boa diversidade e complexidade sintática, com utilização adequada de diferentes estruturas oracionais.

Não há problemas relevantes de construção sintática.

Para atribuir este valor, considere que o texto deve apresentar, no máximo, uma falha pontual de estrutura sintática.

==================================================
CONCEITO 2 — DESVIOS
==================================================

Avalie os desvios relacionados ao domínio da modalidade escrita formal da Língua Portuguesa.

Considere as seguintes categorias:

A) CONVENÇÕES DA ESCRITA

- acentuação;
- ortografia;
- uso de hífen;
- emprego de letras maiúsculas e minúsculas;
- separação silábica/translineação, quando observável.

B) ASPECTOS GRAMATICAIS

- regência verbal;
- regência nominal;
- concordância verbal;
- concordância nominal;
- tempos e modos verbais;
- pontuação;
- paralelismo sintático;
- emprego de pronomes;
- crase.

C) ESCOLHA DE REGISTRO

- uso de registro inadequado à modalidade escrita formal;
- marcas de oralidade.

D) ESCOLHA VOCABULAR

- escolha vocabular inadequada;
- emprego de palavra ou expressão de maneira inadequada ao contexto;
- imprecisão vocabular quando caracterizar efetivamente um desvio da modalidade escrita formal.

IMPORTANTE:

Não considere automaticamente como desvio:

- uma escolha estilística;
- uma construção menos frequente, mas gramaticalmente aceitável;
- uma palavra simplesmente diferente da escolha que você faria;
- uma estrutura sintática complexa;
- uma frase longa;
- uma preferência lexical válida na modalidade escrita formal.

Somente marque um desvio quando houver evidência suficiente de que existe um problema relacionado à modalidade escrita formal ou às categorias especificadas acima.

--------------------------------------------------
ESCALA — DESVIOS
--------------------------------------------------

4 — AUSÊNCIA OU QUASE AUSÊNCIA DE DESVIOS

Não foram identificados desvios relevantes, ou há no máximo ocorrências extremamente pontuais que não caracterizam um padrão de problemas.

O texto demonstra excelente domínio da modalidade escrita formal.

3 — POUCOS DESVIOS

Há poucos desvios pontuais.

Os problemas são pouco frequentes e não comprometem o domínio geral da modalidade escrita formal.

2 — ALGUNS DESVIOS

Há uma quantidade perceptível de desvios.

Os problemas aparecem de maneira recorrente, mas o domínio geral da modalidade escrita formal ainda está razoavelmente preservado.

1 — MUITOS DESVIOS

Há desvios frequentes ao longo do texto.

Os problemas aparecem em diferentes partes da redação e evidenciam domínio insuficiente da modalidade escrita formal.

0 — DESVIOS MUITO FREQUENTES E/OU GRAVES

Os desvios são generalizados, muito frequentes ou apresentam gravidade suficiente para demonstrar domínio muito baixo da modalidade escrita formal.

==================================================
REGRAS PARA DIFERENCIAR ESTRUTURA SINTÁTICA E DESVIOS
==================================================

É fundamental distinguir os dois conceitos.

ESTRUTURA SINTÁTICA:

Avalia principalmente a construção e organização das orações e dos períodos.

DESVIOS:

Avaliam principalmente problemas relacionados às convenções da escrita formal e aspectos gramaticais.

Uma ocorrência deve ser classificada segundo a natureza principal do problema.

Não conte duas vezes a mesma ocorrência simplesmente porque ela pode afetar simultaneamente a leitura e a gramática.

==================================================
PROCEDIMENTO DE ANÁLISE
==================================================

Siga estas etapas internamente:

ETAPA 1 — Leia a redação completa.

ETAPA 2 — Identifique as ocorrências relacionadas à estrutura sintática.

ETAPA 3 — Identifique as ocorrências relacionadas aos desvios.

ETAPA 4 — Diferencie as duas categorias e evite dupla contagem.

ETAPA 5 — Avalie a frequência, gravidade e impacto das ocorrências.

ETAPA 6 — Determine o score de estrutura sintática de 0 a 5.

ETAPA 7 — Determine o score de desvios de 0 a 4.

ETAPA 8 — Produza a saída JSON exatamente no formato especificado.

==================================================
EVIDÊNCIAS
==================================================

Para cada conceito, apresente as principais evidências encontradas na redação.

Para cada evidência, informe:

- trecho original da redação;
- tipo de ocorrência;
- explicação breve.

Não invente evidências.

Se não houver evidências relevantes, retorne uma lista vazia.

Não altere o texto original das evidências.

==================================================
SAÍDA
==================================================

Retorne somente um objeto JSON válido.
"""


# ============================================================
# 3. JSON SCHEMA
# ============================================================

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "estrutura_sintatica": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 5},
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
    },
    "required": ["estrutura_sintatica", "desvios"],
    "additionalProperties": False,
}


# ============================================================
# 4. FUNÇÃO PARA GERAR UMA WEAK LABEL
# ============================================================
def getApiKey(name_api_key):
    with open(API_KEY_SECRET_DIR, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config[name_api_key]


# ============================================================
# 4. FUNÇÃO PARA GERAR UMA WEAK LABEL
# ============================================================


def generate_weak_label(essay):
    client = OpenAI(api_key=getApiKey(API_KEY_SECRET))
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
# 5. CARREGAR DATASET
# ============================================================

dataset = load_dataset(DATASET_NAME, cache_dir="tmp/aes_enem", trust_remote_code=True)['train']

essay = dataset[0]["essay_text"]

print("REDAÇÃO:")
print(essay)

result = generate_weak_label(essay)

print("\nWEAK LABEL:")
print(json.dumps(
    result,
    indent=4,
    ensure_ascii=False
))