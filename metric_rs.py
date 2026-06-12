import random
from collections import defaultdict, Counter

import torch
from torchvision.datasets import MNIST
from torchvision import transforms


# =====================================================
# CONFIGURACION
# =====================================================

DATA_DIR = "./data"
N_SAMPLES = 30000

EASY_SUMS = [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 18]
HARD_SUMS = [7, 8, 9, 10, 11]

LEVELS = {
    "I":  (0.90, 0.10),
    "II": (0.75, 0.25),
    "III": (0.60, 0.40),
    "IV": (0.40, 0.60),
    "V": (0.25, 0.75),
    "VI": (0.10, 0.90)
}


# =====================================================
# CARGAR MNIST
# =====================================================

transform = transforms.ToTensor()

mnist_train = MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)


# =====================================================
# AGRUPAR IMAGENES POR DIGITO
# =====================================================

mnist_by_label = defaultdict(list)

for image, label in mnist_train:
    mnist_by_label[label].append(image)

print("Imagenes por digito")

for digit in range(10):
    print(
        digit,
        len(mnist_by_label[digit])
    )


# =====================================================
# TODAS LAS COMBINACIONES POSIBLES
# =====================================================

sum_to_pairs = defaultdict(list)

for z1 in range(10):

    for z2 in range(10):

        y = z1 + z2

        sum_to_pairs[y].append(
            (z1, z2)
        )


print("\nEjemplos:")

print("Suma 0 :", sum_to_pairs[0])
print("Suma 1 :", sum_to_pairs[1])
print("Suma 9 :", sum_to_pairs[9])


# =====================================================
# FUNCION AUXILIAR
# =====================================================

def build_sum_distribution(
        easy_ratio,
        hard_ratio,
        total_samples
):

    distribution = {}

    n_easy = int(
        total_samples * easy_ratio
    )

    n_hard = total_samples - n_easy

    # ------------------------
    # EASY
    # ------------------------

    easy_per_sum = n_easy // len(EASY_SUMS)

    remainder_easy = (
        n_easy % len(EASY_SUMS)
    )

    for s in EASY_SUMS:
        distribution[s] = easy_per_sum

    for s in EASY_SUMS[:remainder_easy]:
        distribution[s] += 1

    # ------------------------
    # HARD
    # ------------------------

    hard_per_sum = n_hard // len(HARD_SUMS)

    remainder_hard = (
        n_hard % len(HARD_SUMS)
    )

    for s in HARD_SUMS:
        distribution[s] = hard_per_sum

    for s in HARD_SUMS[:remainder_hard]:
        distribution[s] += 1

    return distribution


# =====================================================
# GENERAR UNA MUESTRA
# =====================================================

def sample_example(sum_label):

    pair = random.choice(
        sum_to_pairs[sum_label]
    )

    z1, z2 = pair

    img1 = random.choice(
        mnist_by_label[z1]
    )

    img2 = random.choice(
        mnist_by_label[z2]
    )

    return {
        "x1": img1,
        "x2": img2,
        "z1": z1,
        "z2": z2,
        "y": sum_label
    }


# =====================================================
# GENERAR DATASET COMPLETO
# =====================================================

def generate_level(
        level_name,
        easy_ratio,
        hard_ratio,
        total_samples=30000
):

    distribution = build_sum_distribution(
        easy_ratio,
        hard_ratio,
        total_samples
    )

    dataset = []

    for sum_label, amount in distribution.items():

        print(
            f"Nivel {level_name} "
            f"Suma {sum_label} "
            f"Cantidad {amount}"
        )

        for _ in range(amount):

            sample = sample_example(
                sum_label
            )

            dataset.append(sample)

    random.shuffle(dataset)

    return dataset


# =====================================================
# GENERAR LOS 6 NIVELES
# =====================================================

datasets = {}

for level_name, ratios in LEVELS.items():

    easy_ratio, hard_ratio = ratios

    print(
        "\n===================="
    )

    print(
        f"Generando nivel {level_name}"
    )

    datasets[level_name] = generate_level(
        level_name,
        easy_ratio,
        hard_ratio,
        N_SAMPLES
    )

    print(
        f"Tamaño final: "
        f"{len(datasets[level_name])}"
    )


# =====================================================
# VERIFICACION
# =====================================================

for level_name in datasets:

    counter = Counter()

    for sample in datasets[level_name]:

        counter[
            sample["y"]
        ] += 1

    print(
        f"\nDistribucion Nivel "
        f"{level_name}"
    )

    for s in sorted(counter.keys()):

        print(
            f"Suma {s}: "
            f"{counter[s]}"
        )


# =====================================================
# GUARDAR DATASETS
# =====================================================

for level_name in datasets:

    torch.save(
        datasets[level_name],
        f"mnist_addition_level_{level_name}.pt"
    )

    print(
        f"Guardado: "
        f"mnist_addition_level_{level_name}.pt"
    )