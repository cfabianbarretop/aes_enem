import torch
from collections import defaultdict


# ==========================================
# TODAS LAS COMBINACIONES POSIBLES
# ==========================================

sum_to_pairs = defaultdict(list)

for z1 in range(10):
    for z2 in range(10):

        y = z1 + z2

        sum_to_pairs[y].append(
            (z1, z2)
        )


# ==========================================
# DKB EMPIRICO
# ==========================================

def compute_empirical_dkb(dataset):

    total = 0

    for sample in dataset:

        y = sample["y"]

        valid_pairs = len(
            sum_to_pairs[y]
        )

        incompatible_pairs = (
            100 - valid_pairs
        )

        total += incompatible_pairs

    return total / len(dataset)


# ==========================================
# DATASETS
# ==========================================

files = [
    "mnist_addition_level_I.pt",
    "mnist_addition_level_II.pt",
    "mnist_addition_level_III.pt",
    "mnist_addition_level_IV.pt",
    "mnist_addition_level_V.pt",
    "mnist_addition_level_VI.pt"
]


# ==========================================
# CALCULO
# ==========================================

for file in files:

    dataset = torch.load(
        file,
        weights_only=False
    )

    dkb = compute_empirical_dkb(
        dataset
    )

    print(
        f"{file:<30} "
        f"DKB = {dkb:.6f} -> "
        f"(C - DKB) = {(100 - dkb):.6f}"
    )