import torch
from torchvision import datasets, transforms
import random
from typing import List, Dict

# ---------------------------
# Config
# ---------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---------------------------
# Load MNIST
# ---------------------------
def load_mnist(train: bool):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    dataset = datasets.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=transform
    )
    
    return dataset


# ---------------------------
# Core pairing logic
# ---------------------------
def create_unique_pairs(dataset) -> List[Dict]:
    """
    Cada imagen se usa exactamente una vez.
    Si hay N imágenes -> genera floor(N/2) pares.
    """
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # Asegurar número par de índices
    if len(indices) % 2 != 0:
        indices = indices[:-1]

    data = []

    for i in range(0, len(indices), 2):
        i1 = indices[i]
        i2 = indices[i + 1]

        img1, d1 = dataset[i1]
        img2, d2 = dataset[i2]

        data.append({
            "x1": img1,
            "x2": img2,
            "z1": int(d1),
            "z2": int(d2),
            "y": int(d1 + d2)
        })

    return data


# ---------------------------
# Save dataset
# ---------------------------
def save_dataset(data, path):
    """
    Guarda lista de dicts
    """
    torch.save(data, path)
    print(f"Saved {len(data)} samples to {path}")


# ---------------------------
# Main pipeline
# ---------------------------
def generate_mnist_add_datasets():
    # Load original MNIST
    train_dataset = load_mnist(train=True)
    test_dataset = load_mnist(train=False)

    # Create unique pairs
    train_data = create_unique_pairs(train_dataset)
    test_data = create_unique_pairs(test_dataset)

    # Save
    save_dataset(train_data, "mnist-add-unique-train.pt")
    save_dataset(test_data, "mnist-add-unique-test.pt")


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    generate_mnist_add_datasets()