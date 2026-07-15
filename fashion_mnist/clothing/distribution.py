import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# ==============================================
# CONFIG
# ==============================================
DATA_RESULT_PATH = "result"                                 # Result data path
GRAPH_NAME_DIGIT = "digit_graph"                            # Digit name
GRAPH_NAME_DIGIT_DISTRIBUTION = "distribution_digit_graph"  # Distribution digit name
GRAPH_NAME_LABEL_DISTRIBUTION = "distribution_label_graph"      # Distribution digit name
GRAPH_NAME_COMBINATION_DIGIT = "combination_digit_graph"    # Distribution digit name
classes = [
    "T-Shirt/Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandals",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boots"
]
# ==============================================
# GRAPHS
# ==============================================
class Graphs():
    def __init__(self, root: str, class_name, digit_dist: str, label_dist: str, digit_comb: str, train_loader, test_loader):
        self.result_dir = root
        self.class_name = class_name
        self.digit_dist= digit_dist
        self.label_dist = label_dist
        self.digit_comb = digit_comb
        self.train_loader = train_loader
        self.test_loader = test_loader

    def show_img(self):
        samples = {}
        for (img1, img2, img3), (digit1, digit2, digit3), _ in self.train_loader:
            for images, digits in [(img1, digit1), (img2, digit2), (img3, digit3)]:
                for image, label in zip(images, digits):
                    label = label.item()
                    if label not in samples:
                        samples[label] = image
                    if len(samples) == 10:
                        break
                if len(samples) == 10:
                    break
            if len(samples) == 10:
                break

        # Mostrar las imágenes
        plt.figure(figsize=(12, 4))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(samples[i].squeeze(), cmap="gray")
            plt.title(f"{i}\n{classes[i]}")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(self.class_name, dpi=300, bbox_inches="tight")
        plt.show()


    def digit_distribution(self):
        train_counts = Counter()
        test_counts = Counter()
        
        # Train
        for _, (digit1, digit2, digit3), _ in self.train_loader:
            train_counts.update(digit1.tolist())
            train_counts.update(digit2.tolist())
            train_counts.update(digit3.tolist())

        # Test
        for _, (digit1, digit2, digit3), _ in self.test_loader:
            test_counts.update(digit1.tolist())
            test_counts.update(digit2.tolist())
            test_counts.update(digit3.tolist())

        digits = range(10)

        train_values = [train_counts[d] for d in digits]
        test_values = [test_counts[d] for d in digits]

        plt.figure(figsize=(8,5))

        width = 0.4
        x = range(10)

        plt.bar([i - width/2 for i in x], train_values,
                width=width, label="Train")

        plt.bar([i + width/2 for i in x], test_values,
                width=width, label="Test")

        plt.xticks(x, classes, rotation=30)
        plt.xlabel("Class")
        plt.ylabel("Number of smples")
        plt.title("Distribution of Fashion-MNIST")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.digit_dist, dpi=300, bbox_inches="tight")
        plt.show()

    def label_distribution(self):
        train_sum_counts = Counter()
        test_sum_counts = Counter()

        for _, _, (_, label) in self.train_loader:
            train_sum_counts.update(label.tolist())

        for _, _, (_, label) in self.test_loader:
            test_sum_counts.update(label.tolist())

        values = range(2)

        train_values = [train_sum_counts[i] for i in values]
        test_values = [test_sum_counts[i] for i in values]

        plt.figure(figsize=(10,5))

        width = 0.4
        x = range(2)

        plt.bar([i - width/3 for i in x], train_values,
                width=width, label="Train")

        plt.bar([i + width/3 for i in x], test_values,
                width=width, label="Test")

        plt.xticks(x)
        plt.xlabel("Label (y)")
        plt.ylabel("Number of smples")
        plt.title("Label Distribution (y)")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.label_dist, dpi=300, bbox_inches="tight")
        plt.show()

    def show_valid_outfits(self, n_samples=3):
        # Tomar algunos batches del loader
        for (img1, img2, img3), (digit1, digit2, digit3), (_, label) in self.train_loader:
            # Filtrar solo válidos
            valid_mask = label == 1
            img1_valid = img1[valid_mask]
            img2_valid = img2[valid_mask]
            img3_valid = img3[valid_mask]
            d1_valid = digit1[valid_mask]
            d2_valid = digit2[valid_mask]
            d3_valid = digit3[valid_mask]

            # Mostrar hasta n_samples
            n = min(n_samples, len(img1_valid))
            plt.figure(figsize=(12, 3*n))
            for i in range(n):
                plt.subplot(n, 3, i*3 + 1)
                plt.imshow(img1_valid[i].squeeze().numpy(), cmap="gray")
                plt.title(f"UPPER ({d1_valid[i].item()})")
                plt.axis("off")

                plt.subplot(n, 3, i*3 + 2)
                plt.imshow(img2_valid[i].squeeze().numpy(), cmap="gray")
                plt.title(f"LOWER ({d2_valid[i].item()})")
                plt.axis("off")

                plt.subplot(n, 3, i*3 + 3)
                plt.imshow(img3_valid[i].squeeze().numpy(), cmap="gray")
                plt.title(f"SHOES ({d3_valid[i].item()})")
                plt.axis("off")

            plt.suptitle("Exemples of outfits valid")
            plt.tight_layout()
            # plt.savefig(, dpi=300, bbox_inches="tight")
            plt.show()
            break  # solo primer batch


def main_distribution(train_loader, test_loader):
  # Obtiene el directorio donde está este archivo.py
  base_dir = os.path.dirname(os.path.abspath(__file__))
  # Une el directorio de base_dir con las carpetas "data" y "result"
  result_dir = os.path.join(base_dir, DATA_RESULT_PATH)
  digit_dist = os.path.join(result_dir, f"{GRAPH_NAME_DIGIT_DISTRIBUTION}.png")
  label_dist = os.path.join(result_dir, f"{GRAPH_NAME_LABEL_DISTRIBUTION}.png")
  digit_comb = os.path.join(result_dir, f"{GRAPH_NAME_COMBINATION_DIGIT}.png")
  class_name = os.path.join(result_dir, f"{GRAPH_NAME_DIGIT}.png")
  graph = Graphs(result_dir, class_name, digit_dist, label_dist, digit_comb, train_loader, test_loader)
#   graph.show_img()
#   graph.digit_distribution()
#   graph.label_distribution()
  graph.show_valid_outfits()
