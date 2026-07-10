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
        for (img1, _, _), (digit1, _, _), _ in self.train_loader:
            for image, label in zip(img1, digit1):
                label = label.item()

                if label not in samples:
                    samples[label] = image

                # Cuando ya tengamos las 10 clases, terminamos
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

    def digit_combination(self):

        train_counts = np.zeros((10, 10), dtype=int)
        test_counts = np.zeros((10, 10), dtype=int)

        # Train
        for _, (a_digits, b_digits, _) in self.train_loader:
            for a, b in zip(a_digits.tolist(), b_digits.tolist()):
                train_counts[a, b] += 1

        # Test
        for _, (a_digits, b_digits, _) in self.test_loader:
            for a, b in zip(a_digits.tolist(), b_digits.tolist()):
                test_counts[a, b] += 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Train
        im1 = ax1.imshow(train_counts, cmap="Blues")
        ax1.set_title("Train")
        ax1.set_xlabel("Second digit")
        ax1.set_ylabel("First digit")
        ax1.invert_yaxis()
        ax1.set_xticks(range(10))
        ax1.set_yticks(range(10))

        for i in range(10):
            for j in range(10):
                ax1.text(
                    j, i,
                    train_counts[i, j],
                    ha="center",
                    va="center",
                    color="black"
                )

        # Test
        im2 = ax2.imshow(test_counts, cmap="Blues")
        ax2.set_title("Test")
        ax2.set_xlabel("Second digit")
        ax2.set_ylabel("First digit")
        ax2.set_xticks(range(10))
        ax2.set_yticks(range(10))

        for i in range(10):
            for j in range(10):
                ax2.text(
                    j, i,
                    test_counts[i, j],
                    ha="center",
                    va="center",
                    color="black"
                )

        fig.colorbar(im1, ax=ax1, fraction=0.046)
        fig.colorbar(im2, ax=ax2, fraction=0.046)

        plt.tight_layout()
        plt.savefig(self.digit_comb, dpi=300, bbox_inches="tight")
        plt.show()

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
  graph.label_distribution()
#   graph.digit_combination()
