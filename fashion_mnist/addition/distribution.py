import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# ==============================================
# CONFIG
# ==============================================
RESULT_PATH = "result"                                      # Result path
GRAPH_NAME_DIGIT = "digit_graph"                            # Digit name
GRAPH_NAME_DIGIT_DISTRIBUTION = "digit_distribution_graph"  # Digit distribution
GRAPH_NAME_SUM_DISTRIBUTION = "sum_distribution_graph"      # Sum distribution
GRAPH_NAME_COMBINATION_DIGIT = "digit_combination_graph"    # Digit combination
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
    def __init__(self, root: str, class_name, digit_dist: str, sum_dist: str, digit_comb: str, train_loader, test_loader):
        self.result_dir = root
        self.class_name = class_name
        self.digit_dist= digit_dist
        self.sum_dist = sum_dist
        self.digit_comb = digit_comb
        self.train_loader = train_loader
        self.test_loader = test_loader

    def show_img(self):
        
        samples = {}
        for (img1, img2, img3), (digit1, digit2, digit3, sums) in self.train_loader:
            for image, label in zip(img1, digit1):
                label = label.item()

                if label not in samples:
                    samples[label] = image

                if len(samples) == 10:
                    break

            if len(samples) == 10:
                break

        # Show images
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
        for _, (digit1, digit2, digit3, _) in self.train_loader:
            train_counts.update(digit1.tolist())
            train_counts.update(digit2.tolist())
            train_counts.update(digit3.tolist())

        # Test
        for _, (digit1, digit2, digit3, _) in self.test_loader:
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
        plt.ylabel("Number of samples")
        plt.title("Digit Distribution of Fashion-MNIST dataset")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.digit_dist, dpi=300, bbox_inches="tight")
        plt.show()

    def sum_distribution(self):
        train_sum_counts = Counter()
        test_sum_counts = Counter()

        for _, (_, _, _, sums) in self.train_loader:
            train_sum_counts.update(sums.tolist())

        for _, (_, _, _, sums) in self.test_loader:
            test_sum_counts.update(sums.tolist())

        values = range(28)

        train_values = [train_sum_counts[i] for i in values]
        test_values = [test_sum_counts[i] for i in values]

        plt.figure(figsize=(10,5))

        width = 0.4
        x = range(28)

        plt.bar([i - width/2 for i in x], train_values,
                width=width, label="Train")

        plt.bar([i + width/2 for i in x], test_values,
                width=width, label="Test")

        plt.xticks(x)
        plt.xlabel("Sum (y)")
        plt.ylabel("Number of smples")
        plt.title("Sum Distribution (y)")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.sum_dist, dpi=300, bbox_inches="tight")
        plt.show()

    def digit_combination(self):

        train_counts = np.zeros((10, 10), dtype=int)
        test_counts = np.zeros((10, 10), dtype=int)

        # Train
        for _, (digit1, digit2, digit3, _) in self.train_loader:
            for a, b in zip(digit1.tolist(), digit2.tolist()):
                train_counts[a, b] += 1

        # Test
        for _, (digit1, digit2, digit3, _) in self.test_loader:
            for a, b in zip(digit1.tolist(), digit2.tolist()):
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
  base_dir = os.path.dirname(os.path.abspath(__file__))
  result_dir = os.path.join(base_dir, RESULT_PATH)

  digit_dist = os.path.join(result_dir, f"{GRAPH_NAME_DIGIT_DISTRIBUTION}.png")
  sum_dist = os.path.join(result_dir, f"{GRAPH_NAME_SUM_DISTRIBUTION}.png")
  digit_comb = os.path.join(result_dir, f"{GRAPH_NAME_COMBINATION_DIGIT}.png")
  class_name = os.path.join(result_dir, f"{GRAPH_NAME_DIGIT}.png")
  
  graph = Graphs(result_dir, class_name, digit_dist, sum_dist, digit_comb, train_loader, test_loader)
#   graph.show_img()
  graph.digit_distribution()
  graph.sum_distribution()
#   graph.digit_combination()
