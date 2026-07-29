import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import string

# ==============================================
# CONFIG
# ==============================================
RESULT_PATH = "result"                                      # Result path
GRAPH_NAME_DIGIT = "classes_graph"                            # Digit name
GRAPH_NAME_DIGIT_DISTRIBUTION = "class_distribution_graph"  # Digit distribution
GRAPH_NAME_SUM_DISTRIBUTION = "label_distribution_graph"      # Sum distribution
GRAPH_NAME_COMBINATION_DIGIT = "class_combination_graph"    # Digit combination

digit_classes = ["0","1","2","3","4","5","6","7","8","9"]
letter_classes = list(string.ascii_uppercase)

class_labels =  digit_classes + letter_classes

# ==============================================
# GRAPHS
# ==============================================
class Graphs():
    def __init__(self, root: str, class_name, class_dist: str, label_dist: str, class_comb: str, train_loader, test_loader):
        self.result_dir = root
        self.class_name = class_name
        self.class_dist= class_dist
        self.label_dist = label_dist
        self.class_comb = class_comb
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
    
    def digit_class_distribution(self):
        train_counts = Counter()
        test_counts = Counter()

        # Train
        for _, digits, _ in self.train_loader:
            train_counts.update(digits[0].tolist())
            train_counts.update(digits[1].tolist())
            train_counts.update(digits[2].tolist())

        # Test
        for _, digits, _ in self.test_loader:
            test_counts.update(digits[0].tolist())
            test_counts.update(digits[1].tolist())
            test_counts.update(digits[2].tolist())

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

        plt.xticks(x, digit_classes, rotation=30)
        plt.xlabel("Class")
        plt.ylabel("Number of samples")
        plt.title("Class Distribution")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.class_dist, dpi=300, bbox_inches="tight")
        plt.show()

    def letter_class_distribution(self):
        train_counts = Counter()
        test_counts = Counter()

        # Train
        for _, letters, _ in self.train_loader:
            train_counts.update(letters[0].tolist())
            train_counts.update(letters[1].tolist())
            train_counts.update(letters[2].tolist())

        # Test
        for _, letters, _ in self.test_loader:
            test_counts.update(letters[0].tolist())
            test_counts.update(letters[1].tolist())
            test_counts.update(letters[2].tolist())

        letters = range(26)

        train_values = [train_counts[d] for d in letters]
        test_values = [test_counts[d] for d in letters]

        plt.figure(figsize=(8,5))

        width = 0.4
        x = range(26)

        plt.bar([i - width/2 for i in x], train_values,
                width=width, label="Train")

        plt.bar([i + width/2 for i in x], test_values,
                width=width, label="Test")

        plt.xticks(x, letter_classes, rotation=30)
        plt.xlabel("Class")
        plt.ylabel("Number of samples")
        plt.title("Class Distribution")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.class_dist, dpi=300, bbox_inches="tight")
        plt.show()

    def class_distribution(self):
        train_counts = Counter()
        test_counts = Counter()

        # Train
        for _, classes, _ in self.train_loader:
            train_counts.update(classes[0].tolist())
            train_counts.update(classes[1].tolist())
            train_counts.update(classes[2].tolist())
            train_counts.update(classes[3].tolist())
            train_counts.update(classes[4].tolist())
            train_counts.update(classes[5].tolist())

        # Test
        for _, classes, _ in self.test_loader:
            test_counts.update(classes[0].tolist())
            test_counts.update(classes[1].tolist())
            test_counts.update(classes[2].tolist())
            test_counts.update(classes[3].tolist())
            test_counts.update(classes[4].tolist())
            test_counts.update(classes[5].tolist())
            
        classes = range(36)

        train_values = [train_counts[d] for d in classes]
        test_values = [test_counts[d] for d in classes]

        plt.figure(figsize=(16,5))

        width = 0.4
        x = range(36)

        plt.bar([i - width/2 for i in x], train_values,
                width=width, label="Train")

        plt.bar([i + width/2 for i in x], test_values,
                width=width, label="Test")

        plt.xticks(x, class_labels, rotation=30)
        plt.xlabel("Class")
        plt.ylabel("Number of samples")
        plt.title("Class Distribution")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.class_dist, dpi=300, bbox_inches="tight")
        plt.show()

    def label_distribution(self):
        train_label_counts = Counter()
        test_label_counts = Counter()

        for _, _, labels in self.train_loader:
            train_label_counts.update(labels.tolist())

        for _, _, labels in self.test_loader:
            test_label_counts.update(labels.tolist())

        values = range(28)

        train_values = [train_label_counts[i] for i in values]
        test_values = [test_label_counts[i] for i in values]

        plt.figure(figsize=(10,5))

        width = 0.4
        x = range(1)

        plt.bar([i - width/2 for i in x], train_values,
                width=width, label="Train")

        plt.bar([i + width/2 for i in x], test_values,
                width=width, label="Test")

        plt.xticks([0,1])
        plt.xlabel("Label")
        plt.ylabel("Number of smples")
        plt.title("Label Distribution")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.label_dist, dpi=300, bbox_inches="tight")
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
#   graph.digit_class_distribution()
#   graph.letter_class_distribution()

  graph.class_distribution()
  graph.label_distribution()
