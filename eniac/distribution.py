from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def digit_distribution(train_loader, test_loader):
    train_counts = Counter()
    test_counts = Counter()
    
    # Train
    for _, (a_digits, b_digits, _) in train_loader:
        train_counts.update(a_digits.tolist())
        train_counts.update(b_digits.tolist())

    # Test
    for _, (a_digits, b_digits, _) in test_loader:
        test_counts.update(a_digits.tolist())
        test_counts.update(b_digits.tolist())

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

    plt.xticks(x)
    plt.xlabel("Digit")
    plt.ylabel("Number of smples")
    plt.title("Distribution of digit")
    plt.legend()

    plt.show()

def sum_distribution(train_loader, test_loader):
    train_sum_counts = Counter()
    test_sum_counts = Counter()

    for _, (_, _, sums) in train_loader:
        train_sum_counts.update(sums.tolist())

    for _, (_, _, sums) in test_loader:
        test_sum_counts.update(sums.tolist())

    values = range(19)

    train_values = [train_sum_counts[i] for i in values]
    test_values = [test_sum_counts[i] for i in values]

    plt.figure(figsize=(10,5))

    width = 0.4
    x = range(19)

    plt.bar([i - width/2 for i in x], train_values,
            width=width, label="Train")

    plt.bar([i + width/2 for i in x], test_values,
            width=width, label="Test")

    plt.xticks(x)
    plt.xlabel("Sum (y)")
    plt.ylabel("Number of smples")
    plt.title("Sum Distribution (y)")
    plt.legend()

    plt.show()

def digit_combination(train_loader, test_loader):

    train_counts = np.zeros((10, 10), dtype=int)
    test_counts = np.zeros((10, 10), dtype=int)

    # Train
    for _, (a_digits, b_digits, _) in train_loader:
        for a, b in zip(a_digits.tolist(), b_digits.tolist()):
            train_counts[a, b] += 1

    # Test
    for _, (a_digits, b_digits, _) in test_loader:
        for a, b in zip(a_digits.tolist(), b_digits.tolist()):
            test_counts[a, b] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Train
    im1 = ax1.imshow(train_counts, cmap="Blues")
    ax1.set_title("Train")
    ax1.set_xlabel("Segundo dígito")
    ax1.set_ylabel("Primer dígito")
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
    ax2.set_xlabel("Segundo dígito")
    ax2.set_ylabel("Primer dígito")
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
    plt.show()
