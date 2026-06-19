from collections import Counter
import matplotlib.pyplot as plt

def split_data(train_loader, test_loader):
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

    print("Train:", train_counts)
    print("Test:", test_counts)

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
    plt.title("Distribution of class")
    plt.legend()

    plt.show()

def split_add(train_loader, test_loader):
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