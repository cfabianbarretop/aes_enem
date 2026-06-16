import json
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

INPUT_FILE = "results/leaf_labels_vlm.json"


# =========================
# RULES
# =========================

def rule_based_species(margin, shape, texture):
    if margin == 'serrate': return 'Ocimum basilicum'
    elif margin == 'indented': return 'Jatropha curcas'
    elif margin == 'lobed': return 'Platanus orientalis'
    elif margin == 'serrulate': return "Citrus limon"
    elif margin == 'entire':
        if shape == 'ovate': return 'Pongamia Pinnata'
        elif shape == 'lanceolate': return 'Mangifera indica'
        elif shape == 'oblong': return 'Syzygium cumini'
        elif shape == 'obovate': return "Psidium guajava"
        else:
            if texture == 'leathery': return "Alstonia Scholaris"
            elif texture == 'rough': return "Terminalia Arjuna"
            elif texture == 'glossy': return "Citrus limon"
            else: return "Punica granatum"
    else:
        if shape == 'elliptical': return 'Terminalia Arjuna'
        elif shape == 'lanceolate': return "Mangifera indica"
        else: return 'Syzygium cumini'


# =========================
# EVALUATION
# =========================

def evaluate_from_file(results):
    correct = 0
    total = len(results)

    y_true = []
    y_pred = []

    for r in results:
        y_true.append(r["species"])
        pred_species = rule_based_species(
            r["margin"],
            r["shape"],
            r["texture"]
        )
        y_pred.append(pred_species)

        if pred_species == r["species"]:
            correct += 1


    accuracy = correct / total

    print("\n=========================")
    print(f"Rule-based Accuracy: {accuracy:.4f}")
    print(f"Total samples: {total}")
    print("=========================\n")

    return accuracy, y_true, y_pred

def print_distribution(results):
    species_counts = Counter(r["species"] for r in results)

    margin_by_species = defaultdict(list)
    shape_by_species = defaultdict(list)
    texture_by_species = defaultdict(list)

    # Group data by species
    for r in results:
        sp = r["species"]
        margin_by_species[sp].append(r["margin"])
        shape_by_species[sp].append(r["shape"])
        texture_by_species[sp].append(r["texture"])

    print("\n=== Species distribution ===")
    for species, count in species_counts.most_common():
        print(f"{species}: {count}")

    # Distributions per species
    for species in species_counts:
        print(f"\n=== {species} ===")

        margin_dist = Counter(margin_by_species[species])
        shape_dist = Counter(shape_by_species[species])
        texture_dist = Counter(texture_by_species[species])

        print("Margin:", dict(margin_dist))
        print("Shape:", dict(shape_dist))
        print("Texture:", dict(texture_dist))

def plot_cm(cm, labels, title):
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=75)
    plt.title(title)
    plt.show()

# =========================
# RUN
# =========================

if __name__ == "__main__":
    with open(INPUT_FILE) as f:
        results = json.load(f)
    
    acc, y_true, y_pred = evaluate_from_file(results)
    print_distribution(results)
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_cm(cm, labels, "Confusion Matrix")
