from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data(csv_path, filter=False):
    df = pd.read_csv(csv_path, sep=",")

    if filter:
        df = df[df['y'] == df['p']]
    
    y_true = df[['g_1', 'g_2']].to_numpy()
    y_pred = df[['c_1', 'c_2']].to_numpy()
    
    return y_true, y_pred


# =========================================================
# 1. Codificacion F(C): vector binario → entero
# =========================================================
def encode_concepts(y):
    """
    Convierte vectores de conceptos (multi-dim) a enteros.
    Ej: [3,2] -> "(3, 2)"
    """
    return np.array([f"{tuple(row)}" for row in y])


# =========================================================
# 2. Obtener F(C) y F(C*)
# =========================================================
def get_F_sets(y_true, y_pred):
    F_C_star = set(encode_concepts(y_true))
    F_C = set(encode_concepts(y_pred))
    return F_C_star, F_C


# =========================================================
# 3. Construccion de matriz de confusion
# =========================================================
def build_confusion(F_C_star, F_C, y_true, y_pred):
    labels = sorted(list(F_C_star.union(F_C)))  # m

    y_true_enc = encode_concepts(y_true)
    y_pred_enc = encode_concepts(y_pred)

    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=labels)

    return cm, labels


# =========================================================
# 4. Calculo de p, m y Cls(C)
# =========================================================
def compute_cls(F_C_star, F_C):
    m = len(F_C_star.union(F_C))
    p = len(F_C) 

    cls = 1 - (p / m)
    return cls, p, m


# =========================================================
# 5. Plot
# =========================================================
def plot_cm(cm, labels, title):
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.show()


# =========================================================
# ===================== MAIN ==============================
# =========================================================

#1st scenario
#y_true, y_pred = load_data("e_20_resultados.csv", filter=True)
#2nd scenario
#y_true, y_pred = load_data("c1_A_data_test_resultados.csv", filter=True)
#3th scenario
y_true, y_pred = load_data("e_2_resultados_test.csv", filter=True)
#4th scenario
#y_true, y_pred = load_data("c1_B_data_test_resultados.csv", filter=True)
#5th scenario
#y_true, y_pred = load_data("c1_B_data_train_resultados.csv", filter=True)

"""
y_true = np.array([
[3,1],[2,1],[2,1],[4,2],[4,2],[4,2],[3,2],[4,2],[4,2],[3,2],[4,2],[3,2],[4,2],[3,2]
])

y_pred = np.array([
[2,1],[2,1],[2,1],[3,2],[3,2],[3,2],[3,2],[3,2],[4,2],[3,2],[4,2],[4,2],[4,2],[3,2]
])
"""

# ====== F(C) y F(C*) ======
F_C_star, F_C = get_F_sets(y_true, y_pred)

print("F(C*):", F_C_star)
print("F(C):", F_C)

# ====== Confusion Matrix ======
cm, labels = build_confusion(F_C_star, F_C, y_true, y_pred)

plot_cm(cm, labels, "Joint Concept Confusion Matrix")

# ====== Cls(C) ======
cls, p, m = compute_cls(F_C_star, F_C)

print(f"m = {m}")
print(f"p = {p}")
print(f"Cls(C) = {cls:.4f}")