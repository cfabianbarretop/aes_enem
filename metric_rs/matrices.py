#s,m,pares
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd


def load_data(csv_path, filter=False):
    df = pd.read_csv(csv_path, sep=",")

    if filter:
        df = df[df['y'] == df['p']]
    
    y_true = df[['g_1', 'g_2']].to_numpy()
    y_pred = df[['c_1', 'c_2']].to_numpy()
    
    return y_true, y_pred


#============= MC a nivel de cada concepto =============

#1st scenario
#y_true, y_pred = load_data("e_20_resultados.csv", filter=True)
#2nd scenario
#y_true, y_pred = load_data("c1_A_data_test_resultados.csv", filter=True)
#3th scenario
y_true, y_pred = load_data("e_5_resultados_test.csv", filter=True)
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

# separar conceptos
syntax_true = y_true[:,0]
mistakes_true = y_true[:,1]

syntax_pred = y_pred[:,0]
mistakes_pred = y_pred[:,1]

# labels
syntax_labels = np.unique(np.concatenate((syntax_true, syntax_pred)))
mistakes_labels = np.unique(np.concatenate((mistakes_true, mistakes_pred)))


cm_syntax = confusion_matrix(syntax_true, syntax_pred)

ConfusionMatrixDisplay(cm_syntax, display_labels=syntax_labels).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Syntax")
plt.show()

cm_mistakes = confusion_matrix(mistakes_true, mistakes_pred)

ConfusionMatrixDisplay(cm_mistakes, display_labels=mistakes_labels).plot(cmap=plt.cm.Oranges)
plt.title("Confusion Matrix - Mistakes")
plt.show()

#============= MC a nivel del espacio conjunto =============

# todos los valores posibles
syntax_vals = [0,1,2,3,4]
mistake_vals = [0,1,2,3]

joint_space = list(itertools.product(syntax_vals, mistake_vals))

# mapa: (s,m) → índice
joint_to_idx = {pair:i for i,pair in enumerate(joint_space)}

true_encoded = [joint_to_idx[tuple(x)] for x in y_true]
pred_encoded = [joint_to_idx[tuple(x)] for x in y_pred]

cm_joint = confusion_matrix(true_encoded, pred_encoded, labels=range(len(joint_space)))

labels = [f"({s},{m})" for s,m in joint_space]

disp = ConfusionMatrixDisplay(cm_joint, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Joint Concept Confusion Matrix (Full Space)")
plt.xticks(rotation=90)
plt.show()