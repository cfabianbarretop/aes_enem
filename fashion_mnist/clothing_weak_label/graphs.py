import os
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================
# CONFIG
# ==============================================
GRAPH_RESULT_NAME = "result_graph"      # Result img name
GRAPH_RESULT_CONCEPT_ACC = "concept_acc_graph"      # Result img name

# ==============================================
# COLOR MAP
# ==============================================
cmap = plt.get_cmap("tab10")
KEY_COLOR_A = "aal"
KEY_COLOR_B = "bce"
KEY_COLOR_C = "train"
KEY_COLOR_D = "test"
COLOR_MAP = {
    KEY_COLOR_A: cmap(0),
    KEY_COLOR_B: cmap(1),
    KEY_COLOR_C: cmap(2),
    KEY_COLOR_D: cmap(3)
}

# ==============================================
# GRAPHS
# ==============================================
class Graphs():
    def __init__(self, root: str, img: str, acc_concept_img: str, training: str):
        self.result_dir = root
        self.result_img= img
        self.acc_concept_img = acc_concept_img
        self.training = training
    
    def graph(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        for csv_file in Path(self.result_dir).glob("*.csv"):
            name = csv_file.stem.lower()
            label = csv_file.stem

            if self.training not in name:
                continue
            
            # Extraer clave (aal o bce)
            if KEY_COLOR_A in name:
                key = KEY_COLOR_A
            elif KEY_COLOR_B in name:
                key = KEY_COLOR_B
            elif KEY_COLOR_C in name:
                key = KEY_COLOR_C
            elif KEY_COLOR_D in name:
                key = KEY_COLOR_D
            else:
                key = None

            color = COLOR_MAP.get(key, "black")

            df = pd.read_csv(csv_file)

            # Gráfico de Accuracy
            ax1.plot(
                df["epoch"],
                df["acc"],
                marker="o",
                label=label,
                color=color
            )

            # Gráfico de prob_mod_no
            ax2.plot(
                df["epoch"],
                df["loss"],
                marker="o",
                label=label,
                color=color
            )

            if {"epoch", "RSR"}.issubset(df.columns):
                line_rsr, = ax3.plot(
                    df["epoch"],
                    df["RSR"],
                    marker="o",
                    label=f"{label} - RSR",
                    color=color
                )
            

            if {"epoch", "GAcc"}.issubset(df.columns):
                ax4.plot(
                    df["epoch"],
                    df["GAcc"],
                    marker="o",
                    label=label,
                    color=color
                )

        # Configuración gráfico 1
        ax1.set_title("Acc (Y)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("acc(y)")
        ax1.grid(True)
        ax1.legend()

        # Configuración gráfico 2
        ax2.set_title("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("loss")
        ax2.grid(True)
        ax2.legend()

        # Configuración gráfico 3
        ax3.set_title("RSR")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("RSR (%)")
        ax3.grid(True)
        ax3.legend()

        # Configuración gráfico 3
        ax4.set_title("Acc (C) ")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("acc(C)")
        ax4.grid(True)
        ax4.legend()

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim(0, 21)
            ax.set_xticks(range(0, 22, 2))

        plt.tight_layout()
        plt.savefig(self.result_img, dpi=300, bbox_inches="tight")
        plt.close()
    
    def graph_concept(self):
        for csv_file in Path(self.result_dir).glob("*.csv"):
            name = csv_file.stem.lower()

            if self.training not in name:
                continue

            df = pd.read_csv(csv_file)
            df["acc_mean"] = df[["acc_C1", "acc_C2", "acc_C3"]].mean(axis=1)
            plt.figure(figsize=(8,5))
            plt.plot(df["epoch"], df["acc_C1"], label="C1", marker="o")
            plt.plot(df["epoch"], df["acc_C2"], label="C2", marker="s")
            plt.plot(df["epoch"], df["acc_C3"], label="C3", marker="^")
            plt.plot(df["epoch"], df["GAcc"], label="GAcc", marker="H", linestyle="--")
            plt.plot(df["epoch"], df["acc_mean"], label="Mean", marker="d",  linestyle="--", color="black")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.title("Accuracy by concept")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.acc_concept_img, dpi=300, bbox_inches="tight")
            plt.close()

def main_graph(training="rs", dir_result = ""):
    # Obtiene el directorio donde está este archivo.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Une el directorio de base_dir con las carpetas "data" y "result"
    result_dir = os.path.join(base_dir, dir_result)
    name_img = f"{GRAPH_RESULT_NAME}_{training}.png"
    name_acc_concept_img = f"{GRAPH_RESULT_CONCEPT_ACC}_{training}.png"
    result_img = os.path.join(result_dir, name_img)
    result_acc_concept_img = os.path.join(result_dir, name_acc_concept_img)
    graph = Graphs(result_dir, result_img, result_acc_concept_img, training)
    graph.graph()
    graph.graph_concept()

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("graph")
  parser.add_argument("--training", type=str, default="rs")
  args = parser.parse_args()