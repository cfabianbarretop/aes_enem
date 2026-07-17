import os
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================
# CONFIG
# ==============================================
DATA_RESULT_PATH = "result"             # Result data path
GRAPH_RESULT_NAME = "result_graph"      # Result img name
GRAPH_RESULT_ACC_CONCEPT = "acc_graph" 
EPOCHS=20

# ==============================================
# COLOR MAP
# ==============================================
cmap = plt.get_cmap("tab10")
COLOR_MAP = {
    "aal": cmap(1),
    "bce": cmap(0),
    "3_net": cmap(1),
    "single_net": cmap(0),
    "single_3net": cmap(2),
    "concat_3net": cmap(3),
    "none": cmap(9)
}

# ==============================================
# GRAPHS
# ==============================================
class Graphs():
    def __init__(self, root: str, img: str, result_accC_img: str, training: str):
        self.result_dir = root
        self.result_img= img
        self.result_accC_img = result_accC_img
        self.training = training
    
    def graph(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        for csv_file in Path(self.result_dir).glob("*.csv"):
            name = csv_file.stem.lower()
            label = csv_file.stem

            if self.training not in name:
                continue
            
            # Extraer clave (aal o bce)
            print(f"Processing name: {name}")
            if "aal" in name:
                key = "aal"
            elif "bce" in name:
                key = "bce"
            elif "3_net" in name:
                key = "3_net"
            elif "single_net" in name:
                key = "single_net"
            elif "single_3net" in name:
                key = "single_3net"
            else:
                key = "none"

            color = COLOR_MAP.get(key, "black")

            df = pd.read_csv(csv_file)

            # Gráfico de Accuracy
            ax1.plot(
                df["epoch"],
                df["accY"],
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
            
            # if {"epoch", "RSRw"}.issubset(df.columns):
            #     ax3.plot(
            #         df["epoch"],
            #         df["RSRw"],
            #         color = line_rsr.get_color(),
            #         marker="s",
            #         linestyle="--",
            #         label=f"{label} - RSRw"
            #     )

            if {"epoch", "accC"}.issubset(df.columns):
                ax4.plot(
                    df["epoch"],
                    df["accC"],
                    marker="o",
                    label=label,
                    color=color
                )

        # Configuración gráfico 1
        ax1.set_title("accY (Y)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("accY(y)")
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
        ax4.set_title("accY (C) ")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("accY(C)")
        ax4.grid(True)
        ax4.legend()

        for ax in [ax1, ax4, ax2, ax3]:
            ax.set_xlim(0, EPOCHS+1)
            ax.set_xticks(range(0, EPOCHS+2, 2))

        plt.tight_layout()
        plt.savefig(self.result_img, dpi=300, bbox_inches="tight")
        plt.show()
    
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
            plt.plot(df["epoch"], df["accC"], label="accC", marker="H", linestyle="--")
            plt.plot(df["epoch"], df["acc_mean"], label="Mean", marker="d",  linestyle="--", color="black")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.title("Accuracy by concept")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.result_accC_img, dpi=300, bbox_inches="tight")
            plt.close()

def main_graph(training="rs"):
  # Obtiene el directorio donde está este archivo.py
  base_dir = os.path.dirname(os.path.abspath(__file__))
  # Une el directorio de base_dir con las carpetas "data" y "result"
  result_dir = os.path.join(base_dir, DATA_RESULT_PATH)
  name_img = f"{GRAPH_RESULT_NAME}_{training}.png"
  name_acc_concept_img = f"{GRAPH_RESULT_ACC_CONCEPT}_{training}.png"
  result_img = os.path.join(result_dir, name_img)
  result_accC_img = os.path.join(result_dir, name_acc_concept_img)
  graph = Graphs(result_dir, result_img, result_accC_img, training)
  graph.graph()
#   graph.graph_concept()

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("graph")
  parser.add_argument("--training", type=str, default="rs")
  args = parser.parse_args()