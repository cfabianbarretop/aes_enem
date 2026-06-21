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

# ==============================================
# GRAPHS
# ==============================================
class Graphs():
    def __init__(self, root: str, img: str, training: str):
        self.result_dir = root
        self.result_img= img
        self.training = training
    
    def graph(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        for csv_file in Path(self.result_dir).glob("*.csv"):
            name = csv_file.stem.lower()
            label = csv_file.stem

            if "train" not in name:
                continue

            df = pd.read_csv(csv_file)

            # Gráfico de Accuracy
            ax1.plot(
                df["epoch"],
                df["acc"],
                marker="o",
                label=label
            )

            # Gráfico de prob_mod_no
            ax2.plot(
                df["epoch"],
                df["loss"],
                marker="o",
                label=label
            )

            if {"epoch", "RSR"}.issubset(df.columns):
                line_rsr, = ax3.plot(
                    df["epoch"],
                    df["RSR"],
                    marker="o",
                    label=f"{label} - RSR"
                )
            
            if {"epoch", "RSRw"}.issubset(df.columns):
                ax3.plot(
                    df["epoch"],
                    df["RSRw"],
                    color = line_rsr.get_color(),
                    marker="s",
                    linestyle="--",
                    label=f"{label} - RSRw"
                )

            if {"epoch", "GAcc"}.issubset(df.columns):
                ax4.plot(
                    df["epoch"],
                    df["GAcc"],
                    marker="o",
                    label=label
                )

        # Configuración gráfico 1
        ax1.set_title("Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Acc (%)")
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
        ax4.set_title("GAcc")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("gacc")
        ax4.grid(True)
        ax4.legend()

        plt.tight_layout()
        plt.savefig(self.result_img, dpi=300, bbox_inches="tight")
        plt.show()

def main_graph(training="rs"):
  # Obtiene el directorio donde está este archivo.py
  base_dir = os.path.dirname(os.path.abspath(__file__))
  # Une el directorio de base_dir con las carpetas "data" y "result"
  result_dir = os.path.join(base_dir, DATA_RESULT_PATH)
  name_img = f"{GRAPH_RESULT_NAME}_{training}.png"
  result_img = os.path.join(result_dir, name_img)
  graph = Graphs(result_dir, result_img, training)
  graph.graph()

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("graph")
  parser.add_argument("--training", type=str, default="rs")
  args = parser.parse_args()