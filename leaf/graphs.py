from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

folder = "./result"

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

for csv_file in Path(folder).glob("*.csv"):
    name = csv_file.stem.lower()

    if not name.startswith("e"):
        continue

    if "train" not in name:
        continue

    if name.startswith("e1"):
        label = 'Unsupervised'
    elif name.startswith("e2"):
        label = 'Supervised'
    elif name.startswith("e3"):
        label = 'Loss function unsupervised'
    elif name.startswith("e4"):
        label = 'Dinov2Model supervised'
    elif name.startswith("e5"):
        label = 'Dinov2Model unsupervised'
    elif name.startswith("e6"):
        label = 'CLIPVisionModel supervised'
    elif name.startswith("e7"):
        label = 'SiglipVisionModel supervised'
    else:
        label = csv_file.stem

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
        df["prob_mod_no"],
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
ax2.set_title("Prob Model No")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("prob_mod_no")
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
plt.show()