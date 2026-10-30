import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Results from the 15-run OptIForest experiments
datasets = ["AD", "Mammography", "Synthetic v2"]

branch0_mean = [0.7737948928, 0.8621728322, 0.9799414286]
branch0_std = [0.0067303880, 0.0031232917, 0.0050363902]

branch2_mean = [0.7718995710, 0.8622895584, 0.9771110053]
branch2_std = [0.0044841902, 0.0031284068, 0.0050461420]

x = np.arange(len(datasets))
width = 0.34

fig, ax = plt.subplots(figsize=(6.5, 3.8))

ax.bar(
    x - width / 2,
    branch0_mean,
    width,
    yerr=branch0_std,
    capsize=4,
    label="Branch 0 (probabilistic)",
    color="#3274A1",
    edgecolor="black",
    linewidth=0.5
)

ax.bar(
    x + width / 2,
    branch2_mean,
    width,
    yerr=branch2_std,
    capsize=4,
    label="Branch 2 (binary)",
    color="#E1812C",
    edgecolor="black",
    linewidth=0.5
)

ax.set_ylabel("Mean ROC-AUC")
ax.set_xlabel("Dataset")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 1.05)
ax.legend(loc="upper left", fontsize=8)
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

fig.tight_layout()

Path("results").mkdir(exist_ok=True)

fig.savefig(
    "results/branch_roc_auc.pdf",
    bbox_inches="tight"
)

fig.savefig(
    "results/branch_roc_auc.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("ROC-AUC comparison graph created successfully.")
print("PDF: results/branch_roc_auc.pdf")
print("PNG: results/branch_roc_auc.png")
