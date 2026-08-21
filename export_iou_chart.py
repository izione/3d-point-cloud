import matplotlib.pyplot as plt

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

DATA = {
    "Overall": {
        "color": "#2a78d6",
        "precision": [0.7084, 0.6858, 0.6319, 0.5252, 0.3657, 0.1761, 0.0427, 0.0030, 0.0000],
        "recall":    [0.7363, 0.7129, 0.6568, 0.5459, 0.3802, 0.1830, 0.0444, 0.0031, 0.0000],
    },
    "Person1": {
        "color": "#eb6834",
        "precision": [0.7457, 0.7205, 0.6619, 0.5460, 0.3879, 0.1909, 0.0457, 0.0034, 0.0000],
        "recall":    [0.7207, 0.6963, 0.6397, 0.5277, 0.3748, 0.1845, 0.0441, 0.0033, 0.0000],
    },
    "Person2": {
        "color": "#1baf7a",
        "precision": [0.4591, 0.4545, 0.4318, 0.3864, 0.2182, 0.0773, 0.0227, 0.0000, 0.0000],
        "recall":    [0.9619, 0.9524, 0.9048, 0.8095, 0.4571, 0.1619, 0.0476, 0.0000, 0.0000],
    },
}

PAGE_BG = "#f5f8f9"
CARD_BG = "#ffffff"
GRID = "#dde5e7"
AXIS = "#c4ced0"
TEXT = "#0c1417"
MUTED = "#4c5a5f"

plt.rcParams["font.family"] = "monospace"

fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), facecolor=PAGE_BG)

for ax, metric, title in zip(axes, ("precision", "recall"), ("Precision", "Recall")):
    ax.set_facecolor(CARD_BG)
    for name, d in DATA.items():
        ax.plot(THRESHOLDS, d[metric], color=d["color"], linewidth=2, marker="o",
                 markersize=5.5, markeredgecolor=CARD_BG, markeredgewidth=1.2, label=name)
    ax.set_title(title, color=TEXT, fontsize=15, fontweight="bold", family="sans-serif", loc="left", pad=12)
    ax.set_xlabel("IoU threshold (≥)", color=MUTED, fontsize=10)
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    for spine_name, spine in ax.spines.items():
        spine.set_visible(spine_name in ("left", "bottom"))
        spine.set_color(AXIS)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.995), ncol=3,
           frameon=False, fontsize=12, labelcolor=TEXT, handlelength=1.6, columnspacing=1.4)

fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig("../figures/iou_threshold_sweep.png", dpi=200, facecolor=PAGE_BG)
print("saved to ../figures/iou_threshold_sweep.png")
