import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
RUN_DIR = ROOT / "sam2/checkpoints/lora_runs/lora_r8_lr0.0003_e25_20260314_153210"
LOG_PATH = RUN_DIR / "training_log.jsonl"

FIG4_3_PATH = ROOT / "results/fig4_3_lora_train_loss.png"
FIG4_4_PATH = ROOT / "results/fig4_4_val_dice_curves.png"

TEXT = "#1F2937"
GRID = "#D1D5DB"
LOSS = "#CC6B2C"
MEAN = "#243B53"
ARTERY = "#D1495B"
VEIN = "#2B6CB0"


def load_history():
    rows = [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]
    train_epochs = np.array([row["epoch"] for row in rows], dtype=float)
    train_loss = np.array([row["train_loss"] for row in rows], dtype=float)

    val_rows = [row for row in rows if "val_mean_dice" in row]
    val_epochs = np.array([row["epoch"] for row in val_rows], dtype=float)
    val_mean = np.array([row["val_mean_dice"] for row in val_rows], dtype=float)
    val_artery = np.array([row["val_artery_dice"] for row in val_rows], dtype=float)
    val_vein = np.array([row["val_vein_dice"] for row in val_rows], dtype=float)
    return train_epochs, train_loss, val_epochs, val_mean, val_artery, val_vein


def style_axis(ax, font):
    ax.grid(True, axis="y", linestyle=(0, (4, 4)), linewidth=0.8, color=GRID, alpha=0.85)
    ax.tick_params(axis="both", labelsize=11, colors=TEXT)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")


def generate_fig4_3(font, train_epochs, train_loss):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    style_axis(ax, font)

    ax.plot(
        train_epochs,
        train_loss,
        color=LOSS,
        linewidth=2.4,
        marker="o",
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.5,
        markeredgecolor=LOSS,
        zorder=3,
    )
    ax.fill_between(train_epochs, train_loss, train_loss.min() - 0.03, color=LOSS, alpha=0.10, zorder=1)

    best_idx = int(np.argmin(train_loss))
    best_epoch = int(train_epochs[best_idx])
    best_loss = float(train_loss[best_idx])
    ax.scatter([best_epoch], [best_loss], s=54, color=LOSS, zorder=4)
    ax.annotate(
        f"最低损失 {best_loss:.3f}",
        xy=(best_epoch, best_loss),
        xytext=(best_epoch - 5.0, best_loss + 0.14),
        textcoords="data",
        fontsize=10.8,
        color=TEXT,
        fontproperties=font,
        arrowprops=dict(arrowstyle="-", color=LOSS, lw=1.2),
    )

    ax.set_xlabel("Epoch", fontsize=12.5, color=TEXT, fontproperties=font)
    ax.set_ylabel("Train Loss", fontsize=12.5, color=TEXT, fontproperties=font)
    ax.set_xlim(1, 25)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    y_min = max(0.45, train_loss.min() - 0.08)
    y_max = train_loss.max() + 0.10
    ax.set_ylim(y_min, y_max)

    fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.13)
    FIG4_3_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG4_3_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def generate_fig4_4(font, val_epochs, val_mean, val_artery, val_vein):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    style_axis(ax, font)

    series = [
        ("Mean Dice", val_mean, MEAN),
        ("Artery Dice", val_artery, ARTERY),
        ("Vein Dice", val_vein, VEIN),
    ]
    for label, values, color in series:
        ax.plot(
            val_epochs,
            values,
            label=label,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=5.0,
            markerfacecolor="white",
            markeredgewidth=1.4,
            markeredgecolor=color,
            zorder=3,
        )

    best_idx = int(np.argmax(val_mean))
    best_epoch = int(val_epochs[best_idx])
    best_mean = float(val_mean[best_idx])
    ax.scatter([best_epoch], [best_mean], s=60, color=MEAN, zorder=4)
    ax.annotate(
        f"最佳 Mean Dice {best_mean:.3f}",
        xy=(best_epoch, best_mean),
        xytext=(9.0, 0.818),
        textcoords="data",
        fontsize=10.6,
        color=TEXT,
        fontproperties=font,
        arrowprops=dict(arrowstyle="-", color=MEAN, lw=1.2),
    )

    legend = ax.legend(loc="lower right", frameon=False, fontsize=10.8)
    for text in legend.get_texts():
        text.set_fontproperties(font)

    ax.set_xlabel("Epoch", fontsize=12.5, color=TEXT, fontproperties=font)
    ax.set_ylabel("Validation Dice", fontsize=12.5, color=TEXT, fontproperties=font)
    ax.set_xlim(4, 26)
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_ylim(0.66, 0.94)

    fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.13)
    FIG4_4_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG4_4_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    train_epochs, train_loss, val_epochs, val_mean, val_artery, val_vein = load_history()
    generate_fig4_3(font, train_epochs, train_loss)
    generate_fig4_4(font, val_epochs, val_mean, val_artery, val_vein)


if __name__ == "__main__":
    main()
