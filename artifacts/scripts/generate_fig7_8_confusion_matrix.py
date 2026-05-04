import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.metrics import confusion_matrix


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
RESULT_CSV = ROOT / "artifacts/e2e_classify_v3/per_case_results.csv"
OUTPUT_PATHS = [
    ROOT / "artifacts/figures/fig7_8.png",
    ROOT / "artifacts/figures/fig7_8_confusion_matrix.png",
]

THRESHOLD = 0.05

TEXT = "#253041"
SUBTEXT = "#5B6574"
EDGE = "#D1D5DB"
PANEL = "#F8FAFC"
BLUE_LOW = "#E7F0FF"
BLUE_HIGH = "#2F6FED"
ORANGE_LOW = "#FFF1E7"
ORANGE_HIGH = "#E67E22"


def lerp_color(hex_a: str, hex_b: str, t: float):
    a = np.asarray(colors.to_rgb(hex_a))
    b = np.asarray(colors.to_rgb(hex_b))
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * a + t * b


def add_chip(fig, xy, wh, text, font, facecolor, edgecolor, color):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=fig.transFigure,
    )
    fig.patches.append(patch)
    fig.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=11.0,
        color=color,
        fontproperties=font,
    )


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    df = pd.read_csv(RESULT_CSV)
    pred = (df["pred_prob"] >= THRESHOLD).astype(int)

    cm = confusion_matrix(df["true_label"], pred, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = cm / row_sums

    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    accuracy = (tn + tp) / cm.sum()
    specificity = tn / row_sums[0, 0]
    sensitivity = tp / row_sums[1, 0]

    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    fig.patch.set_facecolor("white")

    note = FancyBboxPatch(
        (0.08, 0.91),
        0.84,
        0.06,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.0,
        edgecolor=EDGE,
        facecolor=PANEL,
        transform=fig.transFigure,
    )
    fig.patches.append(note)
    fig.text(
        0.10,
        0.94,
        "验证集共 76 例（正常 38，DVT 38），采用阈值 0.05 的高灵敏度判定；格内百分比按每个真实类别内占比计算。",
        ha="left",
        va="center",
        fontsize=11.0,
        color=TEXT,
        fontproperties=font,
    )

    for r in range(2):
        for c in range(2):
            pct = float(row_pct[r, c])
            is_correct = r == c
            face = lerp_color(BLUE_LOW, BLUE_HIGH, pct) if is_correct else lerp_color(ORANGE_LOW, ORANGE_HIGH, pct)
            rect = Rectangle((c, r), 1.0, 1.0, facecolor=face, edgecolor="white", linewidth=3.0)
            ax.add_patch(rect)

            value = int(cm[r, c])
            txt_color = "white" if pct >= 0.62 else TEXT
            ax.text(
                c + 0.5,
                r + 0.42,
                f"{value}",
                ha="center",
                va="center",
                fontsize=25,
                color=txt_color,
                fontproperties=font,
                fontweight="bold",
            )
            ax.text(
                c + 0.5,
                r + 0.67,
                f"{pct * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=13.5,
                color=txt_color,
                fontproperties=font,
            )

    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["正常", "DVT"], fontsize=13, color=TEXT, fontproperties=font)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["正常", "DVT"], fontsize=13, color=TEXT, fontproperties=font)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("预测标签", fontsize=13.5, color=TEXT, fontproperties=font, pad=16)
    fig.text(0.075, 0.54, "真实标签", rotation=90, ha="center", va="center", fontsize=13.5, color=TEXT, fontproperties=font)

    add_chip(fig, (0.12, 0.12), (0.22, 0.06), f"Accuracy  {accuracy * 100:.1f}%", font, "#F8FAFC", EDGE, TEXT)
    add_chip(fig, (0.39, 0.12), (0.22, 0.06), f"Specificity  {specificity * 100:.1f}%", font, "#FFF7ED", "#F4C38A", "#C26B18")
    add_chip(fig, (0.66, 0.12), (0.22, 0.06), f"Sensitivity  {sensitivity * 100:.1f}%", font, "#EEF4FF", "#B8CCE6", "#1D4ED8")

    fig.text(
        0.50,
        0.06,
        f"混淆矩阵计数：TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        ha="center",
        va="center",
        fontsize=10.8,
        color=SUBTEXT,
        fontproperties=font,
    )

    fig.subplots_adjust(left=0.16, right=0.96, top=0.84, bottom=0.22)
    OUTPUT_PATHS[0].parent.mkdir(parents=True, exist_ok=True)
    for output_path in OUTPUT_PATHS:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
