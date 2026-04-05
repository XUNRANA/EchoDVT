import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score, roc_curve


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
RESULT_CSV = ROOT / "results/e2e_classify_v3/per_case_results.csv"
OUTPUT_PATH = ROOT / "results/fig5_4.png"

THRESHOLD = 0.05

TEXT = "#253041"
SUBTEXT = "#5B6574"
EDGE = "#D1D5DB"
PANEL = "#F8FAFC"
BLUE = "#2F6FED"
RED = "#D84C54"


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    df = pd.read_csv(RESULT_CSV)

    y_true = df["true_label"].astype(int).values
    y_prob = df["pred_prob"].astype(float).values

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    y_pred = (y_prob >= THRESHOLD).astype(int)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    op_fpr = fp / (fp + tn)
    op_tpr = tp / (tp + fn)

    fig, ax = plt.subplots(figsize=(8.9, 7.1))
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
        "基于验证集 76 例的 Leave-One-Out 概率输出绘制 ROC；红点为当前高灵敏度阈值 0.05 对应的工作点。",
        ha="left",
        va="center",
        fontsize=11.0,
        color=TEXT,
        fontproperties=font,
    )

    ax.fill_between(fpr, tpr, color=BLUE, alpha=0.12, zorder=1)
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.8, label=f"Random Forest ROC  (AUC = {auc:.3f})", zorder=3)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", linewidth=1.5, alpha=0.8, zorder=2)

    ax.scatter([op_fpr], [op_tpr], s=86, color=RED, edgecolor="white", linewidth=1.5, zorder=4)
    ax.annotate(
        f"threshold = 0.05\nFPR = {op_fpr:.3f}, TPR = {op_tpr:.3f}",
        xy=(op_fpr, op_tpr),
        xytext=(0.43, 0.84),
        textcoords="axes fraction",
        fontsize=10.8,
        color=RED,
        fontproperties=font,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#F1C3C7", linewidth=1.0),
    )

    ax.text(
        0.62,
        0.18,
        "理想分类器越接近左上角",
        fontsize=10.6,
        color=SUBTEXT,
        fontproperties=font,
        transform=ax.transAxes,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("假阳性率 FPR（1 - 特异度）", fontsize=12.5, color=TEXT, fontproperties=font)
    ax.set_ylabel("真阳性率 TPR（灵敏度）", fontsize=12.5, color=TEXT, fontproperties=font)
    ax.tick_params(axis="both", labelsize=10.8, colors=TEXT)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(font)

    ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.42, color="#CBD5E1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    legend = ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=1.0, edgecolor=EDGE, facecolor="white", prop=font)
    legend.get_frame().set_linewidth(0.9)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.12)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
