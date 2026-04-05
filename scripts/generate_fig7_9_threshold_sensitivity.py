import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
RESULT_CSV = ROOT / "results/e2e_classify_v3/per_case_results.csv"
OUTPUT_PATHS = [
    ROOT / "results/fig7_9.png",
    ROOT / "results/fig7_9_threshold_sensitivity.png",
]

CURRENT_THRESHOLD = 0.05
THRESHOLDS = np.arange(0.01, 0.51, 0.01)

TEXT = "#253041"
SUBTEXT = "#5B6574"
EDGE = "#D1D5DB"
PANEL = "#F8FAFC"
BLUE = "#2F6FED"
ORANGE = "#E67E22"
GREEN = "#1E9E61"
RED = "#D84C54"


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
        fontsize=10.8,
        color=color,
        fontproperties=font,
    )


def compute_threshold_metrics(df: pd.DataFrame):
    rows = []
    y_true = df["true_label"].astype(int).values
    y_prob = df["pred_prob"].astype(float).values
    for thr in THRESHOLDS:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        rows.append(
            {
                "threshold": thr,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "accuracy": accuracy,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
    return pd.DataFrame(rows)


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    df = pd.read_csv(RESULT_CSV)
    metrics = compute_threshold_metrics(df)

    current = metrics.loc[np.isclose(metrics["threshold"], CURRENT_THRESHOLD)].iloc[0]
    best = metrics.sort_values(["accuracy", "sensitivity", "specificity"], ascending=[False, False, False]).iloc[0]

    fig, ax1 = plt.subplots(figsize=(10.0, 7.0))
    fig.patch.set_facecolor("white")
    ax2 = ax1.twinx()

    note = FancyBboxPatch(
        (0.07, 0.91),
        0.86,
        0.06,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.0,
        edgecolor=EDGE,
        facecolor=PANEL,
        transform=fig.transFigure,
    )
    fig.patches.append(note)
    fig.text(
        0.09,
        0.94,
        "阈值从 0.01 扫描到 0.50。左轴显示灵敏度和特异度，右轴显示总体准确率；红色竖线标示当前选用的 0.05 阈值。",
        ha="left",
        va="center",
        fontsize=11.0,
        color=TEXT,
        fontproperties=font,
    )

    x = metrics["threshold"].values
    sens = metrics["sensitivity"].values
    spec = metrics["specificity"].values
    acc = metrics["accuracy"].values

    ax1.plot(x, sens, color=BLUE, linewidth=2.6, marker="o", markersize=3.6, label="灵敏度 Sensitivity", zorder=4)
    ax1.plot(x, spec, color=ORANGE, linewidth=2.6, marker="o", markersize=3.6, label="特异度 Specificity", zorder=4)
    ax2.plot(x, acc, color=GREEN, linewidth=2.8, linestyle="--", marker="s", markersize=3.3, label="准确率 Accuracy", zorder=5)

    ax1.axvline(CURRENT_THRESHOLD, color=RED, linewidth=2.0, linestyle="--", alpha=0.95, zorder=3)
    ax1.scatter([current["threshold"]], [current["sensitivity"]], color=BLUE, s=44, edgecolor="white", linewidth=1.0, zorder=6)
    ax1.scatter([current["threshold"]], [current["specificity"]], color=ORANGE, s=44, edgecolor="white", linewidth=1.0, zorder=6)
    ax2.scatter([current["threshold"]], [current["accuracy"]], color=GREEN, s=48, edgecolor="white", linewidth=1.0, zorder=7)

    ax2.scatter([best["threshold"]], [best["accuracy"]], color=GREEN, s=56, edgecolor="white", linewidth=1.2, zorder=7)
    ax2.annotate(
        f"准确率峰值\nthreshold={best['threshold']:.2f}\nacc={best['accuracy'] * 100:.1f}%",
        xy=(best["threshold"], best["accuracy"]),
        xytext=(0.66, 0.33),
        textcoords="axes fraction",
        fontsize=10.5,
        color=GREEN,
        fontproperties=font,
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4),
        bbox=dict(boxstyle="round,pad=0.34", facecolor="white", edgecolor="#B9E2CF", linewidth=1.0),
    )

    ax1.annotate(
        f"当前阈值 0.05\nSens={current['sensitivity'] * 100:.1f}%\nSpec={current['specificity'] * 100:.1f}%\nAcc={current['accuracy'] * 100:.1f}%",
        xy=(CURRENT_THRESHOLD, 0.72),
        xytext=(0.12, 0.32),
        textcoords="axes fraction",
        fontsize=10.5,
        color=RED,
        fontproperties=font,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.4),
        bbox=dict(boxstyle="round,pad=0.34", facecolor="white", edgecolor="#F1C3C7", linewidth=1.0),
    )

    ax1.set_xlim(0.01, 0.50)
    ax1.set_ylim(0.0, 1.03)
    ax2.set_ylim(0.48, 0.95)

    ax1.set_xlabel("分类阈值", fontsize=12.5, color=TEXT, fontproperties=font)
    ax1.set_ylabel("灵敏度 / 特异度", fontsize=12.5, color=TEXT, fontproperties=font)
    ax2.set_ylabel("总体准确率", fontsize=12.5, color=GREEN, fontproperties=font)

    ax1.set_xticks(np.arange(0.05, 0.51, 0.05))
    ax1.tick_params(axis="x", labelsize=10.5, colors=TEXT)
    ax1.tick_params(axis="y", labelsize=10.5, colors=TEXT)
    ax2.tick_params(axis="y", labelsize=10.5, colors=GREEN)
    for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
        tick.set_fontproperties(font)
    for tick in ax2.get_yticklabels():
        tick.set_fontproperties(font)

    ax1.grid(True, linestyle="--", linewidth=0.9, alpha=0.42, color="#CBD5E1")
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["left"].set_color("#CBD5E1")
    ax1.spines["bottom"].set_color("#CBD5E1")
    ax2.spines["right"].set_color("#B9E2CF")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower center",
        bbox_to_anchor=(0.50, 1.02),
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor=EDGE,
        facecolor="white",
        prop=font,
    )
    legend.get_frame().set_linewidth(0.9)

    add_chip(
        fig,
        (0.10, 0.10),
        (0.24, 0.06),
        f"0.05 阈值: Sens {current['sensitivity'] * 100:.1f}%",
        font,
        "#EEF4FF",
        "#B8CCE6",
        BLUE,
    )
    add_chip(
        fig,
        (0.38, 0.10),
        (0.24, 0.06),
        f"0.05 阈值: Spec {current['specificity'] * 100:.1f}%",
        font,
        "#FFF7ED",
        "#F4C38A",
        ORANGE,
    )
    add_chip(
        fig,
        (0.66, 0.10),
        (0.24, 0.06),
        f"最佳准确率: {best['accuracy'] * 100:.1f}% @ {best['threshold']:.2f}",
        font,
        "#ECFDF3",
        "#B9E2CF",
        GREEN,
    )

    fig.subplots_adjust(left=0.11, right=0.89, top=0.82, bottom=0.20)
    OUTPUT_PATHS[0].parent.mkdir(parents=True, exist_ok=True)
    for output_path in OUTPUT_PATHS:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
