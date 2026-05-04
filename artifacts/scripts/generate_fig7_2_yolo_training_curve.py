import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
RESULT_CSV = ROOT / "yolo/runs/detect/runs/detect/dvt_runs/aug_step5_speckle_translate_scale/results.csv"
OUTPUT_PATHS = [
    ROOT / "artifacts/figures/fig7_2.png",
    ROOT / "artifacts/figures/fig7_2_yolo_training_curve.png",
]

TEXT = "#253041"
SUBTEXT = "#5B6574"
EDGE = "#D1D5DB"
GRID = "#D8E0EA"
MAP_BLUE = "#2F6FED"
PREC_ORANGE = "#E58A2B"
RECALL_GREEN = "#2FA36B"
LOSS_GRAY = "#6B7280"
PHASE_LIGHT = "#EEF4FF"
PHASE_MID = "#FFF5E8"
PHASE_FINE = "#EDF9F1"


def load_results() -> pd.DataFrame:
    header = None
    rows = []

    with RESULT_CSV.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for raw_row in reader:
            if not raw_row:
                continue
            if header is None:
                header = raw_row
                continue
            if not raw_row[0].strip().isdigit():
                continue
            if len(raw_row) != len(header):
                continue
            rows.append(raw_row)

    if header is None or not rows:
        raise RuntimeError(f"Failed to parse training results from {RESULT_CSV}")

    df = pd.DataFrame(rows, columns=header)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["epoch"]).sort_values("epoch").reset_index(drop=True)
    df = df[df["epoch"].between(1, 50)].copy()
    if len(df) != 50:
        raise RuntimeError(f"Expected 50 epochs, got {len(df)} from {RESULT_CSV}")
    return df


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    df = load_results()

    epochs = df["epoch"].to_numpy(dtype=int)
    precision = df["metrics/precision(B)"].to_numpy(dtype=float) * 100.0
    recall = df["metrics/recall(B)"].to_numpy(dtype=float) * 100.0
    map50 = df["metrics/mAP50(B)"].to_numpy(dtype=float) * 100.0
    train_loss = (
        df["train/box_loss"].to_numpy(dtype=float)
        + df["train/cls_loss"].to_numpy(dtype=float)
        + df["train/dfl_loss"].to_numpy(dtype=float)
    )

    best_idx = int(np.argmax(map50))
    best_epoch = int(epochs[best_idx])
    best_map = float(map50[best_idx])

    fig, ax = plt.subplots(figsize=(11.4, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axvspan(1, 20, color=PHASE_LIGHT, alpha=0.88, zorder=0)
    ax.axvspan(20, 40, color=PHASE_MID, alpha=0.88, zorder=0)
    ax.axvspan(40, 50, color=PHASE_FINE, alpha=0.92, zorder=0)

    ax.axvline(20, color="#94A3B8", linestyle="--", linewidth=1.2, zorder=1)
    ax.axvline(40, color="#94A3B8", linestyle="--", linewidth=1.2, zorder=1)

    ax_loss = ax.twinx()
    ax_loss.plot(
        epochs,
        train_loss,
        color=LOSS_GRAY,
        linewidth=2.2,
        linestyle="--",
        alpha=0.9,
        zorder=2,
    )

    ax.plot(epochs, precision, color=PREC_ORANGE, linewidth=2.3, zorder=4)
    ax.plot(epochs, recall, color=RECALL_GREEN, linewidth=2.3, zorder=4)
    ax.plot(epochs, map50, color=MAP_BLUE, linewidth=2.8, zorder=5)

    ax.scatter(
        [best_epoch],
        [best_map],
        s=76,
        color=MAP_BLUE,
        edgecolor="white",
        linewidth=1.6,
        zorder=6,
    )
    ax.annotate(
        "最佳 mAP50\nEpoch 29: 86.2%\nP=82.9%, R=83.1%",
        xy=(best_epoch, best_map),
        xytext=(23.0, 88.1),
        fontsize=10.4,
        color=MAP_BLUE,
        fontproperties=font,
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="->", color=MAP_BLUE, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#C8D8FF", linewidth=1.0),
    )

    ax.scatter(
        [50],
        [map50[-1]],
        s=58,
        color=MAP_BLUE,
        edgecolor="white",
        linewidth=1.4,
        zorder=6,
    )
    ax.annotate(
        "最终 Epoch 50\nmAP50=85.8%\nP=86.1%, R=81.6%",
        xy=(50, map50[-1]),
        xytext=(38.0, 69.3),
        fontsize=10.2,
        color=TEXT,
        fontproperties=font,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=EDGE, linewidth=1.0),
    )

    phase_y = 88.55
    ax.text(
        10.5,
        phase_y,
        "轻度噪声 0.03",
        ha="center",
        va="top",
        fontsize=10.6,
        color=SUBTEXT,
        fontproperties=font,
        zorder=7,
    )
    ax.text(
        30.0,
        phase_y,
        "中度噪声 0.06",
        ha="center",
        va="top",
        fontsize=10.6,
        color=SUBTEXT,
        fontproperties=font,
        zorder=7,
    )
    ax.text(
        45.0,
        phase_y,
        "轻度噪声 0.03",
        ha="center",
        va="top",
        fontsize=10.6,
        color=SUBTEXT,
        fontproperties=font,
        zorder=7,
    )
    ax.text(
        20.2,
        66.0,
        "40% 切换点",
        ha="left",
        va="bottom",
        fontsize=9.8,
        color=SUBTEXT,
        fontproperties=font,
        rotation=90,
    )
    ax.text(
        40.2,
        66.0,
        "80% 切换点",
        ha="left",
        va="bottom",
        fontsize=9.8,
        color=SUBTEXT,
        fontproperties=font,
        rotation=90,
    )

    ax.set_xlim(1, 50)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xlabel("Epoch", fontsize=12.4, color=TEXT, fontproperties=font)

    ax.set_ylim(64.0, 89.0)
    ax.set_yticks([65, 70, 75, 80, 85])
    ax.set_ylabel("Precision / Recall / mAP50 (%)", fontsize=12.4, color=TEXT, fontproperties=font)

    ax_loss.set_ylim(1.2, 5.6)
    ax_loss.set_yticks([1.5, 2.5, 3.5, 4.5, 5.5])
    ax_loss.set_ylabel("训练损失（box + cls + dfl）", fontsize=12.2, color=TEXT, fontproperties=font)

    ax.tick_params(axis="both", labelsize=10.8, colors=TEXT)
    ax_loss.tick_params(axis="y", labelsize=10.4, colors=TEXT)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(font)
    for tick in ax_loss.get_yticklabels():
        tick.set_fontproperties(font)

    ax.grid(axis="y", linestyle="--", linewidth=0.9, alpha=0.8, color=GRID, zorder=1)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax_loss.spines["top"].set_visible(False)
    ax.spines["left"].set_color(EDGE)
    ax.spines["bottom"].set_color(EDGE)
    ax_loss.spines["right"].set_color(EDGE)

    handles = [
        Line2D([0], [0], color=PREC_ORANGE, linewidth=2.4, label="Precision"),
        Line2D([0], [0], color=RECALL_GREEN, linewidth=2.4, label="Recall"),
        Line2D([0], [0], color=MAP_BLUE, linewidth=2.8, label="mAP50"),
        Line2D([0], [0], color=LOSS_GRAY, linewidth=2.2, linestyle="--", label="训练损失"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor=EDGE,
        facecolor="white",
        prop=font,
    )
    legend.get_frame().set_linewidth(0.9)

    fig.subplots_adjust(left=0.10, right=0.89, top=0.95, bottom=0.12)
    OUTPUT_PATHS[0].parent.mkdir(parents=True, exist_ok=True)
    for output_path in OUTPUT_PATHS:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
