import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")

BASELINE_CSV = ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_091610_lora_r8_baseline/frame_metrics.csv"
MFP_CSV = ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_094056_lora_r8_mfp15/frame_metrics.csv"
OUTPUT_PATHS = [
    ROOT / "artifacts/figures/fig7_5.png",
    ROOT / "artifacts/figures/fig7_5_mfp_dice_decay_curve.png",
]

TEXT = "#253041"
SUBTEXT = "#5B6574"
EDGE = "#D1D5DB"
GRID = "#D8E0EA"
PANEL = "#F8FAFC"
BASELINE_BLUE = "#2F6FED"
MFP_RED = "#D84C54"
ANCHOR_GRAY = "#94A3B8"
LATE_BG = "#F8FAFC"


def load_series(path: Path):
    by_frame = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            by_frame[int(row["frame"])].append(float(row["mean_dice"]))

    xs = np.arange(0, max(by_frame.keys()) + 1, dtype=int)
    sums = np.zeros_like(xs, dtype=float)
    counts = np.zeros_like(xs, dtype=float)
    means = np.full_like(xs, np.nan, dtype=float)

    for frame_idx, values in by_frame.items():
        means[frame_idx] = float(np.mean(values))
        sums[frame_idx] = float(np.sum(values))
        counts[frame_idx] = float(len(values))

    return xs, means, sums, counts


def weighted_smooth(sums: np.ndarray, counts: np.ndarray, radius: int = 4):
    smoothed = np.full_like(sums, np.nan, dtype=float)
    for idx in range(len(sums)):
        lo = max(0, idx - radius)
        hi = min(len(sums), idx + radius + 1)
        total_count = counts[lo:hi].sum()
        if total_count > 0:
            smoothed[idx] = sums[lo:hi].sum() / total_count
    return smoothed


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))

    xs, baseline_mean, baseline_sum, baseline_count = load_series(BASELINE_CSV)
    _, mfp_mean, mfp_sum, mfp_count = load_series(MFP_CSV)
    baseline_smooth = weighted_smooth(baseline_sum, baseline_count, radius=4)
    mfp_smooth = weighted_smooth(mfp_sum, mfp_count, radius=4)

    fig, ax = plt.subplots(figsize=(11.8, 6.7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

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
        "验证集 598 帧按绝对帧号汇总。淡色折线为逐帧均值，深色折线为加权平滑趋势；灰虚线表示 MFP 每隔 15 帧的候选重锚定位置。",
        ha="left",
        va="center",
        fontsize=10.8,
        color=TEXT,
        fontproperties=font,
    )

    ax.axvspan(50, 97.5, color=LATE_BG, alpha=0.95, zorder=0)
    ax.text(
        74,
        0.928,
        "后期帧区间",
        ha="center",
        va="top",
        fontsize=10.5,
        color=SUBTEXT,
        fontproperties=font,
    )

    anchor_positions = [15, 30, 45, 60, 75, 90]
    for idx, x in enumerate(anchor_positions):
        ax.axvline(x, color=ANCHOR_GRAY, linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)
        ax.text(
            x + 0.35,
            0.645,
            f"{x}",
            rotation=90,
            ha="left",
            va="bottom",
            fontsize=9.2,
            color=SUBTEXT,
            fontproperties=font,
        )

    ax.plot(xs, baseline_mean, color=BASELINE_BLUE, linewidth=1.2, alpha=0.22, zorder=2)
    ax.plot(xs, mfp_mean, color=MFP_RED, linewidth=1.2, alpha=0.22, zorder=2)

    ax.plot(xs, baseline_smooth, color=BASELINE_BLUE, linewidth=2.8, zorder=4)
    ax.plot(xs, mfp_smooth, color=MFP_RED, linewidth=2.8, zorder=4)

    ax.scatter(
        xs[baseline_count > 0],
        baseline_mean[baseline_count > 0],
        s=18 + baseline_count[baseline_count > 0] * 0.25,
        color=BASELINE_BLUE,
        alpha=0.28,
        edgecolors="none",
        zorder=3,
    )
    ax.scatter(
        xs[mfp_count > 0],
        mfp_mean[mfp_count > 0],
        s=18 + mfp_count[mfp_count > 0] * 0.25,
        color=MFP_RED,
        alpha=0.28,
        edgecolors="none",
        zorder=3,
    )

    ax.annotate(
        "仅首帧提示在中后段波动更大",
        xy=(54, baseline_smooth[54]),
        xytext=(39, 0.875),
        fontsize=10.2,
        color=BASELINE_BLUE,
        fontproperties=font,
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color=BASELINE_BLUE, lw=1.3),
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#BDD6FF", linewidth=1.0),
        zorder=6,
    )
    ax.annotate(
        "MFP 在 20-60 帧附近整体抬高曲线",
        xy=(45, mfp_smooth[45]),
        xytext=(21, 0.902),
        fontsize=10.2,
        color=MFP_RED,
        fontproperties=font,
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color=MFP_RED, lw=1.3),
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#F3C6CB", linewidth=1.0),
        zorder=6,
    )

    ax.set_xlim(0, 97)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel("Frame Index", fontsize=12.5, color=TEXT, fontproperties=font)
    ax.set_ylim(0.62, 0.94)
    ax.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    ax.set_ylabel("Mean Dice", fontsize=12.5, color=TEXT, fontproperties=font)

    ax.tick_params(axis="both", labelsize=10.8, colors=TEXT)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(font)

    ax.grid(axis="y", linestyle="--", linewidth=0.9, alpha=0.8, color=GRID, zorder=1)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE)
    ax.spines["bottom"].set_color(EDGE)

    legend_handles = [
        Line2D([0], [0], color=BASELINE_BLUE, linewidth=2.8, label="LoRA r8（仅首帧提示）"),
        Line2D([0], [0], color=MFP_RED, linewidth=2.8, label="LoRA r8 + MFP"),
        Line2D([0], [0], color=ANCHOR_GRAY, linewidth=1.2, linestyle="--", label="MFP 候选重锚定帧"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor=EDGE,
        facecolor="white",
        prop=font,
    )
    legend.get_frame().set_linewidth(0.9)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.12)
    OUTPUT_PATHS[0].parent.mkdir(parents=True, exist_ok=True)
    for output_path in OUTPUT_PATHS:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
