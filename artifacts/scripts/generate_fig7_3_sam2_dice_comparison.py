import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
OUTPUT_PATHS = [
    ROOT / "artifacts/figures/fig7_3.png",
    ROOT / "artifacts/figures/fig7_3_sam2_dice_comparison.png",
]

CONFIGS = [
    (
        "Baseline",
        ROOT / "sam2/predictions/sam2_large_yolo_box/val_20260314_111327/summary.json",
    ),
    (
        "LoRA r4",
        ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_034048_lora_r4/summary.json",
    ),
    (
        "LoRA r8",
        ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_091610_lora_r8_baseline/summary.json",
    ),
    (
        "LoRA r8 + MFP",
        ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_094056_lora_r8_mfp15/summary.json",
    ),
]

TEXT = "#253041"
SUBTEXT = "#5B6574"
EDGE = "#D1D5DB"
GRID = "#D8E0EA"
PANEL = "#F8FAFC"
HIGHLIGHT_BG = "#FFF3E6"
HIGHLIGHT_EDGE = "#E58A2B"
MEAN = "#394A6D"
ARTERY = "#D55C6A"
VEIN = "#2F6FED"
MEAN_SOFT = "#C8D1E3"
ARTERY_SOFT = "#F0C5CC"
VEIN_SOFT = "#C5D8FF"


def load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text())
    metrics = data["global_frame_weighted_metrics"]
    return {
        "mean_dice": float(metrics["mean_dice"]),
        "artery_dice": float(metrics["artery_dice"]),
        "vein_dice": float(metrics["vein_dice"]),
        "processed_frames": int(data["processed_frames"]),
    }


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))

    labels = []
    values = []
    processed_frames = None
    for label, path in CONFIGS:
        metrics = load_metrics(path)
        labels.append(label)
        values.append([metrics["mean_dice"], metrics["artery_dice"], metrics["vein_dice"]])
        processed_frames = metrics["processed_frames"]

    data = np.asarray(values, dtype=float)
    x = np.arange(len(labels), dtype=float)
    width = 0.22

    fig, ax = plt.subplots(figsize=(11.2, 6.6))
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
        f"验证集 76 例、{processed_frames} 帧；采用 frame-weighted 口径统计 Mean / Artery / Vein Dice。LoRA r8 + MFP 为当前最佳配置。",
        ha="left",
        va="center",
        fontsize=10.9,
        color=TEXT,
        fontproperties=font,
    )

    soft_palette = [MEAN_SOFT, ARTERY_SOFT, VEIN_SOFT]
    strong_palette = [MEAN, ARTERY, VEIN]
    metric_names = ["Mean Dice", "Artery Dice", "Vein Dice"]
    offsets = np.array([-width, 0.0, width])

    ax.axvspan(x[-1] - 0.48, x[-1] + 0.48, color=HIGHLIGHT_BG, alpha=0.92, zorder=0)
    ax.axvline(x[-1] - 0.48, color=HIGHLIGHT_EDGE, linewidth=1.0, alpha=0.45, zorder=1)
    ax.axvline(x[-1] + 0.48, color=HIGHLIGHT_EDGE, linewidth=1.0, alpha=0.45, zorder=1)

    containers = []
    for metric_idx, offset in enumerate(offsets):
        colors = [soft_palette[metric_idx], soft_palette[metric_idx], soft_palette[metric_idx], strong_palette[metric_idx]]
        edgecolors = [EDGE, EDGE, EDGE, strong_palette[metric_idx]]
        linewidths = [1.0, 1.0, 1.0, 1.6]
        bars = ax.bar(
            x + offset,
            data[:, metric_idx],
            width=width * 0.92,
            color=colors,
            edgecolor=edgecolors,
            linewidth=linewidths,
            zorder=3,
        )
        containers.append(bars)

    for metric_idx, bars in enumerate(containers):
        for cfg_idx, bar in enumerate(bars):
            value = data[cfg_idx, metric_idx]
            label_color = strong_palette[metric_idx] if cfg_idx == 3 else TEXT
            weight = "bold" if cfg_idx == 3 else "normal"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10.0,
                color=label_color,
                fontproperties=font,
                fontweight=weight,
                zorder=5,
            )

    ax.text(
        x[-1],
        0.884,
        "推荐配置",
        ha="center",
        va="bottom",
        fontsize=10.8,
        color=HIGHLIGHT_EDGE,
        fontproperties=font,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#FFF8EF",
            edgecolor="#F3C28D",
            linewidth=1.0,
        ),
        zorder=6,
    )

    ax.annotate(
        "MFP 主要改善静脉传播稳定性\nVein Dice: 0.7029 -> 0.7166",
        xy=(x[-1] + offsets[2], data[-1, 2]),
        xytext=(2.10, 0.60),
        fontsize=10.2,
        color=VEIN,
        fontproperties=font,
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color=VEIN, lw=1.4),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#C5D8FF", linewidth=1.0),
        zorder=6,
    )

    ax.annotate(
        "LoRA 微调显著抬升\n整体 Mean Dice",
        xy=(x[1] - offsets[0], data[1, 0]),
        xytext=(0.18, 0.585),
        fontsize=10.0,
        color=SUBTEXT,
        fontproperties=font,
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=1.2),
        zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.4, color=TEXT, fontproperties=font)
    xticks = ax.get_xticklabels()
    xticks[-1].set_color(HIGHLIGHT_EDGE)

    ax.set_ylabel("Dice Coefficient", fontsize=12.5, color=TEXT, fontproperties=font)
    ax.set_ylim(0.56, 0.89)
    ax.set_yticks([0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
    ax.tick_params(axis="y", labelsize=10.8, colors=TEXT)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font)

    ax.grid(axis="y", linestyle="--", linewidth=0.9, alpha=0.8, color=GRID, zorder=1)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE)
    ax.spines["bottom"].set_color(EDGE)

    legend_handles = [
        Line2D([0], [0], color=MEAN, lw=8, label="Mean Dice"),
        Line2D([0], [0], color=ARTERY, lw=8, label="Artery Dice"),
        Line2D([0], [0], color=VEIN, lw=8, label="Vein Dice"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.885),
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor=EDGE,
        facecolor="white",
        prop=font,
    )
    legend.get_frame().set_linewidth(0.9)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.14)
    OUTPUT_PATHS[0].parent.mkdir(parents=True, exist_ok=True)
    for output_path in OUTPUT_PATHS:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
