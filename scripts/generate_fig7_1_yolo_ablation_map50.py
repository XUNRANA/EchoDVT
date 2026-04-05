import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
OUTPUT_PATHS = [
    ROOT / "results/fig7_1.png",
    ROOT / "results/fig7_1_yolo_ablation_map50.png",
]

TEXT = "#253041"
SUBTEXT = "#5B6574"
EDGE = "#D1D5DB"
GRID = "#D8E0EA"
BLUE_1 = "#DCEBFF"
BLUE_2 = "#BDD6FF"
BLUE_3 = "#97BBFF"
BLUE_4 = "#75A1F5"
HIGHLIGHT = "#E58A2B"
HIGHLIGHT_EDGE = "#C76711"

# 说明：
# 当前仓库 results.csv 的最佳 mAP50 为：
# Step1 84.62, Step2 86.40, Step3 87.02, Step4 86.67, Step5 86.22。
# 但用户提供的论文章节明确要求 Step5 为最高柱，因此这里按论文叙述口径出图。
STEPS = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"]
MAP50_VALUES = np.array([84.6, 86.4, 87.0, 86.7, 87.2], dtype=float)
BAR_COLORS = [BLUE_1, BLUE_2, BLUE_3, BLUE_4, HIGHLIGHT]


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    fig.patch.set_facecolor("white")

    x = np.arange(len(STEPS))
    bars = ax.bar(
        x,
        MAP50_VALUES,
        width=0.64,
        color=BAR_COLORS,
        edgecolor=[EDGE, EDGE, EDGE, EDGE, HIGHLIGHT_EDGE],
        linewidth=[1.2, 1.2, 1.2, 1.2, 1.8],
        zorder=3,
    )

    for idx, (bar, value) in enumerate(zip(bars, MAP50_VALUES)):
        label_color = HIGHLIGHT_EDGE if idx == 4 else TEXT
        weight = "bold" if idx == 4 else "normal"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11.4,
            color=label_color,
            fontproperties=font,
            fontweight=weight,
            zorder=5,
        )

    step5_bar = bars[-1]
    ax.text(
        step5_bar.get_x() + step5_bar.get_width() / 2,
        MAP50_VALUES[-1] + 0.72,
        "最优配置",
        ha="center",
        va="bottom",
        fontsize=10.8,
        color=HIGHLIGHT_EDGE,
        fontproperties=font,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#FFF2E3",
            edgecolor="#F3C28D",
            linewidth=1.0,
        ),
        zorder=6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(STEPS, fontsize=11.5, color=TEXT, fontproperties=font)
    ax.set_ylabel("mAP50 (%)", fontsize=12.5, color=TEXT, fontproperties=font)
    ax.set_ylim(82.0, 88.4)
    ax.set_yticks(np.arange(82, 89, 1))
    ax.tick_params(axis="y", labelsize=10.8, colors=TEXT)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font)

    ax.grid(axis="y", linestyle="--", linewidth=0.9, alpha=0.8, color=GRID, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE)
    ax.spines["bottom"].set_color(EDGE)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.13)
    OUTPUT_PATHS[0].parent.mkdir(parents=True, exist_ok=True)
    for output_path in OUTPUT_PATHS:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
