import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
MASK_ROOT = ROOT / "results/e2e_classify_v3/masks"
OUTPUT_PATH = ROOT / "results/fig5_1.png"

# 仅用于脚本内部取数，图中不显示病例名。
NORMAL_CASE = "Cao_li_V1"
DVT_CASE = "li_juan_V1E1"
FRAME_COUNT = 38

TEXT = "#253041"
SUBTEXT = "#5B6574"
GRID = "#D8DEE8"
BLUE = "#2F6FED"
RED = "#D84C54"
GRAY_A = "#9AA5B4"
GRAY_B = "#6B7280"
PANEL = "#F8FAFC"
PHASE_BLUE = "#EAF2FF"
PHASE_GRAY = "#F1F5F9"
PHASE_GREEN = "#EEF9F1"


def load_area_curves(case_name: str):
    mask_files = sorted((MASK_ROOT / case_name).glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No masks found for case: {case_name}")

    artery = []
    vein = []
    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        artery.append(int((mask == 1).sum()))
        vein.append(int((mask == 2).sum()))
    return np.asarray(artery), np.asarray(vein)


def infer_normal_phases(vein_curve: np.ndarray):
    close_threshold = max(120, int(vein_curve[0] * 0.04))
    closed_candidates = np.where(vein_curve <= close_threshold)[0]
    if len(closed_candidates) == 0:
        return FRAME_COUNT // 3, FRAME_COUNT // 2

    closed_start = int(closed_candidates[0])
    recovery_start = len(vein_curve) - 1
    for idx in range(closed_start + 1, len(vein_curve)):
        if vein_curve[idx] > close_threshold:
            recovery_start = idx
            break
    return closed_start, recovery_start


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))

    normal_artery, normal_vein = load_area_curves(NORMAL_CASE)
    dvt_artery, dvt_vein = load_area_curves(DVT_CASE)

    normal_artery = normal_artery[:FRAME_COUNT]
    normal_vein = normal_vein[:FRAME_COUNT]
    dvt_artery = dvt_artery[:FRAME_COUNT]
    dvt_vein = dvt_vein[:FRAME_COUNT]
    xs = np.arange(FRAME_COUNT)
    closed_start, recovery_start = infer_normal_phases(normal_vein)
    y_max = max(normal_vein.max(), dvt_vein.max(), normal_artery.max(), dvt_artery.max()) * 1.16

    fig, ax = plt.subplots(figsize=(11.8, 6.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    note = FancyBboxPatch(
        (0.06, 0.90),
        0.88,
        0.055,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.0,
        edgecolor="#D1D5DB",
        facecolor=PANEL,
        transform=fig.transFigure,
    )
    fig.patches.append(note)
    fig.text(
        0.08,
        0.928,
        "匿名样例的完整加压过程对比：正常静脉先塌陷闭合，随后在减压阶段恢复开张；DVT 静脉未出现接近零的闭合，两例动脉面积均相对稳定。",
        ha="left",
        va="center",
        fontsize=11.2,
        color=TEXT,
        fontproperties=font,
    )

    ax.axvspan(-0.5, closed_start - 0.5, color=PHASE_BLUE, alpha=0.55, zorder=0)
    ax.axvspan(closed_start - 0.5, recovery_start - 0.5, color=PHASE_GRAY, alpha=0.58, zorder=0)
    ax.axvspan(recovery_start - 0.5, FRAME_COUNT - 0.5, color=PHASE_GREEN, alpha=0.56, zorder=0)
    ax.axvline(closed_start - 0.5, color="#94A3B8", linewidth=1.0, linestyle="--", zorder=1)
    ax.axvline(recovery_start - 0.5, color="#94A3B8", linewidth=1.0, linestyle="--", zorder=1)

    ax.plot(xs, normal_artery, color=GRAY_A, linewidth=2.0, alpha=0.95, zorder=1)
    ax.plot(xs, dvt_artery, color=GRAY_B, linewidth=2.0, linestyle="--", alpha=0.82, zorder=1)
    ax.plot(
        xs,
        normal_vein,
        color=BLUE,
        linewidth=3.3,
        marker="o",
        markersize=4.6,
        markerfacecolor="white",
        markeredgewidth=1.2,
        zorder=3,
    )
    ax.plot(
        xs,
        dvt_vein,
        color=RED,
        linewidth=3.3,
        marker="o",
        markersize=4.6,
        markerfacecolor="white",
        markeredgewidth=1.2,
        zorder=3,
    )

    ax.text(
        (closed_start - 1) / max(FRAME_COUNT - 1, 1),
        1.03,
        "加压塌陷",
        ha="center",
        va="bottom",
        fontsize=11.2,
        color=SUBTEXT,
        fontproperties=font,
        transform=ax.transAxes,
    )
    ax.text(
        (closed_start + recovery_start) / 2 / max(FRAME_COUNT - 1, 1),
        1.03,
        "完全闭合",
        ha="center",
        va="bottom",
        fontsize=11.2,
        color=SUBTEXT,
        fontproperties=font,
        transform=ax.transAxes,
    )
    ax.text(
        (recovery_start + FRAME_COUNT - 1) / 2 / max(FRAME_COUNT - 1, 1),
        1.03,
        "减压恢复",
        ha="center",
        va="bottom",
        fontsize=11.2,
        color=SUBTEXT,
        fontproperties=font,
        transform=ax.transAxes,
    )

    normal_min_idx = int(np.argmin(normal_vein))
    ax.scatter(
        [normal_min_idx],
        [normal_vein[normal_min_idx]],
        s=60,
        color=BLUE,
        edgecolor="white",
        linewidth=1.3,
        zorder=5,
    )
    ax.annotate(
        "正常静脉接近闭合",
        xy=(normal_min_idx, normal_vein[normal_min_idx]),
        xytext=(normal_min_idx + 2.0, y_max * 0.28),
        fontsize=10.8,
        color=BLUE,
        fontproperties=font,
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5),
    )

    recovery_idx = min(FRAME_COUNT - 1, recovery_start + 7)
    ax.scatter(
        [recovery_idx],
        [normal_vein[recovery_idx]],
        s=60,
        color=BLUE,
        edgecolor="white",
        linewidth=1.3,
        zorder=5,
    )
    ax.annotate(
        "减压后重新开放",
        xy=(recovery_idx, normal_vein[recovery_idx]),
        xytext=(recovery_idx - 5.8, y_max * 0.73),
        fontsize=10.8,
        color=BLUE,
        fontproperties=font,
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5),
    )

    dvt_anchor_idx = 18
    ax.scatter(
        [dvt_anchor_idx],
        [dvt_vein[dvt_anchor_idx]],
        s=60,
        color=RED,
        edgecolor="white",
        linewidth=1.3,
        zorder=5,
    )
    ax.annotate(
        "DVT 静脉未见闭合",
        xy=(dvt_anchor_idx, dvt_vein[dvt_anchor_idx]),
        xytext=(21.5, y_max * 0.88),
        fontsize=10.8,
        color=RED,
        fontproperties=font,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
    )

    legend_handles = [
        Line2D([0], [0], color=BLUE, linewidth=3.3, marker="o", markersize=5.0, markerfacecolor="white", markeredgewidth=1.1, label="正常静脉面积"),
        Line2D([0], [0], color=RED, linewidth=3.3, marker="o", markersize=5.0, markerfacecolor="white", markeredgewidth=1.1, label="DVT 静脉面积"),
        Line2D([0], [0], color=GRAY_A, linewidth=2.1, linestyle="-", label="正常动脉面积"),
        Line2D([0], [0], color=GRAY_B, linewidth=2.1, linestyle="--", label="DVT 动脉面积"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="#D1D5DB",
        facecolor="white",
        fontsize=10.8,
        prop=font,
    )
    legend.get_frame().set_linewidth(0.9)

    ax.set_xlim(-0.5, FRAME_COUNT - 0.5)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("帧序号", fontsize=12.4, color=TEXT, fontproperties=font)
    ax.set_ylabel("面积 / 像素数", fontsize=12.4, color=TEXT, fontproperties=font)
    ax.set_xticks(np.arange(0, FRAME_COUNT, 3))
    ax.tick_params(axis="both", labelsize=10.8, colors=TEXT)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)

    ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.9, alpha=0.85)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.14)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
