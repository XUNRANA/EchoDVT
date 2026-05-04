import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
OUTPUT_PATH = ROOT / "artifacts/figures/fig4_7_rpa_principle.png"

TEXT = "#253041"
LINE = "#5B6574"
GRID = "#D7DEE7"
PALE = "#F8FAFC"
ARTERY = "#E45B6C"
VEIN = "#3D7DD8"
DRIFT = "#E8871E"
GOOD = "#2A9D8F"


def add_box(ax, xy, wh, text, font, facecolor, edgecolor=LINE, fontsize=12.2, lw=1.4):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT,
        fontproperties=font,
        linespacing=1.22,
    )
    return patch


def add_arrow(ax, start, end, color=LINE, lw=1.8, style="-|>", rad=0.0, mutation=13):
    arrow = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arrow)
    return arrow


def add_learning_frame(ax, xy, wh, artery_center, vein_center, font, label):
    x, y = xy
    w, h = wh
    frame = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.015",
        linewidth=1.2,
        edgecolor="#CBD5E1",
        facecolor="#FFFFFF",
    )
    ax.add_patch(frame)

    artery_xy = (x + artery_center[0] * w, y + artery_center[1] * h)
    vein_xy = (x + vein_center[0] * w, y + vein_center[1] * h)

    ax.add_patch(Ellipse(artery_xy, w * 0.25, h * 0.19, angle=6, facecolor=ARTERY, edgecolor="none", alpha=0.92))
    ax.add_patch(Ellipse(vein_xy, w * 0.24, h * 0.17, angle=-18, facecolor=VEIN, edgecolor="none", alpha=0.92))
    ax.add_patch(Circle(artery_xy, radius=w * 0.016, facecolor="#FFFFFF", edgecolor=ARTERY, linewidth=1.1))
    ax.add_patch(Circle(vein_xy, radius=w * 0.016, facecolor="#FFFFFF", edgecolor=VEIN, linewidth=1.1))
    add_arrow(ax, artery_xy, vein_xy, color=GOOD, lw=1.6, mutation=11)
    ax.text(x + w / 2, y - 0.028, label, ha="center", va="top", fontsize=10.6, color=TEXT, fontproperties=font)


def add_stage_panel(ax, xy, wh, title, font, edgecolor, facecolor):
    x, y = xy
    w, h = wh
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.6,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(panel)
    chip = FancyBboxPatch(
        (x + 0.02, y + h - 0.065),
        0.18,
        0.048,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor="#FFFFFF",
    )
    ax.add_patch(chip)
    ax.text(x + 0.11, y + h - 0.041, title, ha="center", va="center", fontsize=12.4, color=TEXT, fontproperties=font)
    return panel


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    fig, ax = plt.subplots(figsize=(15.4, 7.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top_note = FancyBboxPatch(
        (0.05, 0.91),
        0.90,
        0.055,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.0,
        edgecolor="#D1D5DB",
        facecolor=PALE,
    )
    ax.add_patch(top_note)
    ax.text(
        0.07,
        0.938,
        "RPA 不改动 SAM2 的传播过程，而是在输出端利用“动脉稳定、动静脉相对位置稳定”的先验来抑制异常静脉掩码。",
        ha="left",
        va="center",
        fontsize=11.5,
        color=TEXT,
        fontproperties=font,
    )

    add_stage_panel(ax, (0.05, 0.12), (0.38, 0.72), "基线学习阶段", font, "#E8C16D", "#FFFDF6")
    add_stage_panel(ax, (0.48, 0.12), (0.47, 0.72), "逐帧检查与漂移判定", font, "#B8CCE6", "#F8FBFF")

    add_learning_frame(ax, (0.10, 0.55), (0.09, 0.16), (0.52, 0.62), (0.40, 0.34), font, "好帧 1")
    add_learning_frame(ax, (0.22, 0.55), (0.09, 0.16), (0.51, 0.61), (0.39, 0.35), font, "好帧 2")
    add_learning_frame(ax, (0.34, 0.55), (0.09, 0.16), (0.53, 0.60), (0.41, 0.33), font, "好帧 3")

    add_box(
        ax,
        (0.11, 0.38),
        (0.30, 0.085),
        "收集前 3 个“动脉与静脉面积均达标”的好帧",
        font,
        "#FFFFFF",
        edgecolor=GRID,
        fontsize=11.0,
    )
    add_box(
        ax,
        (0.13, 0.23),
        (0.26, 0.095),
        "对每个好帧计算\noffset = p_vein - p_artery",
        font,
        "#EAF7F3",
        edgecolor=GOOD,
        fontsize=11.0,
        lw=1.6,
    )
    add_box(
        ax,
        (0.14, 0.15),
        (0.24, 0.06),
        "取中位数得到 baseline_offset",
        font,
        "#F0F9FF",
        edgecolor="#93C5FD",
        fontsize=11.0,
        lw=1.4,
    )
    add_arrow(ax, (0.26, 0.53), (0.26, 0.465), color=LINE, lw=1.5)
    add_arrow(ax, (0.26, 0.375), (0.26, 0.325), color=LINE, lw=1.5)
    add_arrow(ax, (0.26, 0.225), (0.26, 0.205), color=LINE, lw=1.5)

    # Current-frame check panel
    panel = FancyBboxPatch(
        (0.56, 0.23),
        0.26,
        0.44,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        linewidth=1.2,
        edgecolor="#D1D5DB",
        facecolor="#FFFFFF",
    )
    ax.add_patch(panel)
    ax.text(0.69, 0.69, "当前帧", ha="center", va="bottom", fontsize=12.0, color=TEXT, fontproperties=font)

    artery_center = (0.70, 0.52)
    expected_center = (0.65, 0.36)
    actual_center = (0.79, 0.56)

    ax.add_patch(Ellipse(artery_center, 0.12, 0.09, angle=-4, facecolor=ARTERY, edgecolor="none", alpha=0.95))
    ax.add_patch(Ellipse(actual_center, 0.12, 0.08, angle=28, facecolor=VEIN, edgecolor="none", alpha=0.90))
    ax.add_patch(Ellipse(expected_center, 0.12, 0.08, angle=6, facecolor="none", edgecolor=VEIN, linewidth=2.0, linestyle="--"))

    ax.add_patch(Circle(artery_center, radius=0.009, facecolor="#FFFFFF", edgecolor=ARTERY, linewidth=1.3))
    ax.add_patch(Circle(actual_center, radius=0.009, facecolor="#FFFFFF", edgecolor=VEIN, linewidth=1.3))
    ax.add_patch(Circle(expected_center, radius=0.009, facecolor="#FFFFFF", edgecolor=VEIN, linewidth=1.3, linestyle="--"))

    add_arrow(ax, artery_center, expected_center, color=GOOD, lw=2.0, mutation=12)
    add_arrow(ax, expected_center, actual_center, color=DRIFT, lw=2.1, mutation=12)

    ax.text(0.74, 0.61, "实际静脉位置\n$p_{actual}$", ha="left", va="center", fontsize=11.0, color=TEXT, fontproperties=font)
    ax.text(0.55, 0.34, "期望静脉位置\n$p_{expected}$", ha="left", va="center", fontsize=11.0, color=TEXT, fontproperties=font)
    ax.text(0.69, 0.57, "动脉质心\n锚点", ha="center", va="bottom", fontsize=11.0, color=TEXT, fontproperties=font)
    ax.text(0.62, 0.45, "baseline_offset", ha="center", va="bottom", fontsize=10.8, color=GOOD, fontproperties=font)
    ax.text(0.76, 0.47, "drift", ha="center", va="bottom", fontsize=10.8, color=DRIFT, fontproperties=font)

    add_box(
        ax,
        (0.83, 0.46),
        (0.10, 0.10),
        "drift >\nmax_drift",
        font,
        "#FFF7ED",
        edgecolor=DRIFT,
        fontsize=11.0,
        lw=1.6,
    )
    add_arrow(ax, (0.82, 0.51), (0.82, 0.40), color=DRIFT, lw=1.5)
    add_box(
        ax,
        (0.79, 0.26),
        (0.16, 0.10),
        "异常帧抑制：\nvein mask = 0",
        font,
        "#FEF2F2",
        edgecolor=ARTERY,
        fontsize=11.0,
        lw=1.6,
    )

    add_box(
        ax,
        (0.54, 0.15),
        (0.36, 0.055),
        "p_expected = centroid_artery + offset_baseline",
        font,
        "#FFFFFF",
        edgecolor=GRID,
        fontsize=10.8,
    )
    add_box(
        ax,
        (0.58, 0.08),
        (0.29, 0.055),
        "drift = ||p_expected - p_actual|| / diag(image)",
        font,
        "#FFFFFF",
        edgecolor=GRID,
        fontsize=10.6,
    )

    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
