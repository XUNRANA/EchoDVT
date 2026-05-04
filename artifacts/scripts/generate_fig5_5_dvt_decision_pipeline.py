import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
OUTPUT_PATHS = [
    ROOT / "artifacts/figures/fig5_5.png",
    ROOT / "artifacts/figures/fig5_5_dvt_decision_pipeline.png",
]

TEXT = "#253041"
LINE = "#556070"
EDGE = "#C9D3DF"
PALE = "#F8FAFC"
WHITE = "#FFFFFF"
BLUE = "#EAF2FF"
MINT = "#EAFBF2"
LW = 2.2
ARROW_SCALE = 16


def add_box(ax, xy, wh, text, font, fontsize=12.0):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=EDGE,
        facecolor=WHITE,
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
        linespacing=1.18,
    )
    return patch


def add_box_fill(ax, xy, wh, text, font, facecolor, fontsize=12.0):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.020",
        linewidth=1.5,
        edgecolor=EDGE,
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
        linespacing=1.18,
    )
    return patch


def add_box_centered(ax, center, wh, text, font, fontsize=12.0):
    cx, cy = center
    w, h = wh
    return add_box(ax, (cx - w / 2, cy - h / 2), wh, text, font, fontsize=fontsize)


def add_box_fill_centered(ax, center, wh, text, font, facecolor, fontsize=12.0):
    cx, cy = center
    w, h = wh
    return add_box_fill(ax, (cx - w / 2, cy - h / 2), wh, text, font, facecolor, fontsize=fontsize)


def add_chip(ax, xy, wh, text, font, fontsize=10.8):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=1.0,
        edgecolor=EDGE,
        facecolor=WHITE,
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
    )
    return patch


def add_chip_centered(ax, center, wh, text, font, fontsize=10.8):
    cx, cy = center
    w, h = wh
    return add_chip(ax, (cx - w / 2, cy - h / 2), wh, text, font, fontsize=fontsize)


def add_diamond(ax, center, wh, text, font, fontsize=11.8):
    cx, cy = center
    w, h = wh
    pts = [
        (cx, cy + h / 2),
        (cx + w / 2, cy),
        (cx, cy - h / 2),
        (cx - w / 2, cy),
    ]
    patch = Polygon(pts, closed=True, facecolor=WHITE, edgecolor=EDGE, linewidth=1.5)
    ax.add_patch(patch)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT,
        fontproperties=font,
        linespacing=1.16,
    )
    return patch


def add_arrow(ax, start, end, lw=1.8):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ARROW_SCALE,
        linewidth=lw,
        color=LINE,
        shrinkA=0,
        shrinkB=0,
        joinstyle="round",
        capstyle="round",
    )
    ax.add_patch(patch)
    return patch


def add_orth_arrow(ax, points, lw=1.8):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if len(points) > 2:
        ax.plot(
            xs[:-1],
            ys[:-1],
            color=LINE,
            linewidth=lw,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    return add_arrow(ax, points[-2], points[-1], lw=lw)


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))

    fig, ax = plt.subplots(figsize=(24.0, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1.56)
    ax.set_ylim(0, 1)
    ax.axis("off")

    note = FancyBboxPatch(
        (0.05, 0.90),
        1.42,
        0.050,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=EDGE,
        facecolor=PALE,
    )
    ax.add_patch(note)
    ax.text(
        0.08,
        0.924,
        "输入为分割得到的语义掩码序列；主路径优先使用统一 RF 分类器，模型缺失时回退到 VCR 阈值规则。",
        ha="left",
        va="center",
        fontsize=10.6,
        color=TEXT,
        fontproperties=font,
    )

    add_chip_centered(ax, (1.20, 0.84), (0.23, 0.042), "主路径：RF 概率输出 + 阈值判定", font, fontsize=10.1)
    add_chip_centered(ax, (1.20, 0.78), (0.28, 0.042), "降级路径：模型缺失 → VCR fallback", font, fontsize=9.9)

    add_box_fill_centered(ax, (0.15, 0.56), (0.145, 0.105), "分割掩码序列\nsemantic mask[t]\n0=背景，1=动脉，2=静脉", font, BLUE, fontsize=10.5)
    add_box_centered(ax, (0.37, 0.56), (0.14, 0.105), "逐帧面积计算\nartery_area[t]\nvein_area[t]", font, fontsize=10.5)
    add_box_centered(ax, (0.60, 0.56), (0.165, 0.105), "21 维特征提取\nVCR / VDR / vein_cv / VARR\nMVAR / circ_range / 其他特征", font, fontsize=10.1)

    add_diamond(ax, (0.79, 0.56), (0.13, 0.085), "统一模型文件\n存在？", font, fontsize=10.9)
    add_box_centered(ax, (0.99, 0.56), (0.17, 0.105), "StandardScaler\n+\nRF unified\n输出 probability = P(DVT)", font, fontsize=10.1)
    add_diamond(ax, (1.19, 0.56), (0.13, 0.085), "P(DVT) ≥ 0.05 ？", font, fontsize=10.8)

    add_box_fill_centered(ax, (1.01, 0.28), (0.17, 0.095), "VCR fallback\n仅使用静脉压缩比\nVCR > 0.05 ？", font, MINT, fontsize=10.2)
    add_box_fill_centered(ax, (1.40, 0.56), (0.16, 0.17), "最终诊断结论输出\nDVT 疑似 / 正常\nis_dvt / diagnosis\nprobability(若有)\nthreshold / vcr", font, BLUE, fontsize=10.0)
    add_box_centered(ax, (1.41, 0.18), (0.13, 0.085), "离线结果汇总\nper_case_results.csv\nclassification_report.json", font, fontsize=9.8)

    add_arrow(ax, (0.2225, 0.56), (0.30, 0.56), lw=LW)
    add_arrow(ax, (0.44, 0.56), (0.5175, 0.56), lw=LW)
    add_arrow(ax, (0.6825, 0.56), (0.725, 0.56), lw=LW)
    add_arrow(ax, (0.855, 0.56), (0.905, 0.56), lw=LW)
    add_arrow(ax, (1.075, 0.56), (1.125, 0.56), lw=LW)
    add_arrow(ax, (1.255, 0.56), (1.32, 0.56), lw=LW)
    add_arrow(ax, (1.40, 0.475), (1.41, 0.2225), lw=LW)

    add_orth_arrow(ax, [(0.79, 0.5175), (0.79, 0.28), (0.925, 0.28)], lw=LW)
    add_orth_arrow(ax, [(1.095, 0.28), (1.25, 0.28), (1.25, 0.51), (1.32, 0.51)], lw=LW)

    ax.text(0.855, 0.61, "是", ha="center", va="bottom", fontsize=10.0, color=TEXT, fontproperties=font)
    ax.text(0.865, 0.455, "否", ha="left", va="center", fontsize=10.0, color=TEXT, fontproperties=font)
    for output_path in OUTPUT_PATHS:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
