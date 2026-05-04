import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
OUTPUT_PATH = ROOT / "artifacts/figures/fig4_5_mfp_principle.png"

TEXT = "#253041"
LINE = "#5B6574"
GRID = "#D7DEE7"
PROMPT = "#E8871E"
ANCHOR = "#2A9D8F"
MEMORY = "#4F6D8A"
ARTERY = "#E45B6C"
VEIN = "#3D7DD8"
PALE = "#F8FAFC"


def add_round_box(ax, xy, wh, text, font, facecolor, edgecolor=LINE, fontsize=13, lw=1.6):
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


def add_arrow(ax, start, end, color=LINE, lw=1.8, style="-|>", rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arrow)
    return arrow


def add_frame_icon(ax, center, width=0.08, height=0.10):
    x = center[0] - width / 2
    y = center[1] - height / 2
    outer = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.4,
        edgecolor="#9AA5B1",
        facecolor="#EEF2F6",
    )
    ax.add_patch(outer)
    ax.add_patch(Rectangle((x + 0.013, y + 0.020), width * 0.27, height * 0.34, linewidth=1.5, edgecolor=ARTERY, facecolor="none"))
    ax.add_patch(Rectangle((x + width * 0.52, y + height * 0.42), width * 0.24, height * 0.29, linewidth=1.5, edgecolor=VEIN, facecolor="none"))
    return outer


def add_tick(ax, x, y0, y1, label, font):
    ax.plot([x, x], [y0, y1], color="#9CA3AF", linewidth=1.2)
    ax.text(x, y0 - 0.035, label, ha="center", va="top", fontsize=11.2, color=TEXT, fontproperties=font)


def add_chip(ax, xy, wh, text, font):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=1.0,
        edgecolor="#D1D5DB",
        facecolor="#FFFFFF",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.8, color=TEXT, fontproperties=font)


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    fig, ax = plt.subplots(figsize=(15.2, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Top process note
    note = FancyBboxPatch(
        (0.07, 0.90),
        0.86,
        0.055,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.0,
        edgecolor="#D1D5DB",
        facecolor=PALE,
    )
    ax.add_patch(note)
    ax.text(
        0.09,
        0.928,
        "推理前先在多个关键帧上注入 box prompt，然后由 SAM2 一次性执行全视频记忆传播。",
        ha="left",
        va="center",
        fontsize=11.6,
        color=TEXT,
        fontproperties=font,
    )

    # Left lane labels
    ax.text(0.06, 0.73, "YOLO检测帧", ha="right", va="center", fontsize=12.2, color=TEXT, fontproperties=font)
    ax.text(0.06, 0.50, "条件帧", ha="right", va="center", fontsize=12.2, color=TEXT, fontproperties=font)
    ax.text(0.06, 0.27, "记忆传播", ha="right", va="center", fontsize=12.2, color=TEXT, fontproperties=font)

    xs = [0.15, 0.37, 0.59, 0.81]
    tick_labels = ["Frame 0", "Frame 15", "Frame 30", "Frame 45"]

    # Top prompt frames
    top_titles = ["首帧 Prompt", "重锚定 Prompt", "重锚定 Prompt", "重锚定 Prompt"]
    top_subs = ["YOLO box", "YOLO box", "YOLO box", "YOLO box"]
    for x, title, sub in zip(xs, top_titles, top_subs):
        add_frame_icon(ax, (x, 0.72), width=0.082, height=0.102)
        ax.text(x, 0.638, title, ha="center", va="top", fontsize=11.2, color=TEXT, fontproperties=font)
        ax.text(x, 0.612, sub, ha="center", va="top", fontsize=10.1, color=LINE, fontproperties=font)

    # Middle conditioning frames
    for idx, x in enumerate(xs):
        if idx == 0:
            add_round_box(ax, (x - 0.074, 0.43), (0.148, 0.082), "首帧条件帧\nBox Prompt 注入", font, "#FFF4E8", edgecolor=PROMPT, fontsize=11.0, lw=1.8)
        else:
            add_round_box(ax, (x - 0.072, 0.43), (0.144, 0.082), "额外条件帧\nYOLO重锚定注入", font, "#EAF7F3", edgecolor=ANCHOR, fontsize=11.0, lw=1.8)
        add_arrow(ax, (x, 0.665), (x, 0.515), color="#8795A1", lw=1.4)

    # Confidence rule box
    conf_box = FancyBboxPatch(
        (0.23, 0.80),
        0.54,
        0.050,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=1.0,
        edgecolor="#D1D5DB",
        facecolor="#FFFDF6",
    )
    ax.add_patch(conf_box)
    ax.text(
        0.50,
        0.825,
        "候选帧按 interval = 15 采样；仅当 artery 与 vein 置信度均 ≥ 0.3 时才保留；最多额外注入 5 个条件帧。",
        ha="center",
        va="center",
        fontsize=10.5,
        color=TEXT,
        fontproperties=font,
    )

    # Bottom memory propagation intervals
    intervals = [
        (xs[0], xs[1], "中间帧 1-14\n基于记忆传播"),
        (xs[1], xs[2], "中间帧 16-29\n重新锚定后继续传播"),
        (xs[2], xs[3], "中间帧 31-44\n重新锚定后继续传播"),
        (xs[3], 0.91, "后续帧 46-T\n继续传播"),
    ]
    for x0, x1, label in intervals:
        arrow = FancyArrowPatch(
            (x0 + 0.025, 0.24),
            (x1 - 0.025, 0.24),
            arrowstyle="Simple,head_width=10,head_length=12,tail_width=6",
            linewidth=0,
            facecolor=MEMORY,
            edgecolor=MEMORY,
            alpha=0.78,
        )
        ax.add_patch(arrow)
        mid_x = (x0 + x1) / 2
        ax.text(mid_x, 0.292, label, ha="center", va="bottom", fontsize=10.6, color=TEXT, fontproperties=font, linespacing=1.18)

    ax.text(
        0.50,
        0.155,
        "全部条件帧注入完成后，再统一执行 propagate_in_video(start_frame_idx=0)",
        ha="center",
        va="center",
        fontsize=10.6,
        color=TEXT,
        fontproperties=font,
    )

    # Timeline
    axis_y = 0.10
    ax.plot([0.10, 0.92], [axis_y, axis_y], color=LINE, linewidth=1.7)
    add_arrow(ax, (0.92, axis_y), (0.95, axis_y), color=LINE, lw=1.7)
    for x, label in zip(xs, tick_labels):
        add_tick(ax, x, axis_y, 0.40, label, font)
    add_tick(ax, 0.91, axis_y, 0.16, "Frame T", font)
    ax.text(0.51, 0.04, "视频时间轴 / Frame Index", ha="center", va="center", fontsize=12, color=TEXT, fontproperties=font)

    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
