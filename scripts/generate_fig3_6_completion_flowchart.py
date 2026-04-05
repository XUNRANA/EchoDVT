import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Polygon


FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
OUTPUT_PATH = Path("/data1/ouyangxinglong/EchoDVT/results/fig3_6_completion_flowchart.png")

LINE_COLOR = "#4A4A4A"
TEXT_COLOR = "#222222"
LABEL_COLOR = "#666666"

BOX_COLORS = {
    "start": "#A9C9E8",
    "decision": "#FFF2A8",
    "process": "#DDEBDD",
    "output": "#EFB0C5",
}


def add_box(ax, center, size, text, facecolor, font, fontsize=13):
    x, y = center
    w, h = size
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        linewidth=1.5,
        edgecolor=LINE_COLOR,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT_COLOR,
        fontproperties=font,
    )
    return {"kind": "box", "x": x, "y": y, "w": w, "h": h}


def add_diamond(ax, center, size, text, facecolor, font, fontsize=13):
    x, y = center
    w, h = size
    points = [
        (x, y + h / 2),
        (x + w / 2, y),
        (x, y - h / 2),
        (x - w / 2, y),
    ]
    patch = Polygon(
        points,
        closed=True,
        linewidth=1.5,
        edgecolor=LINE_COLOR,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT_COLOR,
        fontproperties=font,
    )
    return {"kind": "diamond", "x": x, "y": y, "w": w, "h": h}


def anchor(node, side):
    x, y, w, h = node["x"], node["y"], node["w"], node["h"]
    if side == "top":
        return (x, y + h / 2)
    if side == "bottom":
        return (x, y - h / 2)
    if side == "left":
        return (x - w / 2, y)
    if side == "right":
        return (x + w / 2, y)
    raise ValueError(f"Unknown anchor side: {side}")


def polyline(ax, points, arrow_end=True, linewidth=1.5):
    if len(points) < 2:
        return
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=LINE_COLOR,
            linewidth=linewidth,
        )
    if arrow_end:
        ax.annotate(
            "",
            xy=points[-1],
            xytext=points[-2],
            arrowprops=dict(
                arrowstyle="->",
                color=LINE_COLOR,
                linewidth=linewidth,
                shrinkA=0,
                shrinkB=0,
            ),
        )
    else:
        start, end = points[-2], points[-1]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=LINE_COLOR,
            linewidth=linewidth,
        )


def add_label(ax, x, y, text, font, fontsize=11, rotation=0, ha="center", va="center"):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=LABEL_COLOR,
        fontproperties=font,
        rotation=rotation,
    )


def add_joint(ax, point, size=3.2):
    ax.plot(point[0], point[1], "o", color=LINE_COLOR, markersize=size)


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))

    fig, ax = plt.subplots(figsize=(15.8, 8.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    start = add_box(ax, (0.50, 0.90), (0.16, 0.065), "首帧YOLO检测", BOX_COLORS["start"], font)
    result = add_diamond(ax, (0.50, 0.74), (0.20, 0.11), "检测结果？", BOX_COLORS["decision"], font)

    iou = add_diamond(ax, (0.16, 0.56), (0.12, 0.085), "IoU>0.3？", BOX_COLORS["decision"], font, fontsize=12)
    use = add_box(ax, (0.07, 0.39), (0.095, 0.072), "直接使用", BOX_COLORS["process"], font, fontsize=12)
    fix = add_box(ax, (0.24, 0.39), (0.125, 0.072), "重叠修正\n调整低置信框", BOX_COLORS["process"], font, fontsize=11)

    artery_to_vein = add_box(ax, (0.39, 0.49), (0.12, 0.08), "相对先验\nartery2vein", BOX_COLORS["process"], font, fontsize=12)
    vein_to_artery = add_box(ax, (0.61, 0.49), (0.12, 0.08), "反向先验\nvein2artery", BOX_COLORS["process"], font, fontsize=12)

    retry = add_box(ax, (0.87, 0.56), (0.105, 0.072), "降低阈值\n0.1→0.01", BOX_COLORS["process"], font, fontsize=12)
    has_result = add_diamond(ax, (0.87, 0.39), (0.12, 0.085), "有结果？", BOX_COLORS["decision"], font, fontsize=12)
    absolute = add_box(ax, (0.87, 0.22), (0.115, 0.08), "绝对先验兜底\nclass_absolute", BOX_COLORS["process"], font, fontsize=12)

    output = add_box(
        ax,
        (0.50, 0.07),
        (0.20, 0.07),
        "输出: {artery: box, vein: box}",
        BOX_COLORS["output"],
        font,
        fontsize=12,
    )

    polyline(ax, [anchor(start, "bottom"), anchor(result, "top")])

    polyline(ax, [anchor(result, "left"), (iou["x"], 0.74), anchor(iou, "top")])
    add_label(ax, 0.29, 0.77, "两类都检到", font)

    polyline(ax, [anchor(iou, "left"), (use["x"], 0.56), anchor(use, "top")])
    add_label(ax, 0.095, 0.47, "否", font)

    polyline(ax, [anchor(iou, "right"), (fix["x"], 0.56), anchor(fix, "top")])
    add_label(ax, 0.235, 0.47, "是", font)

    polyline(ax, [(0.45, 0.68), anchor(artery_to_vein, "top")])
    add_label(ax, 0.42, 0.63, "仅动脉", font)

    polyline(ax, [(0.55, 0.68), anchor(vein_to_artery, "top")])
    add_label(ax, 0.58, 0.63, "仅静脉", font)

    polyline(ax, [anchor(result, "right"), (retry["x"], 0.74), anchor(retry, "top")])
    add_label(ax, 0.75, 0.77, "都没检到", font)

    polyline(ax, [anchor(retry, "bottom"), anchor(has_result, "top")])

    polyline(ax, [anchor(has_result, "bottom"), anchor(absolute, "top")])
    add_label(ax, 0.86, 0.30, "否", font)

    loop_x = 0.97
    polyline(
        ax,
        [anchor(has_result, "right"), (loop_x, 0.39), (loop_x, 0.84), (0.50, 0.84), (0.50, 0.82)],
    )
    add_joint(ax, (0.50, 0.82))
    add_label(ax, 0.93, 0.44, "是", font)
    add_label(ax, loop_x + 0.02, 0.565, "递归处理", font, rotation=90, ha="left")

    bus_y = 0.16
    terminals = [use, fix, artery_to_vein, vein_to_artery, absolute]
    for node in terminals:
        polyline(ax, [anchor(node, "bottom"), (node["x"], bus_y)], arrow_end=False)

    polyline(ax, [(use["x"], bus_y), (absolute["x"], bus_y)], arrow_end=False)
    polyline(ax, [(0.50, bus_y), anchor(output, "top")])

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
