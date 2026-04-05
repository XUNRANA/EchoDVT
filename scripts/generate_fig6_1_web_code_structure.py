import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
OUTPUT_PATHS = [
    ROOT / "results/fig6_1.png",
    ROOT / "results/fig6_1_web_code_structure.png",
]

TEXT = "#253041"
SUBTEXT = "#5B6574"
EDGE = "#C9D3DF"
LINE = "#5B6574"
PANEL = "#F8FAFC"
WHITE = "#FFFFFF"
BLUE = "#EAF2FF"
GREEN = "#EAFBF2"
AMBER = "#FFF6E8"
CYAN = "#ECFEFF"
GRAY = "#F3F4F6"


def add_box(ax, xy, wh, text, font, facecolor=WHITE, fontsize=11.0, lw=1.2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=lw,
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
        linespacing=1.15,
    )
    return patch


def add_chip(ax, center, wh, text, font, facecolor=WHITE, fontsize=10.0):
    cx, cy = center
    w, h = wh
    return add_box(ax, (cx - w / 2, cy - h / 2), wh, text, font, facecolor=facecolor, fontsize=fontsize, lw=1.0)


def add_panel(ax, xy, wh, title, font, facecolor=PANEL):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        linewidth=1.25,
        edgecolor=EDGE,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    tag = FancyBboxPatch(
        (x + 0.02, y + h - 0.065),
        min(0.12, w - 0.04),
        0.045,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=EDGE,
        facecolor=WHITE,
    )
    ax.add_patch(tag)
    ax.text(
        x + 0.02 + min(0.12, w - 0.04) / 2,
        y + h - 0.0425,
        title,
        ha="center",
        va="center",
        fontsize=10.6,
        color=TEXT,
        fontproperties=font,
    )
    return patch


def add_line(ax, p1, p2, lw=2.0):
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color=LINE,
        linewidth=lw,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))

    fig, ax = plt.subplots(figsize=(24.0, 5.6))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1.62)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, (0.72, 0.82), (0.18, 0.065), "web/", font, facecolor=BLUE, fontsize=12.8, lw=1.4)

    x_positions = {
        "app": 0.18,
        "services": 0.48,
        "tabs": 0.83,
        "utils": 1.18,
        "assets": 1.47,
    }
    top_y = 0.48
    bus_y = 0.68

    boxes = {
        "app": (
            (x_positions["app"] - 0.10, top_y - 0.05),
            (0.20, 0.12),
            "app.py\nGradio 应用入口",
            BLUE,
            10.8,
        ),
        "services": (
            (x_positions["services"] - 0.11, top_y - 0.06),
            (0.22, 0.14),
            "services/\ninference.py",
            GREEN,
            10.8,
        ),
        "tabs": (
            (x_positions["tabs"] - 0.13, top_y - 0.10),
            (0.26, 0.22),
            "tabs/\nupload.py\npipeline.py\ndiagnosis.py\ndashboard.py",
            AMBER,
            10.6,
        ),
        "utils": (
            (x_positions["utils"] - 0.11, top_y - 0.06),
            (0.22, 0.15),
            "utils/\nvisualization.py\nmetrics.py",
            CYAN,
            10.7,
        ),
        "assets": (
            (x_positions["assets"] - 0.09, top_y - 0.04),
            (0.18, 0.10),
            "assets/\ncustom.css",
            GRAY,
            10.6,
        ),
    }

    root_center = (0.81, 0.82)
    add_line(ax, (root_center[0], 0.82), (root_center[0], bus_y), lw=2.2)
    add_line(ax, (x_positions["app"], bus_y), (x_positions["assets"], bus_y), lw=2.2)

    for key, xpos in x_positions.items():
        (xy, wh, text, facecolor, fontsize) = boxes[key]
        add_line(ax, (xpos, bus_y), (xpos, xy[1] + wh[1]), lw=2.1)
        add_box(ax, xy, wh, text, font, facecolor=facecolor, fontsize=fontsize, lw=1.25)

    OUTPUT_PATHS[0].parent.mkdir(parents=True, exist_ok=True)
    for output_path in OUTPUT_PATHS:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
