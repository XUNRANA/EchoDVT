import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
OUTPUT_PATH = ROOT / "results/fig4_1_sam2_lora_injection.png"

TEXT = "#1F2937"
LINE = "#4B5563"

BOX = {
    "encoder": "#DFF2E3",
    "memory_attn": "#DDEBFA",
    "mask_decoder": "#FDE7C6",
    "memory_encoder": "#E9E2F7",
    "prompt": "#DDE7D5",
    "memory_bank": "#F7D9E5",
}

ACCENT = {
    "lora": "#E8871E",
    "full": "#D1495B",
    "optional": "#2A9D8F",
}


def add_round_box(ax, xy, wh, facecolor, text, font, fontsize=15, edgecolor=LINE, lw=1.6, linestyle="-"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw,
        linestyle=linestyle,
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
        linespacing=1.28,
    )
    return patch


def add_label_box(ax, xy, wh, text, font, color, face_alpha=0.11, fontsize=12):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=1.4,
        edgecolor=color,
        facecolor=(*plt.matplotlib.colors.to_rgb(color), face_alpha),
        linestyle=(0, (4, 3)),
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        fontproperties=font,
        linespacing=1.2,
    )
    return patch


def add_number(ax, center, num, font, color):
    circle = Circle(center, 0.018, facecolor=color, edgecolor="white", linewidth=1.2, zorder=5)
    ax.add_patch(circle)
    ax.text(
        center[0],
        center[1] - 0.001,
        str(num),
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        fontproperties=font,
        zorder=6,
    )


def add_arrow(ax, start, end, rad=0.0, style="-|>", lw=1.7, color=LINE, linestyle="-"):
    arrow = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
    )
    ax.add_patch(arrow)
    return arrow


def add_memory_bank(ax, xy, wh, font):
    x, y = xy
    w, h = wh
    offsets = [(0.02, 0.02), (0.01, 0.01), (0, 0)]
    for idx, (dx, dy) in enumerate(offsets):
        patch = FancyBboxPatch(
            (x + dx, y + dy),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.5 if idx == 2 else 1.1,
            edgecolor=LINE,
            facecolor=BOX["memory_bank"],
            alpha=0.45 if idx < 2 else 1.0,
        )
        ax.add_patch(patch)
    ax.text(
        x + w / 2 + 0.01,
        y + h / 2 + 0.01,
        "Memory Bank\n历史帧记忆",
        ha="center",
        va="center",
        fontsize=14,
        color=TEXT,
        fontproperties=font,
        linespacing=1.25,
    )


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    fig, ax = plt.subplots(figsize=(14.8, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main architecture layout
    add_round_box(
        ax,
        (0.07, 0.53),
        (0.20, 0.24),
        BOX["encoder"],
        "Image Encoder\n48层 Hiera blocks\nQKV Attention",
        font,
        fontsize=15,
        edgecolor=ACCENT["lora"],
        lw=2.0,
    )
    add_round_box(
        ax,
        (0.33, 0.53),
        (0.22, 0.24),
        BOX["memory_attn"],
        "Memory Attention\n4层 Transformer\nself_attn + cross_attn",
        font,
        fontsize=15,
        edgecolor=ACCENT["lora"],
        lw=2.0,
    )
    add_round_box(
        ax,
        (0.61, 0.53),
        (0.18, 0.24),
        BOX["mask_decoder"],
        "Mask Decoder\nSAM 解码头",
        font,
        fontsize=15,
        edgecolor=ACCENT["full"],
        lw=2.0,
    )
    add_round_box(
        ax,
        (0.61, 0.18),
        (0.18, 0.20),
        BOX["memory_encoder"],
        "Memory Encoder\nmask -> memory\nembedding",
        font,
        fontsize=14,
        edgecolor=ACCENT["optional"],
        lw=2.0,
    )
    add_round_box(
        ax,
        (0.61, 0.82),
        (0.18, 0.10),
        BOX["prompt"],
        "Prompt Encoder\nbox prompt",
        font,
        fontsize=13,
    )
    add_memory_bank(ax, (0.82, 0.22), (0.10, 0.18), font)

    # Module-level injection cues
    ax.text(0.17, 0.585, "注入: Q / V", ha="center", va="center", fontsize=12.2, color=ACCENT["lora"], fontproperties=font)
    ax.text(0.44, 0.575, "注入: self / cross attn", ha="center", va="center", fontsize=11.6, color=ACCENT["lora"], fontproperties=font)
    ax.text(0.70, 0.595, "全量微调", ha="center", va="center", fontsize=12.0, color=ACCENT["full"], fontproperties=font)

    # Connections
    add_arrow(ax, (0.27, 0.65), (0.33, 0.65), lw=1.8)
    add_arrow(ax, (0.55, 0.65), (0.61, 0.65), lw=1.8)
    add_arrow(ax, (0.70, 0.82), (0.70, 0.77), lw=1.6)
    add_arrow(ax, (0.70, 0.53), (0.70, 0.38), lw=1.7)
    add_arrow(ax, (0.79, 0.28), (0.82, 0.31), lw=1.6)
    add_arrow(ax, (0.87, 0.40), (0.87, 0.69), style="-", lw=1.4)
    add_arrow(ax, (0.87, 0.69), (0.55, 0.69), style="-", lw=1.4)
    add_arrow(ax, (0.55, 0.69), (0.55, 0.65), lw=1.4)

    # Input / flow notes
    ax.text(0.05, 0.65, "当前帧图像", ha="right", va="center", fontsize=13, color=TEXT, fontproperties=font)
    add_arrow(ax, (0.055, 0.65), (0.07, 0.65), lw=1.5)
    ax.text(0.875, 0.73, "历史记忆回流", ha="left", va="center", fontsize=12, color=LINE, fontproperties=font, rotation=90)
    ax.text(0.70, 0.48, "分割结果", ha="center", va="center", fontsize=12, color=TEXT, fontproperties=font)
    ax.text(0.70, 0.14, "预测掩码", ha="center", va="center", fontsize=12, color=TEXT, fontproperties=font)

    # Number markers
    add_number(ax, (0.085, 0.79), 1, font, ACCENT["lora"])
    add_number(ax, (0.345, 0.79), 2, font, ACCENT["lora"])
    add_number(ax, (0.625, 0.79), 3, font, ACCENT["full"])
    add_number(ax, (0.625, 0.41), 4, font, ACCENT["optional"])

    # Top legend for four injection points
    legend = FancyBboxPatch(
        (0.08, 0.82),
        0.46,
        0.13,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#D1D5DB",
        facecolor="#FFFFFF",
    )
    ax.add_patch(legend)
    legend_items = [
        (1, ACCENT["lora"], "Image Encoder", "_LoRA_qkv / Q,V"),
        (2, ACCENT["lora"], "Memory Attention", "_LoRA_Linear / Q,K,V,Out"),
        (3, ACCENT["full"], "Mask Decoder", "全量微调"),
        (4, ACCENT["optional"], "Memory Encoder", "可选解冻"),
    ]
    legend_positions = [(0.11, 0.89), (0.34, 0.89), (0.11, 0.845), (0.34, 0.845)]
    for (num, color, title, desc), (x, y) in zip(legend_items, legend_positions):
        add_number(ax, (x, y), num, font, color)
        ax.text(x + 0.03, y + 0.007, title, ha="left", va="center", fontsize=11.8, color=TEXT, fontproperties=font)
        ax.text(x + 0.03, y - 0.020, desc, ha="left", va="center", fontsize=10.2, color=LINE, fontproperties=font)

    # Bottom annotation strip
    note_box = FancyBboxPatch(
        (0.08, 0.04),
        0.84,
        0.08,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#D1D5DB",
        facecolor="#F8FAFC",
    )
    ax.add_patch(note_box)
    ax.text(
        0.10,
        0.08,
        "代码实现对齐: Image Encoder 共48个Hiera blocks；Memory Attention 共4层；"
        "LoRA矩阵约0.489M，Mask Decoder 采用全量微调。",
        ha="left",
        va="center",
        fontsize=11.5,
        color=TEXT,
        fontproperties=font,
    )

    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
