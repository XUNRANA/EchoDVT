import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
OUTPUT_PATH = ROOT / "artifacts/figures/fig4_2_lora_qkv_detail.png"

TEXT = "#253041"
LINE = "#4B5563"
BG = "#FFFFFF"
Q_COLOR = "#E76F51"
K_COLOR = "#9CA3AF"
V_COLOR = "#4F86C6"
LORA_COLOR = "#F4A261"
INPUT_COLOR = "#DDEBFA"
OUTPUT_COLOR = "#E8F3E4"


def add_round_box(
    ax,
    xy,
    wh,
    text,
    font,
    facecolor,
    edgecolor=LINE,
    fontsize=14,
    lw=1.8,
    linestyle="-",
):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linestyle=linestyle,
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
        linespacing=1.25,
    )
    return patch


def add_arrow(ax, start, end, lw=1.8, color=LINE, rad=0.0, style="-|>"):
    arrow = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        mutation_scale=13,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arrow)
    return arrow


def add_elbow_arrow(ax, points, lw=1.8, color=LINE):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=lw,
            solid_capstyle="round",
        )
    add_arrow(ax, points[-2], points[-1], lw=lw, color=color)


def add_step_badge(ax, center, num, font):
    badge = Circle(center, 0.015, facecolor="#F08A1A", edgecolor="white", linewidth=1.1, zorder=5)
    ax.add_patch(badge)
    ax.text(
        center[0],
        center[1] - 0.001,
        str(num),
        ha="center",
        va="center",
        fontsize=9.5,
        color="white",
        fontproperties=font,
        zorder=6,
    )


def add_split_strip(ax, x, y, w, h, font):
    parts = [
        ("Q 分支\nqkv[..., :dim]", Q_COLOR),
        ("K 分支\nqkv[..., dim:2dim]", K_COLOR),
        ("V 分支\nqkv[..., 2dim:]", V_COLOR),
    ]
    part_w = w / 3
    for idx, (label, color) in enumerate(parts):
        add_round_box(
            ax,
            (x + idx * part_w, y),
            (part_w - 0.006, h),
            label,
            font,
            facecolor=(*plt.matplotlib.colors.to_rgb(color), 0.14),
            edgecolor=color,
            fontsize=12.5,
            lw=1.7,
        )


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    fig, ax = plt.subplots(figsize=(15.6, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(BG)

    # Top note
    note_box = FancyBboxPatch(
        (0.07, 0.915),
        0.86,
        0.045,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.0,
        edgecolor="#D1D5DB",
        facecolor="#F8FAFC",
    )
    ax.add_patch(note_box)
    ax.text(
        0.08,
        0.937,
        "对应 _LoRA_qkv.forward()：原始 qkv 投影按通道拆分为 q / k / v，仅在 q 与 v 分支叠加 LoRA 增量。",
        ha="left",
        va="center",
        fontsize=10.5,
        color=TEXT,
        fontproperties=font,
    )

    # Main flow
    add_round_box(
        ax,
        (0.08, 0.61),
        (0.10, 0.10),
        "输入特征\nX",
        font,
        INPUT_COLOR,
        fontsize=13.2,
    )
    add_round_box(
        ax,
        (0.255, 0.585),
        (0.145, 0.15),
        "原始 QKV 投影\nqkv = W_qkv X + b\n输出维度: 3d",
        font,
        "#EEF3F8",
        fontsize=12.8,
    )
    add_round_box(
        ax,
        (0.50, 0.43),
        (0.095, 0.31),
        "通道拆分\n\nq = qkv[..., :d]\n\nk = qkv[..., d:2d]\n\nv = qkv[..., 2d:3d]",
        font,
        "#F9FAFB",
        fontsize=11.0,
        edgecolor="#D1D5DB",
        lw=1.4,
    )
    add_round_box(
        ax,
        (0.91, 0.445),
        (0.065, 0.25),
        "拼接输出\n\n[q', k, v']",
        font,
        OUTPUT_COLOR,
        fontsize=11.0,
    )

    # Q / K / V branches
    add_round_box(
        ax,
        (0.655, 0.665),
        (0.09, 0.075),
        "Query\nq",
        font,
        "#FDF0EC",
        edgecolor=Q_COLOR,
        fontsize=11.8,
    )
    add_round_box(
        ax,
        (0.655, 0.545),
        (0.09, 0.075),
        "Key\nk",
        font,
        "#F4F5F7",
        edgecolor=K_COLOR,
        fontsize=11.8,
    )
    add_round_box(
        ax,
        (0.655, 0.425),
        (0.09, 0.075),
        "Value\nv",
        font,
        "#EEF5FC",
        edgecolor=V_COLOR,
        fontsize=11.8,
    )

    add_round_box(
        ax,
        (0.79, 0.665),
        (0.085, 0.075),
        "更新 Query\nq' = q + Δq",
        font,
        "#FFF2EF",
        edgecolor=Q_COLOR,
        fontsize=10.8,
    )
    add_round_box(
        ax,
        (0.79, 0.545),
        (0.085, 0.075),
        "Key 保持不变\nk",
        font,
        "#F3F4F6",
        edgecolor=K_COLOR,
        fontsize=10.8,
    )
    add_round_box(
        ax,
        (0.79, 0.425),
        (0.085, 0.075),
        "更新 Value\nv' = v + Δv",
        font,
        "#EEF5FC",
        edgecolor=V_COLOR,
        fontsize=10.8,
    )

    # Primary arrows
    add_arrow(ax, (0.18, 0.66), (0.255, 0.66))
    add_arrow(ax, (0.40, 0.66), (0.50, 0.66))
    add_arrow(ax, (0.595, 0.703), (0.655, 0.703), lw=1.6, color=Q_COLOR)
    add_arrow(ax, (0.595, 0.582), (0.655, 0.582), lw=1.6, color=K_COLOR)
    add_arrow(ax, (0.595, 0.462), (0.655, 0.462), lw=1.6, color=V_COLOR)
    add_arrow(ax, (0.745, 0.703), (0.79, 0.703), lw=1.7, color=Q_COLOR)
    add_arrow(ax, (0.745, 0.582), (0.79, 0.582), lw=1.5, color=K_COLOR)
    add_arrow(ax, (0.745, 0.462), (0.79, 0.462), lw=1.7, color=V_COLOR)
    add_arrow(ax, (0.875, 0.703), (0.91, 0.703), lw=1.6, color=Q_COLOR)
    add_arrow(ax, (0.875, 0.582), (0.91, 0.582), lw=1.5, color=K_COLOR)
    add_arrow(ax, (0.875, 0.462), (0.91, 0.462), lw=1.6, color=V_COLOR)

    # LoRA side branches
    add_round_box(
        ax,
        (0.09, 0.26),
        (0.085, 0.065),
        "A_q\nR^d -> R^r",
        font,
        "#FFF6E8",
        edgecolor=LORA_COLOR,
        fontsize=10.8,
    )
    add_round_box(
        ax,
        (0.225, 0.26),
        (0.085, 0.065),
        "B_q\nR^r -> R^d",
        font,
        "#FFF6E8",
        edgecolor=LORA_COLOR,
        fontsize=10.8,
    )
    add_round_box(
        ax,
        (0.09, 0.13),
        (0.085, 0.065),
        "A_v\nR^d -> R^r",
        font,
        "#FFF6E8",
        edgecolor=LORA_COLOR,
        fontsize=10.8,
    )
    add_round_box(
        ax,
        (0.225, 0.13),
        (0.085, 0.065),
        "B_v\nR^r -> R^d",
        font,
        "#FFF6E8",
        edgecolor=LORA_COLOR,
        fontsize=10.8,
    )
    add_round_box(
        ax,
        (0.355, 0.26),
        (0.07, 0.065),
        "缩放\nalpha / r",
        font,
        "#F9FAFB",
        edgecolor="#D1D5DB",
        fontsize=10.6,
    )
    add_round_box(
        ax,
        (0.355, 0.13),
        (0.07, 0.065),
        "缩放\nalpha / r",
        font,
        "#F9FAFB",
        edgecolor="#D1D5DB",
        fontsize=10.6,
    )
    add_round_box(
        ax,
        (0.47, 0.26),
        (0.10, 0.065),
        "LoRA 增量\nΔq = B_q A_q(X)",
        font,
        "#FFF6E8",
        edgecolor=LORA_COLOR,
        fontsize=10.6,
    )
    add_round_box(
        ax,
        (0.47, 0.13),
        (0.10, 0.065),
        "LoRA 增量\nΔv = B_v A_v(X)",
        font,
        "#FFF6E8",
        edgecolor=LORA_COLOR,
        fontsize=10.6,
    )

    # Flow from input to LoRA branches
    add_arrow(ax, (0.13, 0.61), (0.13, 0.325), lw=1.5, color=LORA_COLOR)
    add_arrow(ax, (0.13, 0.26), (0.13, 0.195), lw=1.5, color=LORA_COLOR)
    add_arrow(ax, (0.175, 0.292), (0.225, 0.292), lw=1.6, color=LORA_COLOR)
    add_arrow(ax, (0.175, 0.162), (0.225, 0.162), lw=1.6, color=LORA_COLOR)
    add_arrow(ax, (0.31, 0.292), (0.355, 0.292), lw=1.6, color=LORA_COLOR)
    add_arrow(ax, (0.31, 0.162), (0.355, 0.162), lw=1.6, color=LORA_COLOR)
    add_arrow(ax, (0.425, 0.292), (0.47, 0.292), lw=1.6, color=LORA_COLOR)
    add_arrow(ax, (0.425, 0.162), (0.47, 0.162), lw=1.6, color=LORA_COLOR)

    # Inject delta_q and delta_v into q' / v'
    add_elbow_arrow(ax, [(0.57, 0.292), (0.74, 0.292), (0.74, 0.685), (0.79, 0.685)], lw=1.6, color=LORA_COLOR)
    add_elbow_arrow(ax, [(0.57, 0.162), (0.74, 0.162), (0.74, 0.445), (0.79, 0.445)], lw=1.6, color=LORA_COLOR)

    # Step markers
    add_step_badge(ax, (0.075, 0.755), 1, font)
    add_step_badge(ax, (0.305, 0.772), 2, font)
    add_step_badge(ax, (0.495, 0.805), 3, font)
    add_step_badge(ax, (0.085, 0.365), 4, font)
    add_step_badge(ax, (0.785, 0.778), 5, font)

    # Step captions
    ax.text(0.058, 0.785, "输入", ha="left", va="center", fontsize=10.8, color=TEXT, fontproperties=font)
    ax.text(0.285, 0.805, "原始投影", ha="left", va="center", fontsize=10.8, color=TEXT, fontproperties=font)
    ax.text(0.49, 0.835, "切分", ha="left", va="center", fontsize=10.8, color=TEXT, fontproperties=font)
    ax.text(0.068, 0.392, "低秩旁路", ha="left", va="center", fontsize=10.8, color=TEXT, fontproperties=font)
    ax.text(0.778, 0.815, "融合", ha="left", va="center", fontsize=10.8, color=TEXT, fontproperties=font)

    tail_box = FancyBboxPatch(
        (0.09, 0.83),
        0.17,
        0.06,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.0,
        edgecolor="#D1D5DB",
        facecolor="#FFFDF6",
    )
    ax.add_patch(tail_box)
    ax.text(
        0.175,
        0.86,
        "仅在 Query / Value 分支注入 LoRA\nKey 分支保持冻结",
        ha="center",
        va="center",
        fontsize=10.0,
        color=TEXT,
        fontproperties=font,
        linespacing=1.25,
    )

    ax.text(0.63, 0.26, "注入 Δq", ha="left", va="center", fontsize=10.2, color=LORA_COLOR, fontproperties=font)
    ax.text(0.63, 0.13, "注入 Δv", ha="left", va="center", fontsize=10.2, color=LORA_COLOR, fontproperties=font)

    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
