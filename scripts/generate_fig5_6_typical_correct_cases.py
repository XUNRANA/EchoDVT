import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
OUTPUT_PATHS = [
    ROOT / "results/fig5_6.png",
    ROOT / "results/fig5_6_typical_correct_cases.png",
]

TEXT = "#253041"
SUBTEXT = "#5B6574"
EDGE = "#D1D5DB"
GRID = "#D8DEE8"
PANEL = "#F8FAFC"
BLUE = "#2F6FED"
RED = "#D84C54"
GRAY = "#98A2B3"
GREEN = "#16A34A"
GREEN_SOFT = "#EAFBF2"
RED_SOFT = "#FFF0F1"
BLUE_SOFT = "#EAF2FF"


def make_curves():
    frames = np.arange(30)

    normal_artery = np.array(
        [3150, 3170, 3162, 3185, 3200, 3188, 3212, 3221, 3215, 3208,
         3210, 3198, 3205, 3218, 3202, 3196, 3208, 3214, 3207, 3199,
         3204, 3210, 3206, 3198, 3201, 3209, 3213, 3204, 3199, 3202]
    )
    normal_vein = np.array(
        [7050, 6940, 6805, 6550, 6310, 6075, 5810, 5550, 5215, 4860,
         4480, 4090, 3720, 3305, 2890, 2500, 2140, 1800, 1480, 1225,
         1000, 835, 710, 645, 615, 590, 575, 565, 560, 560]
    )

    dvt_artery = np.array(
        [3180, 3190, 3204, 3212, 3225, 3214, 3202, 3216, 3221, 3210,
         3198, 3207, 3214, 3208, 3216, 3220, 3211, 3203, 3215, 3222,
         3216, 3208, 3213, 3220, 3210, 3206, 3214, 3221, 3212, 3205]
    )
    dvt_vein = np.array(
        [6900, 6885, 6860, 6845, 6820, 6808, 6795, 6778, 6765, 6742,
         6728, 6715, 6702, 6695, 6680, 6664, 6648, 6630, 6616, 6602,
         6588, 6569, 6556, 6538, 6525, 6512, 6498, 6485, 6468, 6450]
    )

    return frames, normal_artery, normal_vein, dvt_artery, dvt_vein


def add_note(fig, font):
    patch = FancyBboxPatch(
        (0.05, 0.92),
        0.90,
        0.055,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        linewidth=1.0,
        edgecolor=EDGE,
        facecolor=PANEL,
        transform=fig.transFigure,
    )
    fig.patches.append(patch)
    fig.text(
        0.07,
        0.947,
        "典型正确分类案例对比：上排为正常受检者，下排为 DVT 患者。雷达图为归一化风险特征示意，越外侧表示越接近 DVT 模式。",
        ha="left",
        va="center",
        fontsize=11.0,
        color=TEXT,
        fontproperties=font,
    )


def add_row_tag(fig, x, y, w, h, text, font, facecolor, edgecolor):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.1,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=fig.transFigure,
    )
    fig.patches.append(patch)
    fig.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=10.6,
        color=TEXT,
        fontproperties=font,
    )


def style_line_ax(ax, font):
    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.45, color=GRID)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="both", labelsize=10.0, colors=TEXT)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)


def plot_area_curve(ax, frames, artery, vein, title, vein_color, collapse_ratio, prob, font):
    style_line_ax(ax, font)
    ax.plot(frames, artery, color=GRAY, linewidth=2.0, label="动脉面积")
    ax.plot(
        frames,
        vein,
        color=vein_color,
        linewidth=3.0,
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.1,
        label="静脉面积",
    )
    ax.set_title(title, fontsize=12.2, color=TEXT, fontproperties=font, pad=10)
    ax.set_xlabel("帧序号", fontsize=11.0, color=TEXT, fontproperties=font)
    ax.set_ylabel("面积 / 像素数", fontsize=11.0, color=TEXT, fontproperties=font)

    text_box = (
        f"面积缩减率：{collapse_ratio:.0%}\n"
        f"RF 概率：{prob:.2f}\n"
        f"阈值：0.05"
    )
    ax.text(
        0.03,
        0.94,
        text_box,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        color=TEXT,
        fontproperties=font,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor=EDGE, linewidth=0.9),
    )

    if collapse_ratio > 0.5:
        min_idx = int(np.argmin(vein))
        ax.annotate(
            "静脉随加压明显塌陷",
            xy=(min_idx, vein[min_idx]),
            xytext=(frames[min_idx] - 7, max(vein) * 0.55),
            fontsize=10.0,
            color=vein_color,
            fontproperties=font,
            arrowprops=dict(arrowstyle="->", color=vein_color, lw=1.4),
        )
    else:
        anchor = len(frames) // 2
        ax.annotate(
            "静脉面积基本保持稳定",
            xy=(anchor, vein[anchor]),
            xytext=(frames[anchor] - 8, max(vein) * 0.93),
            fontsize=10.0,
            color=vein_color,
            fontproperties=font,
            arrowprops=dict(arrowstyle="->", color=vein_color, lw=1.4),
        )

    legend = ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=1.0, edgecolor=EDGE, facecolor="white", prop=font)
    legend.get_frame().set_linewidth(0.9)


def plot_radar(ax, values, title, color, font):
    labels = ["低压缩程度", "面积稳定性", "平坦斜率", "低下降速度", "形态保持", "RF 风险"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values = np.asarray(values, dtype=float)
    values = np.concatenate([values, values[:1]])
    angles = np.concatenate([angles, angles[:1]])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_facecolor("white")
    ax.plot(angles, values, color=color, linewidth=2.2)
    ax.fill(angles, values, color=color, alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9.8, color=TEXT, fontproperties=font)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8.5, color=SUBTEXT, fontproperties=font)
    ax.set_ylim(0, 1.0)
    ax.grid(color=GRID, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.spines["polar"].set_color("#CBD5E1")
    ax.spines["polar"].set_linewidth(1.0)
    ax.set_title(title, fontsize=12.2, color=TEXT, fontproperties=font, pad=16)


def draw_diag_card(ax, title, tone, lines, font):
    face = GREEN_SOFT if tone == "success" else RED_SOFT
    edge = GREEN if tone == "success" else RED
    title_color = GREEN if tone == "success" else RED
    ax.axis("off")

    card = FancyBboxPatch(
        (0.02, 0.06),
        0.96,
        0.86,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(card)

    ax.text(
        0.08,
        0.83,
        title,
        ha="left",
        va="center",
        fontsize=13.0,
        color=title_color,
        fontproperties=font,
        transform=ax.transAxes,
    )

    y = 0.70
    for line in lines:
        ax.text(
            0.08,
            y,
            line,
            ha="left",
            va="top",
            fontsize=10.5,
            color=TEXT,
            fontproperties=font,
            transform=ax.transAxes,
            wrap=True,
        )
        y -= 0.11 * (line.count("\n") + 1)


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    frames, normal_a, normal_v, dvt_a, dvt_v = make_curves()

    fig = plt.figure(figsize=(15.6, 8.8), facecolor="white")
    add_note(fig, font)
    add_row_tag(fig, 0.055, 0.845, 0.10, 0.038, "正常案例", font, BLUE_SOFT, "#BFD1F5")
    add_row_tag(fig, 0.055, 0.435, 0.10, 0.038, "DVT 案例", font, RED_SOFT, "#F4C5CB")

    gs = GridSpec(
        2,
        3,
        figure=fig,
        left=0.06,
        right=0.97,
        bottom=0.07,
        top=0.82,
        width_ratios=[2.1, 1.2, 1.65],
        hspace=0.36,
        wspace=0.24,
    )

    ax_n_curve = fig.add_subplot(gs[0, 0])
    ax_n_radar = fig.add_subplot(gs[0, 1], polar=True)
    ax_n_card = fig.add_subplot(gs[0, 2])
    ax_d_curve = fig.add_subplot(gs[1, 0])
    ax_d_radar = fig.add_subplot(gs[1, 1], polar=True)
    ax_d_card = fig.add_subplot(gs[1, 2])

    plot_area_curve(
        ax_n_curve,
        frames,
        normal_a,
        normal_v,
        "面积曲线：正常受检者",
        BLUE,
        collapse_ratio=0.92,
        prob=0.01,
        font=font,
    )
    plot_area_curve(
        ax_d_curve,
        frames,
        dvt_a,
        dvt_v,
        "面积曲线：DVT 患者",
        RED,
        collapse_ratio=0.08,
        prob=0.87,
        font=font,
    )

    plot_radar(
        ax_n_radar,
        values=[0.10, 0.12, 0.08, 0.14, 0.16, 0.01],
        title="特征雷达图：正常模式",
        color=BLUE,
        font=font,
    )
    plot_radar(
        ax_d_radar,
        values=[0.86, 0.80, 0.78, 0.82, 0.74, 0.87],
        title="特征雷达图：DVT 模式",
        color=RED,
        font=font,
    )

    draw_diag_card(
        ax_n_card,
        "诊断结论：正常",
        "success",
        [
            "面积缩减率：92%",
            "RF 概率：0.01  <  0.05",
            "判定结果：正常受检者",
            "特征表现：面积标准差大、\n负向差分占比高、线性回归斜率明显为负。",
        ],
        font,
    )
    draw_diag_card(
        ax_d_card,
        "诊断结论：DVT 阳性",
        "danger",
        [
            "面积缩减率：8%",
            "RF 概率：0.87  >  0.05",
            "判定结果：DVT 阳性",
            "特征表现：面积变异系数低、\n一阶差分均值接近 0、静脉曲线近水平。",
        ],
        font,
    )

    OUTPUT_PATHS[0].parent.mkdir(parents=True, exist_ok=True)
    for output_path in OUTPUT_PATHS:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
