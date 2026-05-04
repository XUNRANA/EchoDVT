import os
import json
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
OUTPUT_PATHS = [
    ROOT / "artifacts/figures/fig7_10.png",
    ROOT / "artifacts/figures/fig7_10_runtime_breakdown.png",
]
BENCHMARK_PATH = ROOT / "artifacts/metadata/pipeline_timing_benchmark.json"
MODULE_COLORS = {
    "YOLO 首帧检测": "#E67E22",
    "SAM2 LoRA+MFP 视频分割": "#2F6FED",
    "特征提取 + RF 分类": "#1E9E61",
}

TEXT = "#253041"
SUBTEXT = "#5B6574"
EDGE = "#D1D5DB"
PANEL = "#F8FAFC"


def add_chip(fig, xy, wh, text, font, facecolor, edgecolor, color):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.0,
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
        fontsize=10.8,
        color=color,
        fontproperties=font,
    )


def load_benchmark():
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    summary = benchmark["summary"]
    modules = [
        ("YOLO 首帧检测", float(summary["detection_s"]["mean"]), MODULE_COLORS["YOLO 首帧检测"]),
        ("SAM2 LoRA+MFP 视频分割", float(summary["segmentation_s"]["mean"]), MODULE_COLORS["SAM2 LoRA+MFP 视频分割"]),
        ("特征提取 + RF 分类", float(summary["diagnosis_s"]["mean"]), MODULE_COLORS["特征提取 + RF 分类"]),
    ]
    return benchmark, modules


def build_xticks(total):
    upper = max(12, int(np.ceil(total)))
    step = 1 if upper <= 12 else 2
    limit = int(np.ceil(upper / step) * step)
    return np.arange(0, limit + step, step)


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    benchmark, modules = load_benchmark()
    case_name = benchmark["case_name"]
    num_frames = int(benchmark["num_frames"])
    timed_runs = int(benchmark["timed_runs"])
    warmup_runs = int(benchmark["warmup_runs"])
    device_name = benchmark["environment"]["cuda_device_0"]
    total_mean = float(benchmark["summary"]["total_s"]["mean"])
    total_std = float(benchmark["summary"]["total_s"]["std"])

    values = np.array([item[1] for item in modules], dtype=float)
    total = float(values.sum())
    percents = values / total * 100.0

    fig, ax = plt.subplots(figsize=(10.4, 6.8))
    fig.patch.set_facecolor("white")

    note = FancyBboxPatch(
        (0.07, 0.90),
        0.86,
        0.075,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.0,
        edgecolor=EDGE,
        facecolor=PANEL,
        transform=fig.transFigure,
    )
    fig.patches.append(note)
    fig.text(
        0.09,
        0.940,
        f"基于 {device_name} 的实测稳态 benchmark：病例 {case_name}（{num_frames} 帧），预热 {warmup_runs} 次后连续计时 {timed_runs} 次并取均值。\n"
        f"完整流程平均耗时 {total_mean:.2f} s（std={total_std:.2f} s），其中 SAM2 LoRA+MFP 视频分割为主要瓶颈。",
        ha="left",
        va="center",
        fontsize=11.0,
        color=TEXT,
        fontproperties=font,
    )

    left = 0.0
    bar_y = 0.58
    bar_h = 0.28
    centers = []
    for name, val, color in modules:
        ax.barh(bar_y, val, left=left, height=bar_h, color=color, edgecolor="white", linewidth=2.5)
        centers.append(left + val / 2)
        left += val

    # Main dominant label inside SAM2 segment.
    sam2_idx = 1
    ax.text(
        centers[sam2_idx],
        bar_y,
        f"SAM2 分割\n{modules[sam2_idx][1]:.2f} s  ({percents[sam2_idx]:.2f}%)",
        ha="center",
        va="center",
        fontsize=15.0,
        color="white",
        fontproperties=font,
        fontweight="bold",
    )

    # Tiny modules: annotate outside the bar with leader lines.
    tiny_specs = [
        (0, 0.14, 0.24, "YOLO 检测"),
        (2, 0.82, 0.26, "特征+分类"),
    ]
    for idx, x_text, y_text, short_name in tiny_specs:
        _, val, color = modules[idx]
        ax.annotate(
            f"{short_name}\n{val:.3f} s\n{percents[idx]:.02f}%",
            xy=(centers[idx], bar_y + bar_h / 2),
            xytext=(x_text, y_text),
            textcoords="axes fraction",
            ha="center",
            va="center",
            fontsize=11.2,
            color=color,
            fontproperties=font,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6),
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white", edgecolor=color, linewidth=1.0),
        )

    xticks = build_xticks(total)
    ax.set_xlim(0, xticks[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x)} s" for x in xticks], fontsize=10.8, color=TEXT, fontproperties=font)
    ax.set_xlabel("单病例完整分析耗时（秒）", fontsize=12.5, color=TEXT, fontproperties=font, labelpad=12)
    ax.tick_params(axis="x", colors=TEXT)

    ax.grid(True, axis="x", linestyle="--", linewidth=0.9, alpha=0.42, color="#CBD5E1")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")

    add_chip(fig, (0.08, 0.06), (0.20, 0.07), f"总耗时  {total:.2f} s", font, "#F8FAFC", EDGE, TEXT)
    add_chip(fig, (0.31, 0.06), (0.20, 0.07), f"YOLO 检测  {modules[0][1]:.03f} s", font, "#FFF7ED", "#F4C38A", modules[0][2])
    add_chip(fig, (0.54, 0.06), (0.20, 0.07), f"SAM2 分割  {modules[1][1]:.2f} s", font, "#EEF4FF", "#B8CCE6", modules[1][2])
    add_chip(fig, (0.77, 0.06), (0.15, 0.07), f"特征+分类  {modules[2][1]:.03f} s", font, "#ECFDF3", "#B9E2CF", modules[2][2])

    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.24)
    OUTPUT_PATHS[0].parent.mkdir(parents=True, exist_ok=True)
    for output_path in OUTPUT_PATHS:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
