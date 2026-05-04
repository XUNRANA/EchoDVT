import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
VIS_ROOT = ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_094056_lora_r8_mfp15/visualizations"
FRAME_METRICS = ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_094056_lora_r8_mfp15/frame_metrics.csv"

OUTPUT_PATHS = [
    ROOT / "artifacts/figures/fig7_7.png",
    ROOT / "artifacts/figures/fig7_7_failure_case_analysis.png",
]

CASES = [
    {
        "case": "YANG_MUCHEN-V1E1-6",
        "frame": 0,
        "label": "案例 A",
        "title": "失败模式 1：首帧检测框偏差",
        "desc": "静脉框由先验补全生成，初始位置偏离真实目标。",
        "strip_face": "#FFF4E5",
        "strip_edge": "#F3C677",
        "top_extra": 0.02,
    },
    {
        "case": "Ren_min_V1",
        "frame": 16,
        "label": "案例 B",
        "title": "失败模式 2：极端形变导致丢失",
        "desc": "后期静脉被压成细缝，预测已完全消失。",
        "strip_face": "#EEF4FF",
        "strip_edge": "#BFD1F5",
        "top_extra": 0.10,
    },
    {
        "case": "ZHAO_SUFANG-V1E1-3",
        "frame": 71,
        "label": "案例 C",
        "title": "失败模式 3：图像质量过低",
        "desc": "严重信号衰减导致边界对比极弱，模型未输出有效掩码。",
        "strip_face": "#F3F4F6",
        "strip_edge": "#D1D5DB",
        "top_extra": 0.10,
    },
]

TEXT = "#253041"
SUBTEXT = "#5B6574"
EDGE = "#D1D5DB"
PANEL = "#F8FAFC"


def load_metric_map():
    metric_map = {}
    with FRAME_METRICS.open() as f:
        for row in csv.DictReader(f):
            metric_map[(row["case"], int(row["frame"]))] = {
                "mean_dice": float(row["mean_dice"]),
                "artery_dice": float(row["artery_dice"]),
                "vein_dice": float(row["vein_dice"]),
            }
    return metric_map


def split_panels(viz_rgb: np.ndarray):
    _, w = viz_rgb.shape[:2]
    panel_w = (w - 16) // 3
    return (
        viz_rgb[:, :panel_w],
        viz_rgb[:, panel_w + 8 : panel_w * 2 + 8],
        viz_rgb[:, panel_w * 2 + 16 : panel_w * 3 + 16],
    )


def colored_mask(panel_rgb: np.ndarray):
    hsv = cv2.cvtColor(cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    mask = (s >= 55) & (v >= 60) & (
        ((h <= 10) | (h >= 170)) | ((h >= 35) & (h <= 95)) | ((h >= 100) & (h <= 140))
    )
    # Ignore the metric text block in the top-left corner.
    mask[: int(panel_rgb.shape[0] * 0.26), : int(panel_rgb.shape[1] * 0.66)] = False
    return mask


def compute_crop_box(shape, masks, min_side: int = 190, scale: float = 2.15, top_extra: float = 0.08):
    h, w = shape[:2]
    xs = []
    ys = []
    for mask in masks:
        ys_idx, xs_idx = np.where(mask)
        if len(xs_idx) == 0:
            continue
        xs.extend([int(xs_idx.min()), int(xs_idx.max())])
        ys.extend([int(ys_idx.min()), int(ys_idx.max())])

    if not xs:
        return 0, 0, w, h

    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    side = max(x2 - x1 + 1, y2 - y1 + 1)
    side = max(int(side * scale), min_side)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0 - side * top_extra
    crop_x1 = max(0, min(w - side, int(round(cx - side / 2))))
    crop_y1 = max(0, min(h - side, int(round(cy - side / 2))))
    return crop_x1, crop_y1, min(w, crop_x1 + side), min(h, crop_y1 + side)


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    metric_map = load_metric_map()

    fig, axes = plt.subplots(
        nrows=len(CASES),
        ncols=3,
        figsize=(11.8, 11.2),
        gridspec_kw={"wspace": 0.05, "hspace": 0.28},
    )
    fig.patch.set_facecolor("white")

    header = FancyBboxPatch(
        (0.06, 0.946),
        0.88,
        0.040,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.0,
        edgecolor=EDGE,
        facecolor=PANEL,
        transform=fig.transFigure,
    )
    fig.patches.append(header)
    fig.text(
        0.08,
        0.966,
        "典型失败案例。每行依次展示原始帧、真实标注和模型预测；红色为动脉，绿色为静脉。首行保留 YOLO 首帧框，用于说明初始化偏差。",
        ha="left",
        va="center",
        fontsize=10.4,
        color=TEXT,
        fontproperties=font,
    )

    row_meta = []
    for row_idx, case_info in enumerate(CASES):
        frame = case_info["frame"]
        stem = f"{frame:05d}"
        viz_rgb = cv2.cvtColor(cv2.imread(str(VIS_ROOT / case_info["case"] / f"{stem}_viz.jpg")), cv2.COLOR_BGR2RGB)
        raw_panel, gt_panel, pred_panel = split_panels(viz_rgb)
        crop = compute_crop_box(
            raw_panel.shape,
            [colored_mask(raw_panel), colored_mask(gt_panel), colored_mask(pred_panel)],
            top_extra=case_info["top_extra"],
        )
        x1, y1, x2, y2 = crop

        for ax, panel in zip(axes[row_idx], [raw_panel, gt_panel, pred_panel]):
            ax.imshow(panel[y1:y2, x1:x2])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(EDGE)
                spine.set_linewidth(0.9)

        metrics = metric_map[(case_info["case"], frame)]
        axes[row_idx, 0].text(
            0.02,
            0.02,
            f"第 {frame} 帧",
            ha="left",
            va="bottom",
            fontsize=9.5,
            color=TEXT,
            fontproperties=font,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=EDGE, linewidth=0.8),
            transform=axes[row_idx, 0].transAxes,
        )
        row_meta.append((case_info, metrics, axes[row_idx, 0], axes[row_idx, -1]))

    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.04)

    for case_info, metrics, ax_l, ax_r in row_meta:
        pos_l = ax_l.get_position()
        pos_r = ax_r.get_position()
        strip_y = pos_l.y1 + 0.006
        strip_h = 0.031
        strip = FancyBboxPatch(
            (pos_l.x0, strip_y),
            pos_r.x1 - pos_l.x0,
            strip_h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=1.0,
            edgecolor=case_info["strip_edge"],
            facecolor=case_info["strip_face"],
            transform=fig.transFigure,
            zorder=2,
        )
        fig.patches.append(strip)

        fig.text(
            pos_l.x0 + 0.012,
            strip_y + strip_h / 2,
            f"{case_info['label']}  {case_info['title']}  {case_info['desc']}",
            ha="left",
            va="center",
            fontsize=10.3,
            color=TEXT,
            fontproperties=font,
            zorder=3,
        )
        fig.text(
            pos_r.x1 - 0.012,
            strip_y + strip_h / 2,
            f"Mean Dice={metrics['mean_dice']:.3f} | A={metrics['artery_dice']:.3f} | V={metrics['vein_dice']:.3f}",
            ha="right",
            va="center",
            fontsize=10.0,
            color=SUBTEXT,
            fontproperties=font,
            zorder=3,
        )

    OUTPUT_PATHS[0].parent.mkdir(parents=True, exist_ok=True)
    for output_path in OUTPUT_PATHS:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
