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
DATA_ROOT = ROOT / "sam2/dataset/val"

BASELINE_VIS_ROOT = ROOT / "sam2/predictions/sam2_large_yolo_box/val_20260314_111327/visualizations"
LORA_VIS_ROOT = ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_091610_lora_r8_baseline/visualizations"
BASELINE_METRICS = ROOT / "sam2/predictions/sam2_large_yolo_box/val_20260314_111327/frame_metrics.csv"
LORA_METRICS = ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_091610_lora_r8_baseline/frame_metrics.csv"

OUTPUT_PATHS = [
    ROOT / "artifacts/figures/fig7_4.png",
    ROOT / "artifacts/figures/fig7_4_lora_quality_comparison.png",
]

CASE_NAME = "CHEN_HUI-V1"
FRAME_STEMS = ["00051", "00055", "00058", "00062"]
FRAME_NOTES = {
    "00051": "静脉漏分割修复",
    "00055": "形变帧更完整",
    "00058": "边界更平滑",
    "00062": "时序更稳定",
}

TEXT = "#253041"
SUBTEXT = "#5B6574"
EDGE = "#D1D5DB"
GRID = "#D8E0EA"
PANEL = "#F8FAFC"
BASELINE_CHIP = "#FFF4F4"
BASELINE_EDGE = "#F1C3C7"
LORA_CHIP = "#EEF4FF"
LORA_EDGE = "#BFD6FF"
ARTERY = np.array([218, 76, 92], dtype=np.uint8)
VEIN = np.array([47, 111, 237], dtype=np.uint8)


def load_metrics(path: Path):
    metrics = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            metrics[(row["case"], row["frame"])] = {
                "mean_dice": float(row["mean_dice"]),
                "vein_dice": float(row["vein_dice"]),
                "artery_dice": float(row["artery_dice"]),
            }
    return metrics


def split_panels(viz_rgb: np.ndarray):
    _, w = viz_rgb.shape[:2]
    panel_w = (w - 16) // 3
    raw = viz_rgb[:, :panel_w]
    gt = viz_rgb[:, panel_w + 8 : panel_w * 2 + 8]
    pred = viz_rgb[:, panel_w * 2 + 16 : panel_w * 3 + 16]
    return raw, gt, pred


def extract_colored_masks(panel_rgb: np.ndarray):
    bgr = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    artery = (((h <= 10) | (h >= 170)) & (s >= 80) & (v >= 70))
    green_vein = ((h >= 35) & (h <= 95) & (s >= 80) & (v >= 70))
    blue_vein = ((h >= 100) & (h <= 140) & (s >= 80) & (v >= 70))
    vein = green_vein | blue_vein

    artery = cv2.morphologyEx(artery.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
    vein = cv2.morphologyEx(vein.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    return artery, vein


def render_clean_overlay(raw_panel: np.ndarray, overlay_panel: np.ndarray, alpha: float = 0.70) -> np.ndarray:
    out = raw_panel.astype(np.float32).copy()
    artery_mask, vein_mask = extract_colored_masks(overlay_panel)
    for mask, color in ((artery_mask, ARTERY), (vein_mask, VEIN)):
        if mask.any():
            out[mask] = out[mask] * (1.0 - alpha) + color.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8), artery_mask, vein_mask


def compute_crop_box(shape, masks):
    h, w = shape[:2]
    xs = []
    ys = []
    for mask in masks:
        mask_ys, mask_xs = np.where(mask)
        if len(mask_xs) == 0:
            continue
        xs.extend([int(mask_xs.min()), int(mask_xs.max())])
        ys.extend([int(mask_ys.min()), int(mask_ys.max())])

    if not xs:
        return 0, 0, w, h

    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    side = max(x2 - x1 + 1, y2 - y1 + 1)
    side = max(int(side * 2.0), 180)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    crop_x1 = int(round(cx - side / 2))
    crop_y1 = int(round(cy - side / 2))
    crop_x1 = max(0, min(w - side, crop_x1))
    crop_y1 = max(0, min(h - side, crop_y1))
    return crop_x1, crop_y1, min(w, crop_x1 + side), min(h, crop_y1 + side)


def crop(arr: np.ndarray, box):
    x1, y1, x2, y2 = box
    return arr[y1:y2, x1:x2]


def add_chip(ax, text, xy, facecolor, edgecolor, font, text_color=TEXT, width=0.34, height=0.14):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
        clip_on=False,
        zorder=10,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.02,
        y + height / 2,
        text,
        ha="left",
        va="center",
        fontsize=9.5,
        color=text_color,
        fontproperties=font,
        transform=ax.transAxes,
        zorder=11,
    )


def load_row(case_name: str, frame_stem: str):
    baseline_viz = cv2.cvtColor(cv2.imread(str(BASELINE_VIS_ROOT / case_name / f"{frame_stem}_viz.jpg")), cv2.COLOR_BGR2RGB)
    lora_viz = cv2.cvtColor(cv2.imread(str(LORA_VIS_ROOT / case_name / f"{frame_stem}_viz.jpg")), cv2.COLOR_BGR2RGB)
    raw_panel = cv2.cvtColor(cv2.imread(str(DATA_ROOT / case_name / "images" / f"{frame_stem}.jpg")), cv2.COLOR_BGR2RGB)

    _, gt_panel, baseline_pred_panel = split_panels(baseline_viz)
    _, _, lora_pred_panel = split_panels(lora_viz)

    baseline_overlay, b_artery, b_vein = render_clean_overlay(raw_panel, baseline_pred_panel)
    lora_overlay, l_artery, l_vein = render_clean_overlay(raw_panel, lora_pred_panel)
    _, gt_artery, gt_vein = render_clean_overlay(raw_panel, gt_panel, alpha=0.0)

    crop_box = compute_crop_box(raw_panel.shape, [gt_artery, gt_vein, b_artery, b_vein, l_artery, l_vein])
    return crop(baseline_overlay, crop_box), crop(lora_overlay, crop_box)


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    baseline_metrics = load_metrics(BASELINE_METRICS)
    lora_metrics = load_metrics(LORA_METRICS)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(FRAME_STEMS),
        figsize=(14.0, 7.2),
        gridspec_kw={"wspace": 0.06, "hspace": 0.14},
    )
    fig.patch.set_facecolor("white")

    note = FancyBboxPatch(
        (0.08, 0.935),
        0.84,
        0.048,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.0,
        edgecolor=EDGE,
        facecolor=PANEL,
        transform=fig.transFigure,
    )
    fig.patches.append(note)
    fig.text(
        0.10,
        0.958,
        "同一匿名病例的 4 个关键帧横向对比。上排为 Baseline，下排为 LoRA r8；可见 LoRA 在困难帧中减少漏分割，并使掩码边界更平滑、形态更完整。",
        ha="left",
        va="center",
        fontsize=10.8,
        color=TEXT,
        fontproperties=font,
    )

    for col_idx, frame_stem in enumerate(FRAME_STEMS):
        baseline_panel, lora_panel = load_row(CASE_NAME, frame_stem)
        frame_no = int(frame_stem)
        col_note = FRAME_NOTES.get(frame_stem, "")

        for row_idx, panel in enumerate([baseline_panel, lora_panel]):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(EDGE)
                spine.set_linewidth(0.9)

        axes[0, col_idx].set_title(
            f"第 {frame_no} 帧\n{col_note}",
            fontsize=11.4,
            color=TEXT,
            fontproperties=font,
            pad=10,
        )

        b_metrics = baseline_metrics[(CASE_NAME, frame_stem)]
        l_metrics = lora_metrics[(CASE_NAME, frame_stem)]

        add_chip(
            axes[0, col_idx],
            f"M {b_metrics['mean_dice']:.3f}\nV {b_metrics['vein_dice']:.3f}",
            (0.02, 0.02),
            BASELINE_CHIP,
            BASELINE_EDGE,
            font,
            width=0.31,
            height=0.14,
        )
        add_chip(
            axes[1, col_idx],
            f"M {l_metrics['mean_dice']:.3f}\nV {l_metrics['vein_dice']:.3f}",
            (0.02, 0.02),
            LORA_CHIP,
            LORA_EDGE,
            font,
            width=0.31,
            height=0.14,
        )

    axes[0, 0].text(
        0.02,
        0.98,
        "Baseline",
        ha="left",
        va="top",
        fontsize=11.0,
        color=TEXT,
        fontproperties=font,
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor=EDGE, linewidth=0.8),
        transform=axes[0, 0].transAxes,
        zorder=12,
    )
    axes[1, 0].text(
        0.02,
        0.98,
        "LoRA r8",
        ha="left",
        va="top",
        fontsize=11.0,
        color=TEXT,
        fontproperties=font,
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor=EDGE, linewidth=0.8),
        transform=axes[1, 0].transAxes,
        zorder=12,
    )

    fig.subplots_adjust(left=0.04, right=0.99, top=0.86, bottom=0.06)
    OUTPUT_PATHS[0].parent.mkdir(parents=True, exist_ok=True)
    for output_path in OUTPUT_PATHS:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
