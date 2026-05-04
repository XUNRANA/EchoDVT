import csv
import os
from collections import defaultdict
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
VIS_ROOT = ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_094056_lora_r8_mfp15/visualizations"
FRAME_METRICS = ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_094056_lora_r8_mfp15/frame_metrics.csv"

OUTPUT_PATHS = [
    ROOT / "artifacts/figures/fig7_6.png",
    ROOT / "artifacts/figures/fig7_6_segmentation_case_grid.png",
]

CASES = [
    {"case": "Wen_ju_hua_V1", "label": "正常病例 A", "group": "normal"},
    {"case": "Ren_min_V1", "label": "正常病例 B", "group": "normal"},
    {"case": "CHEN_GENDI-V1E", "label": "DVT 病例 A", "group": "dvt"},
    {"case": "NI_YUTONG-V1E-2", "label": "DVT 病例 B", "group": "dvt"},
]

TEXT = "#253041"
SUBTEXT = "#5B6574"
EDGE = "#D1D5DB"
PANEL = "#F8FAFC"
NORMAL_BG = "#EEF4FF"
NORMAL_EDGE = "#BFD1F5"
DVT_BG = "#FFF0F1"
DVT_EDGE = "#F1C3C7"
ARTERY = np.array([220, 76, 92], dtype=np.uint8)
VEIN = np.array([47, 111, 237], dtype=np.uint8)


def load_case_frames():
    frame_map = defaultdict(list)
    with FRAME_METRICS.open() as f:
        for row in csv.DictReader(f):
            frame_map[row["case"]].append(int(row["frame"]))
    return {case: sorted(frames) for case, frames in frame_map.items()}


def split_panels(viz_rgb: np.ndarray):
    _, w = viz_rgb.shape[:2]
    panel_w = (w - 16) // 3
    raw = viz_rgb[:, :panel_w]
    gt = viz_rgb[:, panel_w + 8 : panel_w * 2 + 8]
    pred = viz_rgb[:, panel_w * 2 + 16 : panel_w * 3 + 16]
    return raw, gt, pred


def keep_dense_components(mask: np.ndarray, min_area: int = 120, min_fill: float = 0.28) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    refined = np.zeros_like(mask, dtype=bool)
    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area < min_area:
            continue
        fill_ratio = area / max(w * h, 1)
        if fill_ratio < min_fill:
            continue
        refined |= labels == label_idx
    return refined


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
    artery = keep_dense_components(artery)
    vein = keep_dense_components(vein)
    return artery, vein


def render_overlay(raw_rgb: np.ndarray, pred_panel: np.ndarray, alpha: float = 0.70):
    out = raw_rgb.astype(np.float32).copy()
    artery_mask, vein_mask = extract_colored_masks(pred_panel)
    if artery_mask.any():
        out[artery_mask] = out[artery_mask] * (1.0 - alpha) + ARTERY.astype(np.float32) * alpha
    if vein_mask.any():
        out[vein_mask] = out[vein_mask] * (1.0 - alpha) + VEIN.astype(np.float32) * alpha
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
    side = max(int(side * 2.05), 180)
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


def load_case_panels(case_name: str, frame_stems: list[str]):
    raw_images = []
    overlays = []
    masks = []
    for idx, frame_stem in enumerate(frame_stems):
        raw_rgb = cv2.cvtColor(cv2.imread(str(DATA_ROOT / case_name / "images" / f"{frame_stem}.jpg")), cv2.COLOR_BGR2RGB)
        viz_rgb = cv2.cvtColor(cv2.imread(str(VIS_ROOT / case_name / f"{frame_stem}_viz.jpg")), cv2.COLOR_BGR2RGB)
        _, gt_panel, pred_panel = split_panels(viz_rgb)
        # The first-frame prediction panel still contains the box prompt outline.
        # Use the clean overlay panel there so the thesis figure focuses on the mask itself.
        source_panel = gt_panel if idx == 0 else pred_panel
        overlay, artery_mask, vein_mask = render_overlay(raw_rgb, source_panel)
        raw_images.append(raw_rgb)
        overlays.append(overlay)
        masks.extend([artery_mask, vein_mask])

    crop_box = compute_crop_box(raw_images[0].shape, masks)
    return [crop(panel, crop_box) for panel in overlays]


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    frame_map = load_case_frames()

    fig, axes = plt.subplots(
        nrows=len(CASES),
        ncols=3,
        figsize=(11.6, 13.6),
        gridspec_kw={"wspace": 0.06, "hspace": 0.20},
    )
    fig.patch.set_facecolor("white")

    note = FancyBboxPatch(
        (0.06, 0.942),
        0.88,
        0.045,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.0,
        edgecolor=EDGE,
        facecolor=PANEL,
        transform=fig.transFigure,
    )
    fig.patches.append(note)
    fig.text(
        0.08,
        0.964,
        "LoRA r8 + MFP 分割结果可视化。红色半透明为动脉掩码，蓝色半透明为静脉掩码；每行对应 1 个匿名病例，依次展示首帧、中间帧和末帧。",
        ha="left",
        va="center",
        fontsize=10.7,
        color=TEXT,
        fontproperties=font,
    )

    col_titles = ["首帧", "中间帧", "末帧"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12.8, color=TEXT, fontproperties=font, pad=10)

    for row_idx, case_info in enumerate(CASES):
        case_name = case_info["case"]
        frames = frame_map[case_name]
        selected = [frames[0], frames[len(frames) // 2], frames[-1]]
        frame_stems = [f"{frame:05d}" for frame in selected]
        panels = load_case_panels(case_name, frame_stems)

        for col_idx, (panel, frame_idx) in enumerate(zip(panels, selected)):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(EDGE)
                spine.set_linewidth(0.9)

            ax.text(
                0.02,
                0.98,
                f"第 {frame_idx} 帧",
                ha="left",
                va="top",
                fontsize=9.9,
                color=TEXT,
                fontproperties=font,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=EDGE, linewidth=0.8),
                transform=ax.transAxes,
                zorder=10,
            )

        row_bg = NORMAL_BG if case_info["group"] == "normal" else DVT_BG
        row_edge = NORMAL_EDGE if case_info["group"] == "normal" else DVT_EDGE
        axes[row_idx, 0].text(
            -0.16,
            0.50,
            case_info["label"],
            ha="center",
            va="center",
            rotation=90,
            fontsize=11.0,
            color=TEXT,
            fontproperties=font,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=row_bg, edgecolor=row_edge, linewidth=1.0),
            transform=axes[row_idx, 0].transAxes,
            zorder=12,
        )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.04)
    OUTPUT_PATHS[0].parent.mkdir(parents=True, exist_ok=True)
    for output_path in OUTPUT_PATHS:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
