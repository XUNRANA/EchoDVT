import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
OUTPUT_PATH = ROOT / "artifacts/figures/fig4_9_segmentation_visualization.png"

DATA_ROOT = ROOT / "sam2/dataset/val"
VIS_ROOT = ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_123019_lora_r8_mfp15_rpa/visualizations"
MASK_ROOT = ROOT / "artifacts/e2e_classify_v3/masks"

CASE_NAME = "WANG_WEIFU-V1E"
FRAME_INDICES = [0, 39, 85]

TEXT = "#253041"
SUBTEXT = "#5B6574"
PANEL_EDGE = "#D1D5DB"
PALE = "#F8FAFC"
ARTERY = np.array([228, 91, 108], dtype=np.float32)
VEIN = np.array([61, 125, 216], dtype=np.float32)


def overlay_masks(raw_rgb: np.ndarray, artery_mask: np.ndarray, vein_mask: np.ndarray):
    out = raw_rgb.astype(np.float32).copy()
    if artery_mask.any():
        out[artery_mask] = out[artery_mask] * 0.34 + ARTERY * 0.66
    if vein_mask.any():
        out[vein_mask] = out[vein_mask] * 0.34 + VEIN * 0.66
    return np.clip(out, 0, 255).astype(np.uint8)


def compute_union_crop(shape, masks):
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
    side = max(int(side * 1.95), 180)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    crop_x1 = int(round(cx - side / 2))
    crop_y1 = int(round(cy - side / 2))
    crop_x1 = max(0, min(w - side, crop_x1))
    crop_y1 = max(0, min(h - side, crop_y1))
    return crop_x1, crop_y1, min(w, crop_x1 + side), min(h, crop_y1 + side)


def load_frame_data(frame_idx: int):
    frame_stem = f"{frame_idx:05d}"
    raw_rgb = cv2.cvtColor(cv2.imread(str(DATA_ROOT / CASE_NAME / "images" / f"{frame_stem}.jpg")), cv2.COLOR_BGR2RGB)
    semantic_mask = cv2.imread(str(MASK_ROOT / CASE_NAME / f"{frame_stem}.png"), cv2.IMREAD_GRAYSCALE)
    if semantic_mask is None:
        raise FileNotFoundError(MASK_ROOT / CASE_NAME / f"{frame_stem}.png")
    artery_mask = semantic_mask == 1
    vein_mask = semantic_mask == 2
    return {
        "frame_idx": frame_idx,
        "raw_rgb": raw_rgb,
        "artery_mask": artery_mask,
        "vein_mask": vein_mask,
        "artery_area": int(artery_mask.sum()),
        "vein_area": int(vein_mask.sum()),
    }


def add_chip(ax, xy, wh, text, font, facecolor, edgecolor, color):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=10.8,
        color=color,
        fontproperties=font,
        transform=ax.transAxes,
    )


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    frame_data = [load_frame_data(idx) for idx in FRAME_INDICES]
    crop = compute_union_crop(frame_data[0]["raw_rgb"].shape, [d["artery_mask"] for d in frame_data] + [d["vein_mask"] for d in frame_data])
    x1, y1, x2, y2 = crop

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 6.1))
    titles = ["首帧", "中间帧", "末帧"]

    for ax, title, data in zip(axes, titles, frame_data):
        panel = overlay_masks(data["raw_rgb"], data["artery_mask"], data["vein_mask"])[y1:y2, x1:x2]
        ax.imshow(panel)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title} / 第{data['frame_idx']}帧", fontsize=13.2, color=TEXT, fontproperties=font, pad=10)
        for spine in ax.spines.values():
            spine.set_color(PANEL_EDGE)
            spine.set_linewidth(0.9)

        add_chip(ax, (0.04, -0.16), (0.34, 0.11), f"动脉面积 {data['artery_area']:,} px", font, "#FFF1F2", "#F5B6BE", "#B42337")
        add_chip(ax, (0.42, -0.16), (0.34, 0.11), f"静脉面积 {data['vein_area']:,} px", font, "#EEF4FF", "#B8CCE6", "#1D4ED8")

    note = FancyBboxPatch(
        (0.06, 0.90),
        0.88,
        0.05,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.0,
        edgecolor="#D1D5DB",
        facecolor=PALE,
        transform=fig.transFigure,
    )
    fig.patches.append(note)
    fig.text(
        0.08,
        0.925,
        "匿名示例视频：展示首帧、中间帧与末帧的分割输出。红色半透明为动脉，蓝色半透明为静脉，可见静脉形态随加压过程发生明显变化。",
        ha="left",
        va="center",
        fontsize=11.0,
        color=TEXT,
        fontproperties=font,
    )

    fig.text(0.5, 0.14, "视频时间推进", ha="center", va="center", fontsize=11.6, color=SUBTEXT, fontproperties=font)
    timeline = FancyArrowPatch(
        (0.20, 0.12),
        (0.80, 0.12),
        transform=fig.transFigure,
        arrowstyle="Simple,head_width=10,head_length=12,tail_width=4",
        linewidth=0,
        facecolor="#64748B",
        edgecolor="#64748B",
        alpha=0.82,
    )
    fig.patches.append(timeline)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.82, bottom=0.23, wspace=0.08)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
