import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")

BASELINE_VIS_ROOT = ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_091610_lora_r8_baseline/visualizations"
MFP_VIS_ROOT = ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_094056_lora_r8_mfp15/visualizations"
BASELINE_METRICS = ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_091610_lora_r8_baseline/frame_metrics.csv"
MFP_METRICS = ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_094056_lora_r8_mfp15/frame_metrics.csv"

PRESETS = {
    "default": {
        "output_path": ROOT / "results/fig4_6_mfp_improvement.png",
        "selected_examples": [
            ("XU_MEIYING-V1E", "00038"),
            ("YANG_MUCHEN-V1E1-6", "00034"),
            ("Huang_si_lei_V1", "00022"),
        ],
        "anonymous": False,
        "caption": "红色表示动脉，蓝色表示静脉；所示均为视频后期关键帧。",
    },
    "anonymous_alt": {
        "output_path": ROOT / "results/fig4_6_1.png",
        "selected_examples": [
            ("AN_LEKA-V1E", "00033"),
            ("Zhu_hai_yuan_V1E1", "00030"),
        ],
        "anonymous": True,
        "show_frame": True,
        "caption": "红色表示动脉，蓝色表示静脉；样例名称已隐去以保护隐私。",
    },
}

TEXT = "#1F2937"
SUBTEXT = "#4B5563"
VEIN_BLUE = np.array([80, 80, 255], dtype=np.uint8)
ARTERY_RED = np.array([214, 32, 32], dtype=np.uint8)


def load_metrics(path: Path):
    metrics = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            metrics[(row["case"], row["frame"])] = {
                "mean_dice": float(row["mean_dice"]),
                "vein_dice": float(row["vein_dice"]),
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


def render_clean_overlay(raw_panel: np.ndarray, overlay_panel: np.ndarray, alpha: float = 0.74) -> np.ndarray:
    out = raw_panel.astype(np.float32).copy()
    artery_mask, vein_mask = extract_colored_masks(overlay_panel)
    for mask, color in ((artery_mask, ARTERY_RED), (vein_mask, VEIN_BLUE)):
        if mask.any():
            out[mask] = out[mask] * (1.0 - alpha) + color.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def colored_bbox(panel_rgb: np.ndarray):
    rgb = panel_rgb.astype(np.int16)
    red_mask = (rgb[..., 0] > 150) & (rgb[..., 0] > rgb[..., 1] + 35) & (rgb[..., 0] > rgb[..., 2] + 35)
    green_mask = (rgb[..., 1] > 140) & (rgb[..., 1] > rgb[..., 0] + 20) & (rgb[..., 1] > rgb[..., 2] + 20)
    blue_mask = (rgb[..., 2] > 140) & (rgb[..., 2] > rgb[..., 0] + 20) & (rgb[..., 2] > rgb[..., 1] + 20)
    mask = red_mask | green_mask | blue_mask
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = panel_rgb.shape[:2]
        return 0, 0, w, h
    return xs.min(), ys.min(), xs.max(), ys.max()


def compute_crop_box(gt_panel: np.ndarray, base_pred: np.ndarray, mfp_pred: np.ndarray):
    boxes = [colored_bbox(gt_panel), colored_bbox(base_pred), colored_bbox(mfp_pred)]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    h, w = gt_panel.shape[:2]

    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1
    side = int(max(box_w, box_h) * 1.9)
    side = max(side, 180)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    crop_x1 = int(round(cx - side / 2))
    crop_y1 = int(round(cy - side / 2))
    crop_x2 = crop_x1 + side
    crop_y2 = crop_y1 + side

    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0
    if crop_x2 > w:
        shift = crop_x2 - w
        crop_x1 = max(0, crop_x1 - shift)
        crop_x2 = w
    if crop_y2 > h:
        shift = crop_y2 - h
        crop_y1 = max(0, crop_y1 - shift)
        crop_y2 = h

    crop_y1 = max(crop_y1, 70)
    crop_x1 = max(crop_x1, 30)
    return crop_x1, crop_y1, crop_x2, crop_y2


def crop(arr: np.ndarray, box):
    x1, y1, x2, y2 = box
    return arr[y1:y2, x1:x2]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Figure 4-6 comparison panels.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="default",
        help="Preset layout and sample selection to render.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path override.",
    )
    return parser.parse_args()


def resolve_config(args):
    config = dict(PRESETS[args.preset])
    if args.output is not None:
        config["output_path"] = args.output
    return config


def format_row_text(case_name: str, frame_stem: str, baseline: dict, mfp: dict, row_idx: int, anonymous: bool, show_frame: bool):
    if anonymous:
        prefix = f"样例 {row_idx + 1}"
        if show_frame:
            prefix += f" / 第{int(frame_stem)}帧"
    else:
        prefix = f"{case_name} / Frame {int(frame_stem)}"
    return (
        f"{prefix}"
        f"    Mean Dice: {baseline['mean_dice']:.3f} -> {mfp['mean_dice']:.3f}"
        f"    Vein Dice: {baseline['vein_dice']:.3f} -> {mfp['vein_dice']:.3f}"
    )


def render_figure(config: dict):
    selected_examples = config["selected_examples"]
    output_path = Path(config["output_path"])
    anonymous = config.get("anonymous", False)
    show_frame = config.get("show_frame", False)
    caption = config["caption"]

    font = font_manager.FontProperties(fname=str(FONT_PATH))
    baseline_metrics = load_metrics(BASELINE_METRICS)
    mfp_metrics = load_metrics(MFP_METRICS)

    fig_height = 3.2 * len(selected_examples) + 1.25
    bottom_margin = 0.12 if len(selected_examples) <= 2 else 0.08
    fig, axes = plt.subplots(
        nrows=len(selected_examples),
        ncols=3,
        figsize=(11.4, fig_height),
        gridspec_kw={"wspace": 0.04, "hspace": 0.32},
    )
    if len(selected_examples) == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["真实标注", "仅首帧提示", "MFP"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontproperties=font, fontsize=13.2, color=TEXT, pad=10)

    for row_idx, (case_name, frame_stem) in enumerate(selected_examples):
        base_viz = cv2.cvtColor(cv2.imread(str(BASELINE_VIS_ROOT / case_name / f"{frame_stem}_viz.jpg")), cv2.COLOR_BGR2RGB)
        mfp_viz = cv2.cvtColor(cv2.imread(str(MFP_VIS_ROOT / case_name / f"{frame_stem}_viz.jpg")), cv2.COLOR_BGR2RGB)

        raw_panel, gt_overlay_panel, base_pred_overlay_panel = split_panels(base_viz)
        _, _, mfp_pred_overlay_panel = split_panels(mfp_viz)

        gt_panel = render_clean_overlay(raw_panel, gt_overlay_panel)
        base_pred_panel = render_clean_overlay(raw_panel, base_pred_overlay_panel)
        mfp_pred_panel = render_clean_overlay(raw_panel, mfp_pred_overlay_panel)

        crop_box = compute_crop_box(gt_panel, base_pred_panel, mfp_pred_panel)
        panels = [crop(gt_panel, crop_box), crop(base_pred_panel, crop_box), crop(mfp_pred_panel, crop_box)]

        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#D1D5DB")
                spine.set_linewidth(0.8)

        baseline = baseline_metrics[(case_name, frame_stem)]
        mfp = mfp_metrics[(case_name, frame_stem)]
        row_text = format_row_text(case_name, frame_stem, baseline, mfp, row_idx, anonymous, show_frame)
        axes[row_idx, 1].text(
            0.5,
            -0.075,
            row_text,
            transform=axes[row_idx, 1].transAxes,
            ha="center",
            va="top",
            fontsize=10.6,
            color=TEXT,
            fontproperties=font,
        )

    fig.text(0.5, 0.025, caption, ha="center", va="center", fontsize=10.8, color=SUBTEXT, fontproperties=font)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=bottom_margin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(output_path)


def main():
    args = parse_args()
    config = resolve_config(args)
    render_figure(config)


if __name__ == "__main__":
    main()
