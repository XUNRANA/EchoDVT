import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
OUTPUT_PATH = ROOT / "results/fig4_8_rpa_suppression.png"

DATA_ROOT = ROOT / "sam2/dataset/val"
MFP_VIS_ROOT = ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_094056_lora_r8_mfp15/visualizations"
RPA_VIS_ROOT = ROOT / "sam2/predictions/sam2_lora_yolo_box/val_20260315_123019_lora_r8_mfp15_rpa/visualizations"

CASE_NAME = "QIAN_SHUANGSHUANG-V1E"
FRAME_STEM = "00089"
FRAME_INDEX = int(FRAME_STEM)
MAX_DRIFT = 0.15

TEXT = "#253041"
LINE = "#5B6574"
PALE = "#F8FAFC"
ARTERY = "#E45B6C"
VEIN = "#3D7DD8"
DRIFT = "#E8871E"


def split_panels(viz_rgb: np.ndarray):
    _, w = viz_rgb.shape[:2]
    panel_w = (w - 16) // 3
    return (
        viz_rgb[:, :panel_w],
        viz_rgb[:, panel_w + 8 : panel_w * 2 + 8],
        viz_rgb[:, panel_w * 2 + 16 : panel_w * 3 + 16],
    )


def keep_largest(mask: np.ndarray, min_area: int):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    best = np.zeros_like(mask, dtype=bool)
    best_area = 0
    for idx in range(1, n_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area or area <= best_area:
            continue
        best_area = area
        best = labels == idx
    return best


def extract_prediction_masks(pred_panel: np.ndarray):
    h, w = pred_panel.shape[:2]
    hsv = cv2.cvtColor(cv2.cvtColor(pred_panel, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]

    artery = (((hue <= 10) | (hue >= 170)) & (sat >= 60) & (val >= 70))
    vein = ((hue >= 35) & (hue <= 95) & (sat >= 60) & (val >= 70))

    # Prediction panel carries metric text at top-left; drop it before component extraction.
    artery[: int(h * 0.36), : int(w * 0.62)] = False
    vein[: int(h * 0.22), : int(w * 0.40)] = False

    artery = cv2.morphologyEx(artery.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    vein = cv2.morphologyEx(vein.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    return keep_largest(artery, 200), keep_largest(vein, 100)


def load_gt_artery(case_name: str, frame_stem: str):
    mask_path = DATA_ROOT / case_name / "masks" / f"{frame_stem}.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    return mask == 1


def centroid(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def compute_baseline_offset(case_name: str):
    files = sorted((MFP_VIS_ROOT / case_name).glob("*_viz.jpg"), key=lambda p: int(p.stem.split("_")[0]))
    offsets = []
    for fp in files:
        pred_panel = split_panels(cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB))[2]
        artery, vein = extract_prediction_masks(pred_panel)
        if artery.sum() < 100 or vein.sum() < 100:
            continue
        artery_c = centroid(artery)
        vein_c = centroid(vein)
        offsets.append((vein_c[0] - artery_c[0], vein_c[1] - artery_c[1]))
        if len(offsets) >= 3:
            break

    if len(offsets) < 3:
        raise RuntimeError(f"{case_name} 无法从前三个好帧建立 RPA baseline。")

    return float(np.median([o[0] for o in offsets])), float(np.median([o[1] for o in offsets]))


def crop_box(shape, points):
    h, w = shape[:2]
    ys = [p[0] for p in points]
    xs = [p[1] for p in points]
    center_y = sum(ys) / len(ys)
    center_x = sum(xs) / len(xs)
    span = max(max(ys) - min(ys), max(xs) - min(xs), 90.0)
    side = max(int(span * 2.15), 190)
    y1 = int(round(center_y - side / 2))
    x1 = int(round(center_x - side / 2))
    y1 = max(0, min(h - side, y1))
    x1 = max(0, min(w - side, x1))
    return x1, y1, min(w, x1 + side), min(h, y1 + side)


def overlay_vein(raw_rgb: np.ndarray, vein_mask: np.ndarray):
    out = raw_rgb.astype(np.float32).copy()
    if vein_mask.any():
        color = np.array([61, 125, 216], dtype=np.float32)
        out[vein_mask] = out[vein_mask] * 0.34 + color * 0.66
    return np.clip(out, 0, 255).astype(np.uint8)


def add_arrow(ax, start, end, color, lw=2.0, style="-|>", mutation=13):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    return arrow


def add_chip(ax, xy, wh, text, font, facecolor, edgecolor, fontsize=11.2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=1.2,
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
        fontsize=fontsize,
        color=TEXT,
        fontproperties=font,
        transform=ax.transAxes,
    )
    return patch


def render_panel(ax, image, crop, anchor, expected, actual, font, title, show_actual_mask):
    x1, y1, x2, y2 = crop
    panel = image[y1:y2, x1:x2]
    ax.imshow(panel)
    ax.axis("off")
    ax.set_title(title, fontsize=13.2, color=TEXT, fontproperties=font, pad=8)

    anchor_y, anchor_x = anchor
    expected_y, expected_x = expected
    ax.add_patch(Circle((anchor_x - x1, anchor_y - y1), radius=5.8, facecolor=ARTERY, edgecolor="white", linewidth=1.0))
    ax.add_patch(Circle((expected_x - x1, expected_y - y1), radius=17.0, fill=False, edgecolor=VEIN, linewidth=2.0, linestyle="--"))
    add_arrow(ax, (anchor_x - x1, anchor_y - y1), (expected_x - x1, expected_y - y1), color=LINE, lw=1.8, mutation=12)

    ax.text(anchor_x - x1 + 10, anchor_y - y1 - 8, "动脉锚点", color=ARTERY, fontsize=11.3, fontproperties=font)
    ax.text(expected_x - x1 - 30, expected_y - y1 - 26, "期望静脉位置", color=VEIN, fontsize=11.0, fontproperties=font)

    if show_actual_mask and actual is not None:
        actual_y, actual_x = actual
        ax.add_patch(Circle((actual_x - x1, actual_y - y1), radius=4.5, facecolor=VEIN, edgecolor="white", linewidth=0.9))
        add_arrow(ax, (expected_x - x1 + 12, expected_y - y1 - 8), (actual_x - x1 - 12, actual_y - y1 + 10), color=DRIFT, lw=2.2, mutation=12)
        label_x = max(actual_x - x1 - 18, 12)
        label_y = max(actual_y - y1 - 18, 20)
        ax.text(label_x, label_y, "漂移静脉掩码", color=VEIN, fontsize=11.0, fontproperties=font)
        ax.text((expected_x + actual_x) / 2 - x1 - 6, (expected_y + actual_y) / 2 - y1 - 8, "drift", color=DRIFT, fontsize=10.8, fontproperties=font)
    else:
        add_chip(ax, (0.60, 0.07), (0.28, 0.12), "RPA：静脉掩码置零", font, "#FEF2F2", ARTERY, fontsize=11.0)


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))

    raw_rgb = cv2.cvtColor(cv2.imread(str(DATA_ROOT / CASE_NAME / "images" / f"{FRAME_STEM}.jpg")), cv2.COLOR_BGR2RGB)
    gt_artery = load_gt_artery(CASE_NAME, FRAME_STEM)
    gt_artery_c = centroid(gt_artery)

    mfp_pred_panel = split_panels(cv2.cvtColor(cv2.imread(str(MFP_VIS_ROOT / CASE_NAME / f"{FRAME_STEM}_viz.jpg")), cv2.COLOR_BGR2RGB))[2]
    rpa_pred_panel = split_panels(cv2.cvtColor(cv2.imread(str(RPA_VIS_ROOT / CASE_NAME / f"{FRAME_STEM}_viz.jpg")), cv2.COLOR_BGR2RGB))[2]

    artery_mfp, vein_mfp = extract_prediction_masks(mfp_pred_panel)
    artery_rpa, vein_rpa = extract_prediction_masks(rpa_pred_panel)
    artery_c = centroid(artery_mfp)
    vein_c = centroid(vein_mfp)

    if artery_c is None or vein_c is None:
        raise RuntimeError("无法从选定帧提取 MFP 的 artery/vein 质心。")

    baseline_dy, baseline_dx = compute_baseline_offset(CASE_NAME)
    expected = (artery_c[0] + baseline_dy, artery_c[1] + baseline_dx)
    drift = ((vein_c[0] - expected[0]) ** 2 + (vein_c[1] - expected[1]) ** 2) ** 0.5
    drift_norm = drift / ((raw_rgb.shape[0] ** 2 + raw_rgb.shape[1] ** 2) ** 0.5)

    crop = crop_box(raw_rgb.shape, [gt_artery_c, vein_c, expected])
    left_img = overlay_vein(raw_rgb, vein_mfp)
    right_img = overlay_vein(raw_rgb, vein_rpa)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2))

    render_panel(axes[0], left_img, crop, artery_c, expected, vein_c, font, "未启用 RPA", True)
    render_panel(axes[1], right_img, crop, artery_c, expected, None, font, "启用 RPA", False)

    note = FancyBboxPatch(
        (0.08, 0.915),
        0.84,
        0.052,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.0,
        edgecolor="#D1D5DB",
        facecolor=PALE,
        transform=fig.transFigure,
    )
    fig.patches.append(note)
    fig.text(
        0.10,
        0.941,
        f"匿名示例帧：第 {FRAME_INDEX} 帧。当前帧 drift = {drift_norm:.3f}，超过阈值 {MAX_DRIFT:.2f}，因此 RPA 将静脉掩码抑制为 0。",
        ha="left",
        va="center",
        fontsize=11.1,
        color=TEXT,
        fontproperties=font,
    )
    fig.text(
        0.50,
        0.06,
        "红点表示动脉锚点，蓝色虚线圆表示根据 baseline_offset 推算的期望静脉位置；病例名已隐去以保护隐私。",
        ha="center",
        va="center",
        fontsize=10.8,
        color=LINE,
        fontproperties=font,
    )

    fig.subplots_adjust(left=0.03, right=0.99, top=0.87, bottom=0.12, wspace=0.06)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
