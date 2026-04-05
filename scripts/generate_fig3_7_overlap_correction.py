import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
OUTPUT_PATH = ROOT / "results/fig3_7_overlap_correction.png"
OUTPUT_PATH_ALT = ROOT / "results/fig3_7_1.png"

SOURCE_MODE = "sam2"
YOLO_CASE_NAME = "QIAN_SHUANGSHUANG-V1E_00000.jpg"
SAM2_CASE_DIR = "QIAN_SHUANGSHUANG-V1E"
SAM2_FRAME_ID = "00000"

ARTERY_COLOR = "#FF5050"
VEIN_COLOR = "#4A90E2"
TEXT_COLOR = "#222222"
LABEL_BG = (1.0, 1.0, 1.0, 0.86)


def load_detector():
    import sys

    yolo_dir = ROOT / "yolo"
    sys.path.insert(0, str(yolo_dir))
    os.chdir(yolo_dir)
    import inference  # noqa: PLC0415

    detector = inference.VesselDetector(
        inference.MODEL_PATH,
        device="cpu",
        prior_path=inference.PRIOR_PATH,
    )
    return inference, detector


def resolve_image_path(inference):
    if SOURCE_MODE == "sam2":
        return ROOT / "sam2" / "dataset" / "val" / SAM2_CASE_DIR / "images" / f"{SAM2_FRAME_ID}.jpg"
    return ROOT / "yolo" / inference.INPUT_DIR / YOLO_CASE_NAME


def get_raw_and_fixed(inference, detector):
    img_path = resolve_image_path(inference)
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(img_path)

    h, w = image.shape[:2]
    results = detector.model(image, conf=inference.CONF_THRESHOLD, device="cpu", verbose=False)[0]

    artery_list, vein_list = [], []
    for box in results.boxes:
        cls_id = int(box.cls)
        conf_score = float(box.conf)
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        item = {"box": xyxy, "conf": conf_score}
        if cls_id == 0:
            artery_list.append(item)
        else:
            vein_list.append(item)

    artery = max(artery_list, key=lambda x: x["conf"]) if artery_list else None
    vein = max(vein_list, key=lambda x: x["conf"]) if vein_list else None

    if artery is None or vein is None:
        artery, vein = detector._retry_lower_conf(image, artery, vein)
    if artery is None or vein is None:
        raise RuntimeError(f"{img_path.name} does not contain both artery and vein detections.")

    raw = {"artery": artery, "vein": vein}
    fixed_artery, fixed_vein = detector._check_and_fix_overlap(dict(artery), dict(vein), w, h)
    fixed = {"artery": fixed_artery, "vein": fixed_vein}

    raw_iou = detector._compute_iou_boxes(raw["artery"]["box"], raw["vein"]["box"])
    fixed_iou = detector._compute_iou_boxes(fixed["artery"]["box"], fixed["vein"]["box"])
    return image, raw, fixed, raw_iou, fixed_iou


def make_square_crop(boxes, image_shape, pad=44):
    h, w = image_shape[:2]
    x1 = min(box[0] for box in boxes) - pad
    y1 = min(box[1] for box in boxes) - pad
    x2 = max(box[2] for box in boxes) + pad
    y2 = max(box[3] for box in boxes) + pad

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1)
    side = min(max(side, 250), min(w, h))

    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = int(round(cx + side / 2))
    y2 = int(round(cy + side / 2))

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 -= x2 - w
        x2 = w
    if y2 > h:
        y1 -= y2 - h
        y2 = h

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w)
    y2 = min(y2, h)
    return x1, y1, x2, y2


def draw_panel(ax, crop_rgb, crop_bounds, boxes, font, panel_text, iou_value):
    x1, y1, _, _ = crop_bounds
    ax.imshow(crop_rgb)
    ax.axis("off")

    styles = [
        ("artery", ARTERY_COLOR),
        ("vein", VEIN_COLOR),
    ]
    for key, color in styles:
        box = boxes[key]["box"]
        rect = Rectangle(
            (box[0] - x1, box[1] - y1),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            linewidth=2.8,
            edgecolor=color,
        )
        ax.add_patch(rect)

    ax.text(
        0.03,
        0.97,
        panel_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=TEXT_COLOR,
        fontsize=12,
        fontproperties=font,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=LABEL_BG, edgecolor="none"),
    )
    ax.text(
        0.5,
        -0.07,
        f"IoU = {iou_value:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=TEXT_COLOR,
        fontsize=13,
        fontproperties=font,
    )


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    inference, detector = load_detector()
    image_bgr, raw, fixed, raw_iou, fixed_iou = get_raw_and_fixed(inference, detector)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    crop_bounds = make_square_crop(
        [
            raw["artery"]["box"],
            raw["vein"]["box"],
            fixed["artery"]["box"],
            fixed["vein"]["box"],
        ],
        image_rgb.shape,
    )
    x1, y1, x2, y2 = crop_bounds
    crop_rgb = image_rgb[y1:y2, x1:x2]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.7))
    draw_panel(axes[0], crop_rgb, crop_bounds, raw, font, "修正前", raw_iou)
    draw_panel(axes[1], crop_rgb, crop_bounds, fixed, font, "修正后", fixed_iou)

    handles = [
        Line2D([0], [0], color=ARTERY_COLOR, linewidth=2.8, label="动脉框"),
        Line2D([0], [0], color=VEIN_COLOR, linewidth=2.8, label="静脉框"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        prop=font,
        fontsize=11,
        handlelength=2.2,
        columnspacing=1.8,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.22, wspace=0.06)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUTPUT_PATH_ALT, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
