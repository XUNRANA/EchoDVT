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
OUTPUT_PATH = ROOT / "artifacts/figures/fig3_8_prior_completion_examples.png"

ARTERY_COLOR = "#FF5050"
VEIN_COLOR = "#4A90E2"
TEXT_COLOR = "#222222"
LABEL_BG = (1.0, 1.0, 1.0, 0.86)

CASES = [
    {
        "case_name": "WENG_QINGYA-V1E-1_00000.jpg",
        "label": "例1",
    },
    {
        "case_name": "YU_GUIZHONG-V1E-_(3)_00000.jpg",
        "label": "例2",
    },
    {
        "case_name": "ZHAO_SUFANG-V1E1-3_00000.jpg",
        "label": "例3",
    },
]


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


def resolve_image_path(inference, case_name):
    return ROOT / "yolo" / inference.INPUT_DIR / case_name


def draw_boxes(ax, image_rgb, crop_bounds, boxes, font, row_label=None, panel_label=None, status_label=None):
    x1, y1, x2, y2 = crop_bounds
    crop_rgb = image_rgb[y1:y2, x1:x2]
    ax.imshow(crop_rgb)
    ax.axis("off")

    for key, color in [("artery", ARTERY_COLOR), ("vein", VEIN_COLOR)]:
        box = boxes.get(key)
        if box is None:
            continue
        rect = Rectangle(
            (box[0] - x1, box[1] - y1),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            linewidth=2.6,
            edgecolor=color,
        )
        ax.add_patch(rect)

    if row_label:
        ax.text(
            0.02,
            0.97,
            row_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color=TEXT_COLOR,
            fontproperties=font,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=LABEL_BG, edgecolor="none"),
        )
    if panel_label:
        ax.text(
            0.98,
            0.97,
            panel_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=TEXT_COLOR,
            fontproperties=font,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=LABEL_BG, edgecolor="none"),
        )
    if status_label:
        ax.text(
            0.98,
            0.05,
            status_label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.5,
            color=TEXT_COLOR,
            fontproperties=font,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=LABEL_BG, edgecolor="none"),
        )


def make_square_crop(boxes, image_shape, pad=30):
    h, w = image_shape[:2]
    x1 = min(box[0] for box in boxes) - pad
    y1 = min(box[1] for box in boxes) - pad
    x2 = max(box[2] for box in boxes) + pad
    y2 = max(box[3] for box in boxes) + pad

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1)
    side = min(max(side, 240), min(w, h))

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

    return max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)


def get_before_after(inference, detector, case_name):
    img_path = resolve_image_path(inference, case_name)
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

    before = {
        "artery": None if artery is None else artery["box"],
        "vein": None if vein is None else vein["box"],
    }
    if before["artery"] is None and before["vein"] is None:
        status_label = "无有效框"
    elif before["artery"] is None:
        status_label = "仅静脉"
    elif before["vein"] is None:
        status_label = "仅动脉"
    else:
        status_label = None

    final = detector.predict(image, conf=inference.CONF_THRESHOLD)
    after = {
        "artery": final["artery"]["box"] if final["artery"] is not None else None,
        "vein": final["vein"]["box"] if final["vein"] is not None else None,
    }

    crop_candidates = [b for b in [before["artery"], before["vein"], after["artery"], after["vein"]] if b is not None]
    crop_bounds = make_square_crop(crop_candidates, image.shape)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), before, after, crop_bounds, status_label


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    inference, detector = load_detector()

    fig, axes = plt.subplots(len(CASES), 2, figsize=(6.5, 9.1))
    if len(CASES) == 1:
        axes = [axes]

    for row_idx, case in enumerate(CASES):
        image_rgb, before, after, crop_bounds, status_label = get_before_after(inference, detector, case["case_name"])
        draw_boxes(
            axes[row_idx][0],
            image_rgb,
            crop_bounds,
            before,
            font,
            row_label=case["label"],
            panel_label="补全前",
            status_label=status_label,
        )
        draw_boxes(axes[row_idx][1], image_rgb, crop_bounds, after, font, panel_label="补全后")

    handles = [
        Line2D([0], [0], color=ARTERY_COLOR, linewidth=2.6, label="动脉框"),
        Line2D([0], [0], color=VEIN_COLOR, linewidth=2.6, label="静脉框"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        prop=font,
        fontsize=10,
        handlelength=2.0,
        columnspacing=1.8,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.08, wspace=0.04, hspace=0.05)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
