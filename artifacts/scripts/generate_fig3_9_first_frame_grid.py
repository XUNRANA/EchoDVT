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
OUTPUT_PATH = ROOT / "artifacts/figures/fig3_9_first_frame_grid.png"

ARTERY_COLOR = "#FF5050"
VEIN_COLOR = "#4A90E2"
TEXT_COLOR = "#222222"
LABEL_BG = (1.0, 1.0, 1.0, 0.82)

CASE_NAMES = [
    "AN_LEKA-V1E_00000.jpg",
    "Bao_hua_yu_V1_00000.jpg",
    "CHEN_GENDI-V1E_00000.jpg",
    "Duan_jie_gang_V1_00000.jpg",
    "Fang_ting_hua_V1_00000.jpg",
    "LIU_REN-V1_00000.jpg",
    "Ma_xin_V1_00000.jpg",
    "QU_ZHONGYUN-V1E_00000.jpg",
    "Shen_yue_ping_V1_00000.jpg",
    "Tong_shen_yao_V1_00000.jpg",
    "Wen_ju_hua_V1_00000.jpg",
    "Xiang_li_li_V1_00000.jpg",
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


def short_case_name(case_name):
    stem = case_name.replace("_00000.jpg", "")
    if len(stem) <= 16:
        return stem
    return stem[:16] + "..."


def get_prediction(inference, detector, case_name):
    img_path = ROOT / "yolo" / inference.INPUT_DIR / case_name
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(img_path)
    pred = detector.predict(image, conf=inference.CONF_THRESHOLD)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), pred


def draw_tile(ax, image_rgb, pred, font, case_name):
    ax.imshow(image_rgb)
    ax.axis("off")

    for key, color in [("artery", ARTERY_COLOR), ("vein", VEIN_COLOR)]:
        item = pred.get(key)
        if item is None:
            continue
        box = item["box"]
        rect = Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            linewidth=1.8,
            edgecolor=color,
        )
        ax.add_patch(rect)

    ax.text(
        0.02,
        0.98,
        short_case_name(case_name),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color=TEXT_COLOR,
        fontproperties=font,
        bbox=dict(boxstyle="round,pad=0.18", facecolor=LABEL_BG, edgecolor="none"),
    )


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))
    inference, detector = load_detector()

    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.8, 8.2))

    for ax, case_name in zip(axes.flat, CASE_NAMES):
        image_rgb, pred = get_prediction(inference, detector, case_name)
        draw_tile(ax, image_rgb, pred, font, case_name)

    handles = [
        Line2D([0], [0], color=ARTERY_COLOR, linewidth=2.2, label="动脉框"),
        Line2D([0], [0], color=VEIN_COLOR, linewidth=2.2, label="静脉框"),
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

    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.07, wspace=0.03, hspace=0.04)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
