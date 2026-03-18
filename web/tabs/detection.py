"""
模块 2: YOLO 检测展示
- 固定使用最优 YOLO 权重
- 在首帧上运行检测
- 可视化 artery/vein 检测框（含置信度、推断/修正标记）
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "yolo"))

from web.services.inference import DEFAULT_YOLO_MODEL
from web.utils.visualization import draw_detection_boxes, bgr_to_rgb
from web.utils.ui import render_page_header


FIXED_YOLO_CONFIDENCE = 0.1


def _get_fixed_yolo_model_display() -> str:
    """返回固定最优 YOLO 权重的展示名。"""
    try:
        return str(DEFAULT_YOLO_MODEL.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(DEFAULT_YOLO_MODEL)


def _run_yolo_detection(state: dict, conf_threshold: float):
    """运行 YOLO 检测"""
    if not state.get("frame_files"):
        return state, None, "请先在“数据输入”页加载案例"

    # 加载首帧
    first_frame_path = state["frame_files"][0]
    img = cv2.imread(first_frame_path)
    if img is None:
        return state, None, f"无法读取图像: {first_frame_path}"

    h, w = img.shape[:2]

    if not DEFAULT_YOLO_MODEL.exists():
        # 模拟模式：使用 GT mask 提取框（用于 demo）
        return _demo_detection_from_gt(state, img)

    # 加载 YOLO 模型（通过 InferenceService 单例）
    try:
        from web.services import InferenceService
        result = InferenceService.get().run_detection(img, conf=conf_threshold)
    except ImportError:
        # 如果无法导入 ultralytics，使用 GT mask
        return _demo_detection_from_gt(state, img)
    except Exception as e:
        return state, None, f"YOLO 推理失败: {e}"

    # 更新 state
    state["detections"] = result

    # 可视化
    vis = draw_detection_boxes(img, result)
    vis_rgb = bgr_to_rgb(vis)

    # 生成详细报告
    report = _format_detection_report(result, w, h)

    return state, vis_rgb, report


def _demo_detection_from_gt(state: dict, img: np.ndarray):
    """从 GT mask 中提取检测框作为 demo"""
    masks_dir = Path(state.get("masks_dir", ""))
    first_mask_path = masks_dir / "00000.png"

    if not first_mask_path.exists():
        return state, bgr_to_rgb(img), "固定 YOLO 权重缺失且无 GT mask，无法进行检测"

    gt_mask = cv2.imread(str(first_mask_path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]

    result = {}
    for cls_name, cls_val in [("artery", 1), ("vein", 2)]:
        ys, xs = np.where(gt_mask == cls_val)
        if len(xs) > 0:
            box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            result[cls_name] = {"box": box, "conf": 1.0, "from_gt": True}
        else:
            result[cls_name] = None

    state["detections"] = result

    vis = draw_detection_boxes(img, result)
    vis_rgb = bgr_to_rgb(vis)

    report = "**Demo 模式**: 检测框从 GT mask 中提取（非 YOLO 推理）\n\n"
    report += _format_detection_report(result, w, h)

    return state, vis_rgb, report


def _format_detection_report(result: dict, w: int, h: int) -> str:
    """格式化检测结果报告"""
    cls_cn = {"artery": "动脉", "vein": "静脉"}
    lines = ["### 检测结果\n"]
    lines.append("| 类别 | 置信度 | 位置 (x1,y1,x2,y2) | 状态 |")
    lines.append("|------|--------|---------------------|------|")

    for cls_name in ("artery", "vein"):
        cn = cls_cn[cls_name]
        det = result.get(cls_name)
        if det is None:
            lines.append(f"| {cn} | — | — | 未检测到 |")
            continue

        box = det["box"]
        conf = det.get("conf", 0.0)
        box_str = f"({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})"

        tags = []
        if det.get("from_gt"):
            tags.append("GT 提取")
        if det.get("inferred"):
            tags.append("推断补全")
        if det.get("fixed"):
            tags.append("重叠修正")
        if det.get("prior_all"):
            tags.append("先验补全")
        if not tags:
            tags.append("正常检测")

        status = " ".join(tags)
        lines.append(f"| {cn} | {conf:.3f} | {box_str} | {status} |")

    return "\n".join(lines)


def build_detection_tab(state: gr.State):
    """构建 YOLO 检测展示 Tab"""
    fixed_confidence = gr.State(FIXED_YOLO_CONFIDENCE)

    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            gr.HTML(render_page_header(
                "YOLO 血管检测",
                "在首帧上运行检测，生成动脉和静脉的边界框。",
                eyebrow="Detection",
            ))

            detect_btn = gr.Button("运行检测", variant="primary", size="lg")

            detection_report = gr.Markdown(
                "当前固定使用最优 YOLO 权重。若检测框缺失，系统会自动执行先验补全。"
            )

        with gr.Column(scale=3):
            detection_image = gr.Image(
                label="首帧检测结果（红色=动脉，绿色=静脉）",
                type="numpy",
            )

    detect_btn.click(
        fn=_run_yolo_detection,
        inputs=[state, fixed_confidence],
        outputs=[state, detection_image, detection_report],
    )
