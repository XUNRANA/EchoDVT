"""
模块 2: YOLO 检测展示
- 加载 YOLO 模型（支持选择不同训练步骤的权重）
- 在首帧上运行检测
- 可视化 artery/vein 检测框（含置信度、推断/修正标记）
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "yolo"))

from web.utils.visualization import draw_detection_boxes, bgr_to_rgb


def _find_yolo_weights() -> Dict[str, str]:
    """搜索所有可用的 YOLO 权重"""
    weights = {}
    search_dirs = [
        PROJECT_ROOT / "yolo" / "runs",
        PROJECT_ROOT / "yolo" / "checkpoints",
        Path("/data1/ouyangxinglong/EchoDVT/yolo/runs"),
    ]
    for base in search_dirs:
        if not base.exists():
            continue
        for pt_file in base.rglob("best.pt"):
            # 用相对路径做展示名
            rel = str(pt_file.relative_to(base.parent))
            weights[rel] = str(pt_file)
    # 如果找不到任何权重，提供默认路径
    if not weights:
        default = PROJECT_ROOT / "yolo" / "runs" / "detect" / "runs" / "detect" / "dvt_runs" / "aug_step5_speckle_translate_scale" / "weights" / "best.pt"
        weights["默认 (step5_speckle)"] = str(default)
    return weights


def _find_prior_stats() -> str:
    """查找先验统计文件"""
    candidates = [
        PROJECT_ROOT / "yolo" / "prior_stats.json",
        Path("/data1/ouyangxinglong/EchoDVT/yolo/prior_stats.json"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0])


def _run_yolo_detection(state: dict, weight_key: str, conf_threshold: float):
    """运行 YOLO 检测"""
    if not state.get("frame_files"):
        return state, None, "⚠️ 请先在「视频上传」Tab 中加载案例"

    # 加载首帧
    first_frame_path = state["frame_files"][0]
    img = cv2.imread(first_frame_path)
    if img is None:
        return state, None, f"❌ 无法读取图像: {first_frame_path}"

    h, w = img.shape[:2]

    # 查找权重
    available_weights = _find_yolo_weights()
    model_path = available_weights.get(weight_key)
    if not model_path or not Path(model_path).exists():
        # 模拟模式：使用 GT mask 提取框（用于 demo）
        return _demo_detection_from_gt(state, img)

    # 加载 YOLO 模型
    try:
        from inference import VesselDetector
        prior_path = _find_prior_stats()
        detector = VesselDetector(model_path, device=0, prior_path=prior_path)
        result = detector.predict(img, conf=conf_threshold)
    except ImportError:
        # 如果无法导入 ultralytics，使用 GT mask
        return _demo_detection_from_gt(state, img)
    except Exception as e:
        return state, None, f"❌ YOLO 推理失败: {e}"

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
        return state, bgr_to_rgb(img), "⚠️ 无 YOLO 权重且无 GT mask，无法进行检测"

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

    report = "ℹ️ **Demo 模式**: 检测框从 GT Mask 中提取（非 YOLO 推理）\n\n"
    report += _format_detection_report(result, w, h)

    return state, vis_rgb, report


def _format_detection_report(result: dict, w: int, h: int) -> str:
    """格式化检测结果报告"""
    lines = ["### 🎯 检测结果\n"]
    lines.append("| 类别 | 置信度 | 位置 (x1,y1,x2,y2) | 状态 |")
    lines.append("|------|--------|---------------------|------|")

    for cls_name in ("artery", "vein"):
        det = result.get(cls_name)
        if det is None:
            lines.append(f"| {cls_name} | — | — | ❌ 未检测到 |")
            continue

        box = det["box"]
        conf = det.get("conf", 0.0)
        box_str = f"({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})"

        tags = []
        if det.get("from_gt"):
            tags.append("GT 提取")
        if det.get("inferred"):
            tags.append("🔮 推断补全")
        if det.get("fixed"):
            tags.append("🔧 重叠修正")
        if det.get("prior_all"):
            tags.append("📊 先验补全")
        if not tags:
            tags.append("✅ 正常检测")

        status = " ".join(tags)
        lines.append(f"| {cls_name} | {conf:.3f} | {box_str} | {status} |")

    return "\n".join(lines)


def build_detection_tab(state: gr.State):
    """构建 YOLO 检测展示 Tab"""

    gr.Markdown("""
    ### YOLO 血管检测
    在首帧上运行 YOLO 检测，识别动脉 (artery) 和静脉 (vein) 的边界框。
    缺失框会通过统计先验自动补全。
    """)

    with gr.Row():
        with gr.Column(scale=1):
            available_weights = _find_yolo_weights()
            weight_choices = list(available_weights.keys())

            yolo_weight = gr.Dropdown(
                choices=weight_choices,
                value=weight_choices[0] if weight_choices else None,
                label="🏋️ YOLO 权重",
                info="选择不同训练阶段的模型权重",
            )

            conf_slider = gr.Slider(
                minimum=0.01, maximum=0.95, value=0.1, step=0.01,
                label="置信度阈值",
                info="低阈值可检测更多目标，但可能有误检",
            )

            detect_btn = gr.Button("🎯 运行检测", variant="primary", size="lg")

            detection_report = gr.Markdown("点击「运行检测」查看结果")

        with gr.Column(scale=2):
            detection_image = gr.Image(
                label="首帧检测结果（红色=动脉，绿色=静脉）",
                height=500,
                type="numpy",
            )

    detect_btn.click(
        fn=_run_yolo_detection,
        inputs=[state, yolo_weight, conf_slider],
        outputs=[state, detection_image, detection_report],
    )
