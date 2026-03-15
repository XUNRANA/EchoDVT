"""
EchoDVT 可视化工具
- 检测框绘制
- Mask 叠加
- 对比图生成
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# 颜色定义 (BGR)
COLORS = {
    "artery": (0, 0, 255),       # 红色
    "vein": (0, 255, 0),         # 绿色
    "artery_rgb": (255, 0, 0),
    "vein_rgb": (0, 255, 0),
}


def draw_detection_boxes(
    image: np.ndarray,
    detections: Dict[str, dict],
    line_width: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    在图像上绘制 YOLO 检测框

    Args:
        image: BGR 图像
        detections: {"artery": {"box": [x1,y1,x2,y2], "conf": 0.9, ...}, "vein": {...}}
    """
    out = image.copy()
    for cls_name in ("artery", "vein"):
        det = detections.get(cls_name)
        if det is None:
            continue
        box = det["box"]
        x1, y1, x2, y2 = [int(v) for v in box]
        color = COLORS[cls_name]
        conf = det.get("conf", 0.0)

        # 绘制矩形 (推断框用虚线)
        if det.get("inferred") or det.get("prior_all"):
            _draw_dashed_rect(out, (x1, y1), (x2, y2), color, line_width)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, line_width)

        # 标签
        tags = [f"{cls_name}: {conf:.2f}"]
        if det.get("inferred"):
            tags.append("推断")
        if det.get("fixed"):
            tags.append("修正")
        if det.get("prior_all"):
            tags.append("先验补全")

        label = " | ".join(tags)
        label_y = max(20, y1 - 8)

        # 标签背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, label_y - th - 6), (x1 + tw + 8, label_y + 4), color, -1)
        cv2.putText(out, label, (x1 + 4, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def overlay_masks(
    image: np.ndarray,
    semantic_mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    将分割 mask 半透明叠加在图像上

    Args:
        image: BGR 图像
        semantic_mask: 语义标签 (0=bg, 1=artery, 2=vein)
        alpha: 叠加透明度
    """
    canvas = image.copy()
    for cls_name, cls_value in [("artery", 1), ("vein", 2)]:
        region = semantic_mask == cls_value
        if not np.any(region):
            continue
        color = np.array(COLORS[cls_name], dtype=np.float32)
        src = canvas[region].astype(np.float32)
        blended = src * (1.0 - alpha) + color * alpha
        canvas[region] = blended.astype(np.uint8)

    # 绘制轮廓
    for cls_name, cls_value in [("artery", 1), ("vein", 2)]:
        mask_bin = (semantic_mask == cls_value).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, COLORS[cls_name], 1)

    return canvas


def build_comparison_image(
    original: np.ndarray,
    gt_mask: Optional[np.ndarray],
    pred_mask: np.ndarray,
    detections: Dict[str, dict],
    metrics: Optional[Dict[str, float]] = None,
    frame_idx: int = 0,
    is_prompt_frame: bool = False,
) -> np.ndarray:
    """
    构建三面板对比图: 原图+检测框 | GT | 预测

    Returns:
        拼接后的 BGR 图像
    """
    h, w = original.shape[:2]

    # Panel 1: 原图 + YOLO 框
    panel1 = draw_detection_boxes(original, detections)
    cv2.putText(panel1, f"Frame #{frame_idx} + YOLO", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 2: GT Mask
    if gt_mask is not None:
        panel2 = overlay_masks(original, gt_mask, alpha=0.5)
        cv2.putText(panel2, "Ground Truth", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        panel2 = original.copy()
        cv2.putText(panel2, "No GT Available", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2, cv2.LINE_AA)

    # Panel 3: 预测 Mask
    panel3 = overlay_masks(original, pred_mask, alpha=0.5)
    mode_text = "SAM2 Prediction (Prompt)" if is_prompt_frame else "SAM2 Prediction (Memory)"
    cv2.putText(panel3, mode_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # 添加指标文字
    if metrics:
        y = 56
        for key in ["mean_dice", "miou", "artery_dice", "vein_dice"]:
            if key in metrics:
                pretty_name = key.replace("_", " ").title()
                text = f"{pretty_name}: {metrics[key]:.4f}"
                cv2.putText(panel3, text, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                y += 22

    # 分隔线
    sep = np.full((h, 6, 3), 40, dtype=np.uint8)
    return np.hstack([panel1, sep, panel2, sep, panel3])


def create_area_chart_data(
    frame_areas: Dict[int, Dict[str, float]],
) -> Dict[str, list]:
    """
    从逐帧面积数据生成图表数据

    Args:
        frame_areas: {frame_idx: {"artery_area": float, "vein_area": float}}

    Returns:
        {"frames": [...], "artery": [...], "vein": [...]}
    """
    frames = sorted(frame_areas.keys())
    artery_areas = [frame_areas[f].get("artery_area", 0) for f in frames]
    vein_areas = [frame_areas[f].get("vein_area", 0) for f in frames]
    return {
        "frames": frames,
        "artery": artery_areas,
        "vein": vein_areas,
    }


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR → RGB 转换"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """RGB → BGR 转换"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _draw_dashed_rect(img, pt1, pt2, color, thickness, dash_len=10, gap_len=5):
    """绘制虚线矩形"""
    x1, y1 = pt1
    x2, y2 = pt2
    for p, q in [((x1, y1), (x2, y1)), ((x1, y2), (x2, y2)),
                 ((x1, y1), (x1, y2)), ((x2, y1), (x2, y2))]:
        _draw_dashed_line(img, p, q, color, thickness, dash_len, gap_len)


def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_len=10, gap_len=5):
    """绘制虚线"""
    dist = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    if dist == 0:
        return
    dx, dy = (pt2[0] - pt1[0]) / dist, (pt2[1] - pt1[1]) / dist
    pos = 0
    while pos < dist:
        sx, sy = int(pt1[0] + dx * pos), int(pt1[1] + dy * pos)
        ep = min(pos + dash_len, dist)
        ex, ey = int(pt1[0] + dx * ep), int(pt1[1] + dy * ep)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)
        pos += dash_len + gap_len
