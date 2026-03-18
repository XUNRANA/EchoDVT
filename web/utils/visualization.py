"""
EchoDVT 可视化工具
- 检测框绘制
- Mask 叠加
"""

import cv2
import numpy as np
from typing import Dict


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



def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR → RGB 转换"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



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
