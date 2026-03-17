"""
EchoDVT 指标计算工具
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UNIFIED_MODEL_META = PROJECT_ROOT / "results" / "unified_model" / "rf_unified.json"
DEFAULT_UNIFIED_THRESHOLD = 0.05


def get_unified_threshold(default: float = DEFAULT_UNIFIED_THRESHOLD) -> float:
    """读取最新统一模型阈值；缺失时回退到默认值。"""
    if not UNIFIED_MODEL_META.exists():
        return default
    try:
        meta = json.loads(UNIFIED_MODEL_META.read_text("utf-8"))
        return float(meta.get("threshold", default))
    except Exception:
        return default


def binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算二值 Dice 系数"""
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    intersection = float(np.logical_and(pred_bool, gt_bool).sum())
    total = float(pred_bool.sum() + gt_bool.sum())
    if total < 1e-6:
        return 1.0 if intersection < 1e-6 else 0.0
    return (2.0 * intersection) / total


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算二值 IoU"""
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    intersection = float(np.logical_and(pred_bool, gt_bool).sum())
    union = float(np.logical_or(pred_bool, gt_bool).sum())
    if union < 1e-6:
        return 1.0 if intersection < 1e-6 else 0.0
    return intersection / union


def compute_frame_metrics(
    pred_artery: np.ndarray,
    pred_vein: np.ndarray,
    gt_mask: np.ndarray,
) -> Dict[str, float]:
    """
    计算单帧的分割指标

    Args:
        pred_artery: 预测的动脉二值 mask
        pred_vein: 预测的静脉二值 mask
        gt_mask: GT 语义标注 (0=bg, 1=artery, 2=vein)
    """
    gt_artery = (gt_mask == 1)
    gt_vein = (gt_mask == 2)

    a_dice = binary_dice(pred_artery, gt_artery)
    a_iou = binary_iou(pred_artery, gt_artery)
    v_dice = binary_dice(pred_vein, gt_vein)
    v_iou = binary_iou(pred_vein, gt_vein)

    return {
        "artery_dice": a_dice,
        "artery_iou": a_iou,
        "vein_dice": v_dice,
        "vein_iou": v_iou,
        "mean_dice": float(np.mean([a_dice, v_dice])),
        "miou": float(np.mean([a_iou, v_iou])),
    }


def compute_mask_area(mask: np.ndarray, cls_value: int) -> int:
    """计算 mask 中某类别的像素面积"""
    return int((mask == cls_value).sum())


def compute_dvt_diagnosis(
    frame_vein_areas: list,
    threshold: float = DEFAULT_UNIFIED_THRESHOLD,
    robust: bool = True,
) -> Dict:
    """
    基于静脉面积变化率的 DVT 诊断

    核心逻辑：
    - 正常人：静脉在压缩过程中会塌陷，面积从大变到接近 0
      → min_area / max_area ≈ 0
    - DVT 患者：静脉面积基本不变
      → min_area / max_area ≈ 1

    Args:
        frame_vein_areas: 每帧的静脉面积列表
        threshold: 回退规则阈值，默认与最新统一模型阈值保持一致
        robust: 是否使用鲁棒 VCR（P10 代替 min，过滤分割噪声）

    Returns:
        诊断结果字典
    """
    if not frame_vein_areas or len(frame_vein_areas) < 2:
        return {
            "diagnosis": "数据不足",
            "confidence": 0.0,
            "area_ratio": None,
            "min_area": None,
            "max_area": None,
            "is_dvt": None,
        }

    areas = [a for a in frame_vein_areas if a > 0]
    if not areas:
        return {
            "diagnosis": "未检测到静脉",
            "confidence": 0.0,
            "area_ratio": None,
            "min_area": 0,
            "max_area": 0,
            "is_dvt": None,
        }

    max_area = max(areas)

    if robust and len(areas) >= 5:
        # 过滤分割噪声：去掉面积 < max 的 1% 的异常帧
        noise_floor = max_area * 0.01
        filtered = [a for a in areas if a >= noise_floor]
        if len(filtered) >= 3:
            areas = filtered
        # 用 P5 代替 min，与 classify_dvt.py 保持一致
        p5 = float(np.percentile(areas, 5))
        min_area = int(p5)
    else:
        min_area = min(areas)

    if max_area < 1:
        area_ratio = 0.0
    else:
        area_ratio = min_area / max_area

    is_dvt = area_ratio > threshold

    # 置信度：离阈值越远越高
    distance_from_threshold = abs(area_ratio - threshold)
    confidence = min(1.0, distance_from_threshold / 0.3)

    if is_dvt:
        diagnosis = f"⚠️ DVT 疑似（静脉拒绝塌陷）"
    else:
        diagnosis = f"✅ 正常（静脉正常塌陷）"

    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "area_ratio": area_ratio,
        "min_area": min_area,
        "max_area": max_area,
        "area_change_percent": (1.0 - area_ratio) * 100,
        "is_dvt": is_dvt,
        "threshold": threshold,
    }


def summarize_case_metrics(
    frame_metrics: list,
) -> Dict[str, float]:
    """汇总一个 case 的所有帧指标"""
    if not frame_metrics:
        return {}

    keys = ["artery_dice", "artery_iou", "vein_dice", "vein_iou", "mean_dice", "miou"]
    result = {}
    for key in keys:
        vals = [m[key] for m in frame_metrics if key in m]
        result[key] = float(np.mean(vals)) if vals else 0.0
    result["n_frames"] = len(frame_metrics)
    return result
