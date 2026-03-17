"""
Dashboard 首页模块
- 系统状态卡片（GPU / 内存 / 模型状态）
- 数据集统计卡片（train / val）
- train / val 统一模型指标概览
- 当前流程进度状态（已加载/检测/分割/诊断/评估）
- 数据集分布图表
"""

import gradio as gr
import numpy as np
import cv2
import time
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.utils.chart_style import setup_matplotlib, style_axis
from web.utils.metrics import compute_mask_area, get_unified_threshold
from web.services.inference import DEFAULT_SAM2_VARIANT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── 缓存路径 ───

_CACHE_DIR = Path(__file__).parent.parent / "assets" / "cache"
_VAL_ACC_CACHE = _CACHE_DIR / "val_accuracy.json"
_TEST_ACC_CACHE = _CACHE_DIR / "test_accuracy.json"
_VAL_HPARAM_REPORT = PROJECT_ROOT / "results" / "e2e_classify_v3" / "classification_report.json"
_UNIFIED_MODEL_META = PROJECT_ROOT / "results" / "unified_model" / "rf_unified.json"
_COMPREHENSIVE_VERSION = 3
_DEFAULT_TEST_NORMAL_SAMPLE = 500
_DEFAULT_TEST_PATIENT_SAMPLE = 50


def _get_unified_model_meta() -> Dict:
    """读取统一模型元信息"""
    if _UNIFIED_MODEL_META.exists():
        return json.loads(_UNIFIED_MODEL_META.read_text("utf-8"))
    return {}


def _get_model_meta_signature() -> Optional[str]:
    """返回统一模型元信息文件签名，用于判定缓存是否过期。"""
    if not _UNIFIED_MODEL_META.exists():
        return None
    stat = _UNIFIED_MODEL_META.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _get_model_meta_timestamp() -> str:
    """返回统一模型元信息更新时间。"""
    if not _UNIFIED_MODEL_META.exists():
        return "—"
    return datetime.fromtimestamp(_UNIFIED_MODEL_META.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def _get_val_fixed_threshold() -> float:
    """返回 RF 概率阈值（统一模型）。"""
    meta = _get_unified_model_meta()
    return float(meta.get("threshold", get_unified_threshold()))


def _get_train_case_count() -> int:
    train_normal_file = PROJECT_ROOT / "sam2" / "dataset" / "train_normal.txt"
    if train_normal_file.exists():
        return len(_read_label_file(train_normal_file))
    train_root = PROJECT_ROOT / "sam2" / "dataset" / "train"
    if train_root.exists():
        return len([d for d in train_root.iterdir() if d.is_dir()])
    return 0


def _get_meta_train_val_metrics() -> Optional[Dict]:
    """根据最新统一模型元信息构造 train/val 汇总指标。"""
    meta = _get_unified_model_meta()
    if not meta:
        return None

    try:
        train_total = _get_train_case_count()
        val_normal_set, val_abnormal_set = _get_val_labels()
        val_normal_total = len(val_normal_set)
        val_dvt_total = len(val_abnormal_set)
        val_total = val_normal_total + val_dvt_total

        threshold = float(meta.get("threshold", get_unified_threshold()))
        timestamp = _get_model_meta_timestamp()
        signature = _get_model_meta_signature()

        train_fp = max(0, min(train_total, int(meta.get("train_fp", 0))))
        train_tn = max(0, train_total - train_fp)
        train_metrics = _build_binary_metrics(
            tp=0,
            fp=train_fp,
            tn=train_tn,
            fn=0,
            correct=train_tn,
            total=train_total,
        )
        if train_total > 0 and "train_accuracy" in meta:
            train_metrics["accuracy"] = round(float(meta["train_accuracy"]), 4)
            train_metrics["correct"] = int(round(train_metrics["accuracy"] * train_total))
            train_metrics["tn"] = train_metrics["correct"]
            train_metrics["fp"] = max(0, train_total - train_metrics["correct"])
        train_metrics.update({
            "details": [],
            "available": train_total > 0,
            "timestamp": timestamp,
            "threshold": threshold,
            "classifier": "RF unified",
            "source": "model_meta",
            "model_meta_signature": signature,
            "version": 4,
        })

        val_fp = max(0, min(val_normal_total, int(meta.get("val_fp", 0))))
        val_fn = max(0, min(val_dvt_total, int(meta.get("val_fn", 0))))
        val_tp = max(0, val_dvt_total - val_fn)
        val_tn = max(0, val_normal_total - val_fp)
        val_correct = val_tp + val_tn
        val_metrics = _build_binary_metrics(
            tp=val_tp,
            fp=val_fp,
            tn=val_tn,
            fn=val_fn,
            correct=val_correct,
            total=val_total,
        )
        if "val_accuracy" in meta:
            val_metrics["accuracy"] = round(float(meta["val_accuracy"]), 4)
        if "val_precision" in meta:
            val_metrics["precision"] = round(float(meta["val_precision"]), 4)
        if "val_recall" in meta:
            val_metrics["recall"] = round(float(meta["val_recall"]), 4)
        precision = float(val_metrics.get("precision", 0))
        recall = float(val_metrics.get("recall", 0))
        val_metrics["f1"] = (
            round(2 * precision * recall / (precision + recall), 4)
            if (precision + recall) > 0
            else 0.0
        )
        val_metrics.update({
            "details": [],
            "available": val_total > 0,
            "timestamp": timestamp,
            "threshold": threshold,
            "classifier": "RF unified",
            "source": "model_meta",
            "model_meta_signature": signature,
            "version": 4,
        })

        return {
            "train": train_metrics,
            "val": val_metrics,
            "threshold": threshold,
            "timestamp": timestamp,
            "model_meta_signature": signature,
        }
    except Exception:
        return None


def _is_cache_current_for_model(cache_data: Dict | None) -> bool:
    """检查缓存是否对应当前统一模型。"""
    if not cache_data:
        return False
    current_signature = _get_model_meta_signature()
    if current_signature is None:
        return True
    return cache_data.get("model_meta_signature") == current_signature


# ─── 标签文件读取 ───

def _read_label_file(path: Path) -> List[str]:
    """读取标签文件，返回案例名列表"""
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text("utf-8").strip().split("\n") if line.strip()]


def _get_val_labels():
    """获取验证集标签"""
    normal = _read_label_file(PROJECT_ROOT / "sam2" / "dataset" / "val_normal.txt")
    abnormal = _read_label_file(PROJECT_ROOT / "sam2" / "dataset" / "val_abnormal.txt")
    return set(normal), set(abnormal)


def _get_test_labels():
    """获取测试集标签"""
    normal = _read_label_file(PROJECT_ROOT / "test" / "normal.txt")
    patient = _read_label_file(PROJECT_ROOT / "test" / "patient.txt")
    return set(normal), set(patient)


# ─── 系统信息采集 ───

def _get_gpu_info() -> Dict:
    """获取 GPU 信息"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            mem_used = torch.cuda.memory_allocated(0) / (1024**3)
            return {
                "available": True,
                "name": name,
                "mem_total_gb": round(mem_total, 1),
                "mem_used_gb": round(mem_used, 1),
                "utilization": round(mem_used / mem_total * 100, 1) if mem_total > 0 else 0,
            }
        return {"available": False}
    except ImportError:
        return {"available": False}


def _get_system_memory() -> Dict:
    """获取系统内存信息"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 1),
            "used_gb": round(mem.used / (1024**3), 1),
            "percent": mem.percent,
        }
    except ImportError:
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            info = {}
            for line in lines:
                parts = line.split()
                if parts[0] in ("MemTotal:", "MemAvailable:"):
                    info[parts[0].rstrip(":")] = int(parts[1])
            total = info.get("MemTotal", 0) / (1024**2)
            avail = info.get("MemAvailable", 0) / (1024**2)
            used = total - avail
            return {
                "total_gb": round(total, 1),
                "used_gb": round(used, 1),
                "percent": round(used / total * 100, 1) if total > 0 else 0,
            }
        except Exception:
            return {"total_gb": 0, "used_gb": 0, "percent": 0}


def _get_model_status() -> Dict:
    """检查模型权重是否存在"""
    checks = {
        "YOLO": list((PROJECT_ROOT / "yolo" / "runs").rglob("best.pt")) if (PROJECT_ROOT / "yolo" / "runs").exists() else [],
        "SAM2 Large": [PROJECT_ROOT / "sam2" / "checkpoints" / "sam2_hiera_large.pt"],
        "LoRA r8": list((PROJECT_ROOT / "sam2" / "checkpoints" / "lora_runs").rglob("lora_best.pt")) if (PROJECT_ROOT / "sam2" / "checkpoints" / "lora_runs").exists() else [],
    }
    status = {}
    for name, paths in checks.items():
        found = any(p.exists() for p in paths)
        status[name] = "✅ 就绪" if found else "❌ 缺失"
    return status


def _get_dataset_stats() -> Dict:
    """获取数据集统计信息（使用标签文件）"""
    from web.tabs.upload import _list_cases, _get_dataset_root

    val_cases = _list_cases("val")
    train_cases = _list_cases("train")

    val_normal_set, val_abnormal_set = _get_val_labels()

    val_normal = len(val_normal_set)
    val_patient = len(val_abnormal_set)

    # 估算总帧数
    root = _get_dataset_root()
    sample_sizes = []
    for split in ("val", "train"):
        split_dir = root / split
        if split_dir.exists():
            for case_dir in list(split_dir.iterdir())[:20]:
                img_dir = case_dir / "images"
                if img_dir.exists():
                    n = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
                    sample_sizes.append(n)

    avg_frames = np.mean(sample_sizes) if sample_sizes else 50
    total_frames_est = int(avg_frames * (len(val_cases) + len(train_cases)))

    return {
        "val_count": len(val_cases),
        "train_count": len(train_cases),
        "val_normal": val_normal,
        "val_patient": val_patient,
        "total_cases": len(val_cases) + len(train_cases),
        "total_frames_est": total_frames_est,
        "avg_frames": int(avg_frames),
    }


# ─── 批量准确率计算 ───

def _compute_val_accuracy(force_recompute: bool = False) -> Dict:
    """计算验证集 DVT 诊断准确率（基于 GT mask，无需模型推理）"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force_recompute:
        meta_metrics = _get_meta_train_val_metrics()
        if meta_metrics is not None:
            return meta_metrics["val"]

    # 检查缓存 — version=3 表示使用统一 RF 分类器
    if not force_recompute and _VAL_ACC_CACHE.exists():
        try:
            cached = json.loads(_VAL_ACC_CACHE.read_text("utf-8"))
            if cached.get("version") == 3:
                return cached
        except Exception:
            pass

    val_dir = PROJECT_ROOT / "sam2" / "dataset" / "val"
    if not val_dir.exists():
        return {"correct": 0, "total": 0, "accuracy": 0, "details": [], "version": 3}

    try:
        from classify_dvt import extract_features, predict_dvt
    except Exception as e:
        return {
            "correct": 0,
            "total": 0,
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "threshold": _get_val_fixed_threshold(),
            "classifier": "RF unified",
            "details": [],
            "version": 3,
            "error": f"统一分类器加载失败: {e}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    val_normal_set, val_abnormal_set = _get_val_labels()
    val_threshold = _get_val_fixed_threshold()

    correct = 0
    total = 0
    tp = fp = tn = fn = 0
    details = []

    for case_dir in sorted(val_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        case_name = case_dir.name
        masks_dir = case_dir / "masks"

        # 确定真实标签
        if case_name in val_normal_set:
            actual_dvt = False
        elif case_name in val_abnormal_set:
            actual_dvt = True
        else:
            continue

        if not masks_dir.exists():
            continue

        mask_files = sorted(masks_dir.glob("*.png"), key=lambda p: int(p.stem))
        if len(mask_files) < 2:
            continue

        masks = []
        for mf in mask_files:
            mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks.append(mask)

        if len(masks) < 2:
            continue

        try:
            features = extract_features(masks)
            result = predict_dvt(features)
        except Exception:
            continue
        predicted_dvt = result.get("is_dvt")
        if predicted_dvt is None:
            continue

        total += 1
        is_correct = (predicted_dvt == actual_dvt)
        if is_correct:
            correct += 1

        # 混淆矩阵
        if actual_dvt and predicted_dvt:
            tp += 1
        elif actual_dvt and not predicted_dvt:
            fn += 1
        elif not actual_dvt and predicted_dvt:
            fp += 1
        else:
            tn += 1

        details.append({
            "case": case_name,
            "actual": "DVT" if actual_dvt else "正常",
            "predicted": "DVT" if predicted_dvt else "正常",
            "correct": is_correct,
            "vcr": round(float(features.get("vcr", 0.0)), 4),
            "probability": round(float(result.get("probability", 0.0)), 4),
            "model": result.get("model", "RF unified"),
        })

    accuracy = round(correct / total, 4) if total > 0 else 0
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0

    result = {
        "version": 3,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "threshold": val_threshold,
        "classifier": "RF unified",
        "details": details,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        _VAL_ACC_CACHE.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    except Exception:
        pass

    return result


def _safe_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator > 0 else 0.0


def _build_binary_metrics(tp: int, fp: int, tn: int, fn: int, correct: int, total: int) -> Dict:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
    return {
        "correct": int(correct),
        "total": int(total),
        "accuracy": _safe_ratio(correct, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _is_expected_test_profile(test_acc: Optional[Dict]) -> bool:
    if not test_acc:
        return False
    if int(test_acc.get("total", 0)) <= 0:
        return False
    return (
        int(test_acc.get("normal_sample_size", -1)) == _DEFAULT_TEST_NORMAL_SAMPLE
        and int(test_acc.get("patient_sample_size", -1)) == _DEFAULT_TEST_PATIENT_SAMPLE
    )


def _build_misclassified_reason(
    actual_dvt: bool,
    predicted_dvt: bool,
    area_ratio: Optional[float],
    threshold: float,
) -> str:
    if actual_dvt == predicted_dvt:
        return "判定正确"
    if area_ratio is None:
        return "静脉面积序列不足，无法稳定估计 VCR。"
    collapse_ratio = (1.0 - area_ratio) * 100.0
    if actual_dvt and not predicted_dvt:
        return (
            f"漏诊(FN)：VCR={area_ratio:.3f} ≤ 阈值 {threshold:.3f}。"
            f"静脉压缩幅度约 {collapse_ratio:.1f}% ，系统认为可压缩。"
        )
    return (
        f"误报(FP)：VCR={area_ratio:.3f} > 阈值 {threshold:.3f}。"
        f"静脉压缩幅度约 {collapse_ratio:.1f}% ，系统认为不易压缩。"
    )


def _build_prob_reason(
    actual_dvt: bool,
    predicted_dvt: bool,
    probability: Optional[float],
    threshold: float,
) -> str:
    if actual_dvt == predicted_dvt:
        return "判定正确"
    if probability is None:
        return "模型概率缺失，无法生成概率解释。"
    if actual_dvt and not predicted_dvt:
        return f"漏诊(FN)：RF 概率={probability:.3f} < 阈值 {threshold:.3f}。"
    return f"误报(FP)：RF 概率={probability:.3f} ≥ 阈值 {threshold:.3f}。"


def _evaluate_case_from_gt_masks(
    case_name: str,
    masks_dir: Path,
    actual_dvt: bool,
    split: str,
    threshold: float,
) -> Optional[Dict]:
    if not masks_dir.exists():
        return None
    mask_files = sorted(masks_dir.glob("*.png"), key=lambda p: int(p.stem))
    if len(mask_files) < 2:
        return None

    from classify_dvt import extract_features, predict_dvt

    masks = []
    for mf in mask_files:
        mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks.append(mask)

    if len(masks) < 2:
        return None

    features = extract_features(masks)
    result = predict_dvt(features)
    predicted_dvt = result.get("is_dvt")
    if predicted_dvt is None:
        return None

    vcr = features.get("vcr")
    prob = result.get("probability")
    is_correct = bool(predicted_dvt == actual_dvt)
    reason = _build_prob_reason(actual_dvt, predicted_dvt, prob, threshold)
    return {
        "split": split,
        "case": case_name,
        "actual": "DVT" if actual_dvt else "正常",
        "predicted": "DVT" if predicted_dvt else "正常",
        "correct": is_correct,
        "vcr": round(vcr, 4) if vcr is not None else None,
        "probability": round(prob, 4) if prob is not None else None,
        "reason": reason,
    }


def _summarize_details(details: List[Dict]) -> Dict:
    tp = fp = tn = fn = 0
    for item in details:
        actual_dvt = item.get("actual") == "DVT"
        predicted_dvt = item.get("predicted") == "DVT"
        if actual_dvt and predicted_dvt:
            tp += 1
        elif actual_dvt and not predicted_dvt:
            fn += 1
        elif not actual_dvt and predicted_dvt:
            fp += 1
        else:
            tn += 1
    total = len(details)
    correct = sum(1 for item in details if item.get("correct"))
    metrics = _build_binary_metrics(tp, fp, tn, fn, correct, total)
    metrics["details"] = details
    return metrics


def _collect_train_mask_cases():
    train_root = PROJECT_ROOT / "sam2" / "dataset" / "train"
    if not train_root.exists():
        return []
    return [
        (case_dir.name, case_dir / "masks", False)
        for case_dir in sorted(train_root.iterdir())
        if case_dir.is_dir()
    ]


def _collect_val_mask_cases():
    val_root = PROJECT_ROOT / "sam2" / "dataset" / "val"
    if not val_root.exists():
        return []
    val_normal_set, val_abnormal_set = _get_val_labels()
    cases = []
    for case_dir in sorted(val_root.iterdir()):
        if not case_dir.is_dir():
            continue
        case_name = case_dir.name
        if case_name in val_normal_set:
            actual_dvt = False
        elif case_name in val_abnormal_set:
            actual_dvt = True
        else:
            continue
        cases.append((case_name, case_dir / "masks", actual_dvt))
    return cases


def _normalize_test_detail_item(item: Dict, threshold: float) -> Optional[Dict]:
    case_name = item.get("case")
    actual = item.get("actual")
    predicted = item.get("predicted")
    if not case_name or actual not in ("正常", "DVT") or predicted not in ("正常", "DVT"):
        return None
    actual_dvt = actual == "DVT"
    predicted_dvt = predicted == "DVT"
    try:
        area_ratio = float(item.get("vcr"))
    except Exception:
        area_ratio = None
    try:
        probability = float(item.get("probability"))
    except Exception:
        probability = None
    reason = (
        _build_prob_reason(actual_dvt, predicted_dvt, probability, threshold)
        if probability is not None
        else _build_misclassified_reason(actual_dvt, predicted_dvt, area_ratio, threshold)
    )
    return {
        "split": "test",
        "case": case_name,
        "actual": actual,
        "predicted": predicted,
        "correct": bool(actual_dvt == predicted_dvt),
        "vcr": round(area_ratio, 4) if area_ratio is not None else None,
        "probability": round(probability, 4) if probability is not None else None,
        "reason": reason,
    }


def _compute_comprehensive_accuracy(
    state: Optional[dict] = None,
    force_recompute: bool = False,
    run_test_if_needed: bool = False,
) -> Dict:
    meta_metrics = _get_meta_train_val_metrics()
    if meta_metrics is None:
        return {
            "version": _COMPREHENSIVE_VERSION,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "threshold": _get_val_fixed_threshold(),
            "config": {
                "test_normal_sample": _DEFAULT_TEST_NORMAL_SAMPLE,
                "test_patient_sample": _DEFAULT_TEST_PATIENT_SAMPLE,
            },
            "ready": False,
            "notes": "未找到统一模型元信息，无法构建最新综合指标。",
            "overall": _build_binary_metrics(0, 0, 0, 0, 0, 0),
            "splits": {
                "train": _build_binary_metrics(0, 0, 0, 0, 0, 0),
                "val": _build_binary_metrics(0, 0, 0, 0, 0, 0),
                "test": _build_binary_metrics(0, 0, 0, 0, 0, 0),
            },
            "misclassified": [],
            "details_available": False,
            "details_scope": None,
            "source": "unavailable",
        }

    threshold = float(meta_metrics["threshold"])
    train_metrics = dict(meta_metrics["train"])
    val_metrics = dict(meta_metrics["val"])

    overall = _build_binary_metrics(
        train_metrics["tp"] + val_metrics["tp"],
        train_metrics["fp"] + val_metrics["fp"],
        train_metrics["tn"] + val_metrics["tn"],
        train_metrics["fn"] + val_metrics["fn"],
        train_metrics["correct"] + val_metrics["correct"],
        train_metrics["total"] + val_metrics["total"],
    )

    return {
        "version": _COMPREHENSIVE_VERSION,
        "timestamp": meta_metrics["timestamp"],
        "threshold": threshold,
        "config": {},
        "ready": True,
        "notes": "train / val 指标直接读取最新统一模型元信息。",
        "overall": overall,
        "splits": {
            "train": train_metrics,
            "val": val_metrics,
        },
        "misclassified": [],
        "details_available": False,
        "details_scope": None,
        "source": "model_meta",
        "model_meta_signature": meta_metrics.get("model_meta_signature"),
    }


def _build_comprehensive_summary_html(comp_acc: Dict | None) -> str:
    if not comp_acc:
        return """
        <div class="dashboard-section" style="margin-top: 12px;">
            <h3 class="section-title">🧮 统一模型概览（train + val）</h3>
            <div style="font-size:12px; color:var(--text-secondary);">
                尚未读取到统一模型元信息。
            </div>
        </div>
        """

    splits = comp_acc.get("splits", {})

    def _row(split_key: str, title: str) -> str:
        data = splits.get(split_key, {})
        return (
            f"<tr>"
            f"<td><strong>{title}</strong></td>"
            f"<td>{int(data.get('correct', 0))}/{int(data.get('total', 0))}</td>"
            f"<td>{data.get('accuracy', 0):.1%}</td>"
            f"<td>{data.get('precision', 0):.1%}</td>"
            f"<td>{data.get('recall', 0):.1%}</td>"
            f"<td>{data.get('f1', 0):.3f}</td>"
            f"</tr>"
        )

    return f"""
    <div class="dashboard-section" style="margin-top: 12px;">
        <h3 class="section-title">🧮 统一模型概览（train + val）</h3>
        <div style="font-size:12px; color:var(--text-secondary); line-height:1.7; margin-bottom:10px;">
            分类模型：RF unified (prob &ge; {float(comp_acc.get('threshold', 0)):.2f})<br>
            说明：train / val 指标直接读取最新统一模型元信息<br>
            更新时间：{comp_acc.get('timestamp', '—')}
        </div>
        <table class="recent-table">
            <thead>
                <tr>
                    <th>数据集</th>
                    <th>正确/总数</th>
                    <th>准确率</th>
                    <th>精确率</th>
                    <th>召回率</th>
                    <th>F1</th>
                </tr>
            </thead>
            <tbody>
                {_row('train', 'train')}
                {_row('val', 'val')}
                <tr>
                    <td><strong>train + val</strong></td>
                    <td>{int(comp_acc.get('overall', {}).get('correct', 0))}/{int(comp_acc.get('overall', {}).get('total', 0))}</td>
                    <td>{comp_acc.get('overall', {}).get('accuracy', 0):.1%}</td>
                    <td>{comp_acc.get('overall', {}).get('precision', 0):.1%}</td>
                    <td>{comp_acc.get('overall', {}).get('recall', 0):.1%}</td>
                    <td>{comp_acc.get('overall', {}).get('f1', 0):.3f}</td>
                </tr>
            </tbody>
        </table>
    </div>
    """


def _build_misclassified_html(comp_acc: Dict | None) -> str:
    return f"""
    <div class="dashboard-section" style="border-left: 3px solid var(--primary);">
        <h3 class="section-title" style="color: var(--primary) !important;">📌 首页说明</h3>
        <div style="font-size:12px; color:var(--text-secondary); line-height:1.8;">
            仪表盘当前仅展示 train / val 数据概览与最新统一模型指标。<br>
            如需开始单案例分析，请前往「📤 数据输入」选择数据来源与案例。
        </div>
    </div>
    """


def _run_comprehensive_benchmark(state: dict):
    comp = _compute_comprehensive_accuracy(
        state=state,
        force_recompute=True,
        run_test_if_needed=True,
    )
    status_html, dataset_html, error_html, chart, workflow_html = _refresh_dashboard(state)
    overall = comp.get("overall", {})
    if comp.get("ready") and int(overall.get("total", 0)) > 0:
        status = (
            "✅ 综合评估完成："
            f"准确率 {overall.get('accuracy', 0):.1%} "
            f"（{int(overall.get('correct', 0))}/{int(overall.get('total', 0))}）"
        )
    else:
        status = "⚠️ 综合评估已刷新，但当前未读取到有效的统一模型指标。"
    return status_html, dataset_html, error_html, chart, workflow_html, status


def _run_test_batch_with_limits(
    state: dict,
    normal_sample_size: int = 0,
    patient_sample_size: int = 0,
    progress=gr.Progress(track_tqdm=True),
):
    """运行测试集批量 DVT 诊断（normal/patient 可分别设置取样数）"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    val_acc = _compute_val_accuracy()
    val_fixed_threshold = _get_val_fixed_threshold()

    test_base = PROJECT_ROOT / "test"
    normal_dir = test_base / "normal"
    patient_dir = test_base / "patient"

    # 构建待测试列表
    test_list = []
    normal_dirs = sorted(d for d in normal_dir.iterdir() if d.is_dir()) if normal_dir.exists() else []
    patient_dirs = sorted(d for d in patient_dir.iterdir() if d.is_dir()) if patient_dir.exists() else []

    if normal_sample_size > 0:
        normal_dirs = normal_dirs[:normal_sample_size]
    if patient_sample_size > 0:
        patient_dirs = patient_dirs[:patient_sample_size]

    for d in normal_dirs:
        test_list.append((d, False))
    for d in patient_dirs:
        test_list.append((d, True))

    normal_target = len(normal_dirs)
    patient_target = len(patient_dirs)
    if not test_list:
        return _build_accuracy_html(val_acc, _get_cached_test_accuracy()), "⚠️ 未找到测试数据目录，无法执行批量测试。"

    # 尝试加载推理服务
    inference_svc = None
    inference_error = None
    try:
        from web.services import InferenceService
        inference_svc = InferenceService.get()
    except Exception as e:
        inference_error = f"InferenceService 初始化失败: {e}"

    if inference_svc is None:
        cache_data = {
            "version": 1,
            "correct": 0,
            "total": 0,
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "sample_size": normal_sample_size if normal_sample_size == patient_sample_size else None,
            "normal_sample_size": normal_sample_size,
            "patient_sample_size": patient_sample_size,
            "normal_total": 0,
            "patient_total": 0,
            "normal_specificity": 0,
            "patient_recall": 0,
            "patient_miss_rate": 0,
            "threshold": val_fixed_threshold,
            "classifier": "RF unified",
            "details": [],
            "error": inference_error or "InferenceService 不可用",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_meta_signature": _get_model_meta_signature(),
        }
        _TEST_ACC_CACHE.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), "utf-8")
        return _build_accuracy_html(val_acc, cache_data), f"⚠️ {cache_data['error']}"

    try:
        from classify_dvt import extract_features, predict_dvt
    except Exception as e:
        cache_data = {
            "version": 1,
            "correct": 0,
            "total": 0,
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "sample_size": normal_sample_size if normal_sample_size == patient_sample_size else None,
            "normal_sample_size": normal_sample_size,
            "patient_sample_size": patient_sample_size,
            "normal_total": 0,
            "patient_total": 0,
            "normal_specificity": 0,
            "patient_recall": 0,
            "patient_miss_rate": 0,
            "threshold": val_fixed_threshold,
            "classifier": "RF unified",
            "details": [],
            "error": f"统一分类器加载失败: {e}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_meta_signature": _get_model_meta_signature(),
        }
        _TEST_ACC_CACHE.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), "utf-8")
        return _build_accuracy_html(val_acc, cache_data), f"⚠️ {cache_data['error']}"

    correct = 0
    total = 0
    tp = fp = tn = fn = 0
    details = []
    skipped_detection = 0
    skipped_segmentation = 0
    skipped_other = 0

    for idx, (case_dir, actual_dvt) in enumerate(test_list):
        progress((idx + 1) / len(test_list), desc=f"测试进度: {idx+1}/{len(test_list)} - {case_dir.name}")
        case_name = case_dir.name
        images_dir = case_dir / "images"

        if not images_dir.exists():
            skipped_other += 1
            continue

        frame_files = sorted(images_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        if not frame_files:
            frame_files = sorted(images_dir.glob("*.png"), key=lambda p: int(p.stem))
        if len(frame_files) < 2:
            skipped_other += 1
            continue

        vein_areas = []
        masks_list = []

        try:
            first_frame = cv2.imread(str(frame_files[0]))
            if first_frame is None:
                skipped_other += 1
                continue
            h, w = first_frame.shape[:2]

            # YOLO 检测
            detections = inference_svc.run_detection(first_frame, conf=0.1)
            if not detections or detections.get("artery") is None or detections.get("vein") is None:
                skipped_detection += 1
                continue

            # SAM2 分割
            pred_masks = inference_svc.run_segmentation(
                images_dir=images_dir,
                detections=detections,
                num_frames=len(frame_files),
                use_mfp=False,
                variant="LoRA r8",
            )

            if not pred_masks:
                skipped_segmentation += 1
                continue

            for i in range(len(frame_files)):
                pred = pred_masks.get(i)
                if pred is not None and "semantic" in pred:
                    semantic = pred["semantic"]
                else:
                    semantic = np.zeros((h, w), dtype=np.uint8)
                masks_list.append(semantic)
                vein_areas.append(compute_mask_area(semantic, 2))
        except Exception as e:
            print(f"[TestBatch] {case_name} 推理失败: {e}")
            skipped_other += 1
            continue

        if len(vein_areas) < 2 or len(masks_list) < 2:
            skipped_other += 1
            continue

        try:
            features = extract_features(masks_list)
            result = predict_dvt(features)
        except Exception as e:
            print(f"[TestBatch] {case_name} 分类失败: {e}")
            skipped_other += 1
            continue

        raw_pred = result.get("is_dvt")
        if raw_pred is None:
            skipped_other += 1
            continue
        predicted_dvt = bool(raw_pred)

        total += 1
        is_correct = bool(predicted_dvt == actual_dvt)
        if is_correct:
            correct += 1

        if actual_dvt and predicted_dvt:
            tp += 1
        elif actual_dvt and not predicted_dvt:
            fn += 1
        elif not actual_dvt and predicted_dvt:
            fp += 1
        else:
            tn += 1

        details.append({
            "case": case_name,
            "actual": "DVT" if actual_dvt else "正常",
            "predicted": "DVT" if predicted_dvt else "正常",
            "correct": is_correct,
            "vcr": round(float(result.get("vcr", 0.0)), 4),
            "probability": round(float(result.get("probability", 0.0)), 4),
            "model": result.get("model", "RF unified"),
        })

    accuracy = round(correct / total, 4) if total > 0 else 0
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0

    error_msg = None
    if total == 0:
        error_msg = (
            "本次批量测试未产出有效结果。"
            f"检测失败 {skipped_detection} 例，分割失败 {skipped_segmentation} 例，其它跳过 {skipped_other} 例。"
        )

    normal_total = tn + fp
    patient_total = tp + fn
    normal_specificity = round(tn / normal_total, 4) if normal_total > 0 else 0
    patient_recall = round(tp / patient_total, 4) if patient_total > 0 else 0
    patient_miss_rate = round(fn / patient_total, 4) if patient_total > 0 else 0

    cache_data = {
        "version": 1,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sample_size": normal_sample_size if normal_sample_size == patient_sample_size else None,
        "normal_sample_size": normal_sample_size,
        "patient_sample_size": patient_sample_size,
        "normal_total": normal_total,
        "patient_total": patient_total,
        "normal_specificity": normal_specificity,
        "patient_recall": patient_recall,
        "patient_miss_rate": patient_miss_rate,
        "normal_target": normal_target,
        "patient_target": patient_target,
        "threshold": val_fixed_threshold,
        "classifier": "RF unified",
        "details": details,
        "error": error_msg,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_meta_signature": _get_model_meta_signature(),
    }

    _TEST_ACC_CACHE.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), "utf-8")

    if total > 0:
        normal_note = f"normal={normal_target if normal_sample_size > 0 else '全部'}"
        patient_note = f"patient={patient_target if patient_sample_size > 0 else '全部'}"
        status = (
            f"✅ 测试集批量测试完成：准确率 {accuracy:.1%}（正确 {correct}/{total}）"
            f"，患者召回 {patient_recall:.1%}，{normal_note} / {patient_note}，阈值 prob≥{val_fixed_threshold:.3f}"
        )
    else:
        status = f"⚠️ {error_msg}（阈值 prob≥{val_fixed_threshold:.3f}）"

    return _build_accuracy_html(val_acc, cache_data), status


def _run_test_batch(state: dict, sample_size: int = 0, progress=gr.Progress(track_tqdm=True)):
    """运行测试集批量 DVT 诊断（每类统一取样）"""
    return _run_test_batch_with_limits(
        state=state,
        normal_sample_size=int(sample_size),
        patient_sample_size=int(sample_size),
        progress=progress,
    )


def _run_test_profile_batch(
    state: dict,
    normal_sample_size: int = 500,
    patient_sample_size: int = 50,
    progress=gr.Progress(track_tqdm=True),
):
    """运行 test 分层取样评估（normal/patient 可独立设置）"""
    return _run_test_batch_with_limits(
        state=state,
        normal_sample_size=max(0, int(normal_sample_size)),
        patient_sample_size=max(0, int(patient_sample_size)),
        progress=progress,
    )


def _get_cached_test_accuracy() -> Dict | None:
    """获取缓存的测试集准确率"""
    if _TEST_ACC_CACHE.exists():
        try:
            data = json.loads(_TEST_ACC_CACHE.read_text("utf-8"))
            if not _is_cache_current_for_model(data):
                return None
            return data
        except Exception:
            pass
    return None


# ─── Dashboard 快捷联动 ───

def _build_workflow_status_html(state: Optional[dict]) -> str:
    """构建当前案例流程状态卡片"""
    state = state or {}
    case_name = state.get("current_case") or "未加载"
    split = state.get("split") or ("upload" if state.get("from_video") else "—")
    frame_count = len(state.get("frame_files") or [])

    detections = state.get("detections") or {}
    detect_ready = detections.get("artery") is not None and detections.get("vein") is not None
    seg_ready = bool(state.get("pred_masks"))
    diag_ready = bool(state.get("vein_areas"))
    eval_ready = bool(state.get("frame_metrics"))
    load_ready = frame_count > 0

    if not load_ready:
        next_step = "请先在「📤 数据输入」中加载案例。"
    elif not detect_ready:
        next_step = "下一步建议：运行「🎯 目标检测」。"
    elif not seg_ready:
        next_step = "下一步建议：运行「🔬 视频分割」。"
    elif not diag_ready:
        next_step = "下一步建议：运行「🩺 DVT 诊断」。"
    elif not eval_ready:
        next_step = "下一步建议：运行「📈 定量评估」。"
    else:
        next_step = "✅ 当前案例流程已完整跑通。"

    def _badge(ok: bool, text: str) -> str:
        badge_cls = "badge-success" if ok else "badge-warning"
        icon = "✅" if ok else "⏳"
        return f'<span class="badge {badge_cls}" style="margin-right:6px; margin-bottom:6px;">{icon} {text}</span>'

    return f"""
    <div class="dashboard-section" id="workflow-status-card" style="margin-top: 16px;">
        <h3 class="section-title">🧭 当前流程状态</h3>
        <div style="font-size: 13px; color: var(--text-secondary); margin-bottom: 10px;">
            <b>案例</b>: <code>{case_name}</code> &nbsp;|&nbsp;
            <b>来源</b>: {split} &nbsp;|&nbsp;
            <b>帧数</b>: {frame_count}
        </div>
        <div style="display:flex; flex-wrap:wrap; margin-bottom:10px;">
            {_badge(load_ready, "数据输入")}
            {_badge(detect_ready, "目标检测")}
            {_badge(seg_ready, "视频分割")}
            {_badge(diag_ready, "DVT 诊断")}
            {_badge(eval_ready, "定量评估")}
        </div>
        <div style="font-size: 12px; color: var(--text-muted);">{next_step}</div>
    </div>
    """


def _quick_load_next_val_case(state: dict):
    """自动加载下一个验证集案例，并切换到数据输入页"""
    from web.tabs.upload import _get_dataset_selector_updates, _list_cases, _on_case_selected

    val_cases = _list_cases("val")
    if not val_cases:
        msg = "⚠️ 验证集为空，无法自动加载案例。"
        (
            split_update,
            case_update,
            subset_update,
            train_btn_update,
            val_btn_update,
            test_btn_update,
            selector_status,
        ) = _get_dataset_selector_updates("val", "normal")
        return (
            state,
            split_update,
            case_update,
            None,
            msg,
            [],
            msg,
            _build_workflow_status_html(state),
            gr.Tabs(selected="upload"),
            subset_update,
            train_btn_update,
            val_btn_update,
            test_btn_update,
            selector_status,
        )

    current = state.get("current_case")
    if current in val_cases:
        next_idx = (val_cases.index(current) + 1) % len(val_cases)
    else:
        next_idx = 0
    next_case = val_cases[next_idx]

    new_state, preview, info_md, gallery = _on_case_selected(next_case, "val", "normal", state)
    status = f"✅ 已自动加载验证集案例 `{next_case}`，请继续检测或一键分析。"
    (
        split_update,
        case_update,
        subset_update,
        train_btn_update,
        val_btn_update,
        test_btn_update,
        selector_status,
    ) = _get_dataset_selector_updates("val", "normal", selected_case=next_case)

    return (
        new_state,
        split_update,
        case_update,
        preview,
        info_md,
        gallery,
        status,
        _build_workflow_status_html(new_state),
        gr.Tabs(selected="upload"),
        subset_update,
        train_btn_update,
        val_btn_update,
        test_btn_update,
        selector_status,
    )


def _quick_run_pipeline_from_dashboard(state: dict):
    """使用固定最优参数运行全流程，并切换到一键分析页查看结果"""
    if not state.get("frame_files"):
        msg = "⚠️ 请先加载案例，再执行全流程分析。"
        return (
            state,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            msg,
            _build_workflow_status_html(state),
            gr.Tabs(selected="upload"),
        )

    from web.tabs.pipeline import _run_full_pipeline

    new_state, det_vis, gallery_images, area_fig, report_html, summary_html = _run_full_pipeline(
        state=state,
        model_variant=DEFAULT_SAM2_VARIANT,
        use_mfp=True,
        conf_threshold=0.1,
    )
    case_name = new_state.get("current_case", "Unknown")
    status = f"✅ 案例 `{case_name}` 全流程分析完成（固定最优 {DEFAULT_SAM2_VARIANT}）。"

    return (
        new_state,
        det_vis,
        gallery_images,
        area_fig,
        report_html,
        summary_html,
        status,
        _build_workflow_status_html(new_state),
        gr.Tabs(selected="pipeline"),
    )


# ─── 准确率展示 HTML ───

def _build_accuracy_html(
    val_acc: Dict,
    test_acc: Dict | None,
    comprehensive_acc: Dict | None = None,
) -> str:
    """构建准确率展示卡片 HTML"""
    train_acc = (comprehensive_acc or {}).get("splits", {}).get("train", {}) if comprehensive_acc else {}

    train_pct = f"{train_acc.get('accuracy', 0):.1%}" if train_acc.get("total", 0) > 0 else "—"
    train_detail = (
        f"正确 {int(train_acc.get('correct', 0))}/{int(train_acc.get('total', 0))}<br>"
        f"精确率 {train_acc.get('precision', 0):.1%} / 召回率 {train_acc.get('recall', 0):.1%}<br>"
        f"F1 = {train_acc.get('f1', 0):.3f}"
        if train_acc.get("total", 0) > 0
        else "计算中..."
    )
    train_color = "green" if train_acc.get("accuracy", 0) >= 0.8 else "orange"

    train_html = f"""
    <div class="stat-card stat-card-{train_color}">
        <div class="stat-icon">📚</div>
        <div class="stat-value">{train_pct}</div>
        <div class="stat-label">训练集准确率</div>
        <div class="stat-detail">{train_detail}</div>
    </div>"""

    val_pct = f"{val_acc['accuracy']:.1%}" if val_acc.get("total", 0) > 0 else "—"
    val_detail = (
        f"正确 {val_acc['correct']}/{val_acc['total']}<br>"
        f"精确率 {val_acc.get('precision', 0):.1%} / 召回率 {val_acc.get('recall', 0):.1%}<br>"
        f"F1 = {val_acc.get('f1', 0):.3f}"
        if val_acc.get("total", 0) > 0
        else "计算中..."
    )
    val_color = "green" if val_acc.get("accuracy", 0) >= 0.8 else "orange"

    val_html = f"""
    <div class="stat-card stat-card-{val_color}">
        <div class="stat-icon">🎯</div>
        <div class="stat-value">{val_pct}</div>
        <div class="stat-label">验证集准确率</div>
        <div class="stat-detail">{val_detail}</div>
    </div>"""

    comp_data = (comprehensive_acc or {}).get("overall", {}) if comprehensive_acc else {}
    if comprehensive_acc and int(comp_data.get("total", 0)) > 0:
        comp_pct = f"{comp_data.get('accuracy', 0):.1%}"
        comp_detail = (
            f"正确 {int(comp_data.get('correct', 0))}/{int(comp_data.get('total', 0))}<br>"
            f"精确率 {comp_data.get('precision', 0):.1%} / 召回率 {comp_data.get('recall', 0):.1%}<br>"
            f"F1 = {comp_data.get('f1', 0):.3f}<br>"
            f"train + val 汇总"
        )
        comp_color = "green" if comp_data.get("accuracy", 0) >= 0.8 else "orange"
        comp_html = f"""
        <div class="stat-card stat-card-{comp_color}">
            <div class="stat-icon">🧮</div>
            <div class="stat-value">{comp_pct}</div>
            <div class="stat-label">train + val 准确率</div>
            <div class="stat-detail">{comp_detail}</div>
        </div>"""
    else:
        comp_html = """
        <div class="stat-card stat-card-purple">
            <div class="stat-icon">🧮</div>
            <div class="stat-value">待读取</div>
            <div class="stat-label">train + val 准确率</div>
            <div class="stat-detail">等待统一模型元信息</div>
        </div>"""

    return f'<div class="stat-row">{train_html}{val_html}{comp_html}</div>'


# ─── 图表生成 ───

def _build_distribution_chart(
    stats: Dict,
    val_acc: Dict | None = None,
    test_acc: Dict | None = None,
    comprehensive_acc: Dict | None = None,
):
    """train / val 数据概览图"""
    setup_matplotlib()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 左: 验证集 Normal vs Patient 饼图
    ax1 = axes[0]
    labels = ["正常", "患者 (DVT)"]
    sizes = [stats["val_normal"], stats["val_patient"]]
    colors = ["#10b981", "#ef4444"]
    explode = (0.05, 0.05)

    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct="%1.0f%%", startangle=90,
        textprops={"color": "#1e293b", "fontsize": 12},
        pctdistance=0.65, labeldistance=1.15,
    )
    for t in autotexts:
        t.set_fontsize(13)
        t.set_fontweight("bold")
    ax1.set_title("验证集分布", fontsize=14,
                  fontweight="bold", color="#1e293b", pad=16)

    # 中: 数据集概览柱状图
    ax2 = axes[1]
    categories = ["训练集", "验证集"]
    values = [
        stats["train_count"],
        stats["val_count"],
    ]
    bar_colors = ["#3b82f6", "#10b981"]

    bars = ax2.bar(categories, values, color=bar_colors, width=0.6,
                   edgecolor=[c + "88" for c in bar_colors], linewidth=2)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 3,
                 str(val), ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color="#1e293b")
    style_axis(ax2, title="数据集概览", ylabel="案例数")
    ax2.set_ylim(0, max(values) * 1.25)

    # 右: train / val / 汇总准确率
    ax3 = axes[2]
    train_acc = (comprehensive_acc or {}).get("splits", {}).get("train", {}).get("accuracy", 0)
    val_acc_value = val_acc.get("accuracy", 0) if val_acc else 0
    comp_data = (comprehensive_acc or {}).get("overall", {}) if comprehensive_acc else {}
    acc_labels = ["训练集", "验证集", "train + val"]
    acc_values = [train_acc * 100, val_acc_value * 100, float(comp_data.get("accuracy", 0)) * 100]
    acc_colors = ["#3b82f6", "#10b981", "#8b5cf6"]

    bars = ax3.bar(acc_labels, acc_values, color=acc_colors, width=0.5,
                   edgecolor=[c + "88" for c in acc_colors], linewidth=2)
    for bar, val in zip(bars, acc_values):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=14, fontweight="bold", color="#1e293b")
    style_axis(ax3, title="统一模型准确率", ylabel="准确率 (%)")
    ax3.set_ylim(0, 110)
    ax3.axhline(y=80, color="#f59e0b", linestyle="--", alpha=0.5, label="80% 基准线")
    ax3.legend(fontsize=9)

    plt.tight_layout(pad=2.0)
    return fig


# ─── Dashboard 面板构建 ───

def _refresh_dashboard(state: dict):
    """刷新 Dashboard 数据"""
    gpu = _get_gpu_info()
    mem = _get_system_memory()
    models = _get_model_status()
    stats = _get_dataset_stats()

    # 计算验证集准确率（带缓存）
    val_acc = _compute_val_accuracy()
    comprehensive_acc = _compute_comprehensive_accuracy(
        state=state,
        force_recompute=False,
        run_test_if_needed=False,
    )

    # ========== GPU 卡片 ==========
    if gpu.get("available"):
        gpu_html = f"""
        <div class="stat-card stat-card-blue">
            <div class="stat-icon">🖥️</div>
            <div class="stat-value">{gpu['utilization']}%</div>
            <div class="stat-label">GPU 显存</div>
            <div class="stat-detail">{gpu['name']}<br>{gpu['mem_used_gb']:.1f} / {gpu['mem_total_gb']:.1f} GB</div>
        </div>"""
    else:
        gpu_html = """
        <div class="stat-card stat-card-blue">
            <div class="stat-icon">🖥️</div>
            <div class="stat-value">N/A</div>
            <div class="stat-label">GPU</div>
            <div class="stat-detail">CUDA 不可用</div>
        </div>"""

    # ========== 内存卡片 ==========
    mem_html = f"""
    <div class="stat-card stat-card-purple">
        <div class="stat-icon">💾</div>
        <div class="stat-value">{mem['percent']}%</div>
        <div class="stat-label">系统内存</div>
        <div class="stat-detail">{mem['used_gb']:.1f} / {mem['total_gb']:.1f} GB</div>
    </div>"""

    # ========== 模型状态 ==========
    model_lines = "<br>".join(f"{name}: {status}" for name, status in models.items())
    all_ready = all("✅" in s for s in models.values())
    model_html = f"""
    <div class="stat-card {'stat-card-green' if all_ready else 'stat-card-orange'}">
        <div class="stat-icon">🧠</div>
        <div class="stat-value">{'全部就绪' if all_ready else '部分就绪'}</div>
        <div class="stat-label">模型状态</div>
        <div class="stat-detail">{model_lines}</div>
    </div>"""

    status_html = f"""
    <div class="stat-row">
        {gpu_html}{mem_html}{model_html}
    </div>"""

    # ========== 数据集统计 + 准确率卡片 ==========
    comprehensive_summary_html = _build_comprehensive_summary_html(comprehensive_acc)

    dataset_html = f"""
    <div class="stat-row">
        <div class="stat-card stat-card-cyan">
            <div class="stat-icon">📊</div>
            <div class="stat-value">{stats['val_count']}</div>
            <div class="stat-label">验证集</div>
            <div class="stat-detail">{stats['val_normal']} 正常 + {stats['val_patient']} 患者</div>
        </div>
        <div class="stat-card stat-card-blue">
            <div class="stat-icon">📚</div>
            <div class="stat-value">{stats['train_count']}</div>
            <div class="stat-label">训练集</div>
            <div class="stat-detail">全部正常</div>
        </div>
        <div class="stat-card stat-card-pink">
            <div class="stat-icon">🎞️</div>
            <div class="stat-value">~{stats['total_frames_est']:,}</div>
            <div class="stat-label">估计总帧数</div>
            <div class="stat-detail">平均 {stats['avg_frames']} 帧/案例</div>
        </div>
    </div>
    """ + _build_accuracy_html(val_acc, None, comprehensive_acc) + comprehensive_summary_html

    # ========== 误判案例分析 ==========
    error_html = _build_misclassified_html(comprehensive_acc)

    # ========== 图表 ==========
    chart = _build_distribution_chart(
        stats,
        val_acc=val_acc,
        comprehensive_acc=comprehensive_acc,
    )
    workflow_html = _build_workflow_status_html(state)

    return status_html, dataset_html, error_html, chart, workflow_html


def build_dashboard_panel(state: gr.State):
    """构建 Dashboard 首页面板"""

    # 欢迎区
    gr.HTML("""
    <div class="welcome-banner">
        <h2>欢迎使用 EchoDVT 智能诊断系统</h2>
        <p>基于超声影像的深静脉血栓 (DVT) 智能辅助诊断平台。
        请从左侧导航栏选择功能模块开始使用。</p>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            status_cards = gr.HTML(elem_id="status-cards")

        with gr.Column(scale=2):
            gr.HTML("""
            <div class="dashboard-section dashboard-actions-panel">
                <h3 class="section-title">⚡ 快速开始</h3>
                <p class="dashboard-muted-copy">先自动加载一个验证集案例，再继续一键分析或分步查看。</p>
            """)
            quick_load_btn = gr.Button("📂 自动加载下一个验证集案例", variant="secondary")
            quick_analyze_btn = gr.Button("🚀 运行全流程分析", variant="primary")
            quick_action_status = gr.Markdown(
                "> 💡 仪表盘默认展示最新统一模型的 train / val 指标；点击上方按钮即可开始案例分析。"
            )
            gr.HTML("</div>")

            workflow_status = gr.HTML(elem_id="workflow-status")

    dataset_cards = gr.HTML(elem_id="dataset-cards")
    distribution_chart = gr.Plot(label="train / val 数据概览")

    error_records = gr.HTML(elem_id="error-records")

    refresh_btn = gr.Button("🔄 刷新仪表盘数据", variant="secondary", size="sm")

    refresh_btn.click(
        fn=_refresh_dashboard,
        inputs=[state],
        outputs=[status_cards, dataset_cards, error_records, distribution_chart, workflow_status],
    )

    return (
        status_cards,
        dataset_cards,
        error_records,
        distribution_chart,
        workflow_status,
        refresh_btn,
        quick_load_btn,
        quick_analyze_btn,
        quick_action_status,
    )
