#!/usr/bin/env python3
"""
EchoDVT 二分类脚本 — 基于分割结果判断 DVT
==========================================
原理：压缩超声中，正常静脉在探头压力下塌陷（面积骤减/消失），
        有血栓的静脉拒绝塌陷（面积基本不变）。

支持两种模式：
  1. 端到端模式（默认）：加载 SAM2 LoRA + YOLO → 推理全帧 → 提取特征 → 分类
  2. 预计算模式：--pred-dir 指定已有 mask 目录（向后兼容）

输入：SAM2 分割结果目录（每个 case 包含逐帧 semantic mask PNG）
输出：分类结果、指标报告、可视化图表
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from PIL import Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────── 配置 ───────────────────────

_REPO_ROOT = Path(__file__).resolve().parent
_SAM2_DIR = _REPO_ROOT / "sam2"

# 数据集默认路径
DEFAULT_DATA_ROOT = "/data1/ouyangxinglong/EchoDVT/data/DVT_segmentation"
DEFAULT_DATA_DIR = str(_SAM2_DIR / "dataset" / "val")
DEFAULT_NORMAL_LIST = str(_SAM2_DIR / "dataset" / "val_normal.txt")
DEFAULT_ABNORMAL_LIST = str(_SAM2_DIR / "dataset" / "val_abnormal.txt")

# 推理默认路径
DEFAULT_SAM2_CONFIG = "configs/sam2/sam2_hiera_l.yaml"
DEFAULT_SAM2_CHECKPOINT = str(_SAM2_DIR / "checkpoints" / "sam2_hiera_large.pt")
DEFAULT_LORA_WEIGHTS = str(
    _SAM2_DIR / "checkpoints" / "lora_runs"
    / "lora_r8_lr0.0003_e25_20260314_153210" / "lora_best.pt"
)
DEFAULT_YOLO_MODEL = str(
    _REPO_ROOT / "yolo" / "runs" / "detect" / "runs" / "detect"
    / "dvt_runs" / "aug_step5_speckle_translate_scale" / "weights" / "best.pt"
)
DEFAULT_YOLO_PRIOR = str(_REPO_ROOT / "yolo" / "prior_stats.json")

# mask 像素值定义
BG_VAL, ARTERY_VAL, VEIN_VAL = 0, 1, 2


# ─────────────────────── 延迟导入推理模块 ───────────────────────

def _import_inference_modules():
    """延迟导入 SAM2 推理相关模块，仅在端到端模式下需要"""
    if str(_SAM2_DIR) not in sys.path:
        sys.path.insert(0, str(_SAM2_DIR))

    from inference_lora import LoRASAM2VideoSegmenter
    from inference_box_prompt_large import (
        VesselDetector, resolve_sam2_device, resolve_yolo_device,
    )
    from sam2.postprocess import MultiFramePrompter

    return LoRASAM2VideoSegmenter, VesselDetector, resolve_sam2_device, resolve_yolo_device, MultiFramePrompter


# ─────────────────────── 特征提取 ───────────────────────

def load_masks(case_mask_dir):
    """加载一个 case 所有帧的 predicted semantic mask"""
    mask_files = sorted(
        [f for f in os.listdir(case_mask_dir) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    masks = []
    for mf in mask_files:
        mask = np.array(Image.open(os.path.join(case_mask_dir, mf)))
        masks.append(mask)
    return masks, mask_files


def compute_area_series(masks):
    """计算每帧的动脉面积和静脉面积序列"""
    artery_areas = []
    vein_areas = []
    for mask in masks:
        artery_areas.append(np.sum(mask == ARTERY_VAL))
        vein_areas.append(np.sum(mask == VEIN_VAL))
    return np.array(artery_areas, dtype=np.float64), np.array(vein_areas, dtype=np.float64)


def extract_features(masks):
    """
    从一个 case 的全帧分割结果中提取二分类特征。
    
    返回: dict of feature_name -> value
    """
    artery_areas, vein_areas = compute_area_series(masks)
    n_frames = len(masks)

    features = {}

    # ── 1. 静脉压缩比 (Vein Compression Ratio, VCR) ──
    # = min_area / max_area，越小说明压缩越充分（正常）
    vein_max = np.max(vein_areas) if np.max(vein_areas) > 0 else 1
    vein_min = np.min(vein_areas)
    features["vcr"] = vein_min / vein_max

    # ── 2. 静脉消失率 (Vein Disappearance Rate, VDR) ──
    # 面积 < 阈值的帧占比，越高说明静脉被压扁了（正常）
    vein_thresh = 0.1 * vein_max  # 面积 < 最大值的 10% 视为"消失"
    features["vdr"] = np.mean(vein_areas < vein_thresh) if vein_max > 0 else 0

    # ── 3. 静脉面积变异系数 (Vein Area CV) ──
    # 标准差/均值，越大说明变化越剧烈（正常）
    vein_mean = np.mean(vein_areas)
    features["vein_cv"] = np.std(vein_areas) / vein_mean if vein_mean > 0 else 0

    # ── 4. 静脉面积相对范围 (Vein Area Relative Range, VARR) ──
    # = (max - min) / max，越大说明压缩幅度越大（正常）
    features["varr"] = (vein_max - vein_min) / vein_max if vein_max > 0 else 0

    # ── 5. 最小静脉/动脉面积比 (Min Vein-to-Artery Ratio, MVAR) ──
    # 动脉不受压缩影响，用作参考。最小比值越小说明静脉被压得越小（正常）
    va_ratios = []
    for a_area, v_area in zip(artery_areas, vein_areas):
        if a_area > 0:
            va_ratios.append(v_area / a_area)
    if va_ratios:
        features["mvar"] = np.min(va_ratios)
        features["mean_var"] = np.mean(va_ratios)
    else:
        features["mvar"] = 1.0
        features["mean_var"] = 1.0

    # ── 6. 静脉面积下降趋势 (Vein Trend Slope) ──
    # 线性拟合斜率，负斜率说明面积在减小（正常）
    if n_frames > 1 and vein_max > 0:
        x = np.arange(n_frames)
        vein_norm = vein_areas / vein_max  # 归一化到 [0, 1]
        slope = np.polyfit(x, vein_norm, 1)[0]
        features["vein_slope"] = slope
    else:
        features["vein_slope"] = 0

    # ── 7. 静脉最小面积出现的相对位置 ──
    # 正常情况下最小面积出现在中后段（压缩最深处）
    if vein_max > 0:
        min_idx = np.argmin(vein_areas)
        features["vein_min_position"] = min_idx / max(n_frames - 1, 1)
    else:
        features["vein_min_position"] = 0.5

    # ── 8. 动脉稳定性指数 (Artery Stability Index) ──
    # 动脉应该很稳定（不受压缩），用来验证分割质量
    artery_mean = np.mean(artery_areas)
    features["artery_stability"] = 1 - (np.std(artery_areas) / artery_mean) if artery_mean > 0 else 0

    # ── 9. 连续帧静脉面积最大下降比 ──
    # 正常人在某两帧之间会有一个剧烈的面积下降
    if n_frames > 1 and vein_max > 0:
        diffs = np.diff(vein_areas)
        features["max_drop_ratio"] = -np.min(diffs) / vein_max
    else:
        features["max_drop_ratio"] = 0

    # ── 10. 静脉面积百分位数特征 ──
    if vein_max > 0:
        features["vein_p10"] = np.percentile(vein_areas, 10) / vein_max
        features["vein_p25"] = np.percentile(vein_areas, 25) / vein_max
        features["vein_p50"] = np.percentile(vein_areas, 50) / vein_max
    else:
        features["vein_p10"] = features["vein_p25"] = features["vein_p50"] = 0

    # ── 11. 分割质量感知特征 ──
    # 分割差的 case 静脉检出率低、面积跳变大 → 分类器应倾向判 DVT（安全侧）
    vein_present = vein_areas > 10  # 面积>10像素视为"检测到静脉"
    features["vein_detect_rate"] = np.mean(vein_present)

    artery_present = artery_areas > 10
    features["artery_detect_rate"] = np.mean(artery_present)

    # 静脉面积时序平滑度：相邻帧面积变化的均值（归一化），越大越不平滑
    if n_frames > 1 and vein_max > 0:
        frame_diffs = np.abs(np.diff(vein_areas))
        features["vein_jitter"] = np.mean(frame_diffs) / vein_max
    else:
        features["vein_jitter"] = 0

    # 静脉面积自相关（lag=1）：真正的压缩过程是平滑的(高自相关)
    if n_frames > 2 and np.std(vein_areas) > 0:
        vein_centered = vein_areas - np.mean(vein_areas)
        autocorr = np.correlate(vein_centered, vein_centered, mode="full")
        autocorr = autocorr / autocorr[len(vein_centered) - 1]  # normalize
        features["vein_autocorr"] = autocorr[len(vein_centered)]  # lag=1
    else:
        features["vein_autocorr"] = 0

    # ── 12. 静脉形状变化（圆度变化）──
    # 被压扁的静脉会变得更扁（圆度降低）
    circularities = []
    for mask in masks:
        vein_mask = (mask == VEIN_VAL).astype(np.uint8)
        if np.sum(vein_mask) < 10:
            continue
        # 计算周长（边界像素数）和面积
        area = np.sum(vein_mask)
        # 用 erosion 近似计算周长
        eroded = ndimage.binary_erosion(vein_mask)
        perimeter = np.sum(vein_mask) - np.sum(eroded)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            circularities.append(min(circularity, 1.0))

    if circularities:
        features["circ_cv"] = np.std(circularities) / np.mean(circularities) if np.mean(circularities) > 0 else 0
        features["circ_min"] = np.min(circularities)
        features["circ_range"] = np.max(circularities) - np.min(circularities)
    else:
        features["circ_cv"] = features["circ_min"] = features["circ_range"] = 0

    return features


# ─────────────────────── 标签加载 ───────────────────────

def load_labels(data_root, split="val"):
    """
    加载标签。
    假设数据集结构：
        data_root/
        val/
            normal/   (case_xxx, ...)
            abnormal/ (case_yyy, ...)
    或者存在 labels.csv：case_id, label (0=normal, 1=dvt)
    """
    labels = {}

    # 方式1: 检查是否存在 labels.csv
    csv_path = os.path.join(data_root, f"{split}_labels.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(data_root, "labels.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            case_id = str(row.iloc[0])
            label = int(row.iloc[1])  # 0=normal, 1=dvt
            labels[case_id] = label
        return labels

    # 方式2: 目录结构推断
    split_dir = os.path.join(data_root, split)

    # 检查是否有 normal/abnormal 子目录
    normal_dir = os.path.join(split_dir, "normal")
    abnormal_dir = os.path.join(split_dir, "abnormal")
    if os.path.isdir(normal_dir) and os.path.isdir(abnormal_dir):
        for case in os.listdir(normal_dir):
            labels[case] = 0
        for case in os.listdir(abnormal_dir):
            labels[case] = 1
        return labels

    # 方式3: 从 case 目录名推断 (如 case_normal_001, case_dvt_001)
    if os.path.isdir(split_dir):
        for case in os.listdir(split_dir):
            case_lower = case.lower()
            if "normal" in case_lower or "neg" in case_lower:
                labels[case] = 0
            elif "dvt" in case_lower or "pos" in case_lower or "abnormal" in case_lower:
                labels[case] = 1

    return labels


def load_labels_from_file(label_file):
    """从指定的标签文件加载"""
    labels = {}
    if label_file.endswith(".csv"):
        df = pd.read_csv(label_file)
        for _, row in df.iterrows():
            labels[str(row.iloc[0])] = int(row.iloc[1])
    elif label_file.endswith(".json"):
        with open(label_file) as f:
            raw = json.load(f)
            labels = {str(k): int(v) for k, v in raw.items()}
    elif label_file.endswith(".txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels[parts[0]] = int(parts[1])
    return labels


def load_labels_from_split_files(normal_file, abnormal_file):
    """从 val_normal.txt / val_abnormal.txt 加载标签"""
    labels = {}
    if os.path.exists(normal_file):
        with open(normal_file) as f:
            for line in f:
                name = line.strip()
                if name:
                    labels[name] = 0
    if os.path.exists(abnormal_file):
        with open(abnormal_file) as f:
            for line in f:
                name = line.strip()
                if name:
                    labels[name] = 1
    return labels


# ─────────────────────── 端到端推理 ───────────────────────

def run_inference(
    data_dir,
    case_names,
    output_mask_dir,
    *,
    sam2_config,
    sam2_checkpoint,
    lora_weights,
    lora_r,
    yolo_model,
    yolo_prior,
    device,
    use_mfp,
    mfp_interval,
    skip_existing=True,
):
    """对指定 case 列表运行 SAM2 LoRA + YOLO 全帧推理，保存语义 mask"""
    import cv2

    (
        LoRASAM2VideoSegmenter,
        VesselDetector,
        resolve_sam2_device,
        resolve_yolo_device,
        MultiFramePrompter,
    ) = _import_inference_modules()

    sam2_device = resolve_sam2_device(device)
    yolo_device = resolve_yolo_device("auto", sam2_device)

    print(f"\n[Inference] device={sam2_device}, yolo_device={yolo_device}")
    print(f"[Inference] LoRA weights: {lora_weights} (r={lora_r})")
    print(f"[Inference] MFP: {'ON interval=' + str(mfp_interval) if use_mfp else 'OFF'}")

    # 构建模型（只构建一次）
    detector = VesselDetector(
        model_path=Path(yolo_model),
        yolo_device=yolo_device,
        prior_path=Path(yolo_prior),
    )
    segmenter = LoRASAM2VideoSegmenter(
        model_cfg=sam2_config,
        checkpoint=Path(sam2_checkpoint),
        lora_weights=Path(lora_weights),
        device=sam2_device,
        lora_r=lora_r,
    )
    prompter = MultiFramePrompter(interval=mfp_interval) if use_mfp else None

    os.makedirs(output_mask_dir, exist_ok=True)
    total = len(case_names)

    for idx, case_name in enumerate(case_names, 1):
        case_out = os.path.join(output_mask_dir, case_name)

        # 跳过已有结果
        if skip_existing and os.path.isdir(case_out):
            existing = [f for f in os.listdir(case_out) if f.endswith(".png")]
            if existing:
                print(f"  [{idx}/{total}] {case_name}: skip ({len(existing)} masks exist)")
                continue

        images_dir = Path(data_dir) / case_name / "images"
        if not images_dir.is_dir():
            print(f"  [{idx}/{total}] {case_name}: SKIP (no images/ dir)")
            continue

        image_files = sorted(images_dir.glob("*.jpg"))
        if not image_files:
            print(f"  [{idx}/{total}] {case_name}: SKIP (no jpg files)")
            continue

        num_frames = len(image_files)

        # YOLO 检测首帧
        first_img = cv2.imread(str(image_files[0]))
        boxes = detector.predict(first_img, conf=0.1)

        # MFP: 选择额外提示帧
        extra_prompt_frames = None
        if prompter is not None:
            extra_prompt_frames = prompter.select_prompt_frames(
                num_frames=num_frames,
                detector=detector,
                images_dir=images_dir,
                image_files=image_files,
            )

        # 推理所有帧
        pred_masks = segmenter.segment_video_with_first_frame_prompt(
            images_dir=images_dir,
            first_frame_boxes=boxes,
            target_frame_indices=set(range(num_frames)),
            extra_prompt_frames=extra_prompt_frames,
        )

        # 保存语义 mask
        os.makedirs(case_out, exist_ok=True)
        for fi, frame_data in pred_masks.items():
            sem = frame_data["semantic"]  # uint8: 0=bg, 1=artery, 2=vein
            out_path = os.path.join(case_out, f"{fi:05d}.png")
            Image.fromarray(sem).save(out_path)

        print(f"  [{idx}/{total}] {case_name}: {num_frames} frames -> {case_out}")

    print(f"[Inference] Done. Masks saved to {output_mask_dir}\n")


# ─────────────────────── 分类器 ───────────────────────

def threshold_classifier(features_df, feature_name="vcr", threshold=0.5):
    """
    基于单特征阈值的简单分类器。
    VCR > threshold → DVT (1)，VCR <= threshold → Normal (0)
    """
    predictions = (features_df[feature_name] > threshold).astype(int)
    return predictions


def find_best_threshold(values, labels, higher_is_positive=True):
    """网格搜索最佳阈值"""
    best_f1 = 0
    best_thresh = 0
    for thresh in np.linspace(np.min(values), np.max(values), 200):
        if higher_is_positive:
            preds = (values > thresh).astype(int)
        else:
            preds = (values < thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def run_ml_classifiers(X, y, feature_names):
    """使用 Leave-One-Out 交叉验证运行多种 ML 分类器"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DVT 代价敏感: class_weight 加大漏判 DVT(FN) 的惩罚
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "LR (cost-sensitive)": LogisticRegression(
            max_iter=1000, C=1.0, class_weight={0: 1, 1: 3}),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
        "RF (cost-sensitive)": RandomForestClassifier(
            n_estimators=100, max_depth=3, random_state=42,
            class_weight={0: 1, 1: 3}),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42),
    }

    results = {}
    loo = LeaveOneOut()

    for name, clf in classifiers.items():
        # LOO 交叉验证
        y_pred = cross_val_predict(clf, X_scaled, y, cv=loo)
        y_prob = cross_val_predict(clf, X_scaled, y, cv=loo, method="predict_proba")[:, 1]

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        auc = roc_auc_score(y, y_prob)
        cm = confusion_matrix(y, y_pred)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        # 特征重要性
        clf.fit(X_scaled, y)
        if hasattr(clf, "feature_importances_"):
            results[name]["feature_importance"] = dict(
                zip(feature_names, clf.feature_importances_)
            )
        elif hasattr(clf, "coef_"):
            results[name]["feature_importance"] = dict(
                zip(feature_names, np.abs(clf.coef_[0]))
            )

    return results


# ─────────────────────── 可视化 ───────────────────────

def plot_results(features_df, labels, ml_results, output_dir):
    """生成可视化图表"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 核心特征分布对比
    key_features = ["vcr", "vdr", "varr", "vein_cv", "mvar", "max_drop_ratio"]
    available = [f for f in key_features if f in features_df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, feat in enumerate(available[:6]):
        ax = axes[i]
        normal_vals = features_df.loc[labels == 0, feat]
        dvt_vals = features_df.loc[labels == 1, feat]
        ax.hist(normal_vals, bins=15, alpha=0.6, label="Normal", color="steelblue")
        ax.hist(dvt_vals, bins=15, alpha=0.6, label="DVT", color="tomato")
        ax.set_title(feat, fontsize=12)
        ax.legend()
    plt.suptitle("Feature Distribution: Normal vs DVT", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_distributions.png"), dpi=150)
    plt.close()

    # 2. ROC 曲线
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in ml_results.items():
        fpr, tpr, _ = roc_curve(labels, res["y_prob"])
        ax.plot(fpr, tpr, label=f'{name} (AUC={res["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Leave-One-Out CV)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close()

    # 3. 特征重要性（取最佳模型）
    best_name = max(ml_results, key=lambda k: ml_results[k]["auc"])
    if "feature_importance" in ml_results[best_name]:
        imp = ml_results[best_name]["feature_importance"]
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        names, values = zip(*sorted_imp[:10])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(len(names)), values, color="steelblue")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_title(f"Top Feature Importance ({best_name})")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
        plt.close()

    # 4. VCR 散点图（最直观的单特征）
    if "vcr" in features_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        normal_mask = labels == 0
        dvt_mask = labels == 1
        ax.scatter(
            features_df.loc[normal_mask].index,
            features_df.loc[normal_mask, "vcr"],
            c="steelblue", label="Normal", s=60, alpha=0.7
        )
        ax.scatter(
            features_df.loc[dvt_mask].index,
            features_df.loc[dvt_mask, "vcr"],
            c="tomato", label="DVT", s=60, alpha=0.7
        )
        # 画最佳阈值线
        best_thresh, _ = find_best_threshold(
            features_df["vcr"].values, labels.values, higher_is_positive=True
        )
        ax.axhline(y=best_thresh, color="green", linestyle="--",
                    label=f"Threshold={best_thresh:.3f}")
        ax.set_ylabel("Vein Compression Ratio (VCR)")
        ax.set_xlabel("Case")
        ax.set_title("VCR per Case (higher = less compression = DVT)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "vcr_scatter.png"), dpi=150)
        plt.close()


# ─────────────────────── 主流程 ───────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EchoDVT 二分类：端到端 SAM2 LoRA 推理 + DVT 分类"
    )

    # ── 数据/输出 ──
    parser.add_argument(
        "--pred-dir", default=None,
        help="预计算 mask 目录（指定时跳过推理，向后兼容）"
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help="数据集目录（含各 case 子目录，每个有 images/）"
    )
    parser.add_argument(
        "--data-root", default=DEFAULT_DATA_ROOT,
        help="数据集根目录（仅用于旧式标签加载）"
    )
    parser.add_argument(
        "--output-dir", default="results/e2e_classify",
        help="输出目录"
    )
    parser.add_argument(
        "--mask-cache-dir", default=None,
        help="推理 mask 缓存目录（默认: output-dir/masks/）"
    )

    # ── 标签 ──
    parser.add_argument(
        "--label-file", default=None,
        help="标签文件路径 (csv/json/txt)，格式: case_id, label(0/1)"
    )
    parser.add_argument(
        "--normal-list", default=DEFAULT_NORMAL_LIST,
        help="正常 case 列表文件（每行一个 case 名）"
    )
    parser.add_argument(
        "--abnormal-list", default=DEFAULT_ABNORMAL_LIST,
        help="异常 case 列表文件（每行一个 case 名）"
    )
    parser.add_argument(
        "--split", default="val", choices=["train", "val"],
    )

    # ── 推理参数 ──
    infer_group = parser.add_argument_group("inference", "SAM2 LoRA + YOLO 推理参数")
    infer_group.add_argument(
        "--sam2-config", default=DEFAULT_SAM2_CONFIG,
        help="SAM2 配置文件"
    )
    infer_group.add_argument(
        "--sam2-checkpoint", default=DEFAULT_SAM2_CHECKPOINT,
        help="SAM2 base checkpoint 路径"
    )
    infer_group.add_argument(
        "--lora-weights", default=DEFAULT_LORA_WEIGHTS,
        help="LoRA 权重路径"
    )
    infer_group.add_argument(
        "--lora-r", type=int, default=8,
        help="LoRA rank（须与训练时一致）"
    )
    infer_group.add_argument(
        "--yolo-model", default=DEFAULT_YOLO_MODEL,
        help="YOLO 模型路径"
    )
    infer_group.add_argument(
        "--yolo-prior", default=DEFAULT_YOLO_PRIOR,
        help="YOLO prior_stats.json 路径"
    )
    infer_group.add_argument(
        "--device", default="auto",
        help="推理设备: auto / cuda / cuda:0 / cpu"
    )
    infer_group.add_argument(
        "--mfp", action="store_true", default=True,
        help="启用 MultiFramePrompter（默认启用）"
    )
    infer_group.add_argument(
        "--no-mfp", action="store_false", dest="mfp",
        help="禁用 MultiFramePrompter"
    )
    infer_group.add_argument(
        "--mfp-interval", type=int, default=15,
        help="MFP 采样间隔（帧）"
    )
    infer_group.add_argument(
        "--no-skip-existing", action="store_true", default=False,
        help="不跳过已有 mask 的 case（强制重新推理）"
    )

    # ── 分类参数 ──
    parser.add_argument(
        "--vcr-threshold", type=float, default=None,
        help="手动指定 VCR 阈值（不指定则自动搜索最佳）"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── 加载标签 ──
    if args.label_file:
        labels_dict = load_labels_from_file(args.label_file)
    elif os.path.exists(args.normal_list) and os.path.exists(args.abnormal_list):
        labels_dict = load_labels_from_split_files(args.normal_list, args.abnormal_list)
    else:
        labels_dict = load_labels(args.data_root, args.split)

    # ── 确定 pred_dir：预计算 or 端到端推理 ──
    if args.pred_dir:
        pred_dir = args.pred_dir
        print(f"[Mode] Pre-computed masks from {pred_dir}")
    else:
        mask_cache_dir = args.mask_cache_dir or os.path.join(output_dir, "masks")
        case_names = sorted(labels_dict.keys()) if labels_dict else sorted(
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
        )
        print(f"[Mode] End-to-end inference on {len(case_names)} cases")
        run_inference(
            data_dir=args.data_dir,
            case_names=case_names,
            output_mask_dir=mask_cache_dir,
            sam2_config=args.sam2_config,
            sam2_checkpoint=args.sam2_checkpoint,
            lora_weights=args.lora_weights,
            lora_r=args.lora_r,
            yolo_model=args.yolo_model,
            yolo_prior=args.yolo_prior,
            device=args.device,
            use_mfp=args.mfp,
            mfp_interval=args.mfp_interval,
            skip_existing=not args.no_skip_existing,
        )
        pred_dir = mask_cache_dir

    if not labels_dict:
        print("⚠ 未找到标签文件！将只提取特征，不做分类评估。")
        print(f"  请提供 --label-file 或确保 {args.data_root} 下有标签信息")

    # ── 遍历 case 提取特征 ──
    case_dirs = sorted([
        d for d in os.listdir(pred_dir)
        if os.path.isdir(os.path.join(pred_dir, d)) and d != "classification"
    ])

    print(f"Found {len(case_dirs)} cases in {pred_dir}")

    all_features = {}
    for case_id in case_dirs:
        case_mask_dir = os.path.join(pred_dir, case_id)

        # 检查是否有 mask 文件
        png_files = [f for f in os.listdir(case_mask_dir) if f.endswith(".png")]
        if not png_files:
            # 可能 mask 在子目录中 (e.g., masks/ 或 pred_masks/)
            for subdir in ["masks", "pred_masks", "semantic_masks"]:
                sub_path = os.path.join(case_mask_dir, subdir)
                if os.path.isdir(sub_path):
                    case_mask_dir = sub_path
                    break

        masks, mask_files = load_masks(case_mask_dir)
        if len(masks) == 0:
            print(f"  [SKIP] {case_id}: no mask files found")
            continue

        features = extract_features(masks)
        features["n_frames"] = len(masks)
        all_features[case_id] = features

        print(f"  {case_id}: {len(masks)} frames, "
                f"VCR={features['vcr']:.3f}, VDR={features['vdr']:.3f}, "
                f"VARR={features['varr']:.3f}")

    if not all_features:
        print("No valid cases found!")
        sys.exit(1)

    # ── 构建 DataFrame ──
    features_df = pd.DataFrame.from_dict(all_features, orient="index")
    features_df.index.name = "case_id"

    # 匹配标签
    matched_labels = {}
    for case_id in features_df.index:
        if case_id in labels_dict:
            matched_labels[case_id] = labels_dict[case_id]
        else:
            # 尝试模糊匹配（去掉前缀/后缀）
            for lk in labels_dict:
                if case_id in lk or lk in case_id:
                    matched_labels[case_id] = labels_dict[lk]
                    break

    has_labels = len(matched_labels) > 0
    if has_labels:
        labels_series = pd.Series(matched_labels).reindex(features_df.index)
        # 只保留有标签的 case
        valid_mask = labels_series.notna()
        features_df = features_df[valid_mask]
        labels_series = labels_series[valid_mask].astype(int)

        n_normal = (labels_series == 0).sum()
        n_dvt = (labels_series == 1).sum()
        print(f"\nMatched {len(labels_series)} cases with labels: "
                f"{n_normal} normal, {n_dvt} DVT")

    # 保存特征
    features_df.to_csv(os.path.join(output_dir, "features.csv"))
    print(f"\nFeatures saved to {output_dir}/features.csv")

    if not has_labels:
        print("\nNo labels available — skipping classification evaluation.")
        print("Features extracted. Use --label-file to enable classification.")
        sys.exit(0)

    # ─────────── 分类评估 ───────────

    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)

    # 1. 单特征阈值分类
    print("\n── 1. Threshold-based Classification ──")

    # 对核心特征分别找最佳阈值
    threshold_features = {
        "vcr":  (True,  "VCR > thresh → DVT"),   # 高VCR = 不压缩 = DVT
        "varr": (False, "VARR < thresh → DVT"),   # 低VARR = 压缩幅度小 = DVT
        "vdr":  (False, "VDR < thresh → DVT"),    # 低VDR = 不消失 = DVT
        "vein_cv": (False, "CV < thresh → DVT"),  # 低CV = 面积不变 = DVT
    }

    best_single = {"name": "", "f1": 0, "thresh": 0}
    for feat, (higher_is_pos, desc) in threshold_features.items():
        if feat not in features_df.columns:
            continue
        vals = features_df[feat].values
        thresh, f1 = find_best_threshold(vals, labels_series.values, higher_is_pos)
        if higher_is_pos:
            preds = (vals > thresh).astype(int)
        else:
            preds = (vals < thresh).astype(int)
        acc = accuracy_score(labels_series, preds)
        prec = precision_score(labels_series, preds, zero_division=0)
        rec = recall_score(labels_series, preds, zero_division=0)

        print(f"  {feat:12s}: thresh={thresh:.4f}  "
                f"Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  "
                f"({desc})")

        if f1 > best_single["f1"]:
            best_single = {"name": feat, "f1": f1, "thresh": thresh,
                            "higher_is_pos": higher_is_pos}

    print(f"\n  Best single feature: {best_single['name']} "
            f"(F1={best_single['f1']:.3f}, threshold={best_single['thresh']:.4f})")

    # 用户指定阈值
    if args.vcr_threshold is not None:
        print(f"\n  [User threshold] VCR > {args.vcr_threshold}:")
        preds = (features_df["vcr"] > args.vcr_threshold).astype(int)
        print(f"    Acc={accuracy_score(labels_series, preds):.3f}  "
                f"F1={f1_score(labels_series, preds, zero_division=0):.3f}")

    # 2. ML 分类器（LOO-CV）
    print("\n── 2. ML Classifiers (Leave-One-Out CV) ──")

    feature_cols = [c for c in features_df.columns if c != "n_frames"]
    X = features_df[feature_cols].values
    y = labels_series.values
    feature_names = feature_cols

    ml_results = run_ml_classifiers(X, y, feature_names)

    for name, res in ml_results.items():
        print(f"\n  {name}:")
        print(f"    Acc={res['accuracy']:.3f}  Prec={res['precision']:.3f}  "
                f"Rec={res['recall']:.3f}  F1={res['f1']:.3f}  AUC={res['auc']:.3f}")
        cm = res["confusion_matrix"]
        print(f"    Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

        if "feature_importance" in res:
            top3 = sorted(res["feature_importance"].items(),
                            key=lambda x: x[1], reverse=True)[:3]
            top3_str = ", ".join(f"{n}={v:.3f}" for n, v in top3)
            print(f"    Top features: {top3_str}")

    # 3. 找最佳模型（按 AUC）
    best_model = max(ml_results, key=lambda k: ml_results[k]["auc"])
    print(f"\n  Best model (AUC): {best_model} (AUC={ml_results[best_model]['auc']:.3f})")

    # ── 3. 高召回率阈值优化（重点：不漏判 DVT 病人）──
    print("\n── 3. High-Recall Threshold Optimization ──")
    print("  目标：最大化 Recall（不漏判 DVT），同时 Accuracy ≥ 90%\n")

    best_hr_model = None
    best_hr_recall = 0
    best_hr_acc = 0
    best_hr_thresh = 0.5

    for name, res in ml_results.items():
        y_prob = res["y_prob"]
        # 搜索最优阈值：在 Accuracy ≥ 90% 约束下最大化 Recall
        opt_thresh, opt_recall, opt_acc = 0.5, 0, 0
        for t in np.linspace(0.01, 0.99, 500):
            preds_t = (y_prob >= t).astype(int)
            acc_t = accuracy_score(y, preds_t)
            rec_t = recall_score(y, preds_t, zero_division=0)
            if acc_t >= 0.90 and rec_t > opt_recall:
                opt_recall = rec_t
                opt_thresh = t
                opt_acc = acc_t
            elif acc_t >= 0.90 and rec_t == opt_recall and acc_t > opt_acc:
                opt_thresh = t
                opt_acc = acc_t

        preds_opt = (y_prob >= opt_thresh).astype(int)
        prec_opt = precision_score(y, preds_opt, zero_division=0)
        f1_opt = f1_score(y, preds_opt, zero_division=0)
        cm_opt = confusion_matrix(y, preds_opt)
        fn_count = cm_opt[1, 0]

        print(f"  {name}:")
        print(f"    Optimal threshold={opt_thresh:.3f}  "
              f"Acc={opt_acc:.3f}  Prec={prec_opt:.3f}  "
              f"Rec={opt_recall:.3f}  F1={f1_opt:.3f}")
        print(f"    TN={cm_opt[0,0]} FP={cm_opt[0,1]} FN={fn_count} TP={cm_opt[1,1]}")

        # 记录到 results
        ml_results[name]["high_recall_threshold"] = opt_thresh
        ml_results[name]["high_recall_acc"] = opt_acc
        ml_results[name]["high_recall_recall"] = opt_recall
        ml_results[name]["high_recall_preds"] = preds_opt

        if opt_recall > best_hr_recall or (
            opt_recall == best_hr_recall and opt_acc > best_hr_acc
        ):
            best_hr_model = name
            best_hr_recall = opt_recall
            best_hr_acc = opt_acc
            best_hr_thresh = opt_thresh

    print(f"\n  ** Best high-recall model: {best_hr_model}")
    print(f"     threshold={best_hr_thresh:.3f}  "
          f"Recall={best_hr_recall:.3f}  Acc={best_hr_acc:.3f}")

    # 使用 high-recall 最佳模型作为最终推荐
    final_model = best_hr_model
    final_preds = ml_results[final_model]["high_recall_preds"]
    final_probs = ml_results[final_model]["y_prob"]

    # 4. 可视化
    print("\n── 4. Generating Visualizations ──")
    plot_results(features_df, labels_series, ml_results, output_dir)
    print(f"  Plots saved to {output_dir}/")

    # 5. 保存完整报告
    report = {
        "n_cases": len(features_df),
        "n_normal": int(n_normal),
        "n_dvt": int(n_dvt),
        "best_single_feature": {
            "name": best_single["name"],
            "threshold": float(best_single["thresh"]),
            "f1": float(best_single["f1"]),
        },
        "ml_results": {
            name: {
                "accuracy": float(res["accuracy"]),
                "precision": float(res["precision"]),
                "recall": float(res["recall"]),
                "f1": float(res["f1"]),
                "auc": float(res["auc"]),
                "high_recall_threshold": float(res.get("high_recall_threshold", 0.5)),
                "high_recall_accuracy": float(res.get("high_recall_acc", 0)),
                "high_recall_recall": float(res.get("high_recall_recall", 0)),
            }
            for name, res in ml_results.items()
        },
        "best_model_auc": best_model,
        "best_auc": float(ml_results[best_model]["auc"]),
        "recommended_model": final_model,
        "recommended_threshold": float(best_hr_thresh),
        "recommended_accuracy": float(best_hr_acc),
        "recommended_recall": float(best_hr_recall),
    }

    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Per-case 预测结果（使用 high-recall 推荐模型）
    case_results = features_df.copy()
    case_results["true_label"] = labels_series
    case_results["pred_label"] = final_preds
    case_results["pred_prob"] = final_probs
    case_results["correct"] = (case_results["true_label"] == case_results["pred_label"])
    case_results.to_csv(os.path.join(output_dir, "per_case_results.csv"))

    print(f"\n  Report: {output_dir}/classification_report.json")
    print(f"  Per-case: {output_dir}/per_case_results.csv")

    # 打印错误案例
    errors = case_results[~case_results["correct"]]
    fn_cases = errors[(errors["true_label"] == 1) & (errors["pred_label"] == 0)]
    fp_cases = errors[(errors["true_label"] == 0) & (errors["pred_label"] == 1)]

    print(f"\n── Final Results (model={final_model}, "
          f"threshold={best_hr_thresh:.3f}) ──")
    total_correct = case_results["correct"].sum()
    print(f"  Accuracy: {total_correct}/{len(case_results)} "
          f"= {total_correct/len(case_results):.1%}")
    print(f"  False Negatives (DVT→Normal, 漏判): {len(fn_cases)}")
    print(f"  False Positives (Normal→DVT, 误判): {len(fp_cases)}")

    if len(fn_cases) > 0:
        print(f"\n  *** DVT 漏判 ({len(fn_cases)} 例) — 需重点关注 ***")
        for case_id, row in fn_cases.iterrows():
            print(f"    {case_id}: prob={row['pred_prob']:.3f}, "
                  f"VCR={row['vcr']:.3f}, VDR={row['vdr']:.3f}")

    if len(fp_cases) > 0:
        print(f"\n  Normal 误判为 DVT ({len(fp_cases)} 例):")
        for case_id, row in fp_cases.iterrows():
            print(f"    {case_id}: prob={row['pred_prob']:.3f}, "
                  f"VCR={row['vcr']:.3f}")

    if len(errors) == 0:
        print("  ** 零错误！所有病例分类正确 **")

    print("\nDone!")


if __name__ == "__main__":
    main()




"""
    核心设计思路

    提取的关键特征（共 15 个）

    ┌───────────────────┬───────────────────────┬────────────────┬────────────────┐
    │       特征        │         含义          │     正常人     │    DVT 患者    │
    ├───────────────────┼───────────────────────┼────────────────┼────────────────┤
    │ VCR (核心)        │ min面积/max面积       │ 接近 0（塌陷） │ 接近 1（不变） │
    ├───────────────────┼───────────────────────┼────────────────┼────────────────┤
    │ VDR               │ 静脉"消失"帧占比      │ 高（会消失）   │ 低（始终可见） │
    ├───────────────────┼───────────────────────┼────────────────┼────────────────┤
    │ VARR              │ 面积相对变化幅度      │ 大             │ 小             │
    ├───────────────────┼───────────────────────┼────────────────┼────────────────┤
    │ vein_cv           │ 面积变异系数          │ 大（剧烈变化） │ 小（平稳）     │
    ├───────────────────┼───────────────────────┼────────────────┼────────────────┤
    │ MVAR              │ 最小 静脉/动脉 面积比 │ 小             │ 大             │
    ├───────────────────┼───────────────────────┼────────────────┼────────────────┤
    │ vein_slope        │ 面积线性趋势斜率      │ 负（缩小）     │ 接近 0         │
    ├───────────────────┼───────────────────────┼────────────────┼────────────────┤
    │ max_drop_ratio    │ 最大单帧下降比        │ 大             │ 小             │
    ├───────────────────┼───────────────────────┼────────────────┼────────────────┤
    │ circ_cv/min/range │ 圆度变化              │ 大（被压扁）   │ 小             │
    └───────────────────┴───────────────────────┴────────────────┴────────────────┘

    分类方法

    1. 单特征阈值 — 自动搜索最佳阈值，适合可解释性要求高的场景
    2. ML 分类器 — Logistic Regression / Random Forest / Gradient Boosting，使用 Leave-One-Out 交叉验证（因为样本量只有 76）

    使用方式

    # 端到端模式（默认）：推理 + 分类
    python classify_dvt.py --output-dir results/e2e_classify/

    # 指定 LoRA 权重和参数
    python classify_dvt.py \
      --lora-weights sam2/checkpoints/lora_runs/.../lora_best.pt \
      --lora-r 8 --mfp --mfp-interval 15

    # 预计算 mask 模式（向后兼容）
    python classify_dvt.py \
      --pred-dir results/lora_r8_mfp15/ \
      --label-file /path/to/labels.csv

    # 手动指定 VCR 阈值
    python classify_dvt.py --vcr-threshold 0.4

"""
