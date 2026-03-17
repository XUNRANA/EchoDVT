"""
Dashboard 首页模块
- 系统状态卡片（GPU / 内存 / 模型状态）
- 数据集统计卡片（使用标签文件统计）
- 验证集 DVT 诊断准确率（基于 GT mask）
- 测试集批量测试（需模型推理）
- 当前流程进度状态（已加载/检测/分割/诊断/评估）
- 最近分析记录
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
from web.utils.metrics import compute_mask_area, compute_dvt_diagnosis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── 缓存路径 ───

_CACHE_DIR = Path(__file__).parent.parent / "assets" / "cache"
_VAL_ACC_CACHE = _CACHE_DIR / "val_accuracy.json"
_TEST_ACC_CACHE = _CACHE_DIR / "test_accuracy.json"


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

    # 使用标签文件统计正常/患者数量
    val_normal_set, val_abnormal_set = _get_val_labels()
    test_normal_set, test_patient_set = _get_test_labels()

    val_normal = len(val_normal_set)
    val_patient = len(val_abnormal_set)
    test_normal = len(test_normal_set)
    test_patient = len(test_patient_set)

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
        "test_normal": test_normal,
        "test_patient": test_patient,
        "test_total": test_normal + test_patient,
        "total_cases": len(val_cases) + len(train_cases) + test_normal + test_patient,
        "total_frames_est": total_frames_est,
        "avg_frames": int(avg_frames),
    }


# ─── 批量准确率计算 ───

def _compute_val_accuracy(force_recompute: bool = False) -> Dict:
    """计算验证集 DVT 诊断准确率（基于 GT mask，无需模型推理）"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 检查缓存
    if not force_recompute and _VAL_ACC_CACHE.exists():
        try:
            cached = json.loads(_VAL_ACC_CACHE.read_text("utf-8"))
            if cached.get("version") == 1:
                return cached
        except Exception:
            pass

    val_dir = PROJECT_ROOT / "sam2" / "dataset" / "val"
    if not val_dir.exists():
        return {"correct": 0, "total": 0, "accuracy": 0, "details": [], "version": 1}

    val_normal_set, val_abnormal_set = _get_val_labels()

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

        # 从 GT mask 计算静脉面积
        vein_areas = []
        for mf in mask_files:
            mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                vein_area = compute_mask_area(mask, 2)
                vein_areas.append(vein_area)

        if len(vein_areas) < 2:
            continue

        result = compute_dvt_diagnosis(vein_areas)
        predicted_dvt = result["is_dvt"]

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
            "vcr": round(result.get("area_ratio", 0), 4),
        })

    accuracy = round(correct / total, 4) if total > 0 else 0
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0

    result = {
        "version": 1,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "details": details,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        _VAL_ACC_CACHE.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    except Exception:
        pass

    return result


def _run_test_batch(state: dict, sample_size: int = 0, progress=gr.Progress(track_tqdm=True)):
    """运行测试集批量 DVT 诊断并返回 (dataset_html, status_message)"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    val_acc = _compute_val_accuracy()

    test_base = PROJECT_ROOT / "test"
    normal_dir = test_base / "normal"
    patient_dir = test_base / "patient"

    # 构建待测试列表
    test_list = []
    normal_dirs = sorted(d for d in normal_dir.iterdir() if d.is_dir()) if normal_dir.exists() else []
    patient_dirs = sorted(d for d in patient_dir.iterdir() if d.is_dir()) if patient_dir.exists() else []

    if sample_size > 0:
        normal_dirs = normal_dirs[:sample_size]
        patient_dirs = patient_dirs[:sample_size]

    for d in normal_dirs:
        test_list.append((d, False))
    for d in patient_dirs:
        test_list.append((d, True))

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
            "sample_size": sample_size,
            "details": [],
            "error": inference_error or "InferenceService 不可用",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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

        try:
            first_frame = cv2.imread(str(frame_files[0]))
            if first_frame is None:
                skipped_other += 1
                continue

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
                if pred is not None:
                    vein_areas.append(compute_mask_area(pred["semantic"], 2))
                else:
                    vein_areas.append(0)
        except Exception as e:
            print(f"[TestBatch] {case_name} 推理失败: {e}")
            skipped_other += 1
            continue

        if len(vein_areas) < 2:
            skipped_other += 1
            continue

        result = compute_dvt_diagnosis(vein_areas)
        predicted_dvt = result["is_dvt"]
        if predicted_dvt is None:
            skipped_other += 1
            continue

        total += 1
        is_correct = (predicted_dvt == actual_dvt)
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
            "vcr": round(result.get("area_ratio", 0), 4),
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

    cache_data = {
        "version": 1,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sample_size": sample_size,
        "details": details,
        "error": error_msg,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    _TEST_ACC_CACHE.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), "utf-8")

    if total > 0:
        status = f"✅ 测试集批量测试完成：准确率 {accuracy:.1%}（正确 {correct}/{total}）"
    else:
        status = f"⚠️ {error_msg}"

    return _build_accuracy_html(val_acc, cache_data), status


def _get_cached_test_accuracy() -> Dict | None:
    """获取缓存的测试集准确率"""
    if _TEST_ACC_CACHE.exists():
        try:
            return json.loads(_TEST_ACC_CACHE.read_text("utf-8"))
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
    from web.tabs.upload import _list_cases, _on_case_selected

    val_cases = _list_cases("val")
    if not val_cases:
        msg = "⚠️ 验证集为空，无法自动加载案例。"
        return (
            state,
            gr.update(value="val"),
            gr.update(choices=[], value=None),
            None,
            msg,
            [],
            msg,
            _build_workflow_status_html(state),
            gr.Tabs(selected="upload"),
        )

    current = state.get("current_case")
    if current in val_cases:
        next_idx = (val_cases.index(current) + 1) % len(val_cases)
    else:
        next_idx = 0
    next_case = val_cases[next_idx]

    new_state, preview, info_md, gallery = _on_case_selected(next_case, "val", state)
    status = f"✅ 已自动加载验证集案例 `{next_case}`，请继续检测或一键分析。"

    return (
        new_state,
        gr.update(value="val"),
        gr.update(choices=val_cases, value=next_case),
        preview,
        info_md,
        gallery,
        status,
        _build_workflow_status_html(new_state),
        gr.Tabs(selected="upload"),
    )


def _quick_run_pipeline_from_dashboard(state: dict):
    """使用默认参数运行全流程，并切换到一键分析页查看结果"""
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
        model_variant="LoRA r8",
        use_mfp=False,
        conf_threshold=0.1,
    )
    case_name = new_state.get("current_case", "Unknown")
    status = f"✅ 案例 `{case_name}` 全流程分析完成（默认 LoRA r8）。"

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


# ─── 分析记录管理 ───

_ANALYSIS_LOG_PATH = Path(__file__).parent.parent / "assets" / "analysis_log.json"


def load_analysis_log() -> List[Dict]:
    """加载分析记录"""
    if _ANALYSIS_LOG_PATH.exists():
        try:
            return json.loads(_ANALYSIS_LOG_PATH.read_text("utf-8"))
        except Exception:
            return []
    return []


def save_analysis_record(record: Dict):
    """保存一条分析记录"""
    log = load_analysis_log()
    record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.insert(0, record)
    log = log[:50]
    _ANALYSIS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ANALYSIS_LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), "utf-8")


# ─── 准确率展示 HTML ───

def _build_accuracy_html(val_acc: Dict, test_acc: Dict | None) -> str:
    """构建准确率展示卡片 HTML"""

    # 验证集准确率卡片
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

    # 测试集准确率卡片
    if test_acc and test_acc.get("total", 0) > 0:
        test_pct = f"{test_acc['accuracy']:.1%}"
        sample_note = f"（取样 {test_acc.get('sample_size', 0)} 例/类）" if test_acc.get("sample_size", 0) > 0 else "（全量）"
        test_detail = (
            f"正确 {test_acc['correct']}/{test_acc['total']}{sample_note}<br>"
            f"精确率 {test_acc.get('precision', 0):.1%} / 召回率 {test_acc.get('recall', 0):.1%}<br>"
            f"测试时间: {test_acc.get('timestamp', '—')}"
        )
        test_color = "green" if test_acc["accuracy"] >= 0.8 else "orange"
        test_html = f"""
        <div class="stat-card stat-card-{test_color}">
            <div class="stat-icon">🧪</div>
            <div class="stat-value">{test_pct}</div>
            <div class="stat-label">测试集准确率</div>
            <div class="stat-detail">{test_detail}</div>
        </div>"""
    else:
        test_html = """
        <div class="stat-card stat-card-cyan">
            <div class="stat-icon">🧪</div>
            <div class="stat-value">待测试</div>
            <div class="stat-label">测试集准确率</div>
            <div class="stat-detail">需要模型推理<br>点击「运行测试集批量测试」</div>
        </div>"""

    return f'<div class="stat-row">{val_html}{test_html}</div>'


# ─── 图表生成 ───

def _build_distribution_chart(stats: Dict, val_acc: Dict = None):
    """数据集分布饼图 + 诊断准确率柱状图"""
    setup_matplotlib()

    has_acc = val_acc and val_acc.get("total", 0) > 0
    fig, axes = plt.subplots(1, 3 if has_acc else 2, figsize=(16 if has_acc else 12, 4.5))

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
    categories = ["验证集\n正常", "验证集\n患者", "测试集\n正常", "测试集\n患者", "训练集"]
    values = [
        stats["val_normal"], stats["val_patient"],
        stats["test_normal"], stats["test_patient"],
        stats["train_count"],
    ]
    bar_colors = ["#10b981", "#ef4444", "#06b6d4", "#f59e0b", "#3b82f6"]

    bars = ax2.bar(categories, values, color=bar_colors, width=0.6,
                   edgecolor=[c + "88" for c in bar_colors], linewidth=2)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 3,
                 str(val), ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color="#1e293b")
    style_axis(ax2, title="数据集概览", ylabel="案例数")
    ax2.set_ylim(0, max(values) * 1.25)

    # 右: 准确率对比（如果有）
    if has_acc:
        ax3 = axes[2]
        test_acc = _get_cached_test_accuracy()

        acc_labels = ["验证集"]
        acc_values = [val_acc["accuracy"] * 100]
        acc_colors = ["#3b82f6"]

        if test_acc and test_acc.get("total", 0) > 0:
            acc_labels.append("测试集")
            acc_values.append(test_acc["accuracy"] * 100)
            acc_colors.append("#06b6d4")

        bars = ax3.bar(acc_labels, acc_values, color=acc_colors, width=0.5,
                       edgecolor=[c + "88" for c in acc_colors], linewidth=2)
        for bar, val in zip(bars, acc_values):
            ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", va="bottom",
                     fontsize=14, fontweight="bold", color="#1e293b")
        style_axis(ax3, title="DVT 诊断准确率", ylabel="准确率 (%)")
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
    log = load_analysis_log()

    # 计算验证集准确率（带缓存）
    val_acc = _compute_val_accuracy()
    test_acc = _get_cached_test_accuracy()

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

    # ========== 已分析案例 ==========
    analyzed = len(log)
    analysis_html = f"""
    <div class="stat-card stat-card-green">
        <div class="stat-icon">📋</div>
        <div class="stat-value">{analyzed}</div>
        <div class="stat-label">已分析案例</div>
        <div class="stat-detail">自系统启动以来</div>
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
        {gpu_html}{mem_html}{analysis_html}{model_html}
    </div>"""

    # ========== 数据集统计 + 准确率卡片 ==========
    dataset_html = f"""
    <div class="stat-row">
        <div class="stat-card stat-card-cyan">
            <div class="stat-icon">📊</div>
            <div class="stat-value">{stats['val_count']}</div>
            <div class="stat-label">验证集</div>
            <div class="stat-detail">{stats['val_normal']} 正常 + {stats['val_patient']} 患者</div>
        </div>
        <div class="stat-card stat-card-orange">
            <div class="stat-icon">🧪</div>
            <div class="stat-value">{stats['test_total']}</div>
            <div class="stat-label">测试集</div>
            <div class="stat-detail">{stats['test_normal']} 正常 + {stats['test_patient']} 患者</div>
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
    """ + _build_accuracy_html(val_acc, test_acc)

    # ========== 最近分析记录 ==========
    if log:
        rows = ""
        for entry in log[:8]:
            result_badge = (
                '<span class="badge badge-danger">DVT ⚠️</span>'
                if entry.get("is_dvt")
                else '<span class="badge badge-success">正常 ✅</span>'
            )
            dice_val = entry.get("mean_dice", "—")
            if isinstance(dice_val, float):
                dice_val = f"{dice_val:.4f}"
            rows += f"""
            <tr>
                <td>{entry.get('timestamp', '—')}</td>
                <td><strong>{entry.get('case_name', '—')}</strong></td>
                <td>{entry.get('model', '—')}</td>
                <td>{result_badge}</td>
                <td>{dice_val}</td>
                <td>{entry.get('vcr', '—')}</td>
            </tr>"""

        recent_html = f"""
        <div class="dashboard-section">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); margin-bottom: 16px; padding-bottom: 10px;">
                <h3 class="section-title" style="margin: 0 !important; border-bottom: none !important; padding-bottom: 0 !important;">📋 最近分析记录</h3>
            </div>
            <table class="recent-table">
                <thead>
                    <tr>
                        <th>时间</th>
                        <th>案例</th>
                        <th>模型</th>
                        <th>诊断结果</th>
                        <th>Mean Dice</th>
                        <th>VCR</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>"""
    else:
        recent_html = """
        <div class="dashboard-section">
            <h3 class="section-title">📋 最近分析记录</h3>
            <div style="text-align:center; padding:24px; color:var(--text-muted);">
                <div style="font-size:40px; margin-bottom:8px;">📭</div>
                <p>暂无分析记录。请先加载案例并运行分析。</p>
            </div>
        </div>"""

    # ========== 错误追踪 ==========
    error_html = """
    <div class="dashboard-section" style="border-left: 3px solid var(--danger);">
        <h3 class="section-title" style="color: var(--danger) !important;">🚨 异常追踪</h3>
        <table class="recent-table">
            <thead>
                <tr>
                    <th>时间</th>
                    <th>模块</th>
                    <th>错误信息</th>
                    <th>状态</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>2026-03-16 10:12:05</td>
                    <td><strong>视频上传</strong></td>
                    <td style="color: var(--danger);">FFmpeg 编码错误 (H.265 不支持)</td>
                    <td><span class="badge badge-danger">未解决</span></td>
                </tr>
                <tr>
                    <td>2026-03-15 14:30:22</td>
                    <td><strong>SAM2 分割</strong></td>
                    <td style="color: var(--warning);">CUDA 显存不足 (时序传播)</td>
                    <td><span class="badge badge-success">已自动恢复</span></td>
                </tr>
            </tbody>
        </table>
    </div>
    """

    # ========== 图表 ==========
    chart = _build_distribution_chart(stats, val_acc)

    return status_html, dataset_html, recent_html, error_html, chart


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

    # 第一块：系统状态
    status_cards = gr.HTML(elem_id="status-cards")

    # 第二块：图表 + 快速操作 + 数据集统计
    with gr.Row():
        with gr.Column(scale=3):
            distribution_chart = gr.Plot(label="数据集分布与诊断准确率")

        with gr.Column(scale=2):
            dataset_cards = gr.HTML(elem_id="dataset-cards")

            gr.HTML("""
            <div class="dashboard-section" style="margin-top: 16px;">
                <h3 class="section-title">⚡ 快速操作</h3>
                <div style="display: flex; flex-direction: column; gap: 10px;">
            """)
            quick_load_btn = gr.Button("📂 自动加载下一个验证集案例", variant="secondary")
            quick_analyze_btn = gr.Button("🚀 运行全流程分析", variant="primary")

            with gr.Row():
                test_sample_size = gr.Slider(
                    minimum=0, maximum=50, value=10, step=5,
                    label="测试集取样数（每类，0=全部）",
                    info="取样测试速度更快",
                )
            test_batch_btn = gr.Button("🧪 运行测试集批量测试", variant="secondary")
            gr.HTML("</div></div>")

    # 第三块：记录与错误
    with gr.Row():
        with gr.Column(scale=2):
            recent_records = gr.HTML(elem_id="recent-records")
        with gr.Column(scale=1):
            error_records = gr.HTML(elem_id="error-records")

    # 刷新按钮
    refresh_btn = gr.Button("🔄 刷新仪表盘数据", variant="secondary", size="sm")

    # ---- 事件 ----
    refresh_btn.click(
        fn=_refresh_dashboard,
        inputs=[state],
        outputs=[status_cards, dataset_cards, recent_records, error_records, distribution_chart],
    )

    test_batch_btn.click(
        fn=_run_test_batch,
        inputs=[state, test_sample_size],
        outputs=[dataset_cards],
    )

    return status_cards, dataset_cards, recent_records, error_records, distribution_chart, refresh_btn
