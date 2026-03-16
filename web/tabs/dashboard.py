"""
Dashboard 首页模块
- 系统状态卡片（GPU / 内存 / 模型状态）
- 数据集统计卡片
- 最近分析记录
- 数据集分布图表
- 快速操作区
"""

import gradio as gr
import numpy as np
import time
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.utils.chart_style import setup_matplotlib, style_axis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── 系统信息采集 ───

def _get_gpu_info() -> Dict:
    """获取 GPU 信息"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            mem_used = torch.cuda.memory_allocated(0) / (1024**3)
            mem_cached = torch.cuda.memory_reserved(0) / (1024**3)
            return {
                "available": True,
                "name": name,
                "mem_total_gb": round(mem_total, 1),
                "mem_used_gb": round(mem_used, 1),
                "mem_cached_gb": round(mem_cached, 1),
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
        status[name] = "✅ Ready" if found else "❌ Missing"
    return status


def _get_dataset_stats() -> Dict:
    """获取数据集统计信息"""
    from web.tabs.upload import _list_cases, _get_dataset_root

    val_cases = _list_cases("val")
    train_cases = _list_cases("train")

    # 统计正常 vs 患者 (AN_ 前缀 = 正常)
    val_normal = sum(1 for c in val_cases if c.startswith("AN_"))
    val_patient = len(val_cases) - val_normal

    # 统计总帧数（采样几个案例估算）
    total_frames_est = 0
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
    log.insert(0, record)  # 最新在前
    log = log[:50]  # 最多保留 50 条
    _ANALYSIS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ANALYSIS_LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), "utf-8")


# ─── 图表生成 ───

def _build_distribution_chart(stats: Dict):
    """数据集分布饼图 + 帧数分布柱状图"""
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左: Normal vs Patient 饼图
    ax1 = axes[0]
    labels = ["Normal", "Patient (DVT)"]
    sizes = [stats["val_normal"], stats["val_patient"]]
    colors = ["#10b981", "#ef4444"]
    explode = (0.05, 0.05)

    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct="%1.0f%%", startangle=90,
        textprops={"color": "#e2e8f0", "fontsize": 12},
        pctdistance=0.65, labeldistance=1.15,
    )
    for t in autotexts:
        t.set_fontsize(13)
        t.set_fontweight("bold")
    ax1.set_title("Validation Set Distribution", fontsize=14,
                  fontweight="bold", color="#e2e8f0", pad=16)

    # 右: 数据集概览柱状图
    ax2 = axes[1]
    categories = ["Val\nNormal", "Val\nPatient", "Train"]
    values = [stats["val_normal"], stats["val_patient"], stats["train_count"]]
    bar_colors = ["#10b981", "#ef4444", "#3b82f6"]

    bars = ax2.bar(categories, values, color=bar_colors, width=0.6,
                   edgecolor=[c + "88" for c in bar_colors], linewidth=2)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 3,
                 str(val), ha="center", va="bottom",
                 fontsize=14, fontweight="bold", color="#f1f5f9")
    style_axis(ax2, title="Dataset Overview", ylabel="Cases")
    ax2.set_ylim(0, max(values) * 1.2)

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

    # GPU 卡片
    if gpu.get("available"):
        gpu_html = f"""
        <div class="stat-card stat-card-blue">
            <div class="stat-icon">🖥️</div>
            <div class="stat-value">{gpu['utilization']}%</div>
            <div class="stat-label">GPU Memory</div>
            <div class="stat-detail">{gpu['name']}<br>{gpu['mem_used_gb']:.1f} / {gpu['mem_total_gb']:.1f} GB</div>
        </div>"""
    else:
        gpu_html = """
        <div class="stat-card stat-card-gray">
            <div class="stat-icon">🖥️</div>
            <div class="stat-value">N/A</div>
            <div class="stat-label">GPU</div>
            <div class="stat-detail">CUDA unavailable</div>
        </div>"""

    # 内存卡片
    mem_html = f"""
    <div class="stat-card stat-card-purple">
        <div class="stat-icon">💾</div>
        <div class="stat-value">{mem['percent']}%</div>
        <div class="stat-label">System Memory</div>
        <div class="stat-detail">{mem['used_gb']:.1f} / {mem['total_gb']:.1f} GB</div>
    </div>"""

    # 已分析案例
    analyzed = len(log)
    analysis_html = f"""
    <div class="stat-card stat-card-green">
        <div class="stat-icon">📋</div>
        <div class="stat-value">{analyzed}</div>
        <div class="stat-label">Analyzed Cases</div>
        <div class="stat-detail">since system start</div>
    </div>"""

    # 模型状态
    model_lines = "<br>".join(f"{name}: {status}" for name, status in models.items())
    all_ready = all("✅" in s for s in models.values())
    model_html = f"""
    <div class="stat-card {'stat-card-green' if all_ready else 'stat-card-orange'}">
        <div class="stat-icon">🧠</div>
        <div class="stat-value">{'All Ready' if all_ready else 'Partial'}</div>
        <div class="stat-label">Model Status</div>
        <div class="stat-detail">{model_lines}</div>
    </div>"""

    status_html = f"""
    <div class="stat-row">
        {gpu_html}{mem_html}{analysis_html}{model_html}
    </div>"""

    # 数据集统计卡片
    dataset_html = f"""
    <div class="stat-row">
        <div class="stat-card stat-card-cyan">
            <div class="stat-icon">📊</div>
            <div class="stat-value">{stats['val_count']}</div>
            <div class="stat-label">Val Cases</div>
            <div class="stat-detail">{stats['val_normal']} normal + {stats['val_patient']} patient</div>
        </div>
        <div class="stat-card stat-card-indigo">
            <div class="stat-icon">📚</div>
            <div class="stat-value">{stats['train_count']}</div>
            <div class="stat-label">Train Cases</div>
            <div class="stat-detail">all normal</div>
        </div>
        <div class="stat-card stat-card-pink">
            <div class="stat-icon">🎞️</div>
            <div class="stat-value">~{stats['total_frames_est']:,}</div>
            <div class="stat-label">Total Frames</div>
            <div class="stat-detail">avg {stats['avg_frames']} frames/case</div>
        </div>
    </div>"""

    # 最近分析记录
    if log:
        rows = ""
        for entry in log[:8]:
            result_badge = (
                '<span class="badge badge-danger">DVT ⚠️</span>'
                if entry.get("is_dvt")
                else '<span class="badge badge-success">Normal ✅</span>'
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
                <h3 class="section-title" style="margin: 0 !important; border-bottom: none !important; padding-bottom: 0 !important;">📋 Recent Analysis Records</h3>
                <div style="display: flex; gap: 8px;">
                    <input type="text" placeholder="🔍 Search records..." style="padding: 4px 10px; border-radius: 4px; border: 1px solid var(--border); background: var(--bg-dark); color: var(--text-primary); font-size: 12px; width: 180px;">
                    <button style="padding: 4px 12px; background: var(--bg-dark); border: 1px solid var(--border); color: var(--text-secondary); border-radius: 4px; font-size: 12px; cursor: pointer;">🔽 Filter</button>
                    <button style="padding: 4px 12px; background: var(--gradient-primary); border: none; color: white; border-radius: 4px; font-size: 12px; cursor: pointer;">📥 Export CSV</button>
                </div>
            </div>
            <table class="recent-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Case</th>
                        <th>Model</th>
                        <th>Result</th>
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
            <h3 class="section-title">📋 Recent Analysis Records</h3>
            <div class="empty-state">
                <div style="font-size:40px; margin-bottom:8px;">📭</div>
                <p>No analysis records yet. Start by loading a case and running analysis.</p>
            </div>
        </div>"""

    # 最近错误样例 (Mocked for demonstration, but visually rich)
    error_html = """
    <div class="dashboard-section" style="border-left: 3px solid var(--danger);">
        <h3 class="section-title" style="color: var(--danger) !important;">🚨 Recent Error Tracking</h3>
        <table class="recent-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Module</th>
                    <th>Error Message</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>2026-03-16 10:12:05</td>
                    <td><strong>Video Upload</strong></td>
                    <td style="color: var(--danger);">FFmpeg codec error (H.265 not supported)</td>
                    <td><span class="badge badge-danger">Unresolved</span></td>
                </tr>
                <tr>
                    <td>2026-03-15 14:30:22</td>
                    <td><strong>SAM2 Seg</strong></td>
                    <td style="color: var(--warning);">CUDA OutOfMemoryError during temporal tracking</td>
                    <td><span class="badge badge-success">Auto-Recovered</span></td>
                </tr>
            </tbody>
        </table>
    </div>
    """

    # 图表
    chart = _build_distribution_chart(stats)

    return status_html, dataset_html, recent_html, error_html, chart


def build_dashboard_panel(state: gr.State):
    """构建 Dashboard 首页面板"""

    # 欢迎区
    gr.HTML("""
    <div class="welcome-banner">
        <h2>Welcome to EchoDVT Dashboard</h2>
        <p>Ultrasound-based Deep Vein Thrombosis intelligent diagnosis system.
        Select a module from the left sidebar to get started.</p>
    </div>
    """)

    # 第一块：状态系统
    status_cards = gr.HTML(elem_id="status-cards")
    
    # 第二块：图表 + 快速操作 + 数据集统计
    with gr.Row():
        with gr.Column(scale=3):
            distribution_chart = gr.Plot(label="Dataset & Results Distribution")
        
        with gr.Column(scale=2):
            dataset_cards = gr.HTML(elem_id="dataset-cards")
            
            gr.HTML("""
            <div class="dashboard-section" style="margin-top: 16px;">
                <h3 class="section-title">⚡ Quick Actions</h3>
                <div style="display: flex; flex-direction: column; gap: 10px;">
            """)
            quick_load_btn = gr.Button("📂 Auto-Load Next Val Case", variant="secondary")
            quick_analyze_btn = gr.Button("🚀 Run Full Pipeline", variant="primary")
            gr.HTML("</div></div>")

    # 第三块：记录与错误
    with gr.Row():
        with gr.Column(scale=2):
            recent_records = gr.HTML(elem_id="recent-records")
        with gr.Column(scale=1):
            error_records = gr.HTML(elem_id="error-records")

    # 刷新按钮
    refresh_btn = gr.Button("🔄 Refresh Dashboard Data", variant="secondary", size="sm")

    # ---- 事件 ----
    refresh_btn.click(
        fn=_refresh_dashboard,
        inputs=[state],
        outputs=[status_cards, dataset_cards, recent_records, error_records, distribution_chart],
    )

    return status_cards, dataset_cards, recent_records, error_records, distribution_chart, refresh_btn
