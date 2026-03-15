"""
模块 4: DVT 诊断结果
- 基于 19 维时序特征的智能诊断
- 面积变化曲线可视化
- 诊断结果展示（含关键特征表）
"""

import gradio as gr
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.utils.metrics import compute_dvt_diagnosis
from web.utils.chart_style import setup_matplotlib, style_axis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _run_diagnosis(state: dict, threshold: float):
    """执行 DVT 诊断（支持完整特征提取）"""
    vein_areas = state.get("vein_areas", [])
    artery_areas = state.get("artery_areas", [])
    pred_masks = state.get("pred_masks")

    if not vein_areas:
        return (
            "⚠️ 请先在「🔬 SAM2 分割」Tab 中完成分割",
            None,
            "等待分割结果...",
        )

    # 尝试使用 InferenceService 进行完整 19 维特征提取
    full_features = None
    ml_result = None
    if pred_masks:
        try:
            from web.services import InferenceService
            # 收集所有帧的 semantic mask（按帧序排列）
            num_frames = len(state.get("frame_files", []))
            masks_list = []
            for i in range(num_frames):
                entry = pred_masks.get(i)
                if entry is not None:
                    masks_list.append(entry["semantic"])
                else:
                    # 用零 mask 填充缺失帧
                    if masks_list:
                        masks_list.append(np.zeros_like(masks_list[-1]))
                    else:
                        masks_list.append(np.zeros((256, 256), dtype=np.uint8))

            ml_result = InferenceService.get().run_diagnosis(masks_list)
            full_features = ml_result.get("features")
        except Exception:
            pass

    # 兜底：简单 VCR 阈值诊断
    result = compute_dvt_diagnosis(vein_areas, threshold=threshold)

    # 如果 ML 诊断可用，覆盖部分结果
    if ml_result is not None:
        result["is_dvt"] = ml_result["is_dvt"]
        result["confidence"] = ml_result["confidence"]
        result["ml_threshold"] = ml_result["threshold"]
        if ml_result["is_dvt"]:
            result["diagnosis"] = "⚠️ DVT 疑似（静脉拒绝塌陷）"
        else:
            result["diagnosis"] = "✅ 正常（静脉正常塌陷）"

    # 生成面积曲线图
    setup_matplotlib()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    frames = list(range(len(vein_areas)))

    # 上图：面积变化
    ax1 = axes[0]
    ax1.fill_between(frames, artery_areas, alpha=0.3, color="#ef4444", label="Artery Area")
    ax1.fill_between(frames, vein_areas, alpha=0.3, color="#22c55e", label="Vein Area")
    ax1.plot(frames, artery_areas, color="#ef4444", linewidth=2)
    ax1.plot(frames, vein_areas, color="#22c55e", linewidth=2)
    style_axis(ax1, title="Artery / Vein Area Over Frames",
               xlabel="Frame Index", ylabel="Area (px)")
    ax1.legend(fontsize=11, loc="upper right")

    # 标注最小面积帧
    if vein_areas:
        min_idx = int(np.argmin(vein_areas))
        ax1.axvline(x=min_idx, color="#f59e0b", linestyle="--", alpha=0.7,
                    label=f"Min Vein (Frame {min_idx})")
        ax1.annotate(f"Min: {vein_areas[min_idx]}", (min_idx, vein_areas[min_idx]),
                     textcoords="offset points", xytext=(10, 10),
                     fontsize=10, color="#f59e0b",
                     arrowprops=dict(arrowstyle="->", color="#f59e0b"))

    # 下图：静脉/动脉面积比
    ax2 = axes[1]
    if artery_areas:
        ratios = []
        for v, a in zip(vein_areas, artery_areas):
            ratios.append(v / a if a > 0 else 0)
        ax2.plot(frames, ratios, color="#06b6d4", linewidth=2)
        ax2.fill_between(frames, ratios, alpha=0.2, color="#06b6d4")
    style_axis(ax2, title="Vein-to-Artery Area Ratio",
               xlabel="Frame Index", ylabel="V/A Ratio")

    plt.tight_layout()

    # 诊断报告
    report = _format_diagnosis_report(result, len(vein_areas), full_features)

    return report, fig, _diagnosis_summary_html(result)


def _format_diagnosis_report(result: dict, n_frames: int, features: dict = None) -> str:
    """格式化诊断报告（含完整特征表）"""
    lines = ["### 🩺 DVT Diagnosis Report\n"]

    if result["is_dvt"] is None:
        lines.append(f"**Status**: {result['diagnosis']}")
        return "\n".join(lines)

    lines.append(f"**Result**: {result['diagnosis']}\n")

    # 关键指标表
    lines.append("#### Key Features\n")
    lines.append("| Feature | Value | Description |")
    lines.append("|---------|-------|-------------|")

    if features:
        lines.append(f"| **VCR** | {features.get('vcr', 0):.4f} | Vein Compression Ratio (min/max) |")
        lines.append(f"| **VDR** | {features.get('vdr', 0):.4f} | Vein Disappearance Rate |")
        lines.append(f"| **VARR** | {features.get('varr', 0):.4f} | (max-min)/max |")
        lines.append(f"| **vein_cv** | {features.get('vein_cv', 0):.4f} | std/mean |")
        lines.append(f"| **MVAR** | {features.get('mvar', 0):.4f} | min(vein/artery) |")
        lines.append(f"| vein_slope | {features.get('vein_slope', 0):.6f} | Negative = normal |")
        lines.append(f"| max_drop_ratio | {features.get('max_drop_ratio', 0):.4f} | Larger = rapid collapse |")
        lines.append(f"| vein_detect_rate | {features.get('vein_detect_rate', 0):.2%} | |")
        lines.append(f"| artery_stability | {features.get('artery_stability', 0):.4f} | |")
    else:
        lines.append(f"| Area ratio (min/max) | {result['area_ratio']:.4f} | |")
        lines.append(f"| Area reduction | {result['area_change_percent']:.1f}% | |")

    lines.append(f"| Min vein area | {result.get('min_area', 'N/A')} px | |")
    lines.append(f"| Max vein area | {result.get('max_area', 'N/A')} px | |")

    # 判断依据
    ml_thresh = result.get("ml_threshold")
    if ml_thresh is not None:
        lines.append(f"| ML threshold (VCR) | {ml_thresh:.3f} | High-recall optimized |")
    lines.append(f"| Simple threshold | {result.get('threshold', 0.4):.2f} | min/max threshold |")
    lines.append(f"| Confidence | {result['confidence']:.2f} | |")
    lines.append(f"| Frames analyzed | {n_frames} | |")

    return "\n".join(lines)


def _diagnosis_summary_html(result: dict) -> str:
    """生成诊断摘要 HTML"""
    if result["is_dvt"] is None:
        return f'<div class="status-pending">{result["diagnosis"]}</div>'

    if result["is_dvt"]:
        return f"""
        <div style="text-align:center; padding:28px 20px; background:linear-gradient(135deg, rgba(220,38,38,0.12), rgba(239,68,68,0.06));
                    border:2px solid #dc2626; border-radius:16px; margin:8px 0;">
            <div style="font-size:52px; margin-bottom:8px;">⚠️</div>
            <div style="font-size:24px; font-weight:800; color:#ef4444; margin-bottom:6px;">
                DVT Suspected
            </div>
            <div style="font-size:14px; color:#fca5a5; margin-bottom:4px;">
                VCR = {result.get('area_ratio', 0):.3f} &nbsp;|&nbsp; Confidence {result['confidence']:.0%}
            </div>
            <div style="font-size:12px; color:#94a3b8; margin-top:8px; line-height:1.6;">
                Vein resists compression during ultrasound examination<br>
                <b>Further clinical examination recommended</b>
            </div>
        </div>"""
    else:
        return f"""
        <div style="text-align:center; padding:28px 20px; background:linear-gradient(135deg, rgba(16,185,129,0.12), rgba(34,197,94,0.06));
                    border:2px solid #059669; border-radius:16px; margin:8px 0;">
            <div style="font-size:52px; margin-bottom:8px;">✅</div>
            <div style="font-size:24px; font-weight:800; color:#10b981; margin-bottom:6px;">
                Normal
            </div>
            <div style="font-size:14px; color:#6ee7b7; margin-bottom:4px;">
                VCR = {result.get('area_ratio', 0):.3f} &nbsp;|&nbsp; Confidence {result['confidence']:.0%}
            </div>
            <div style="font-size:12px; color:#94a3b8; margin-top:8px; line-height:1.6;">
                Vein collapses normally, area reduction {result.get('area_change_percent', 0):.0f}%<br>
                No thrombosis signs detected
            </div>
        </div>"""


def build_diagnosis_tab(state: gr.State):
    """构建 DVT 诊断 Tab"""

    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            gr.HTML("""
            <div style="padding:16px 20px; background:linear-gradient(135deg, #1f2937, #1e293b);
                        border-radius:12px; border:1px solid #334155; margin-bottom:8px;">
                <h3 style="margin:0 0 4px 0; color:#e2e8f0; font-size:16px;">
                    🩺 DVT Intelligent Diagnosis
                </h3>
                <p style="margin:0; color:#94a3b8; font-size:13px;">
                    Automatic DVT assessment based on 19-dimensional temporal features from vein area changes
                </p>
            </div>
            """)

            threshold_slider = gr.Slider(
                minimum=0.1, maximum=0.8, value=0.4, step=0.05,
                label="⚙️ DVT Threshold",
                info="min_area / max_area > threshold → DVT suspected",
            )

            diagnose_btn = gr.Button("🩺 Run Diagnosis", variant="primary", size="lg")

            diagnosis_html = gr.HTML("""
            <div style="text-align:center; padding:24px; color:#64748b; border:1px dashed #334155; border-radius:12px;">
                Complete segmentation first, then click「Run Diagnosis」
            </div>
            """)

            diagnosis_report = gr.Markdown("""
> 💡 **How it works**: Normal veins collapse under probe pressure (VCR → 0). 
> Thrombotic veins resist compression (VCR → 1).
""")

        with gr.Column(scale=3):
            area_plot = gr.Plot(label="Area Change Curves")

    diagnose_btn.click(
        fn=_run_diagnosis,
        inputs=[state, threshold_slider],
        outputs=[diagnosis_report, area_plot, diagnosis_html],
    )
