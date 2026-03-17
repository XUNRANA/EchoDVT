"""
模块 4: DVT 诊断结果
- 基于 21 维时序特征的智能诊断
- 面积变化曲线可视化
- 诊断结果展示（含关键特征表）
"""

import gradio as gr
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.utils.metrics import compute_dvt_diagnosis, get_unified_threshold
from web.utils.chart_style import setup_matplotlib, style_axis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _run_diagnosis(state: dict):
    """执行 DVT 诊断（支持完整特征提取）"""
    vein_areas = state.get("vein_areas", [])
    artery_areas = state.get("artery_areas", [])
    pred_masks = state.get("pred_masks")
    unified_threshold = get_unified_threshold()

    if not vein_areas:
        return (
            "⚠️ 请先在「🔬 SAM2 分割」Tab 中完成分割",
            None,
            "等待分割结果...",
        )

    # 尝试使用 InferenceService 进行完整 21 维特征提取
    full_features = None
    ml_result = None
    if pred_masks:
        try:
            from web.services import InferenceService
            import signal

            # 收集所有帧的 semantic mask（按帧序排列）
            num_frames = len(state.get("frame_files", []))
            masks_list = []
            for i in range(num_frames):
                entry = pred_masks.get(i)
                if entry is not None:
                    masks_list.append(entry["semantic"])
                else:
                    if masks_list:
                        masks_list.append(np.zeros_like(masks_list[-1]))
                    else:
                        masks_list.append(np.zeros((256, 256), dtype=np.uint8))

            # 设置 30 秒超时，防止模型加载卡死
            def _timeout_handler(signum, frame):
                raise TimeoutError("ML diagnosis timed out")

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(30)
            try:
                ml_result = InferenceService.get().run_diagnosis(masks_list)
                full_features = ml_result.get("features")
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except (TimeoutError, Exception) as e:
            print(f"[Diagnosis] ML inference skipped: {e}")
            ml_result = None

    # 兜底：简单 VCR 阈值诊断
    result = compute_dvt_diagnosis(vein_areas, threshold=unified_threshold)
    result["model"] = "VCR fallback"
    result["vcr"] = result.get("area_ratio")

    # 如果 ML 诊断可用，覆盖部分结果
    if ml_result is not None:
        result["is_dvt"] = ml_result["is_dvt"]
        result["confidence"] = ml_result["confidence"]
        result["ml_threshold"] = ml_result["threshold"]
        result["probability"] = ml_result.get("probability")
        result["model"] = ml_result.get("model", "RF unified")
        result["vcr"] = ml_result.get("vcr", result.get("area_ratio"))
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
    ax1.fill_between(frames, artery_areas, alpha=0.3, color="#ef4444", label="动脉面积")
    ax1.fill_between(frames, vein_areas, alpha=0.3, color="#22c55e", label="静脉面积")
    ax1.plot(frames, artery_areas, color="#ef4444", linewidth=2)
    ax1.plot(frames, vein_areas, color="#22c55e", linewidth=2)
    style_axis(ax1, title="动脉 / 静脉面积变化",
               xlabel="帧序号", ylabel="面积 (px)")
    ax1.legend(fontsize=11, loc="upper right")

    # 标注最小面积帧
    if vein_areas:
        min_idx = int(np.argmin(vein_areas))
        ax1.axvline(x=min_idx, color="#f59e0b", linestyle="--", alpha=0.7,
                    label=f"静脉最小值 (第 {min_idx} 帧)")
        ax1.annotate(f"最小: {vein_areas[min_idx]}", (min_idx, vein_areas[min_idx]),
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
    style_axis(ax2, title="静脉/动脉面积比",
               xlabel="帧序号", ylabel="V/A 比值")

    plt.tight_layout()

    # 诊断报告
    report = _format_diagnosis_report(result, len(vein_areas), full_features)

    return report, fig, _diagnosis_summary_html(result)


def _format_diagnosis_report(result: dict, n_frames: int, features: dict = None) -> str:
    """格式化诊断报告（含完整特征表）"""
    lines = ["### 🩺 DVT 诊断报告\n"]
    model_name = result.get("model", "")
    probability = result.get("probability")

    if result["is_dvt"] is None:
        lines.append(f"**Status**: {result['diagnosis']}")
        return "\n".join(lines)

    lines.append(f"**诊断结果**: {result['diagnosis']}\n")

    # 关键指标表
    lines.append("#### 关键特征\n")
    lines.append("| 特征 | 值 | 说明 |")
    lines.append("|------|-----|------|")

    if features:
        lines.append(f"| **VCR** | {features.get('vcr', 0):.4f} | 静脉压缩比 (min/max) |")
        lines.append(f"| **VDR** | {features.get('vdr', 0):.4f} | 静脉消失率 |")
        lines.append(f"| **VARR** | {features.get('varr', 0):.4f} | (max-min)/max |")
        lines.append(f"| **vein_cv** | {features.get('vein_cv', 0):.4f} | 变异系数 std/mean |")
        lines.append(f"| **MVAR** | {features.get('mvar', 0):.4f} | 最小静脉/动脉比 |")
        lines.append(f"| vein_slope | {features.get('vein_slope', 0):.6f} | 负值=正常 |")
        lines.append(f"| max_drop_ratio | {features.get('max_drop_ratio', 0):.4f} | 越大=越快塌陷 |")
        lines.append(f"| vein_detect_rate | {features.get('vein_detect_rate', 0):.2%} | 静脉检出率 |")
        lines.append(f"| artery_stability | {features.get('artery_stability', 0):.4f} | 动脉稳定性 |")
    else:
        lines.append(f"| 面积比 (min/max) | {result['area_ratio']:.4f} | |")
        lines.append(f"| 面积缩减率 | {result['area_change_percent']:.1f}% | |")

    lines.append(f"| 静脉最小面积 | {result.get('min_area', 'N/A')} px | |")
    lines.append(f"| 静脉最大面积 | {result.get('max_area', 'N/A')} px | |")
    if probability is not None:
        lines.append(f"| RF 概率 | {probability:.4f} | 统一模型输出 |")

    # 判断依据
    ml_thresh = result.get("ml_threshold")
    if "RF" in model_name and ml_thresh is not None:
        lines.append(f"| 统一模型阈值 (prob) | {ml_thresh:.3f} | RF unified |")
    else:
        lines.append(f"| 回退阈值 (VCR) | {result.get('threshold', get_unified_threshold()):.3f} | 简单 min/max 规则 |")
    lines.append(f"| 置信度 | {result['confidence']:.2f} | |")
    lines.append(f"| 分析帧数 | {n_frames} | |")

    return "\n".join(lines)


def _diagnosis_summary_html(result: dict) -> str:
    """生成诊断摘要 HTML"""
    if result["is_dvt"] is None:
        return f'<div class="status-pending">{result["diagnosis"]}</div>'

    prob_val = result.get("probability")
    metric_label = f"RF prob = {prob_val:.3f}" if prob_val is not None else f"VCR = {result.get('area_ratio', 0):.3f}"

    if result["is_dvt"]:
        return f"""
        <div style="text-align:center; padding:28px 20px; background:linear-gradient(135deg, rgba(220,38,38,0.12), rgba(239,68,68,0.06));
                    border:2px solid #dc2626; border-radius:16px; margin:8px 0;">
            <div style="font-size:52px; margin-bottom:8px;">⚠️</div>
            <div style="font-size:24px; font-weight:800; color:#ef4444; margin-bottom:6px;">
                DVT 疑似
            </div>
            <div style="font-size:14px; color:#b91c1c; margin-bottom:4px;">
                {metric_label} &nbsp;|&nbsp; 置信度 {result['confidence']:.0%}
            </div>
            <div style="font-size:12px; color:#64748b; margin-top:8px; line-height:1.6;">
                静脉在超声压缩检查中拒绝塌陷<br>
                <b>建议进一步临床检查</b>
            </div>
        </div>"""
    else:
        return f"""
        <div style="text-align:center; padding:28px 20px; background:linear-gradient(135deg, rgba(16,185,129,0.12), rgba(34,197,94,0.06));
                    border:2px solid #059669; border-radius:16px; margin:8px 0;">
            <div style="font-size:52px; margin-bottom:8px;">✅</div>
            <div style="font-size:24px; font-weight:800; color:#10b981; margin-bottom:6px;">
                正常
            </div>
            <div style="font-size:14px; color:#047857; margin-bottom:4px;">
                {metric_label} &nbsp;|&nbsp; 置信度 {result['confidence']:.0%}
            </div>
            <div style="font-size:12px; color:#64748b; margin-top:8px; line-height:1.6;">
                静脉正常塌陷，面积缩减 {result.get('area_change_percent', 0):.0f}%<br>
                未见血栓征象
            </div>
        </div>"""


def build_diagnosis_tab(state: gr.State):
    """构建 DVT 诊断 Tab"""
    unified_threshold = get_unified_threshold()

    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            gr.HTML("""
            <div style="padding:16px 20px; background:linear-gradient(135deg, #f0f9ff, #eff6ff);
                        border-radius:12px; border:1px solid #e2e8f0; margin-bottom:8px;">
                <h3 style="margin:0 0 4px 0; color:#1e293b; font-size:16px;">
                    🩺 DVT 智能诊断
                </h3>
                <p style="margin:0; color:#64748b; font-size:13px;">
                    基于 21 维时序特征的 DVT 自动评估，默认使用最新统一模型阈值
                </p>
            </div>
            """)

            gr.Markdown(
                f"> 当前固定使用最新统一模型 `RF unified`，判断阈值为 `prob ≥ {unified_threshold:.2f}`。"
            )

            diagnose_btn = gr.Button("🩺 运行诊断", variant="primary", size="lg")

            diagnosis_html = gr.HTML("""
            <div style="text-align:center; padding:24px; color:#64748b; border:1px dashed #cbd5e1; border-radius:12px;">
                请先完成分割，然后点击「运行诊断」
            </div>
            """)

            diagnosis_report = gr.Markdown("""
> 💡 **诊断原理**: 默认使用最新 `RF unified` 模型输出的 DVT 概率进行判断。
> 当统一模型不可用时，回退到与当前阈值对齐的简单 VCR 规则。
""")

        with gr.Column(scale=3):
            area_plot = gr.Plot(label="面积变化曲线")

    diagnose_btn.click(
        fn=_run_diagnosis,
        inputs=[state],
        outputs=[diagnosis_report, area_plot, diagnosis_html],
    )
