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


def _run_diagnosis(state: dict, threshold: float):
    """执行 DVT 诊断（支持完整特征提取）"""
    vein_areas = state.get("vein_areas", [])
    artery_areas = state.get("artery_areas", [])
    pred_masks = state.get("pred_masks")

    if not vein_areas:
        return (
            "⚠️ 请先在「SAM2 分割」Tab 中完成分割",
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor="#1e293b")

    frames = list(range(len(vein_areas)))

    # 上图：面积变化
    ax1 = axes[0]
    ax1.set_facecolor("#0f172a")
    ax1.fill_between(frames, artery_areas, alpha=0.3, color="#ef4444", label="动脉面积")
    ax1.fill_between(frames, vein_areas, alpha=0.3, color="#22c55e", label="静脉面积")
    ax1.plot(frames, artery_areas, color="#ef4444", linewidth=2)
    ax1.plot(frames, vein_areas, color="#22c55e", linewidth=2)
    ax1.set_xlabel("帧序号", fontsize=12, color="#94a3b8")
    ax1.set_ylabel("像素面积", fontsize=12, color="#94a3b8")
    ax1.set_title("动脉/静脉面积变化曲线", fontsize=14, fontweight="bold", color="#f1f5f9")
    ax1.legend(fontsize=11, loc="upper right",
               facecolor="#334155", edgecolor="#475569", labelcolor="#f1f5f9")
    ax1.tick_params(colors="#94a3b8")
    for spine in ax1.spines.values():
        spine.set_color("#334155")
    ax1.grid(True, alpha=0.2, color="#475569")

    # 标注最小面积帧
    if vein_areas:
        min_idx = int(np.argmin(vein_areas))
        ax1.axvline(x=min_idx, color="#f59e0b", linestyle="--", alpha=0.7,
                    label=f"最小静脉面积 (Frame {min_idx})")
        ax1.annotate(f"Min: {vein_areas[min_idx]}", (min_idx, vein_areas[min_idx]),
                     textcoords="offset points", xytext=(10, 10),
                     fontsize=10, color="#f59e0b",
                     arrowprops=dict(arrowstyle="->", color="#f59e0b"))

    # 下图：静脉/动脉面积比
    ax2 = axes[1]
    ax2.set_facecolor("#0f172a")
    if artery_areas:
        ratios = []
        for v, a in zip(vein_areas, artery_areas):
            ratios.append(v / a if a > 0 else 0)
        ax2.plot(frames, ratios, color="#06b6d4", linewidth=2)
        ax2.fill_between(frames, ratios, alpha=0.2, color="#06b6d4")
    ax2.set_xlabel("帧序号", fontsize=12, color="#94a3b8")
    ax2.set_ylabel("静脉/动脉面积比", fontsize=12, color="#94a3b8")
    ax2.set_title("静脉/动脉面积比变化", fontsize=14, fontweight="bold", color="#f1f5f9")
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_color("#334155")
    ax2.grid(True, alpha=0.2, color="#475569")

    plt.tight_layout()

    # 诊断报告
    report = _format_diagnosis_report(result, len(vein_areas), full_features)

    return report, fig, _diagnosis_summary_html(result)


def _format_diagnosis_report(result: dict, n_frames: int, features: dict = None) -> str:
    """格式化诊断报告（含完整特征表）"""
    lines = ["### 🩺 DVT 诊断报告\n"]

    if result["is_dvt"] is None:
        lines.append(f"**状态**: {result['diagnosis']}")
        return "\n".join(lines)

    lines.append(f"**诊断结果**: {result['diagnosis']}\n")

    # 关键指标表
    lines.append("#### 关键指标\n")
    lines.append("| 指标 | 值 | 说明 |")
    lines.append("|------|------|------|")

    if features:
        lines.append(f"| **VCR** (静脉压缩比) | {features.get('vcr', 0):.4f} | min/max，越小越正常 |")
        lines.append(f"| **VDR** (静脉消失率) | {features.get('vdr', 0):.4f} | 面积<10%最大值的帧占比 |")
        lines.append(f"| **VARR** (相对范围) | {features.get('varr', 0):.4f} | (max-min)/max |")
        lines.append(f"| **vein_cv** (变异系数) | {features.get('vein_cv', 0):.4f} | std/mean |")
        lines.append(f"| **MVAR** (最小V/A比) | {features.get('mvar', 0):.4f} | min(vein/artery) |")
        lines.append(f"| 面积趋势斜率 | {features.get('vein_slope', 0):.6f} | 负=面积下降(正常) |")
        lines.append(f"| 最大帧间下降 | {features.get('max_drop_ratio', 0):.4f} | 越大=急剧塌陷 |")
        lines.append(f"| 静脉检出率 | {features.get('vein_detect_rate', 0):.2%} | |")
        lines.append(f"| 动脉稳定性 | {features.get('artery_stability', 0):.4f} | |")
    else:
        lines.append(f"| 面积变化率 (min/max) | {result['area_ratio']:.4f} | |")
        lines.append(f"| 面积缩减百分比 | {result['area_change_percent']:.1f}% | |")

    lines.append(f"| 静脉最小面积 | {result.get('min_area', 'N/A')} px | |")
    lines.append(f"| 静脉最大面积 | {result.get('max_area', 'N/A')} px | |")

    # 判断依据
    ml_thresh = result.get("ml_threshold")
    if ml_thresh is not None:
        lines.append(f"| ML 判断阈值 (VCR) | {ml_thresh:.3f} | 高召回优化 |")
    lines.append(f"| 简单阈值 | {result.get('threshold', 0.4):.2f} | min/max 阈值 |")
    lines.append(f"| 置信度 | {result['confidence']:.2f} | |")
    lines.append(f"| 分析帧数 | {n_frames} | |")

    lines.append("\n---\n")
    lines.append("**诊断原理**: 在压缩超声检查中，正常静脉会被探头压瘪（面积大幅缩小），"
                 "而有血栓的静脉会拒绝塌陷（面积基本不变）。")
    if ml_thresh is not None:
        lines.append(f"\n当 `VCR > {ml_thresh:.3f}` 时判为 DVT 疑似（高召回优化阈值）。")
    else:
        lines.append(f"\n当 `min_area / max_area > {result.get('threshold', 0.4):.2f}` 时判为 DVT 疑似。")

    return "\n".join(lines)


def _diagnosis_summary_html(result: dict) -> str:
    """生成诊断摘要 HTML"""
    if result["is_dvt"] is None:
        return f'<div class="status-pending">{result["diagnosis"]}</div>'

    if result["is_dvt"]:
        return f"""
        <div style="text-align:center; padding:24px; background:rgba(239,68,68,0.1);
                    border:2px solid #ef4444; border-radius:16px; margin:8px 0;">
            <div style="font-size:48px;">⚠️</div>
            <div style="font-size:22px; font-weight:700; color:#ef4444; margin:8px 0;">
                DVT 疑似
            </div>
            <div style="font-size:14px; color:#f87171;">
                静脉面积变化率: {result['area_ratio']:.3f} (面积缩减仅 {result['area_change_percent']:.1f}%)
            </div>
            <div style="font-size:12px; color:#94a3b8; margin-top:8px;">
                静脉拒绝塌陷 — 建议进一步临床检查
            </div>
        </div>
        """
    else:
        return f"""
        <div style="text-align:center; padding:24px; background:rgba(16,185,129,0.1);
                    border:2px solid #10b981; border-radius:16px; margin:8px 0;">
            <div style="font-size:48px;">✅</div>
            <div style="font-size:22px; font-weight:700; color:#10b981; margin:8px 0;">
                正常
            </div>
            <div style="font-size:14px; color:#34d399;">
                静脉面积变化率: {result['area_ratio']:.3f} (面积缩减 {result['area_change_percent']:.1f}%)
            </div>
            <div style="font-size:12px; color:#94a3b8; margin-top:8px;">
                静脉正常塌陷 — 未见血栓征象
            </div>
        </div>
        """


def build_diagnosis_tab(state: gr.State):
    """构建 DVT 诊断 Tab"""

    gr.Markdown("""
    ### DVT 智能诊断
    基于静脉在压缩超声过程中的面积变化率进行自动判断。
    正常静脉会塌陷（面积大幅缩小），有血栓的静脉会拒绝塌陷。
    """)

    with gr.Row():
        with gr.Column(scale=1):
            threshold_slider = gr.Slider(
                minimum=0.1, maximum=0.8, value=0.4, step=0.05,
                label="⚙️ DVT 判断阈值",
                info="min_area / max_area > 阈值 → DVT",
            )

            diagnose_btn = gr.Button("🩺 执行诊断", variant="primary", size="lg")

            diagnosis_html = gr.HTML("等待诊断...")

            diagnosis_report = gr.Markdown("完成分割后点击「执行诊断」")

        with gr.Column(scale=2):
            area_plot = gr.Plot(label="面积变化曲线")

    diagnose_btn.click(
        fn=_run_diagnosis,
        inputs=[state, threshold_slider],
        outputs=[diagnosis_report, area_plot, diagnosis_html],
    )
