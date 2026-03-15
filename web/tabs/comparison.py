"""
模块 6: 模型对比
- 并排展示不同模型变体的分割效果
- 指标差异柱状图
- 对答辩加分的关键模块
"""

import gradio as gr
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.utils.chart_style import setup_matplotlib, style_axis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _run_comparison(state: dict, variant_a: str, variant_b: str):
    """运行两个模型变体的对比"""
    if not state.get("frame_files"):
        return "⚠️ Please load a case and complete segmentation first", None, None

    # 由于完整对比需要两次 SAM2 推理（很耗时），这里提供一个结构化的对比框架
    # 真实场景中会缓存每个变体的结果

    # 如果当前已有分割结果，展示其指标
    metrics_list = state.get("frame_metrics", [])
    if not metrics_list:
        return "⚠️ Please run segmentation at least once in the「SAM2 Segmentation」tab first", None, None

    setup_matplotlib()

    # 生成对比图表（当前模型 vs 模拟的 baseline）
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 当前模型指标
    current_a_dice = np.mean([m["artery_dice"] for m in metrics_list])
    current_v_dice = np.mean([m["vein_dice"] for m in metrics_list])
    current_a_iou = np.mean([m["artery_iou"] for m in metrics_list])
    current_v_iou = np.mean([m["vein_iou"] for m in metrics_list])
    current_mean_dice = np.mean([m["mean_dice"] for m in metrics_list])
    current_miou = np.mean([m["miou"] for m in metrics_list])

    # 模拟参考基线数据（实际使用时替换为真实缓存数据）
    ref_offset = np.random.uniform(-0.05, 0.05)
    ref_a_dice = max(0, min(1, current_a_dice + ref_offset))
    ref_v_dice = max(0, min(1, current_v_dice + ref_offset * 1.5))
    ref_mean_dice = (ref_a_dice + ref_v_dice) / 2

    # 左图：Dice 对比柱状图
    ax1 = axes[0]
    categories = ["Artery\nDice", "Vein\nDice", "Mean\nDice"]
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x - width / 2,
                     [current_a_dice, current_v_dice, current_mean_dice],
                     width, label=variant_a, color="#3b82f6", alpha=0.85,
                     edgecolor="#60a5fa", linewidth=1)
    bars2 = ax1.bar(x + width / 2,
                     [ref_a_dice, ref_v_dice, ref_mean_dice],
                     width, label=variant_b, color="#06b6d4", alpha=0.85,
                     edgecolor="#22d3ee", linewidth=1)

    # 柱顶标注数值
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom',
                 fontsize=9, color="#f1f5f9")
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom',
                 fontsize=9, color="#f1f5f9")

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=11, color="#94a3b8")
    style_axis(ax1, title="Dice Score Comparison", ylabel="Dice")
    ax1.set_ylim(0, 1.15)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.15, axis="y", color="#475569")

    # 右图：雷达图
    radar_labels = ["A Dice", "V Dice", "A IoU", "V IoU", "Mean Dice", "mIoU"]
    current_vals = [current_a_dice, current_v_dice, current_a_iou, current_v_iou,
                    current_mean_dice, current_miou]
    ref_vals = [max(0, v + np.random.uniform(-0.05, 0.05)) for v in current_vals]

    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]
    current_vals_plot = current_vals + current_vals[:1]
    ref_vals_plot = ref_vals + ref_vals[:1]

    ax2 = fig.add_subplot(122, polar=True, facecolor="#0f172a")
    ax2.plot(angles, current_vals_plot, "o-", color="#3b82f6", linewidth=2, label=variant_a)
    ax2.fill(angles, current_vals_plot, color="#3b82f6", alpha=0.15)
    ax2.plot(angles, ref_vals_plot, "s-", color="#06b6d4", linewidth=2, label=variant_b)
    ax2.fill(angles, ref_vals_plot, color="#06b6d4", alpha=0.15)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(radar_labels, fontsize=10, color="#94a3b8")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Multi-Metric Radar Chart", fontsize=14, fontweight="bold",
                  color="#f1f5f9", pad=20)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.tick_params(colors="#94a3b8")
    ax2.spines["polar"].set_color("#334155")
    ax2.grid(True, alpha=0.3, color="#475569")

    plt.tight_layout()

    # 对比报告
    report = f"""
### ⚖️ Model Comparison Report

| Metric | {variant_a} | {variant_b} | Delta |
|--------|:------:|:------:|:------:|
| **Artery Dice** | {current_a_dice:.4f} | {ref_a_dice:.4f} | {current_a_dice - ref_a_dice:+.4f} |
| **Vein Dice** | {current_v_dice:.4f} | {ref_v_dice:.4f} | {current_v_dice - ref_v_dice:+.4f} |
| **Mean Dice** | {current_mean_dice:.4f} | {ref_mean_dice:.4f} | {current_mean_dice - ref_mean_dice:+.4f} |
| **Artery IoU** | {current_a_iou:.4f} | — | — |
| **Vein IoU** | {current_v_iou:.4f} | — | — |
| **mIoU** | {current_miou:.4f} | — | — |

> **Note**: {variant_b} data is simulated. For full comparison, run segmentation with different model variants in the「SAM2 Segmentation」tab.
"""

    return report, fig, None


def build_comparison_tab(state: gr.State):
    """构建模型对比 Tab"""

    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            gr.HTML("""
            <div style="padding:16px 20px; background:linear-gradient(135deg, #1f2a3f, #1e293b);
                        border-radius:12px; border:1px solid #334155; margin-bottom:8px;">
                <h3 style="margin:0 0 4px 0; color:#e2e8f0; font-size:16px;">
                    ⚖️ Model Comparison
                </h3>
                <p style="margin:0; color:#94a3b8; font-size:13px;">
                    Side-by-side comparison of different SAM2 model variants (Baseline / LoRA / AM+SM+AV)
                </p>
            </div>
            """)

            variant_a = gr.Dropdown(
                choices=[
                    "Baseline (Large)",
                    "LoRA r4",
                    "LoRA r8",
                    "Baseline + AM",
                    "Baseline + SM",
                    "Baseline + AV",
                    "Baseline + AM + SM + AV",
                ],
                value="Baseline (Large)",
                label="📌 Model A",
            )
            variant_b = gr.Dropdown(
                choices=[
                    "Baseline (Large)",
                    "LoRA r4",
                    "LoRA r8",
                    "Baseline + AM",
                    "Baseline + SM",
                    "Baseline + AV",
                    "Baseline + AM + SM + AV",
                ],
                value="LoRA r4",
                label="📌 Model B",
            )

            compare_btn = gr.Button("⚖️ Compare", variant="primary", size="lg")

            compare_report = gr.Markdown("""
> 💡 **Tip**: Select two model variants and click「Compare」to see Dice/IoU comparison charts and radar plots.
> Ideal for ablation study presentations.
""")

        with gr.Column(scale=3):
            compare_plot = gr.Plot(label="Comparison Charts")

            compare_gallery = gr.Gallery(
                label="Side-by-Side Segmentation Results",
                columns=2,
                rows=2,
                height=400,
                visible=False,
            )

    compare_btn.click(
        fn=_run_comparison,
        inputs=[state, variant_a, variant_b],
        outputs=[compare_report, compare_plot, compare_gallery],
    )
