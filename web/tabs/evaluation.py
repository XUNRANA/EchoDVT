"""
模块 5: 定量评估面板
- Dice / mIoU 指标展示
- 逐帧指标 + case 汇总
- 图表可视化
"""

import gradio as gr
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.utils.metrics import summarize_case_metrics
from web.utils.chart_style import setup_matplotlib, style_axis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _show_evaluation(state: dict):
    """展示评估结果"""
    metrics_list = state.get("frame_metrics", [])

    if not metrics_list:
        return "⚠️ Please complete segmentation first (requires GT-annotated frames for evaluation)", None, None

    # Case 汇总
    summary = summarize_case_metrics(metrics_list)

    # 指标图表
    setup_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    frame_indices = [m["frame_idx"] for m in metrics_list]

    # 左图：Dice 变化
    ax1 = axes[0]
    ax1.plot(frame_indices, [m["artery_dice"] for m in metrics_list],
             "o-", color="#ef4444", linewidth=2, markersize=5, label="Artery Dice")
    ax1.plot(frame_indices, [m["vein_dice"] for m in metrics_list],
             "s-", color="#22c55e", linewidth=2, markersize=5, label="Vein Dice")
    ax1.plot(frame_indices, [m["mean_dice"] for m in metrics_list],
             "^-", color="#3b82f6", linewidth=2, markersize=5, label="Mean Dice")
    style_axis(ax1, title="Per-Frame Dice Score", xlabel="Frame Index", ylabel="Dice")
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.05)

    # 右图：mIoU 变化
    ax2 = axes[1]
    ax2.plot(frame_indices, [m["artery_iou"] for m in metrics_list],
             "o-", color="#ef4444", linewidth=2, markersize=5, label="Artery IoU")
    ax2.plot(frame_indices, [m["vein_iou"] for m in metrics_list],
             "s-", color="#22c55e", linewidth=2, markersize=5, label="Vein IoU")
    ax2.plot(frame_indices, [m["miou"] for m in metrics_list],
             "^-", color="#3b82f6", linewidth=2, markersize=5, label="mIoU")
    style_axis(ax2, title="Per-Frame mIoU", xlabel="Frame Index", ylabel="IoU")
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()

    # 报告
    report = _format_eval_report(summary, metrics_list)

    # 逐帧数据表格 (Markdown 格式)
    table_lines = ["| Frame | A Dice | A IoU | V Dice | V IoU | Mean Dice | mIoU |"]
    table_lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for m in metrics_list:
        table_lines.append(
            f"| {m['frame_idx']} | {m['artery_dice']:.4f} | {m['artery_iou']:.4f} "
            f"| {m['vein_dice']:.4f} | {m['vein_iou']:.4f} "
            f"| {m['mean_dice']:.4f} | {m['miou']:.4f} |"
        )
    table_md = "\n".join(table_lines)

    return report, fig, table_md


def _format_eval_report(summary: dict, metrics_list: list) -> str:
    """格式化评估报告"""
    lines = ["### 📊 Segmentation Quality Report\n"]
    lines.append(f"**Evaluated Frames**: {summary.get('n_frames', 0)}\n")

    lines.append("| Metric | Artery | Vein | Mean |")
    lines.append("|--------|:------:|:----:|:----:|")
    lines.append(f"| **Dice** | {summary['artery_dice']:.4f} | {summary['vein_dice']:.4f} | {summary['mean_dice']:.4f} |")
    lines.append(f"| **IoU** | {summary['artery_iou']:.4f} | {summary['vein_iou']:.4f} | {summary['miou']:.4f} |")

    # 最佳/最差帧
    if metrics_list:
        worst = min(metrics_list, key=lambda m: m["mean_dice"])
        best = max(metrics_list, key=lambda m: m["mean_dice"])
        lines.append(f"\n- 🏆 **Best Frame**: Frame {best['frame_idx']} (Dice = {best['mean_dice']:.4f})")
        lines.append(f"- ⚠️ **Worst Frame**: Frame {worst['frame_idx']} (Dice = {worst['mean_dice']:.4f})")

    return "\n".join(lines)


def build_evaluation_tab(state: gr.State):
    """构建定量评估 Tab"""

    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            gr.HTML("""
            <div style="padding:16px 20px; background:linear-gradient(135deg, #f0f9ff, #eff6ff);
                        border-radius:12px; border:1px solid #e2e8f0; margin-bottom:8px;">
                <h3 style="margin:0 0 4px 0; color:#1e293b; font-size:16px;">
                    📊 Quantitative Evaluation
                </h3>
                <p style="margin:0; color:#64748b; font-size:13px;">
                    Per-frame Dice / mIoU metrics and case-level summary. Only evaluates frames with GT annotations.
                </p>
            </div>
            """)

            eval_btn = gr.Button("📊 Generate Evaluation Report", variant="primary", size="lg")

            eval_report = gr.Markdown("""
> 💡 **Note**: Complete segmentation first. This tab evaluates segmentation quality against Ground Truth annotations.
""")

            gr.Markdown("### Per-Frame Metrics")
            eval_table = gr.Markdown("*Evaluation data will appear here after running the report.*")

        with gr.Column(scale=3):
            eval_plot = gr.Plot(label="Per-Frame Metric Curves")

    eval_btn.click(
        fn=_show_evaluation,
        inputs=[state],
        outputs=[eval_report, eval_plot, eval_table],
    )
