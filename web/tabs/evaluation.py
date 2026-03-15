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


def _show_evaluation(state: dict):
    """展示评估结果"""
    metrics_list = state.get("frame_metrics", [])

    if not metrics_list:
        return "⚠️ 请先完成分割（确保有 GT 标注的帧用于评估）", None, None

    # Case 汇总
    summary = summarize_case_metrics(metrics_list)

    # 指标图表
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1e293b")

    frame_indices = [m["frame_idx"] for m in metrics_list]

    # 左图：Dice 变化
    ax1 = axes[0]
    ax1.set_facecolor("#0f172a")
    ax1.plot(frame_indices, [m["artery_dice"] for m in metrics_list],
             "o-", color="#ef4444", linewidth=2, markersize=5, label="Artery Dice")
    ax1.plot(frame_indices, [m["vein_dice"] for m in metrics_list],
             "s-", color="#22c55e", linewidth=2, markersize=5, label="Vein Dice")
    ax1.plot(frame_indices, [m["mean_dice"] for m in metrics_list],
             "^-", color="#3b82f6", linewidth=2, markersize=5, label="Mean Dice")
    ax1.set_xlabel("Frame Index", fontsize=11, color="#94a3b8")
    ax1.set_ylabel("Dice", fontsize=11, color="#94a3b8")
    ax1.set_title("逐帧 Dice 系数", fontsize=13, fontweight="bold", color="#f1f5f9")
    ax1.legend(fontsize=10, facecolor="#334155", edgecolor="#475569", labelcolor="#f1f5f9")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(colors="#94a3b8")
    for spine in ax1.spines.values():
        spine.set_color("#334155")
    ax1.grid(True, alpha=0.2, color="#475569")

    # 右图：mIoU 变化
    ax2 = axes[1]
    ax2.set_facecolor("#0f172a")
    ax2.plot(frame_indices, [m["artery_iou"] for m in metrics_list],
             "o-", color="#ef4444", linewidth=2, markersize=5, label="Artery IoU")
    ax2.plot(frame_indices, [m["vein_iou"] for m in metrics_list],
             "s-", color="#22c55e", linewidth=2, markersize=5, label="Vein IoU")
    ax2.plot(frame_indices, [m["miou"] for m in metrics_list],
             "^-", color="#3b82f6", linewidth=2, markersize=5, label="mIoU")
    ax2.set_xlabel("Frame Index", fontsize=11, color="#94a3b8")
    ax2.set_ylabel("IoU", fontsize=11, color="#94a3b8")
    ax2.set_title("逐帧 mIoU", fontsize=13, fontweight="bold", color="#f1f5f9")
    ax2.legend(fontsize=10, facecolor="#334155", edgecolor="#475569", labelcolor="#f1f5f9")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_color("#334155")
    ax2.grid(True, alpha=0.2, color="#475569")

    plt.tight_layout()

    # 报告
    report = _format_eval_report(summary, metrics_list)

    # 逐帧数据表格 (Markdown 格式，避免 pandas 依赖)
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
    lines = ["### 📊 分割质量评估报告\n"]
    lines.append(f"**已评估帧数**: {summary.get('n_frames', 0)}\n")

    lines.append("| 指标 | 动脉 (Artery) | 静脉 (Vein) | 平均 |")
    lines.append("|------|:------:|:------:|:------:|")
    lines.append(f"| **Dice** | {summary['artery_dice']:.4f} | {summary['vein_dice']:.4f} | {summary['mean_dice']:.4f} |")
    lines.append(f"| **IoU** | {summary['artery_iou']:.4f} | {summary['vein_iou']:.4f} | {summary['miou']:.4f} |")

    # 最差帧标注
    if metrics_list:
        worst = min(metrics_list, key=lambda m: m["mean_dice"])
        best = max(metrics_list, key=lambda m: m["mean_dice"])
        lines.append(f"\n- 🏆 **最佳帧**: Frame {best['frame_idx']} (Dice = {best['mean_dice']:.4f})")
        lines.append(f"- ⚠️ **最差帧**: Frame {worst['frame_idx']} (Dice = {worst['mean_dice']:.4f})")

    return "\n".join(lines)


def build_evaluation_tab(state: gr.State):
    """构建定量评估 Tab"""

    gr.Markdown("""
    ### 分割质量评估
    查看逐帧 Dice / mIoU 指标和 case 级汇总统计。仅在有 GT 标注的帧上进行评估。
    """)

    eval_btn = gr.Button("📊 生成评估报告", variant="primary", size="lg")

    eval_report = gr.Markdown("完成分割后点击「生成评估报告」")

    eval_plot = gr.Plot(label="逐帧指标曲线")

    gr.Markdown("### 逐帧指标明细")
    eval_table = gr.Markdown("完成评估后将显示逐帧指标表格")

    eval_btn.click(
        fn=_show_evaluation,
        inputs=[state],
        outputs=[eval_report, eval_plot, eval_table],
    )
