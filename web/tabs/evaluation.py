"""
模块 5: 导出完整报告
- 生成 PDF 格式的诊断报告
- 包含：案例信息、检测结果、分割指标、面积曲线、DVT 诊断
- 使用 matplotlib PdfPages 生成
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import sys
import tempfile
import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.utils.metrics import summarize_case_metrics, compute_dvt_diagnosis, get_unified_threshold
from web.utils.chart_style import setup_matplotlib, style_axis, get_chinese_font
from web.utils.ui import render_page_header

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _generate_pdf_report(state: dict, progress=gr.Progress(track_tqdm=True)):
    """生成完整 PDF 报告"""
    if not state.get("frame_files"):
        return None, "请先在“数据输入”页加载案例"

    vein_areas = state.get("vein_areas", [])
    artery_areas = state.get("artery_areas", [])

    if not vein_areas:
        return None, "请先完成分割和诊断后再导出报告"

    setup_matplotlib()

    case_name = state.get("current_case", "未知案例")
    split = state.get("split", "")
    frame_files = state.get("frame_files", [])
    detections = state.get("detections", {})
    frame_metrics = state.get("frame_metrics", [])
    pred_masks = state.get("pred_masks", {})

    # 获取诊断结果
    unified_threshold = get_unified_threshold()
    diag_result = compute_dvt_diagnosis(vein_areas, threshold=unified_threshold)

    # 尝试 ML 诊断
    ml_result = None
    full_features = None
    if pred_masks:
        try:
            from web.services import InferenceService
            num_frames = len(frame_files)
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
            ml_result = InferenceService.get().run_diagnosis(masks_list)
            full_features = ml_result.get("features")
        except Exception:
            pass

    if ml_result is not None:
        diag_result["is_dvt"] = ml_result["is_dvt"]
        diag_result["confidence"] = ml_result["confidence"]
        diag_result["probability"] = ml_result.get("probability")
        diag_result["model"] = ml_result.get("model", "RF unified")
        diag_result["vcr"] = ml_result.get("vcr", diag_result.get("area_ratio"))
        diag_result["diagnosis"] = (
            "DVT 疑似（静脉拒绝塌陷）" if ml_result["is_dvt"]
            else "正常（静脉正常塌陷）"
        )

    # 生成 PDF
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = Path(tempfile.gettempdir()) / f"EchoDVT_Report_{case_name}_{timestamp}.pdf"

    progress(0.1, desc="生成 PDF 报告...")

    zh_font = get_chinese_font()
    font_props = {"fontsize": 10}
    title_props = {"fontsize": 14, "fontweight": "bold"}
    subtitle_props = {"fontsize": 11, "fontweight": "bold"}

    with PdfPages(str(pdf_path)) as pdf:
        # ── 第 1 页: 封面 + 案例信息 + 诊断结论 ──
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        fig.patch.set_facecolor("white")

        # 标题
        fig.text(0.5, 0.92, "EchoDVT 超声诊断报告", ha="center", va="top", **title_props)
        fig.text(0.5, 0.89, f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 ha="center", va="top", fontsize=9, color="#64748b")

        # 案例信息
        info_y = 0.84
        info_lines = [
            f"案例名称: {case_name}",
            f"数据集分区: {split}",
            f"总帧数: {len(frame_files)}",
            f"标注帧数: {len(frame_metrics)}",
        ]
        for line in info_lines:
            fig.text(0.08, info_y, line, **font_props)
            info_y -= 0.025

        # 诊断结论
        fig.text(0.08, info_y - 0.02, "诊断结论", **subtitle_props)
        info_y -= 0.05

        is_dvt = diag_result.get("is_dvt")
        diag_text = diag_result.get("diagnosis", "未完成诊断")
        model_name = diag_result.get("model", "")
        prob = diag_result.get("probability")
        conf = diag_result.get("confidence", 0)

        result_color = "#dc2626" if is_dvt else "#059669"
        fig.text(0.08, info_y, f"结果: {diag_text}", fontsize=12, fontweight="bold", color=result_color)
        info_y -= 0.03

        metric_parts = []
        if prob is not None:
            metric_parts.append(f"RF 概率 = {prob:.4f}")
        if diag_result.get("vcr") is not None:
            metric_parts.append(f"VCR = {diag_result['vcr']:.4f}")
        metric_parts.append(f"置信度 = {conf:.0%}")
        metric_parts.append(f"模型: {model_name}")
        fig.text(0.08, info_y, " | ".join(metric_parts), **font_props, color="#475569")
        info_y -= 0.02

        # 检测框信息
        if detections:
            fig.text(0.08, info_y - 0.02, "YOLO 检测结果", **subtitle_props)
            info_y -= 0.05
            for cls_name, cn in [("artery", "动脉"), ("vein", "静脉")]:
                det = detections.get(cls_name)
                if det is None:
                    fig.text(0.08, info_y, f"  {cn}: 未检测到", **font_props)
                else:
                    box = det["box"]
                    conf_det = det.get("conf", 0)
                    tags = []
                    if det.get("from_gt"):
                        tags.append("GT")
                    if det.get("inferred"):
                        tags.append("推断")
                    if det.get("prior_all"):
                        tags.append("先验")
                    tag_str = f" [{','.join(tags)}]" if tags else ""
                    fig.text(0.08, info_y,
                             f"  {cn}: conf={conf_det:.3f}  box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}){tag_str}",
                             **font_props)
                info_y -= 0.025

        # 关键特征表
        if full_features:
            fig.text(0.08, info_y - 0.02, "21 维时序特征 (关键)", **subtitle_props)
            info_y -= 0.05
            key_feats = [
                ("VCR", full_features.get("vcr", 0), "静脉压缩比"),
                ("VDR", full_features.get("vdr", 0), "静脉消失率"),
                ("VARR", full_features.get("varr", 0), "(max-min)/max"),
                ("vein_cv", full_features.get("vein_cv", 0), "变异系数"),
                ("MVAR", full_features.get("mvar", 0), "最小静脉/动脉比"),
                ("max_drop_ratio", full_features.get("max_drop_ratio", 0), "最大下降比"),
                ("vein_detect_rate", full_features.get("vein_detect_rate", 0), "静脉检出率"),
            ]
            for name, val, desc in key_feats:
                fig.text(0.08, info_y, f"  {name} = {val:.4f}    ({desc})", **font_props)
                info_y -= 0.022

        # 面积统计
        fig.text(0.08, info_y - 0.02, "面积统计", **subtitle_props)
        info_y -= 0.05
        if vein_areas:
            fig.text(0.08, info_y, f"  静脉面积: min={min(vein_areas):.0f}  max={max(vein_areas):.0f}  mean={np.mean(vein_areas):.0f} px", **font_props)
            info_y -= 0.025
        if artery_areas:
            fig.text(0.08, info_y, f"  动脉面积: min={min(artery_areas):.0f}  max={max(artery_areas):.0f}  mean={np.mean(artery_areas):.0f} px", **font_props)
            info_y -= 0.025
        area_change = diag_result.get("area_change_percent", 0)
        fig.text(0.08, info_y, f"  面积缩减率: {area_change:.1f}%", **font_props)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        progress(0.3, desc="生成面积曲线...")

        # ── 第 2 页: 面积变化曲线 ──
        fig2, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
        fig2.patch.set_facecolor("white")
        fig2.suptitle(f"面积变化曲线 — {case_name}", fontsize=13, fontweight="bold", y=0.96)

        frames = list(range(len(vein_areas)))

        ax1 = axes[0]
        ax1.fill_between(frames, artery_areas, alpha=0.3, color="#ef4444", label="动脉面积")
        ax1.fill_between(frames, vein_areas, alpha=0.3, color="#22c55e", label="静脉面积")
        ax1.plot(frames, artery_areas, color="#ef4444", linewidth=2)
        ax1.plot(frames, vein_areas, color="#22c55e", linewidth=2)
        style_axis(ax1, title="动脉 / 静脉面积变化", xlabel="帧序号", ylabel="面积 (px)")
        ax1.legend(fontsize=10, loc="upper right")

        if vein_areas:
            min_idx = int(np.argmin(vein_areas))
            ax1.axvline(x=min_idx, color="#f59e0b", linestyle="--", alpha=0.7)
            ax1.annotate(f"最小: {vein_areas[min_idx]:.0f}", (min_idx, vein_areas[min_idx]),
                         textcoords="offset points", xytext=(10, 10),
                         fontsize=9, color="#f59e0b",
                         arrowprops=dict(arrowstyle="->", color="#f59e0b"))

        ax2 = axes[1]
        if artery_areas:
            ratios = [v / a if a > 0 else 0 for v, a in zip(vein_areas, artery_areas)]
            ax2.plot(frames, ratios, color="#06b6d4", linewidth=2)
            ax2.fill_between(frames, ratios, alpha=0.2, color="#06b6d4")
        style_axis(ax2, title="静脉/动脉面积比", xlabel="帧序号", ylabel="V/A 比值")

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig2, dpi=150)
        plt.close(fig2)

        progress(0.5, desc="生成分割指标...")

        # ── 第 3 页: 分割指标（如果有 GT） ──
        if frame_metrics:
            summary = summarize_case_metrics(frame_metrics)

            fig3, axes3 = plt.subplots(2, 1, figsize=(8.27, 11.69))
            fig3.patch.set_facecolor("white")
            fig3.suptitle(f"分割质量指标 — {case_name}", fontsize=13, fontweight="bold", y=0.96)

            frame_indices = [m["frame_idx"] for m in frame_metrics]

            # Dice 曲线
            ax_d = axes3[0]
            ax_d.plot(frame_indices, [m["artery_dice"] for m in frame_metrics],
                      "o-", color="#ef4444", linewidth=2, markersize=4, label="动脉 Dice")
            ax_d.plot(frame_indices, [m["vein_dice"] for m in frame_metrics],
                      "s-", color="#22c55e", linewidth=2, markersize=4, label="静脉 Dice")
            ax_d.plot(frame_indices, [m["mean_dice"] for m in frame_metrics],
                      "^-", color="#3b82f6", linewidth=2, markersize=4, label="平均 Dice")
            style_axis(ax_d, title="逐帧 Dice 分数", xlabel="帧序号", ylabel="Dice")
            ax_d.legend(fontsize=9)
            ax_d.set_ylim(0, 1.05)

            # IoU 曲线
            ax_i = axes3[1]
            ax_i.plot(frame_indices, [m["artery_iou"] for m in frame_metrics],
                      "o-", color="#ef4444", linewidth=2, markersize=4, label="动脉 IoU")
            ax_i.plot(frame_indices, [m["vein_iou"] for m in frame_metrics],
                      "s-", color="#22c55e", linewidth=2, markersize=4, label="静脉 IoU")
            ax_i.plot(frame_indices, [m["miou"] for m in frame_metrics],
                      "^-", color="#3b82f6", linewidth=2, markersize=4, label="mIoU")
            style_axis(ax_i, title="逐帧 mIoU", xlabel="帧序号", ylabel="IoU")
            ax_i.legend(fontsize=9)
            ax_i.set_ylim(0, 1.05)

            plt.tight_layout(rect=[0, 0, 1, 0.94])

            # 在图下方添加汇总文字
            fig3.text(0.08, 0.02,
                      f"平均 Dice: {summary['mean_dice']:.4f} | "
                      f"动脉 Dice: {summary['artery_dice']:.4f} | "
                      f"静脉 Dice: {summary['vein_dice']:.4f} | "
                      f"mIoU: {summary['miou']:.4f}",
                      fontsize=9, color="#475569")

            pdf.savefig(fig3, dpi=150)
            plt.close(fig3)

        progress(0.7, desc="生成首帧可视化...")

        # ── 第 4 页: 首帧检测 + 分割可视化 ──
        if frame_files:
            fig4 = plt.figure(figsize=(8.27, 11.69))
            fig4.patch.set_facecolor("white")
            fig4.suptitle(f"首帧可视化 — {case_name}", fontsize=13, fontweight="bold", y=0.96)

            first_img = cv2.imread(frame_files[0])
            if first_img is not None:
                first_rgb = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)

                # 原图
                ax_orig = fig4.add_subplot(2, 1, 1)
                ax_orig.imshow(first_rgb)
                ax_orig.set_title("原始图像", fontsize=11, fontweight="bold")
                ax_orig.axis("off")

                # 检测框
                if detections:
                    for cls_name, color in [("artery", "#ef4444"), ("vein", "#22c55e")]:
                        det = detections.get(cls_name)
                        if det is not None:
                            box = det["box"]
                            import matplotlib.patches as mpatches
                            rect = mpatches.Rectangle(
                                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                linewidth=2, edgecolor=color, facecolor="none"
                            )
                            ax_orig.add_patch(rect)

                # 分割结果
                ax_seg = fig4.add_subplot(2, 1, 2)
                if pred_masks and 0 in pred_masks:
                    from web.utils.visualization import overlay_masks
                    sem = pred_masks[0]["semantic"]
                    vis = overlay_masks(first_img, sem, alpha=0.45)
                    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                    ax_seg.imshow(vis_rgb)
                else:
                    ax_seg.imshow(first_rgb)
                ax_seg.set_title("分割叠加", fontsize=11, fontweight="bold")
                ax_seg.axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig4, dpi=150)
            plt.close(fig4)

    progress(1.0, desc="PDF 生成完成")

    report_md = (
        f"### 📄 报告已生成\n\n"
        f"- **案例**: {case_name}\n"
        f"- **诊断**: {diag_result.get('diagnosis', '—')}\n"
        f"- **页数**: {'4' if frame_files else '3'} 页{' (含分割指标)' if frame_metrics else ''}\n"
        f"- **路径**: `{pdf_path}`\n\n"
        f"> 点击下方「下载」按钮保存 PDF 文件。"
    )

    return str(pdf_path), report_md


def build_report_tab(state: gr.State):
    """构建导出报告 Tab"""

    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            gr.HTML(render_page_header(
                "导出完整报告",
                "将当前案例的检测、分割和诊断结果汇总为 PDF 报告。",
                eyebrow="Report",
            ))

            export_btn = gr.Button("生成 PDF 报告", variant="primary", size="lg")

            report_status = gr.Markdown(
                "完成分割和诊断后，可在此导出 PDF。报告包含案例信息、检测结果、面积曲线、分割指标和诊断结论。"
            )

        with gr.Column(scale=3):
            pdf_file = gr.File(label="PDF 报告下载", file_count="single", height=200)

    export_btn.click(
        fn=_generate_pdf_report,
        inputs=[state],
        outputs=[pdf_file, report_status],
    )
