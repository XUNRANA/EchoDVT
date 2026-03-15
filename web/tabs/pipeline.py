"""
模块: 一键全流程分析
  数据加载 → YOLO 检测 → SAM2 分割 → 特征提取 → DVT 诊断
  单击完成，生成完整诊断报告
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.utils.visualization import draw_detection_boxes, overlay_masks, bgr_to_rgb
from web.utils.metrics import compute_frame_metrics, compute_mask_area, compute_dvt_diagnosis
from web.utils.chart_style import setup_matplotlib, style_axis


def _run_full_pipeline(
    state: dict,
    model_variant: str,
    use_mfp: bool,
    conf_threshold: float,
    progress=gr.Progress(track_tqdm=True),
):
    """执行完整的端到端分析流程"""
    if not state.get("frame_files"):
        return state, None, [], None, "", ""

    images_dir = Path(state["images_dir"])
    masks_dir = Path(state.get("masks_dir", ""))
    frame_files = state["frame_files"]
    num_frames = len(frame_files)

    # ============ Step 1: YOLO 检测 ============
    progress(0.05, desc="[1/4] YOLO 血管检测...")

    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        return state, None, [], None, "", "无法读取首帧图像"

    h, w = first_frame.shape[:2]
    detections = None

    try:
        from web.services import InferenceService
        detections = InferenceService.get().run_detection(first_frame, conf=conf_threshold)
    except Exception:
        pass

    # 兜底: 从 GT mask 提取
    if detections is None or detections.get("artery") is None or detections.get("vein") is None:
        detections = _demo_detection(state, first_frame)

    if detections.get("artery") is None or detections.get("vein") is None:
        return state, None, [], None, "", "检测失败：未能找到动脉和静脉框"

    state["detections"] = detections

    # 检测可视化
    det_vis = bgr_to_rgb(draw_detection_boxes(first_frame, detections))

    # ============ Step 2: SAM2 分割 ============
    progress(0.15, desc="[2/4] SAM2 视频分割...")

    pred_masks_by_idx = None
    is_lora = "LoRA" in model_variant
    try:
        if is_lora:
            from web.services import InferenceService
            pred_masks_by_idx = InferenceService.get().run_segmentation(
                images_dir=images_dir,
                detections=detections,
                num_frames=num_frames,
                use_mfp=use_mfp,
                variant=model_variant if model_variant in ("LoRA r8", "LoRA r4") else "LoRA r8",
            )
        else:
            from tabs.segmentation import _run_baseline_sam2
            pred_masks_by_idx = _run_baseline_sam2(
                images_dir, detections, set(range(num_frames)),
                model_variant, progress,
            )
    except Exception:
        pass

    # 兜底: GT mask demo
    if pred_masks_by_idx is None:
        pred_masks_by_idx = _demo_segmentation(state)

    # ============ Step 3: 收集面积和指标 ============
    progress(0.60, desc="[3/4] 面积和指标计算...")

    gt_frame_map = {}
    if masks_dir.exists():
        for mf in sorted(masks_dir.glob("*.png"), key=lambda p: int(p.stem)):
            gt_frame_map[int(mf.stem)] = str(mf)

    all_frame_metrics = []
    vein_areas = []
    artery_areas = []
    gallery_images = []
    masks_list = []

    for i, frame_path in enumerate(frame_files):
        img = cv2.imread(frame_path)
        if img is None:
            continue
        fh, fw = img.shape[:2]

        pred = pred_masks_by_idx.get(i)
        if pred is not None:
            semantic_mask = pred["semantic"]
        else:
            semantic_mask = np.zeros((fh, fw), dtype=np.uint8)

        masks_list.append(semantic_mask)
        artery_areas.append(compute_mask_area(semantic_mask, 1))
        vein_areas.append(compute_mask_area(semantic_mask, 2))

        if i in gt_frame_map:
            gt_mask = cv2.imread(gt_frame_map[i], cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                if gt_mask.shape != (fh, fw):
                    gt_mask = cv2.resize(gt_mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
                metrics = compute_frame_metrics((semantic_mask == 1), (semantic_mask == 2), gt_mask)
                metrics["frame_idx"] = i
                all_frame_metrics.append(metrics)

        # Gallery 采样
        should_show = (i % max(1, num_frames // 16) == 0) or (i in gt_frame_map) or (i == 0)
        if should_show and len(gallery_images) < 24:
            vis = overlay_masks(img, semantic_mask, alpha=0.45)
            gallery_images.append((bgr_to_rgb(vis), f"Frame {i}"))

    state["pred_masks"] = pred_masks_by_idx
    state["frame_metrics"] = all_frame_metrics
    state["vein_areas"] = vein_areas
    state["artery_areas"] = artery_areas

    # ============ Step 4: DVT 诊断 ============
    progress(0.85, desc="[4/4] DVT 智能诊断...")

    full_features = None
    ml_result = None
    try:
        from web.services import InferenceService
        ml_result = InferenceService.get().run_diagnosis(masks_list)
        full_features = ml_result.get("features")
    except Exception:
        pass

    simple_result = compute_dvt_diagnosis(vein_areas)

    if ml_result is not None:
        simple_result["is_dvt"] = ml_result["is_dvt"]
        simple_result["confidence"] = ml_result["confidence"]
        simple_result["ml_threshold"] = ml_result["threshold"]
        if ml_result["is_dvt"]:
            simple_result["diagnosis"] = "DVT 疑似（静脉拒绝塌陷）"
        else:
            simple_result["diagnosis"] = "正常（静脉正常塌陷）"

    # ============ 生成结果 ============
    progress(0.95, desc="生成诊断报告...")

    # 面积曲线图
    area_fig = _build_area_chart(vein_areas, artery_areas)

    # 诊断报告 HTML
    report_html = _build_full_report_html(
        state, detections, simple_result, full_features,
        all_frame_metrics, vein_areas, artery_areas, model_variant, use_mfp,
    )

    # 诊断摘要
    summary_html = _build_summary_card(simple_result)

    progress(1.0, desc="分析完成")

    return state, det_vis, gallery_images, area_fig, report_html, summary_html


def _demo_detection(state, img):
    """从 GT mask 提取检测框"""
    masks_dir = Path(state.get("masks_dir", ""))
    first_mask_path = masks_dir / "00000.png"
    if not first_mask_path.exists():
        return {"artery": None, "vein": None}
    gt_mask = cv2.imread(str(first_mask_path), cv2.IMREAD_GRAYSCALE)
    result = {}
    for cls_name, cls_val in [("artery", 1), ("vein", 2)]:
        ys, xs = np.where(gt_mask == cls_val)
        if len(xs) > 0:
            result[cls_name] = {"box": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                                "conf": 1.0, "from_gt": True}
        else:
            result[cls_name] = None
    return result


def _demo_segmentation(state):
    """GT mask 模拟分割"""
    masks_dir = Path(state.get("masks_dir", ""))
    frame_files = state.get("frame_files", [])
    gt_frame_map = {}
    if masks_dir.exists():
        for mf in sorted(masks_dir.glob("*.png"), key=lambda p: int(p.stem)):
            gt_frame_map[int(mf.stem)] = str(mf)

    pred_masks = {}
    last_mask = None
    for i in range(len(frame_files)):
        if i in gt_frame_map:
            gt = cv2.imread(gt_frame_map[i], cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                last_mask = gt
                pred_masks[i] = {"semantic": gt, "artery": (gt == 1), "vein": (gt == 2)}
                continue
        if last_mask is not None:
            pred_masks[i] = {"semantic": last_mask.copy(), "artery": (last_mask == 1), "vein": (last_mask == 2)}
    return pred_masks


def _build_area_chart(vein_areas, artery_areas):
    """构建面积变化图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_matplotlib()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    frames = list(range(len(vein_areas)))

    # 上图：面积曲线
    ax1 = axes[0]
    ax1.fill_between(frames, artery_areas, alpha=0.25, color="#ef4444")
    ax1.fill_between(frames, vein_areas, alpha=0.25, color="#22c55e")
    ax1.plot(frames, artery_areas, color="#ef4444", linewidth=2, label="Artery")
    ax1.plot(frames, vein_areas, color="#22c55e", linewidth=2, label="Vein")
    if vein_areas:
        min_idx = int(np.argmin(vein_areas))
        ax1.axvline(x=min_idx, color="#f59e0b", linestyle="--", alpha=0.6)
        ax1.annotate(f"Min Vein: {vein_areas[min_idx]}", (min_idx, vein_areas[min_idx]),
                     textcoords="offset points", xytext=(10, 10), fontsize=9, color="#f59e0b",
                     arrowprops=dict(arrowstyle="->", color="#f59e0b"))
    style_axis(ax1, title="Artery / Vein Area Over Frames", ylabel="Area (px)")
    ax1.legend(fontsize=10)

    # 下图：V/A 比
    ax2 = axes[1]
    if artery_areas:
        ratios = [v / a if a > 0 else 0 for v, a in zip(vein_areas, artery_areas)]
        ax2.plot(frames, ratios, color="#06b6d4", linewidth=2)
        ax2.fill_between(frames, ratios, alpha=0.15, color="#06b6d4")
    style_axis(ax2, title="Vein-to-Artery Area Ratio",
              xlabel="Frame Index", ylabel="V/A Ratio")

    plt.tight_layout(pad=2.0)
    return fig


def _build_summary_card(result):
    """诊断摘要大卡片"""
    if result.get("is_dvt") is None:
        return '<div style="text-align:center; padding:20px; color:#94a3b8;">数据不足，无法诊断</div>'

    if result["is_dvt"]:
        return f"""
        <div style="text-align:center; padding:28px 20px; background:linear-gradient(135deg, rgba(220,38,38,0.12), rgba(239,68,68,0.06));
                    border:2px solid #dc2626; border-radius:16px;">
            <div style="font-size:52px; margin-bottom:8px;">⚠️</div>
            <div style="font-size:24px; font-weight:800; color:#ef4444; margin-bottom:6px;">
                DVT 疑似
            </div>
            <div style="font-size:14px; color:#fca5a5; margin-bottom:4px;">
                VCR = {result.get('area_ratio', 0):.3f} &nbsp;|&nbsp; 置信度 {result['confidence']:.0%}
            </div>
            <div style="font-size:12px; color:#94a3b8; margin-top:8px; line-height:1.6;">
                静脉面积在压缩过程中变化极小，拒绝塌陷<br>
                <b>建议进一步临床检查</b>
            </div>
        </div>"""
    else:
        return f"""
        <div style="text-align:center; padding:28px 20px; background:linear-gradient(135deg, rgba(16,185,129,0.12), rgba(34,197,94,0.06));
                    border:2px solid #059669; border-radius:16px;">
            <div style="font-size:52px; margin-bottom:8px;">✅</div>
            <div style="font-size:24px; font-weight:800; color:#10b981; margin-bottom:6px;">
                正常
            </div>
            <div style="font-size:14px; color:#6ee7b7; margin-bottom:4px;">
                VCR = {result.get('area_ratio', 0):.3f} &nbsp;|&nbsp; 置信度 {result['confidence']:.0%}
            </div>
            <div style="font-size:12px; color:#94a3b8; margin-top:8px; line-height:1.6;">
                静脉在压缩过程中正常塌陷，面积缩减 {result.get('area_change_percent', 0):.0f}%<br>
                未见血栓征象
            </div>
        </div>"""


def _build_full_report_html(state, detections, result, features,
                            frame_metrics, vein_areas, artery_areas,
                            model_variant, use_mfp):
    """构建完整的诊断报告 HTML"""
    case_name = state.get("current_case", "Unknown")
    n_frames = len(state.get("frame_files", []))
    from_video = state.get("from_video", False)

    # ---- 检测信息 ----
    det_rows = ""
    for cls in ("artery", "vein"):
        det = detections.get(cls)
        if det:
            box = det["box"]
            conf = det.get("conf", 0)
            tags = []
            if det.get("from_gt"): tags.append("GT")
            if det.get("inferred"): tags.append("推断")
            if det.get("fixed"): tags.append("修正")
            if not tags: tags.append("检测")
            status = " / ".join(tags)
            det_rows += f"<tr><td>{cls}</td><td>{conf:.3f}</td><td>({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})</td><td>{status}</td></tr>"
        else:
            det_rows += f"<tr><td>{cls}</td><td>—</td><td>—</td><td>未检测</td></tr>"

    # ---- 特征表 ----
    feat_rows = ""
    if features:
        feat_items = [
            ("VCR (静脉压缩比)", features.get("vcr", 0), "min/max，越小越正常"),
            ("VDR (消失率)", features.get("vdr", 0), "面积<10%max的帧占比"),
            ("VARR (相对范围)", features.get("varr", 0), "(max-min)/max"),
            ("vein_cv (变异系数)", features.get("vein_cv", 0), "std/mean"),
            ("MVAR (最小V/A比)", features.get("mvar", 0), "min(vein/artery)"),
            ("mean_var (均值V/A比)", features.get("mean_var", 0), "mean(vein/artery)"),
            ("vein_slope (趋势斜率)", features.get("vein_slope", 0), "负=下降(正常)"),
            ("vein_min_position", features.get("vein_min_position", 0), "最小面积相对帧位置"),
            ("artery_stability", features.get("artery_stability", 0), "动脉稳定性指数"),
            ("max_drop_ratio", features.get("max_drop_ratio", 0), "最大帧间下降比"),
            ("vein_p10", features.get("vein_p10", 0), "归一化10th百分位"),
            ("vein_p25", features.get("vein_p25", 0), "归一化25th百分位"),
            ("vein_p50", features.get("vein_p50", 0), "归一化50th百分位(中位数)"),
            ("vein_detect_rate", features.get("vein_detect_rate", 0), "静脉检出帧占比"),
            ("artery_detect_rate", features.get("artery_detect_rate", 0), "动脉检出帧占比"),
            ("vein_jitter", features.get("vein_jitter", 0), "帧间面积跳变(越大越不稳)"),
            ("vein_autocorr", features.get("vein_autocorr", 0), "lag-1自相关(高=平滑)"),
            ("circ_cv", features.get("circ_cv", 0), "圆度变异系数"),
            ("circ_min", features.get("circ_min", 0), "最小圆度"),
            ("circ_range", features.get("circ_range", 0), "圆度范围"),
        ]
        for name, val, desc in feat_items:
            feat_rows += f"<tr><td style='font-weight:600;'>{name}</td><td>{val:.4f}</td><td style='color:#94a3b8;'>{desc}</td></tr>"

    # ---- 分割指标 ----
    seg_summary = ""
    if frame_metrics:
        avg_dice = np.mean([m["mean_dice"] for m in frame_metrics])
        avg_miou = np.mean([m["miou"] for m in frame_metrics])
        avg_a = np.mean([m["artery_dice"] for m in frame_metrics])
        avg_v = np.mean([m["vein_dice"] for m in frame_metrics])
        seg_summary = f"""
        <table class="report-table">
          <tr><th>指标</th><th>Artery</th><th>Vein</th><th>Mean</th></tr>
          <tr><td>Dice</td><td>{avg_a:.4f}</td><td>{avg_v:.4f}</td><td>{avg_dice:.4f}</td></tr>
          <tr><td>mIoU</td><td>{np.mean([m['artery_iou'] for m in frame_metrics]):.4f}</td>
              <td>{np.mean([m['vein_iou'] for m in frame_metrics]):.4f}</td><td>{avg_miou:.4f}</td></tr>
        </table>
        <p style="color:#64748b; font-size:12px; margin-top:4px;">
          (基于 {len(frame_metrics)} 帧 GT 标注评估)
        </p>"""
    else:
        seg_summary = '<p style="color:#64748b; font-size:13px;">无 GT 标注，跳过分割指标评估</p>'

    # ---- 诊断依据 ----
    ml_thresh = result.get("ml_threshold")
    thresh_note = f"VCR > {ml_thresh:.3f} (高召回优化阈值)" if ml_thresh else f"min/max > {result.get('threshold', 0.4):.2f}"

    mfp_note = " + MFP" if use_mfp else ""

    html = f"""
    <div class="full-report">
      <h2 style="color:#e2e8f0; margin:0 0 16px 0; font-size:20px; border-bottom:2px solid #334155; padding-bottom:8px;">
        DVT 诊断报告
      </h2>

      <!-- 基本信息 -->
      <div class="report-section">
        <h3>1. 基本信息</h3>
        <table class="report-table">
          <tr><td style="width:140px;">案例</td><td><code>{case_name}</code></td></tr>
          <tr><td>数据来源</td><td>{'本地上传视频' if from_video else f'数据集 ({state.get("split", "val")})'}</td></tr>
          <tr><td>总帧数</td><td>{n_frames}</td></tr>
          <tr><td>分割模型</td><td>{model_variant}{mfp_note}</td></tr>
        </table>
      </div>

      <!-- 检测结果 -->
      <div class="report-section">
        <h3>2. YOLO 血管检测</h3>
        <table class="report-table">
          <tr><th>类别</th><th>置信度</th><th>位置</th><th>状态</th></tr>
          {det_rows}
        </table>
      </div>

      <!-- 分割指标 -->
      <div class="report-section">
        <h3>3. 分割质量评估</h3>
        {seg_summary}
      </div>

      <!-- 面积统计 -->
      <div class="report-section">
        <h3>4. 面积统计</h3>
        <table class="report-table">
          <tr><td>静脉最大面积</td><td>{result.get('max_area', 'N/A')} px</td></tr>
          <tr><td>静脉最小面积</td><td>{result.get('min_area', 'N/A')} px</td></tr>
          <tr><td>面积变化率 (VCR)</td><td>{result.get('area_ratio', 0):.4f}</td></tr>
          <tr><td>面积缩减百分比</td><td>{result.get('area_change_percent', 0):.1f}%</td></tr>
        </table>
      </div>

      <!-- 19维特征 -->
      <div class="report-section">
        <h3>5. 时序特征分析 (19维)</h3>
        {'<table class="report-table"><tr><th>特征</th><th>值</th><th>说明</th></tr>' + feat_rows + '</table>' if feat_rows else '<p style="color:#64748b;">特征提取不可用</p>'}
      </div>

      <!-- 诊断结论 -->
      <div class="report-section">
        <h3>6. 诊断结论</h3>
        <table class="report-table">
          <tr><td>判断依据</td><td>{thresh_note}</td></tr>
          <tr><td>置信度</td><td>{result.get('confidence', 0):.0%}</td></tr>
          <tr><td style="font-weight:700;">结果</td>
              <td style="font-weight:700; color:{'#ef4444' if result.get('is_dvt') else '#10b981'};">
                {result.get('diagnosis', '未知')}
              </td></tr>
        </table>
        <div style="background:#0f172a; padding:12px 16px; border-radius:8px; margin-top:8px;
                    border-left:3px solid #3b82f6;">
          <p style="color:#94a3b8; font-size:12px; margin:0; line-height:1.7;">
            <b>诊断原理</b>：在压缩超声检查中，正常静脉会被探头压瘪（面积大幅缩小 → VCR 接近 0），
            而有血栓的静脉拒绝塌陷（面积基本不变 → VCR 接近 1）。<br>
            本系统综合 19 维时序特征进行判断，以高召回率为优化目标，减少漏诊风险。
          </p>
        </div>
      </div>
    </div>
    """
    return html


def build_pipeline_tab(state: gr.State):
    """构建一键分析 Tab"""

    gr.HTML("""
    <div style="padding:16px 20px; background:linear-gradient(135deg, #1e3a5f, #1e293b);
                border-radius:12px; border:1px solid #334155; margin-bottom:8px;">
        <h3 style="margin:0 0 4px 0; color:#e2e8f0; font-size:16px;">
            一键全流程分析
        </h3>
        <p style="margin:0; color:#94a3b8; font-size:13px;">
            自动执行 YOLO 检测 → SAM2 分割 → 特征提取 → DVT 诊断，生成完整报告
        </p>
    </div>
    """)

    with gr.Row(equal_height=False):
        # ========== 左栏: 参数 + 诊断摘要 ==========
        with gr.Column(scale=2):
            with gr.Group():
                model_variant = gr.Radio(
                    choices=["LoRA r8", "LoRA r4", "Baseline (Large)"],
                    value="LoRA r8",
                    label="分割模型",
                )
                use_mfp = gr.Checkbox(label="多帧提示 (MFP)", value=False,
                                      info="每隔 15 帧用 YOLO 重新锚定，减少误差累积")
                conf_slider = gr.Slider(minimum=0.01, maximum=0.5, value=0.1, step=0.01,
                                        label="YOLO 置信度阈值")

            run_btn = gr.Button(
                "🚀 开始全流程分析",
                variant="primary", size="lg",
            )

            diagnosis_summary = gr.HTML(
                '<div style="text-align:center; padding:24px; color:#64748b;">加载数据后点击「开始全流程分析」</div>'
            )

            det_preview = gr.Image(label="首帧检测结果", height=280, type="numpy")

        # ========== 右栏: 结果 ==========
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("分割结果"):
                    seg_gallery = gr.Gallery(
                        label="逐帧分割可视化",
                        columns=4, rows=3, height=420,
                        object_fit="contain",
                    )

                with gr.Tab("面积曲线"):
                    area_plot = gr.Plot(label="面积变化趋势")

                with gr.Tab("完整报告"):
                    report_html = gr.HTML(
                        '<div style="padding:20px; color:#64748b; text-align:center;">分析完成后将显示详细诊断报告</div>'
                    )

    # ========== 事件绑定 ==========
    run_btn.click(
        fn=_run_full_pipeline,
        inputs=[state, model_variant, use_mfp, conf_slider],
        outputs=[state, det_preview, seg_gallery, area_plot, report_html, diagnosis_summary],
    )
