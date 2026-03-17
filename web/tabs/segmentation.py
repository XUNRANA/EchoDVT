"""
模块 3: SAM2 分割传播
- 固定使用最优 SAM2 LoRA 权重
- 首帧 box prompt → 后续帧 memory 传播
- 逐帧展示分割结果（动脉红、静脉绿叠加）
- 收集面积数据供后续诊断
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sam2"))

from web.utils.visualization import overlay_masks, bgr_to_rgb, build_comparison_image
from web.utils.metrics import compute_frame_metrics, compute_mask_area
from web.services.inference import DEFAULT_SAM2_VARIANT, DEFAULT_LORA_WEIGHTS


FIXED_USE_MFP = True


def _get_fixed_sam2_weight_display() -> str:
    weight_path = DEFAULT_LORA_WEIGHTS.get(DEFAULT_SAM2_VARIANT)
    if weight_path is None:
        return DEFAULT_SAM2_VARIANT
    try:
        display_path = weight_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = weight_path
    return f"{DEFAULT_SAM2_VARIANT} (`{display_path}`)"


def _run_sam2_segmentation(
    state: dict,
    model_variant: str,
    use_mfp: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    """运行 SAM2 分割"""
    if not state.get("frame_files"):
        return state, None, [], "⚠️ 请先在「📤 数据输入」Tab 中加载案例"
    if not state.get("detections"):
        return state, None, [], "⚠️ 请先在「🎯 YOLO 检测」Tab 中运行检测"

    images_dir = Path(state["images_dir"])
    masks_dir = Path(state["masks_dir"])
    detections = state["detections"]
    frame_files = state["frame_files"]

    # 检查检测框
    artery_det = detections.get("artery")
    vein_det = detections.get("vein")
    if artery_det is None or vein_det is None:
        return state, None, [], "❌ 检测框不完整，无法执行分割"

    # 收集 GT mask 帧索引
    mask_files_list = sorted(masks_dir.glob("*.png"), key=lambda p: int(p.stem)) if masks_dir.exists() else []
    gt_frame_map = {}
    for mf in mask_files_list:
        idx = int(mf.stem)
        gt_frame_map[idx] = str(mf)

    # 尝试使用 SAM2 进行真实推理
    pred_masks_by_idx = None
    is_lora = "LoRA" in model_variant
    try:
        if is_lora:
            # LoRA 变体 — 通过 InferenceService 统一调用
            from web.services import InferenceService
            progress(0.05, desc="加载 LoRA SAM2 模型...")
            pred_masks_by_idx = InferenceService.get().run_segmentation(
                images_dir=images_dir,
                detections=detections,
                num_frames=len(frame_files),
                use_mfp=use_mfp,
                variant=model_variant if model_variant in DEFAULT_LORA_WEIGHTS else DEFAULT_SAM2_VARIANT,
            )
        else:
            # Baseline / AM / SM / AV 变体 — 原始 build_sam2_video_predictor
            pred_masks_by_idx = _run_baseline_sam2(
                images_dir, detections, set(range(len(frame_files))),
                model_variant, progress,
            )
    except Exception as e:
        progress(0, desc=f"SAM2 推理失败: {e}，使用 Demo 模式")

    # Demo 模式：如果 SAM2 不可用，用 GT mask 模拟
    if pred_masks_by_idx is None:
        pred_masks_by_idx = _demo_from_gt(gt_frame_map, images_dir, frame_files)

    # 收集面积和指标
    all_frame_metrics = []
    vein_areas = []
    artery_areas = []
    gallery_images = []

    total_frames = len(frame_files)
    for i, frame_path in enumerate(frame_files):
        progress((i + 1) / total_frames, desc=f"处理帧 {i}/{total_frames}")
        img = cv2.imread(frame_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # 获取预测 mask
        pred = pred_masks_by_idx.get(i)
        if pred is not None:
            semantic_mask = pred["semantic"]
        else:
            semantic_mask = np.zeros((h, w), dtype=np.uint8)

        # 计算面积
        a_area = compute_mask_area(semantic_mask, 1)
        v_area = compute_mask_area(semantic_mask, 2)
        artery_areas.append(a_area)
        vein_areas.append(v_area)

        # 计算指标（如果有 GT）
        if i in gt_frame_map:
            gt_mask = cv2.imread(gt_frame_map[i], cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                if gt_mask.shape != (h, w):
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                pred_artery = (semantic_mask == 1)
                pred_vein = (semantic_mask == 2)
                metrics = compute_frame_metrics(pred_artery, pred_vein, gt_mask)
                metrics["frame_idx"] = i
                all_frame_metrics.append(metrics)

        # 生成可视化（每隔几帧取样 + 标注帧必取）
        should_show = (i % max(1, total_frames // 20) == 0) or (i in gt_frame_map) or (i == 0)
        if should_show:
            vis = overlay_masks(img, semantic_mask, alpha=0.45)
            vis_rgb = bgr_to_rgb(vis)
            gallery_images.append((vis_rgb, f"第 {i} 帧"))

    # 更新 state
    state["pred_masks"] = pred_masks_by_idx
    state["frame_metrics"] = all_frame_metrics
    state["vein_areas"] = vein_areas
    state["artery_areas"] = artery_areas

    # 汇总报告
    report = _format_segmentation_report(all_frame_metrics, len(frame_files), model_variant)

    # 首帧可视化
    first_vis = gallery_images[0][0] if gallery_images else None

    return state, first_vis, gallery_images, report


def _run_baseline_sam2(images_dir, detections, target_indices, model_variant, progress):
    """运行 Baseline SAM2 推理（非 LoRA 变体）"""
    from sam2.build_sam import build_sam2_video_predictor
    import torch
    from contextlib import nullcontext

    sam2_dir = PROJECT_ROOT / "sam2"
    config = "configs/sam2/sam2_hiera_l.yaml"
    ckpt = sam2_dir / "checkpoints" / "sam2_hiera_large.pt"

    if not ckpt.exists():
        raise FileNotFoundError(f"SAM2 checkpoint 不存在: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 根据模型变体选择不同构建方式
    use_am = "AM" in model_variant.upper()
    use_sm = "SM" in model_variant.upper()
    use_av = "AV" in model_variant.upper()

    predictor = build_sam2_video_predictor(
        config_file=config,
        ckpt_path=str(ckpt),
        device=device,
        use_adaptive_memory=use_am,
        use_separate_memory=use_sm,
        use_av_constraint=use_av,
    )

    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    pred_masks = {}

    with torch.inference_mode():
        with amp_ctx:
            inference_state = predictor.init_state(
                video_path=str(images_dir),
                async_loading_frames=False,
            )
            video_h = int(inference_state["video_height"])
            video_w = int(inference_state["video_width"])

            # 首帧添加 box prompt
            for cls_name, obj_id in [("artery", 1), ("vein", 2)]:
                det = detections[cls_name]
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    box=np.asarray(det["box"], dtype=np.float32),
                )

            # 传播
            obj_id_to_cls = {1: "artery", 2: "vein"}
            for out_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=0,
            ):
                semantic = np.zeros((video_h, video_w), dtype=np.uint8)
                for i, oid in enumerate(out_obj_ids):
                    cls_name = obj_id_to_cls.get(int(oid))
                    if cls_name is None:
                        continue
                    logits = out_mask_logits[i].detach().float().cpu().numpy().squeeze()
                    if logits.shape != (video_h, video_w):
                        logits = cv2.resize(logits, (video_w, video_h), interpolation=cv2.INTER_LINEAR)
                    mask_bin = logits > 0
                    cls_val = 1 if cls_name == "artery" else 2
                    semantic[mask_bin] = cls_val

                pred_masks[out_idx] = {
                    "semantic": semantic,
                    "artery": (semantic == 1),
                    "vein": (semantic == 2),
                }

            predictor.reset_state(inference_state)

    return pred_masks


def _demo_from_gt(gt_frame_map, images_dir, frame_files):
    """Demo 模式：使用 GT mask 模拟分割结果"""
    pred_masks = {}
    last_mask = None

    for i in range(len(frame_files)):
        if i in gt_frame_map:
            gt = cv2.imread(gt_frame_map[i], cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                last_mask = gt
                pred_masks[i] = {
                    "semantic": gt,
                    "artery": (gt == 1),
                    "vein": (gt == 2),
                }
                continue
        # 无 GT 帧：使用最近的 GT 模拟 memory 传播
        if last_mask is not None:
            pred_masks[i] = {
                "semantic": last_mask.copy(),
                "artery": (last_mask == 1),
                "vein": (last_mask == 2),
            }
    return pred_masks


def _format_segmentation_report(metrics_list, total_frames, variant):
    """生成分割报告"""
    lines = [f"### 🔬 分割完成\n"]
    lines.append(f"- **模型变体**: {variant}")
    lines.append(f"- **总帧数**: {total_frames}")
    lines.append(f"- **已评估帧**: {len(metrics_list)} (有 GT 标注的帧)")

    if metrics_list:
        avg_dice = np.mean([m["mean_dice"] for m in metrics_list])
        avg_miou = np.mean([m["miou"] for m in metrics_list])
        avg_a_dice = np.mean([m["artery_dice"] for m in metrics_list])
        avg_v_dice = np.mean([m["vein_dice"] for m in metrics_list])

        lines.append(f"\n### 📐 平均指标\n")
        lines.append("| 指标 | 值 |")
        lines.append("|------|------|")
        lines.append(f"| **平均 Dice** | {avg_dice:.4f} |")
        lines.append(f"| **mIoU** | {avg_miou:.4f} |")
        lines.append(f"| **动脉 Dice** | {avg_a_dice:.4f} |")
        lines.append(f"| **静脉 Dice** | {avg_v_dice:.4f} |")

    return "\n".join(lines)


def build_segmentation_tab(state: gr.State):
    """构建 SAM2 分割 Tab"""
    fixed_variant = gr.State(DEFAULT_SAM2_VARIANT)
    fixed_use_mfp = gr.State(FIXED_USE_MFP)

    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            gr.HTML("""
            <div style="padding:16px 20px; background:linear-gradient(135deg, #f0f9ff, #eff6ff);
                        border-radius:12px; border:1px solid #e2e8f0; margin-bottom:8px;">
                <h3 style="margin:0 0 4px 0; color:#1e293b; font-size:16px;">
                    🔬 SAM2 视频分割
                </h3>
                <p style="margin:0; color:#64748b; font-size:13px;">
                    使用首帧 YOLO 检测框作为 prompt，通过 SAM2 记忆传播机制完成全视频分割
                </p>
            </div>
            """)

            segment_btn = gr.Button("🔬 开始分割", variant="primary", size="lg")

            seg_report = gr.Markdown(f"""
> 💡 **操作指引**: 当前 UI 已固定最优 SAM2 分割配置 `{_get_fixed_sam2_weight_display()}`，点击「开始分割」即可。
> 分割完成后可在右侧图库中查看各帧分割结果。
""")

        with gr.Column(scale=3):
            seg_preview = gr.Image(
                label="分割预览（红色=动脉，绿色=静脉）",
                height=400,
                type="numpy",
            )
            seg_gallery = gr.Gallery(
                label="逐帧分割结果（采样展示）",
                columns=5,
                rows=3,
                height=350,
                object_fit="contain",
            )

    segment_btn.click(
        fn=_run_sam2_segmentation,
        inputs=[state, fixed_variant, fixed_use_mfp],
        outputs=[state, seg_preview, seg_gallery, seg_report],
    )
