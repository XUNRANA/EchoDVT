#!/usr/bin/env python3
"""
SAM2 LoRA 推理评估脚本（YOLO box prompt）

功能：
1) 加载 LoRA 微调后的 SAM2 权重
2) 使用 YOLO 在首帧提供 artery/vein 框（含缺失框补全）
3) 后续帧仅依赖 memory 传播（AM/SM/AV 全部关闭）
4) 仅在有标注帧上评估 Dice / mIoU
5) 输出日志、CSV、summary.json、每帧可视化

用法:
    cd sam2
    python inference_lora.py \
      --lora-weights checkpoints/lora_runs/<run>/lora_best.pt \
      --split val

    # 自定义 LoRA 参数（需和训练时一致）
    python inference_lora.py \
      --lora-weights checkpoints/lora_runs/<run>/lora_best.pt \
      --lora-r 4 \
      --lora-memory-attention
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── 复用 inference_box_prompt_large.py 中的常量和工具类 ──
from inference_box_prompt_large import (
    DETECTION_CLASS_NAMES,
    SEGMENTATION_CLASS_VALUES,
    OBJECT_ID_TO_CLASS,
    CLASS_TO_OBJECT_ID,
    CLASS_COLORS,
    METRIC_KEYS,
    Logger,
    VesselDetector,
    binary_metrics,
    compute_frame_metrics,
    overlay_mask,
    draw_boxes,
    build_visualization,
    write_csv,
    summarize_rows,
    resolve_sam2_device,
    resolve_yolo_device,
    str2bool,
)

# ── LoRA SAM2 构建 ──
from sam2.lora_sam2 import LoRA_SAM2_Video, build_lora_sam2_video


class LoRASAM2VideoSegmenter:
    """
    LoRA 微调后的 SAM2 视频分割器

    与 SAM2MemoryVideoSegmenter 结构一致，
    区别是加载 LoRA 权重，AM/SM/AV 全部关闭。
    """

    def __init__(
        self,
        model_cfg: str,
        checkpoint: Path,
        lora_weights: Path,
        device: str,
        lora_r: int = 4,
        lora_alpha: float = 1.0,
        lora_memory_attention: bool = True,
        lora_memory_encoder: bool = False,
        use_adaptive_memory: bool = False,
        use_separate_memory: bool = False,
        use_av_constraint: bool = False,
        av_prior_stats_path: str = None,
    ) -> None:
        # 构建 LoRA 模型（推理模式, 不用 trainer）
        self.lora_model = build_lora_sam2_video(
            config_file=model_cfg,
            ckpt_path=str(checkpoint),
            r=lora_r,
            lora_alpha=lora_alpha,
            device=device,
            apply_to_image_encoder=True,
            apply_to_memory_attention=lora_memory_attention,
            apply_to_memory_encoder=lora_memory_encoder,
            use_trainer=False,
            use_adaptive_memory=use_adaptive_memory,
            use_separate_memory=use_separate_memory,
            use_av_constraint=use_av_constraint,
            av_prior_stats_path=av_prior_stats_path,
        )

        # 加载 LoRA 权重
        self.lora_model.load_lora_parameters(str(lora_weights))
        self.lora_model.eval()

        # predictor 引用（用于 init_state / add_new_points_or_box / propagate_in_video）
        self.predictor = self.lora_model.predictor
        self.device = device

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if num_labels <= 1:
            return mask
        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest_idx

    @classmethod
    def _postprocess_binary(cls, mask: np.ndarray, mode: str, morph_kernel: int) -> np.ndarray:
        if mode == "none":
            return mask
        out = cls._largest_component(mask)
        if mode == "lcc_morph":
            k = max(1, int(morph_kernel))
            kernel = np.ones((k, k), dtype=np.uint8)
            out_u8 = out.astype(np.uint8)
            out_u8 = cv2.morphologyEx(out_u8, cv2.MORPH_OPEN, kernel)
            out_u8 = cv2.morphologyEx(out_u8, cv2.MORPH_CLOSE, kernel)
            out = cls._largest_component(out_u8.astype(bool))
        return out

    @classmethod
    def _decode_masks_from_logits(
        cls,
        out_obj_ids: List[int],
        out_mask_logits: torch.Tensor,
        height: int,
        width: int,
        artery_thresh: float,
        vein_thresh: float,
        postprocess_mode: str,
        morph_kernel: int,
    ) -> Dict[str, np.ndarray]:
        logits_map = {
            "artery": np.full((height, width), -1e9, dtype=np.float32),
            "vein": np.full((height, width), -1e9, dtype=np.float32),
        }
        for i, out_obj_id in enumerate(out_obj_ids):
            cls_name = OBJECT_ID_TO_CLASS.get(int(out_obj_id))
            if cls_name is None:
                continue
            logits = out_mask_logits[i].detach().float().cpu().numpy().squeeze()
            if logits.shape != (height, width):
                logits = cv2.resize(logits, (width, height), interpolation=cv2.INTER_LINEAR)
            logits_map[cls_name] = logits

        artery_mask = logits_map["artery"] > artery_thresh
        vein_mask = logits_map["vein"] > vein_thresh
        artery_mask = cls._postprocess_binary(artery_mask, postprocess_mode, morph_kernel)
        vein_mask = cls._postprocess_binary(vein_mask, postprocess_mode, morph_kernel)

        semantic = np.zeros((height, width), dtype=np.uint8)
        semantic[artery_mask] = SEGMENTATION_CLASS_VALUES["artery"]
        semantic[vein_mask] = SEGMENTATION_CLASS_VALUES["vein"]

        overlap = artery_mask & vein_mask
        if np.any(overlap):
            oy, ox = np.where(overlap)
            artery_logits = logits_map["artery"][oy, ox]
            vein_logits = logits_map["vein"][oy, ox]
            semantic[oy, ox] = np.where(
                artery_logits >= vein_logits,
                SEGMENTATION_CLASS_VALUES["artery"],
                SEGMENTATION_CLASS_VALUES["vein"],
            ).astype(np.uint8)

        return {"artery": artery_mask, "vein": vein_mask, "semantic": semantic}

    def segment_video_with_first_frame_prompt(
        self,
        images_dir: Path,
        first_frame_boxes: Dict[str, Dict | None],
        target_frame_indices: set[int],
        artery_score_thresh: float = 0.0,
        vein_score_thresh: float = 0.0,
        postprocess_mode: str = "lcc",
        morph_kernel: int = 3,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        amp_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        pred_masks: Dict[int, Dict[str, np.ndarray]] = {}
        with torch.inference_mode():
            with amp_ctx:
                inference_state = self.predictor.init_state(
                    video_path=str(images_dir),
                    async_loading_frames=False,
                )
                video_h = int(inference_state["video_height"])
                video_w = int(inference_state["video_width"])

                artery_box = first_frame_boxes["artery"]["box"]
                vein_box = first_frame_boxes["vein"]["box"]
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=CLASS_TO_OBJECT_ID["artery"],
                    box=np.asarray(artery_box, dtype=np.float32),
                )
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=CLASS_TO_OBJECT_ID["vein"],
                    box=np.asarray(vein_box, dtype=np.float32),
                )

                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=0,
                ):
                    if out_frame_idx not in target_frame_indices:
                        continue
                    pred_masks[out_frame_idx] = self._decode_masks_from_logits(
                        out_obj_ids=out_obj_ids,
                        out_mask_logits=out_mask_logits,
                        height=video_h,
                        width=video_w,
                        artery_thresh=artery_score_thresh,
                        vein_thresh=vein_score_thresh,
                        postprocess_mode=postprocess_mode,
                        morph_kernel=morph_kernel,
                    )

                self.predictor.reset_state(inference_state)
        return pred_masks


# ============================================================
# Main evaluation loop (与 inference_box_prompt_large.py 结构一致)
# ============================================================

def run(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    data_root = Path(args.data_root).resolve()
    split_dir = data_root / args.split
    yolo_model_path = Path(args.yolo_model).resolve()
    yolo_prior_path = Path(args.yolo_prior).resolve()
    sam2_ckpt = Path(args.sam2_checkpoint).resolve()
    lora_weights = Path(args.lora_weights).resolve()

    if not split_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在: {split_dir}")
    if not yolo_model_path.exists():
        raise FileNotFoundError(f"YOLO 权重不存在: {yolo_model_path}")
    if not sam2_ckpt.exists():
        raise FileNotFoundError(f"SAM2 checkpoint 不存在: {sam2_ckpt}")
    if not lora_weights.exists():
        raise FileNotFoundError(f"LoRA 权重不存在: {lora_weights}")

    output_root = Path(args.output_root).resolve()
    run_name = (
        f"{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_lora_r{args.lora_r}"
        f"_am{int(args.use_adaptive_memory)}"
        f"_sm{int(args.use_separate_memory)}"
        f"_av{int(args.use_av_constraint)}"
    )
    run_dir = output_root / run_name
    vis_root = run_dir / "visualizations"
    run_dir.mkdir(parents=True, exist_ok=True)
    vis_root.mkdir(parents=True, exist_ok=True)

    logger = Logger(run_dir / "inference_eval.log")
    sam2_device = resolve_sam2_device(args.device)
    yolo_device = resolve_yolo_device(args.yolo_device, sam2_device)
    artery_thresh = args.mask_score_thresh if args.artery_mask_thresh is None else args.artery_mask_thresh
    vein_thresh = args.mask_score_thresh if args.vein_mask_thresh is None else args.vein_mask_thresh

    logger.log("=" * 80)
    logger.log("SAM2 LoRA + YOLO首帧box提示 + 记忆传播评估")
    logger.log(f"Split: {args.split}")
    logger.log(f"Data root: {data_root}")
    logger.log(f"YOLO model: {yolo_model_path}")
    logger.log(f"SAM2 config: {args.sam2_config}")
    logger.log(f"SAM2 ckpt: {sam2_ckpt}")
    logger.log(f"LoRA weights: {lora_weights}")
    logger.log(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    logger.log(f"LoRA memory_attention: {args.lora_memory_attention}")
    logger.log(f"LoRA memory_encoder: {args.lora_memory_encoder}")
    logger.log(
        f"AM={args.use_adaptive_memory}, SM={args.use_separate_memory}, AV={args.use_av_constraint}"
    )
    logger.log(
        f"Mask thresh: artery={artery_thresh:.3f}, vein={vein_thresh:.3f}, "
        f"postprocess={args.postprocess_mode}"
    )
    logger.log(f"SAM2 device: {sam2_device} | YOLO device: {yolo_device}")
    logger.log(f"Output dir: {run_dir}")
    logger.log("=" * 80)

    # ── 构建 YOLO 检测器 ──
    detector = VesselDetector(
        model_path=yolo_model_path,
        yolo_device=yolo_device,
        prior_path=yolo_prior_path,
        retry_conf=args.retry_conf,
        prompt_min_conf=args.prompt_min_conf,
        prompt_min_area_ratio=args.prompt_min_area_ratio,
        prompt_max_area_ratio=args.prompt_max_area_ratio,
        prompt_max_aspect_ratio=args.prompt_max_aspect_ratio,
    )

    # ── 构建 LoRA SAM2 分割器 ──
    segmenter = LoRASAM2VideoSegmenter(
        model_cfg=args.sam2_config,
        checkpoint=sam2_ckpt,
        lora_weights=lora_weights,
        device=sam2_device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_memory_attention=args.lora_memory_attention,
        lora_memory_encoder=args.lora_memory_encoder,
        use_adaptive_memory=args.use_adaptive_memory,
        use_separate_memory=args.use_separate_memory,
        use_av_constraint=args.use_av_constraint,
        av_prior_stats_path=str(yolo_prior_path),
    )

    # ── 遍历 case ──
    case_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    if args.max_cases > 0:
        case_dirs = case_dirs[: args.max_cases]

    frame_rows: List[Dict] = []
    case_rows: List[Dict] = []
    skipped_frames = 0
    processed_frames = 0

    for case_dir in case_dirs:
        images_dir = case_dir / "images"
        masks_dir = case_dir / "masks"
        if not images_dir.exists() or not masks_dir.exists():
            logger.log(f"[Skip case] {case_dir.name}: 缺少 images 或 masks 目录")
            continue

        image_files = sorted(images_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        if not image_files:
            logger.log(f"[Skip case] {case_dir.name}: 无图像帧")
            continue
        frame_stems = [p.stem for p in image_files]
        stem_to_frame_idx = {stem: idx for idx, stem in enumerate(frame_stems)}

        gt_masks = sorted(masks_dir.glob("*.png"), key=lambda p: int(p.stem))
        if not gt_masks:
            logger.log(f"[Skip case] {case_dir.name}: 无标准掩码")
            continue

        # YOLO 首帧检测
        prompt_frame_idx = 0
        prompt_stem = frame_stems[prompt_frame_idx]
        prompt_image = cv2.imread(str(image_files[prompt_frame_idx]), cv2.IMREAD_COLOR)
        if prompt_image is None:
            logger.log(f"[Skip case] {case_dir.name}: 首帧读取失败")
            continue
        first_frame_boxes = detector.predict(prompt_image, conf=args.yolo_conf)
        logger.log(
            f"[Case Prompt] {case_dir.name}: first_frame={prompt_stem}, "
            f"artery_conf={first_frame_boxes['artery']['conf']:.3f}, "
            f"vein_conf={first_frame_boxes['vein']['conf']:.3f}"
        )

        # 收集需要评估的帧
        eval_targets: Dict[int, Path] = {}
        for mask_path in gt_masks:
            stem = mask_path.stem
            frame_idx = stem_to_frame_idx.get(stem)
            if frame_idx is None:
                skipped_frames += 1
                continue
            eval_targets[frame_idx] = mask_path

        if not eval_targets:
            logger.log(f"[Skip case] {case_dir.name}: 无可评估帧")
            continue

        # SAM2 LoRA 分割
        pred_masks_by_idx = segmenter.segment_video_with_first_frame_prompt(
            images_dir=images_dir,
            first_frame_boxes=first_frame_boxes,
            target_frame_indices=set(eval_targets.keys()),
            artery_score_thresh=artery_thresh,
            vein_score_thresh=vein_thresh,
            postprocess_mode=args.postprocess_mode,
            morph_kernel=args.morph_kernel,
        )

        # 评估 + 可视化
        case_metrics: List[Dict[str, float]] = []
        case_vis_dir = vis_root / case_dir.name
        case_vis_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in sorted(eval_targets):
            mask_path = eval_targets[frame_idx]
            stem = frame_stems[frame_idx]
            image_path = images_dir / f"{stem}.jpg"

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or gt_mask is None:
                skipped_frames += 1
                continue

            h, w = image.shape[:2]
            if gt_mask.shape != (h, w):
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            frame_pred = pred_masks_by_idx.get(frame_idx)
            if frame_pred is None:
                skipped_frames += 1
                continue
            if frame_pred["semantic"].shape != (h, w):
                frame_pred = {
                    "artery": cv2.resize(
                        frame_pred["artery"].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                    ).astype(bool),
                    "vein": cv2.resize(
                        frame_pred["vein"].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                    ).astype(bool),
                    "semantic": cv2.resize(frame_pred["semantic"], (w, h), interpolation=cv2.INTER_NEAREST),
                }
            pred_mask = frame_pred["semantic"]
            pred_by_class = {"artery": frame_pred["artery"], "vein": frame_pred["vein"]}

            metrics = compute_frame_metrics(pred_by_class, gt_mask)

            frame_row = {"case": case_dir.name, "frame": stem, **metrics}
            frame_rows.append(frame_row)
            case_metrics.append(metrics)
            processed_frames += 1

            # 可视化
            vis_boxes = first_frame_boxes if frame_idx == prompt_frame_idx else {"artery": None, "vein": None}
            vis = build_visualization(
                image=image,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                boxes=vis_boxes,
                metrics=metrics,
                is_prompt_frame=(frame_idx == prompt_frame_idx),
            )
            cv2.imwrite(str(case_vis_dir / f"{stem}_viz.jpg"), vis)

            logger.log(
                f"[{case_dir.name}/{stem}] "
                f"Dice={metrics['mean_dice']:.4f}, mIoU={metrics['miou']:.4f}, "
                f"A_dice/mIoU={metrics['artery_dice']:.4f}/{metrics['artery_miou']:.4f}, "
                f"V_dice/mIoU={metrics['vein_dice']:.4f}/{metrics['vein_miou']:.4f}"
            )

        if case_metrics:
            case_summary = summarize_rows(case_metrics)
            case_row = {"case": case_dir.name, "n_frames": len(case_metrics), **case_summary}
            case_rows.append(case_row)
            logger.log(
                f"[Case Summary] {case_dir.name}: n={len(case_metrics)}, "
                f"Dice={case_summary['mean_dice']:.4f}, mIoU={case_summary['miou']:.4f}, "
                f"A={case_summary['artery_dice']:.4f}/{case_summary['artery_miou']:.4f}, "
                f"V={case_summary['vein_dice']:.4f}/{case_summary['vein_miou']:.4f}"
            )

    # ── 汇总 ──
    if not frame_rows:
        logger.close()
        raise RuntimeError("没有可评估帧。")

    global_summary = summarize_rows(frame_rows)
    case_level_metrics = [{k: row[k] for k in METRIC_KEYS + ["artery_iou", "vein_iou"]} for row in case_rows]
    global_case_summary = summarize_rows(case_level_metrics)

    logger.log("-" * 80)
    logger.log(f"Processed frames: {processed_frames}, Skipped frames: {skipped_frames}")
    logger.log(
        f"[Global-FrameWeighted] Dice={global_summary['mean_dice']:.4f}, "
        f"mIoU={global_summary['miou']:.4f}, "
        f"A Dice/mIoU={global_summary['artery_dice']:.4f}/{global_summary['artery_miou']:.4f}, "
        f"V Dice/mIoU={global_summary['vein_dice']:.4f}/{global_summary['vein_miou']:.4f}"
    )
    logger.log(
        f"[Global-CaseWeighted] Dice={global_case_summary['mean_dice']:.4f}, "
        f"mIoU={global_case_summary['miou']:.4f}, "
        f"A Dice/mIoU={global_case_summary['artery_dice']:.4f}/{global_case_summary['artery_miou']:.4f}, "
        f"V Dice/mIoU={global_case_summary['vein_dice']:.4f}/{global_case_summary['vein_miou']:.4f}"
    )
    logger.log("-" * 80)

    # CSV
    frame_csv_fields = ["case", "frame", "artery_dice", "artery_miou", "artery_iou",
                        "vein_dice", "vein_miou", "vein_iou", "mean_dice", "miou"]
    case_csv_fields = ["case", "n_frames", "artery_dice", "artery_miou", "artery_iou",
                       "vein_dice", "vein_miou", "vein_iou", "mean_dice", "miou"]
    write_csv(run_dir / "frame_metrics.csv", frame_rows, frame_csv_fields)
    write_csv(run_dir / "case_metrics.csv", case_rows, case_csv_fields)

    # Summary JSON
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "processed_frames": processed_frames,
                "skipped_frames": skipped_frames,
                "global_metrics": global_summary,
                "global_frame_weighted_metrics": global_summary,
                "global_case_weighted_metrics": global_case_summary,
                "split": args.split,
                "sam2_config": args.sam2_config,
                "sam2_checkpoint": str(sam2_ckpt),
                "lora_weights": str(lora_weights),
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_memory_attention": args.lora_memory_attention,
                "lora_memory_encoder": args.lora_memory_encoder,
                "yolo_model": str(yolo_model_path),
                "use_adaptive_memory": args.use_adaptive_memory,
                "use_separate_memory": args.use_separate_memory,
                "use_av_constraint": args.use_av_constraint,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.log(f"结果已保存到: {run_dir}")
    logger.close()


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_yolo_model = (
        repo_root / "yolo" / "runs" / "detect" / "runs" / "detect"
        / "dvt_runs" / "aug_step5_speckle_translate_scale" / "weights" / "best.pt"
    )
    default_yolo_prior = repo_root / "yolo" / "prior_stats.json"
    default_data_root = script_dir / "dataset"
    default_output_root = script_dir / "predictions" / "sam2_lora_yolo_box"
    default_sam2_ckpt = script_dir / "checkpoints" / "sam2_hiera_large.pt"

    parser = argparse.ArgumentParser(description="SAM2 LoRA + YOLO首帧框 推理评估脚本")

    # Data
    parser.add_argument("--data-root", type=str, default=str(default_data_root))
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output-root", type=str, default=str(default_output_root))
    parser.add_argument("--max-cases", type=int, default=0, help="0=全部")

    # YOLO
    parser.add_argument("--yolo-model", type=str, default=str(default_yolo_model))
    parser.add_argument("--yolo-prior", type=str, default=str(default_yolo_prior))
    parser.add_argument("--yolo-conf", type=float, default=0.1)
    parser.add_argument("--retry-conf", type=float, default=0.01)
    parser.add_argument("--prompt-min-conf", type=float, default=0.05)
    parser.add_argument("--prompt-min-area-ratio", type=float, default=0.0005)
    parser.add_argument("--prompt-max-area-ratio", type=float, default=0.6)
    parser.add_argument("--prompt-max-aspect-ratio", type=float, default=8.0)
    parser.add_argument("--yolo-device", type=str, default="auto")

    # SAM2
    parser.add_argument("--sam2-config", type=str, default="configs/sam2/sam2_hiera_l.yaml")
    parser.add_argument("--sam2-checkpoint", type=str, default=str(default_sam2_ckpt))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mask-score-thresh", type=float, default=0.0)
    parser.add_argument("--artery-mask-thresh", type=float, default=None)
    parser.add_argument("--vein-mask-thresh", type=float, default=None)
    parser.add_argument("--postprocess-mode", type=str, choices=["none", "lcc", "lcc_morph"], default="lcc")
    parser.add_argument("--morph-kernel", type=int, default=3)

    # LoRA (需和训练时一致!)
    parser.add_argument("--lora-weights", type=str, required=True, help="LoRA 权重路径 (.pt)")
    parser.add_argument("--lora-r", type=int, default=4, help="LoRA rank (需和训练一致)")
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--lora-memory-attention", type=str2bool, default=True,
                        help="是否对 memory_attention 使用 LoRA (需和训练一致)")
    parser.add_argument("--lora-memory-encoder", type=str2bool, default=False,
                        help="是否解冻 memory_encoder (需和训练一致)")

    # AM/SM/AV train-free modules
    parser.add_argument("--use-adaptive-memory", type=str2bool, default=False,
                        help="启用自适应记忆筛选 (软权重)")
    parser.add_argument("--use-separate-memory", type=str2bool, default=False,
                        help="启用分离式记忆库 (跨物体记忆)")
    parser.add_argument("--use-av-constraint", type=str2bool, default=False,
                        help="启用动静脉位置约束")

    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
