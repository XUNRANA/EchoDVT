#!/usr/bin/env python3
"""
SAM2 LoRA 微调训练脚本 — DVT 超声视频血管分割

用法:
    # 基础训练 (单 GPU)
    cd sam2
    python train_lora.py

    # 指定参数
    python train_lora.py --lr 5e-4 --epochs 30 --lora-r 4 --max-frames 40 --gpu 0

    # 继续训练
    python train_lora.py --resume checkpoints/lora_best.pt

训练策略:
    1. 冻结 SAM2 绝大部分参数
    2. LoRA 注入 Image Encoder QKV + Memory Attention
    3. 全量微调 Mask Decoder (参数量小)
    4. 可选: 解冻 Memory Encoder
    5. 每个 case: 首帧用 GT box prompt → 后续帧靠 memory 传播 → 在有标注帧上计算 loss
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── 项目导入 ──
from sam2.dvt_dataset import DVTVideoDataset, CLASS_TO_OBJECT_ID, SEGMENTATION_CLASS_VALUES
from sam2.lora_sam2 import LoRA_SAM2_Video, build_lora_sam2_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ============================================================
# Loss Functions
# ============================================================

def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Dice Loss for binary segmentation.
    pred_logits: [B, 1, H, W] (raw logits)
    target:      [B, 1, H, W] (binary 0/1)
    """
    pred = torch.sigmoid(pred_logits)
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def sigmoid_focal_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid Focal Loss."""
    prob = torch.sigmoid(pred_logits)
    ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * ((1 - p_t) ** gamma) * ce
    return loss.mean()


def combined_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice + Focal Loss 组合"""
    return dice_loss(pred_logits, target) + sigmoid_focal_loss(pred_logits, target)


# ============================================================
# Metric computation
# ============================================================

@torch.no_grad()
def compute_dice(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    total = pred_flat.sum() + target_flat.sum()
    if total < 1e-6:
        return 1.0 if intersection < 1e-6 else 0.0
    return (2.0 * intersection / (total + 1e-6)).item()


# ============================================================
# Single-case training step
# ============================================================

def train_one_case(
    lora_model: LoRA_SAM2_Video,
    case_data: Dict,
    device: torch.device,
    amp_ctx,
    vein_weight: float = 1.5,
    artery_weight: float = 1.0,
    max_propagate_frames: int = 0,
) -> Dict:
    """
    对单个 case 执行一次前向 + 损失计算。

    流程:
      1. init_state_train (加载视频帧)
      2. add_new_points_or_box_train (首帧添加 artery/vein box prompt)
      3. propagate_in_video_train (逐帧传播, 在有 GT 的帧上计算 loss)

    Returns:
        {"loss": Tensor, "metrics": {...}}
    """
    images_dir = case_data["images_dir"]
    artery_box = case_data["artery_box"]
    vein_box = case_data["vein_box"]
    gt_masks = case_data["gt_masks"]
    prompt_frame_idx = case_data["prompt_frame_idx"]

    # 将 gt_masks 的 tensor 移到 device
    gt_masks_device = {}
    for fidx, gt in gt_masks.items():
        gt_masks_device[fidx] = gt.to(device)

    # Step 1: 初始化视频状态
    with amp_ctx:
        inference_state = lora_model.init_state_train(
            video_path=images_dir,
            async_loading_frames=False,
        )

        # Step 2: 首帧添加 box prompt
        for cls_name, box in [("artery", artery_box), ("vein", vein_box)]:
            obj_id = CLASS_TO_OBJECT_ID[cls_name]
            lora_model.add_new_points_or_box_train(
                inference_state=inference_state,
                frame_idx=prompt_frame_idx,
                obj_id=obj_id,
                box=box,
            )

        # Step 3: 视频传播并计算损失
        total_loss = torch.zeros((), device=device, requires_grad=True)
        n_loss_frames = 0
        artery_dices = []
        vein_dices = []

        for frame_idx, frame_loss, pred_masks in lora_model.propagate_in_video_train(
            inference_state=inference_state,
            gt_masks_dict=gt_masks_device,
            loss_fn=combined_loss,
            start_frame_idx=prompt_frame_idx,
            max_frame_num_to_track=max_propagate_frames if max_propagate_frames > 0 else None,
            detach_memory=True,
            vein_weight=vein_weight,
            artery_weight=artery_weight,
        ):
            if frame_loss is not None:
                total_loss = total_loss + frame_loss
                n_loss_frames += 1

                # 计算 Dice metric (不影响梯度)
                if frame_idx in gt_masks_device and pred_masks is not None:
                    gt = gt_masks_device[frame_idx]
                    for obj_idx, obj_id in enumerate(inference_state["obj_ids"]):
                        if obj_idx < pred_masks.shape[0]:
                            pred_single = pred_masks[obj_idx:obj_idx+1].unsqueeze(0)
                            gt_single = (gt == obj_id).float().unsqueeze(0).unsqueeze(0)
                            h_p, w_p = pred_single.shape[-2:]
                            gt_resized = F.interpolate(gt_single, size=(h_p, w_p), mode="nearest")
                            d = compute_dice(pred_single, gt_resized)
                            if obj_id == 1:
                                artery_dices.append(d)
                            elif obj_id == 2:
                                vein_dices.append(d)

    # 平均损失
    if n_loss_frames > 0:
        avg_loss = total_loss / n_loss_frames
    else:
        avg_loss = total_loss

    # 重置状态释放显存
    lora_model.reset_state(inference_state)

    return {
        "loss": avg_loss,
        "n_loss_frames": n_loss_frames,
        "artery_dice": float(np.mean(artery_dices)) if artery_dices else 0.0,
        "vein_dice": float(np.mean(vein_dices)) if vein_dices else 0.0,
        "mean_dice": float(np.mean(artery_dices + vein_dices)) if (artery_dices or vein_dices) else 0.0,
    }


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    lora_model: LoRA_SAM2_Video,
    val_dataset: DVTVideoDataset,
    device: torch.device,
    max_cases: int = 0,
) -> Dict:
    """在验证集上评估"""
    lora_model.eval()
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu")

    all_artery_dices = []
    all_vein_dices = []
    n_cases = len(val_dataset) if max_cases <= 0 else min(max_cases, len(val_dataset))

    for i in range(n_cases):
        case_data = val_dataset[i]

        try:
            images_dir = case_data["images_dir"]
            artery_box = case_data["artery_box"]
            vein_box = case_data["vein_box"]
            gt_masks = case_data["gt_masks"]
            prompt_frame_idx = case_data["prompt_frame_idx"]

            with amp_ctx:
                # 使用推理模式 (非训练)
                inference_state = lora_model.init_state(
                    video_path=images_dir,
                    async_loading_frames=False,
                )

                for cls_name, box in [("artery", artery_box), ("vein", vein_box)]:
                    obj_id = CLASS_TO_OBJECT_ID[cls_name]
                    lora_model.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=prompt_frame_idx,
                        obj_id=obj_id,
                        box=np.asarray(box, dtype=np.float32),
                    )

                gt_frame_indices = set(gt_masks.keys())

                for frame_idx, obj_ids, video_res_masks in lora_model.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=prompt_frame_idx,
                ):
                    if frame_idx not in gt_frame_indices:
                        continue
                    gt = gt_masks[frame_idx].numpy()
                    for obj_idx, obj_id in enumerate(obj_ids):
                        if obj_idx >= video_res_masks.shape[0]:
                            continue
                        pred_np = (video_res_masks[obj_idx, 0].cpu().numpy() > 0).astype(np.uint8)
                        gt_np = (gt == obj_id).astype(np.uint8)

                        # Resize if needed
                        if pred_np.shape != gt_np.shape:
                            pred_np = cv2.resize(pred_np, (gt_np.shape[1], gt_np.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)

                        inter = (pred_np & gt_np).sum()
                        total = pred_np.sum() + gt_np.sum()
                        dice = (2.0 * inter) / (total + 1e-6) if total > 0 else 1.0
                        if obj_id == 1:
                            all_artery_dices.append(dice)
                        elif obj_id == 2:
                            all_vein_dices.append(dice)

                lora_model.reset_state(inference_state)

        except Exception as e:
            log.warning(f"Val case {case_data['case_name']} failed: {e}")
            continue

    lora_model.train()
    return {
        "artery_dice": float(np.mean(all_artery_dices)) if all_artery_dices else 0.0,
        "vein_dice": float(np.mean(all_vein_dices)) if all_vein_dices else 0.0,
        "mean_dice": float(np.mean(all_artery_dices + all_vein_dices)) if (all_artery_dices or all_vein_dices) else 0.0,
        "n_cases": n_cases,
    }


# ============================================================
# Main training loop
# ============================================================

def train(args):
    # ── Device ──
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Output dir ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"lora_r{args.lora_r}_lr{args.lr}_e{args.epochs}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output: {output_dir}")

    # ── Save args ──
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Dataset ──
    train_dataset = DVTVideoDataset(
        data_root=args.data_root,
        split="train",
        max_frames=args.max_frames,
        box_margin=0.05,
        augment=True,
        box_jitter=args.box_jitter,
    )
    val_dataset = DVTVideoDataset(
        data_root=args.data_root,
        split="val",
        max_frames=0,
        box_margin=0.05,
        augment=False,
        box_jitter=0.0,
    )
    log.info(f"Train: {len(train_dataset)} cases, Val: {len(val_dataset)} cases")

    # ── Build LoRA model ──
    log.info(f"Building LoRA SAM2 (r={args.lora_r}, alpha={args.lora_alpha}) ...")
    lora_model = build_lora_sam2_video(
        config_file=args.sam2_config,
        ckpt_path=args.sam2_checkpoint,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        device=str(device),
        apply_to_image_encoder=True,
        apply_to_memory_attention=args.lora_memory_attention,
        apply_to_memory_encoder=args.lora_memory_encoder,
        use_trainer=True,
    )

    # ── Resume ──
    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        log.info(f"Resuming from {args.resume}")
        lora_model.load_lora_parameters(args.resume)

    # ── Optimizer ──
    params = lora_model.get_trainable_parameters()
    log.info(f"Trainable parameters: {sum(p.numel() for p in params) / 1e6:.4f}M")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── AMP ──
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu")

    # ── Training Log ──
    train_log_path = output_dir / "training_log.jsonl"

    # ── Training ──
    log.info("=" * 60)
    log.info("Starting LoRA training ...")
    log.info(f"  Epochs: {args.epochs}")
    log.info(f"  LR: {args.lr}")
    log.info(f"  LoRA rank: {args.lora_r}")
    log.info(f"  Grad accum: {args.grad_accum}")
    log.info(f"  Max frames/case: {args.max_frames}")
    log.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        lora_model.train()
        epoch_losses = []
        epoch_artery_dices = []
        epoch_vein_dices = []
        t0 = time.time()

        # 随机打乱 case 顺序
        indices = list(range(len(train_dataset)))
        if args.shuffle:
            import random
            random.shuffle(indices)

        optimizer.zero_grad()
        accum_loss = torch.zeros((), device=device)
        accum_count = 0

        for step, case_idx in enumerate(indices):
            case_data = train_dataset[case_idx]
            case_name = case_data["case_name"]

            try:
                result = train_one_case(
                    lora_model=lora_model,
                    case_data=case_data,
                    device=device,
                    amp_ctx=amp_ctx,
                    vein_weight=args.vein_weight,
                    artery_weight=args.artery_weight,
                    max_propagate_frames=args.max_frames,
                )

                loss = result["loss"]
                if loss.requires_grad and torch.isfinite(loss):
                    # Gradient accumulation
                    scaled_loss = loss / args.grad_accum
                    scaler.scale(scaled_loss).backward()
                    accum_loss = accum_loss + loss.detach()
                    accum_count += 1

                    if accum_count >= args.grad_accum:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                        epoch_losses.append(accum_loss.item() / accum_count)
                        accum_loss = torch.zeros((), device=device)
                        accum_count = 0

                epoch_artery_dices.append(result["artery_dice"])
                epoch_vein_dices.append(result["vein_dice"])

                if (step + 1) % args.log_every == 0:
                    avg_loss = np.mean(epoch_losses[-10:]) if epoch_losses else 0
                    log.info(
                        f"[E{epoch+1}/{args.epochs}] "
                        f"Step {step+1}/{len(indices)} "
                        f"loss={avg_loss:.4f} "
                        f"A_dice={result['artery_dice']:.4f} "
                        f"V_dice={result['vein_dice']:.4f} "
                        f"case={case_name}"
                    )

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log.warning(f"OOM on case {case_name}, skipping")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    accum_loss = torch.zeros((), device=device)
                    accum_count = 0
                    continue
                else:
                    log.error(f"Error on case {case_name}: {e}")
                    continue

        # 处理剩余的 accumulated gradients
        if accum_count > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_losses.append(accum_loss.item() / accum_count)

        scheduler.step()

        epoch_time = time.time() - t0
        avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0
        avg_a_dice = float(np.mean(epoch_artery_dices)) if epoch_artery_dices else 0
        avg_v_dice = float(np.mean(epoch_vein_dices)) if epoch_vein_dices else 0

        log.info(
            f"[Epoch {epoch+1}] "
            f"loss={avg_epoch_loss:.4f} "
            f"A_dice={avg_a_dice:.4f} V_dice={avg_v_dice:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6f} "
            f"time={epoch_time:.0f}s"
        )

        # ── Validation ──
        val_metrics = {}
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            log.info("Running validation ...")
            val_metrics = validate(
                lora_model=lora_model,
                val_dataset=val_dataset,
                device=device,
                max_cases=args.val_max_cases,
            )
            log.info(
                f"[Val] "
                f"A_dice={val_metrics['artery_dice']:.4f} "
                f"V_dice={val_metrics['vein_dice']:.4f} "
                f"Mean_dice={val_metrics['mean_dice']:.4f}"
            )

            # Save best
            if val_metrics["mean_dice"] > best_dice:
                best_dice = val_metrics["mean_dice"]
                best_path = output_dir / "lora_best.pt"
                lora_model.save_lora_parameters(str(best_path))
                log.info(f"New best model saved! dice={best_dice:.4f}")

        # ── Save checkpoint ──
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = output_dir / f"lora_epoch{epoch+1:03d}.pt"
            lora_model.save_lora_parameters(str(ckpt_path))

        # ── Log ──
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_epoch_loss,
            "train_artery_dice": avg_a_dice,
            "train_vein_dice": avg_v_dice,
            "lr": scheduler.get_last_lr()[0],
            "time": epoch_time,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "best_dice": best_dice,
        }
        with open(train_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    log.info("=" * 60)
    log.info(f"Training complete! Best val dice: {best_dice:.4f}")
    log.info(f"Results saved to: {output_dir}")
    log.info("=" * 60)


# ============================================================
# CLI
# ============================================================

def build_parser():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(description="SAM2 LoRA Fine-tuning for DVT")

    # Data
    parser.add_argument("--data-root", type=str, default=str(script_dir / "dataset"))
    parser.add_argument("--max-frames", type=int, default=30,
                        help="每个 case 最大传播帧数 (减少显存, 0=全部)")

    # Model
    parser.add_argument("--sam2-config", type=str, default="configs/sam2/sam2_hiera_l.yaml")
    parser.add_argument("--sam2-checkpoint", type=str,
                        default=str(script_dir / "checkpoints" / "sam2_hiera_large.pt"))

    # LoRA
    parser.add_argument("--lora-r", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=1.0, help="LoRA alpha")
    parser.add_argument("--lora-memory-attention", action="store_true", default=True,
                        help="对 memory_attention 应用 LoRA")
    parser.add_argument("--no-lora-memory-attention", dest="lora_memory_attention", action="store_false")
    parser.add_argument("--lora-memory-encoder", action="store_true", default=False,
                        help="解冻 memory_encoder (参数量小, 全量微调)")

    # Training
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="梯度累积步数 (等效增大 batch size)")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--vein-weight", type=float, default=1.5, help="静脉损失权重")
    parser.add_argument("--artery-weight", type=float, default=1.0, help="动脉损失权重")
    parser.add_argument("--box-jitter", type=float, default=0.1,
                        help="训练时 box prompt 抖动比例 (模拟 YOLO 误差)")
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")

    # Validation
    parser.add_argument("--val-every", type=int, default=5, help="每几个 epoch 验证一次")
    parser.add_argument("--val-max-cases", type=int, default=20,
                        help="验证时最多评估几个 case (0=全部)")

    # Output
    parser.add_argument("--output-dir", type=str,
                        default=str(script_dir / "checkpoints" / "lora_runs"))
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=10)

    # Device
    parser.add_argument("--gpu", type=int, default=0)

    # Resume
    parser.add_argument("--resume", type=str, default=None, help="LoRA checkpoint 路径")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
