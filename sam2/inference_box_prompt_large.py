#!/usr/bin/env python3
"""
SAM2 large 推理评估脚本（YOLO box prompt）

功能：
1) 使用指定 YOLO 权重为每帧提供 artery/vein 框（含缺失框补全）
2) 使用 SAM2 large（sam2.1_hiera_large）做 box prompt 分割
3) 仅在有标准掩码的帧上评估 Dice / mIoU
4) 记录日志、导出 CSV、可视化每个样例（每个有标注帧）
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
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


DETECTION_CLASS_NAMES = {0: "artery", 1: "vein"}
SEGMENTATION_CLASS_VALUES = {"artery": 1, "vein": 2}
CLASS_COLORS = {
    "artery": (0, 0, 255),  # BGR
    "vein": (0, 255, 0),
}


class Logger:
    def __init__(self, log_path: Path) -> None:
        self._terminal = sys.stdout
        self._fp = open(log_path, "w", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message)
        self._fp.write(message + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


class VesselDetector:
    """
    基于 YOLO 的 artery/vein 框检测器：
    - 每帧尽量保证输出两个框
    - 若缺失则根据先验进行补全
    """

    DEFAULT_PRIOR = {
        "class_absolute": {
            "artery": {
                "cx": {"mean": 0.34},
                "cy": {"mean": 0.53},
                "w": {"mean": 0.21},
                "h": {"mean": 0.24},
            },
            "vein": {
                "cx": {"mean": 0.36},
                "cy": {"mean": 0.70},
                "w": {"mean": 0.23},
                "h": {"mean": 0.20},
            },
        },
        "artery2vein": {
            "cx_offset": {"mean": 0.021},
            "cy_offset": {"mean": 0.168},
            "w_ratio": {"mean": 1.117},
            "h_ratio": {"mean": 0.845},
        },
        "vein2artery": {
            "cx_offset": {"mean": -0.021},
            "cy_offset": {"mean": -0.168},
            "w_ratio": {"mean": 0.979},
            "h_ratio": {"mean": 1.625},
        },
    }

    def __init__(
        self,
        model_path: Path,
        yolo_device: str | int,
        prior_path: Path,
        retry_conf: float = 0.01,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.device = yolo_device
        self.retry_conf = retry_conf
        self.prior = self._load_prior(prior_path)

    def _load_prior(self, prior_path: Path) -> dict:
        if prior_path.exists():
            with open(prior_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data, _ = self._normalize_prior_schema(data)
            return data
        return deepcopy(self.DEFAULT_PRIOR)

    def _normalize_prior_schema(self, prior: dict) -> Tuple[dict, bool]:
        data = deepcopy(prior) if isinstance(prior, dict) else {}
        patched = False

        for cls_name in ("artery", "vein"):
            default_cls = self.DEFAULT_PRIOR["class_absolute"][cls_name]
            if "class_absolute" not in data or not isinstance(data["class_absolute"], dict):
                data["class_absolute"] = {}
                patched = True
            if cls_name not in data["class_absolute"] or not isinstance(
                data["class_absolute"][cls_name], dict
            ):
                data["class_absolute"][cls_name] = {}
                patched = True

            for key, default_val in default_cls.items():
                item = data["class_absolute"][cls_name].get(key)
                if not isinstance(item, dict):
                    data["class_absolute"][cls_name][key] = deepcopy(default_val)
                    patched = True
                elif "mean" not in item:
                    data["class_absolute"][cls_name][key]["mean"] = default_val["mean"]
                    patched = True

        for direction in ("artery2vein", "vein2artery"):
            default_dir = self.DEFAULT_PRIOR[direction]
            if direction not in data or not isinstance(data[direction], dict):
                data[direction] = {}
                patched = True
            for key, default_val in default_dir.items():
                item = data[direction].get(key)
                if not isinstance(item, dict):
                    data[direction][key] = deepcopy(default_val)
                    patched = True
                elif "mean" not in item:
                    data[direction][key]["mean"] = default_val["mean"]
                    patched = True
        return data, patched

    def _get_prior(self, direction: str, key: str) -> float:
        return float(self.prior[direction][key]["mean"])

    def _get_class_prior(self, cls_name: str, key: str) -> float:
        return float(self.prior["class_absolute"][cls_name][key]["mean"])

    @staticmethod
    def _clip_box(box: List[float], w: int, h: int) -> List[float]:
        x1, y1, x2, y2 = box
        x1 = float(np.clip(x1, 0, max(0, w - 1)))
        y1 = float(np.clip(y1, 0, max(0, h - 1)))
        x2 = float(np.clip(x2, 0, max(0, w - 1)))
        y2 = float(np.clip(y2, 0, max(0, h - 1)))
        if x2 <= x1:
            x2 = min(float(w - 1), x1 + 1.0)
        if y2 <= y1:
            y2 = min(float(h - 1), y1 + 1.0)
        return [x1, y1, x2, y2]

    @staticmethod
    def _compute_iou_boxes(box1: List[float], box2: List[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union = area1 + area2 - inter_area
        return inter_area / union if union > 0 else 0.0

    @staticmethod
    def _norm_box_to_xyxy(cx: float, cy: float, bw: float, bh: float, w: int, h: int) -> List[float]:
        bw = float(np.clip(bw, 1e-4, 1.0))
        bh = float(np.clip(bh, 1e-4, 1.0))
        cx = float(np.clip(cx, bw / 2, 1.0 - bw / 2))
        cy = float(np.clip(cy, bh / 2, 1.0 - bh / 2))
        return [
            (cx - bw / 2) * w,
            (cy - bh / 2) * h,
            (cx + bw / 2) * w,
            (cy + bh / 2) * h,
        ]

    def _infer_class_box(self, cls_name: str, w: int, h: int) -> Dict:
        box = self._norm_box_to_xyxy(
            self._get_class_prior(cls_name, "cx"),
            self._get_class_prior(cls_name, "cy"),
            self._get_class_prior(cls_name, "w"),
            self._get_class_prior(cls_name, "h"),
            w,
            h,
        )
        return {"box": self._clip_box(box, w, h), "conf": 0.0, "inferred": True, "prior_all": True}

    def _infer_vein(self, a_box: List[float], w: int, h: int) -> Dict:
        a_cx = (a_box[0] + a_box[2]) / 2 / w
        a_cy = (a_box[1] + a_box[3]) / 2 / h
        a_w = (a_box[2] - a_box[0]) / w
        a_h = (a_box[3] - a_box[1]) / h
        v_cx = a_cx + self._get_prior("artery2vein", "cx_offset")
        v_cy = a_cy + self._get_prior("artery2vein", "cy_offset")
        v_w = a_w * self._get_prior("artery2vein", "w_ratio")
        v_h = a_h * self._get_prior("artery2vein", "h_ratio")
        box = self._norm_box_to_xyxy(v_cx, v_cy, v_w, v_h, w, h)
        return {"box": self._clip_box(box, w, h), "conf": 0.0, "inferred": True}

    def _infer_artery(self, v_box: List[float], w: int, h: int) -> Dict:
        v_cx = (v_box[0] + v_box[2]) / 2 / w
        v_cy = (v_box[1] + v_box[3]) / 2 / h
        v_w = (v_box[2] - v_box[0]) / w
        v_h = (v_box[3] - v_box[1]) / h
        a_cx = v_cx + self._get_prior("vein2artery", "cx_offset")
        a_cy = v_cy + self._get_prior("vein2artery", "cy_offset")
        a_w = v_w * self._get_prior("vein2artery", "w_ratio")
        a_h = v_h * self._get_prior("vein2artery", "h_ratio")
        box = self._norm_box_to_xyxy(a_cx, a_cy, a_w, a_h, w, h)
        return {"box": self._clip_box(box, w, h), "conf": 0.0, "inferred": True}

    def _retry_lower_conf(self, img: np.ndarray, artery: Dict | None, vein: Dict | None) -> Tuple[Dict | None, Dict | None]:
        results = self.model(img, conf=self.retry_conf, device=self.device, verbose=False)[0]
        for box in results.boxes:
            cls_id = int(box.cls)
            conf_score = float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            if cls_id == 0 and artery is None:
                artery = {"box": xyxy, "conf": conf_score}
            elif cls_id == 1 and vein is None:
                vein = {"box": xyxy, "conf": conf_score}
        return artery, vein

    def _check_and_fix_overlap(
        self, artery: Dict, vein: Dict, w: int, h: int, max_iou: float = 0.3
    ) -> Tuple[Dict, Dict]:
        iou = self._compute_iou_boxes(artery["box"], vein["box"])
        if iou <= max_iou:
            return artery, vein
        if artery["conf"] >= vein["conf"]:
            vein = self._infer_vein(artery["box"], w, h)
            vein["fixed"] = True
        else:
            artery = self._infer_artery(vein["box"], w, h)
            artery["fixed"] = True
        return artery, vein

    def predict(self, image: np.ndarray, conf: float) -> Dict[str, Dict | None]:
        if image is None:
            raise ValueError("输入图像为空，无法执行 YOLO 检测。")

        h, w = image.shape[:2]
        results = self.model(image, conf=conf, device=self.device, verbose=False)[0]
        artery_list: List[Dict] = []
        vein_list: List[Dict] = []

        for box in results.boxes:
            cls_id = int(box.cls)
            if cls_id not in DETECTION_CLASS_NAMES:
                continue
            conf_score = float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            item = {"box": self._clip_box(xyxy, w, h), "conf": conf_score}
            if cls_id == 0:
                artery_list.append(item)
            else:
                vein_list.append(item)

        artery = max(artery_list, key=lambda x: x["conf"]) if artery_list else None
        vein = max(vein_list, key=lambda x: x["conf"]) if vein_list else None

        if artery is None or vein is None:
            artery, vein = self._retry_lower_conf(image, artery, vein)

        if artery is None and vein is None:
            artery = self._infer_class_box("artery", w, h)
            vein = self._infer_vein(artery["box"], w, h)
            vein["prior_all"] = True
        elif artery is not None and vein is None:
            vein = self._infer_vein(artery["box"], w, h)
        elif vein is not None and artery is None:
            artery = self._infer_artery(vein["box"], w, h)

        artery["box"] = self._clip_box(artery["box"], w, h)
        vein["box"] = self._clip_box(vein["box"], w, h)
        artery, vein = self._check_and_fix_overlap(artery, vein, w, h)
        return {"artery": artery, "vein": vein}


class SAM2BoxSegmenter:
    def __init__(self, model_cfg: str, checkpoint: Path, device: str) -> None:
        model = build_sam2(config_file=model_cfg, ckpt_path=str(checkpoint), device=device)
        self.predictor = SAM2ImagePredictor(model)
        self.device = device

    def segment(self, image_bgr: np.ndarray, boxes: Dict[str, Dict | None]) -> Tuple[np.ndarray, Dict[str, Dict]]:
        if image_bgr is None:
            raise ValueError("输入图像为空，无法执行 SAM2 分割。")
        h, w = image_bgr.shape[:2]
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        score_map = np.full((h, w), -1e9, dtype=np.float32)
        detail: Dict[str, Dict] = {}
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        amp_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with torch.inference_mode():
            with amp_ctx:
                self.predictor.set_image(image_rgb)
                for cls_name in ("artery", "vein"):
                    det = boxes.get(cls_name)
                    if det is None:
                        detail[cls_name] = {"sam_score": 0.0, "pixel_count": 0}
                        continue
                    box = np.asarray(det["box"], dtype=np.float32)
                    masks, scores, _ = self.predictor.predict(
                        box=box,
                        multimask_output=True,
                        return_logits=False,
                    )
                    best_idx = int(np.argmax(scores))
                    best_score = float(scores[best_idx])
                    best_mask = masks[best_idx].astype(bool)
                    cls_value = SEGMENTATION_CLASS_VALUES[cls_name]
                    update_pixels = best_mask & (best_score >= score_map)
                    pred_mask[update_pixels] = cls_value
                    score_map[update_pixels] = best_score
                    detail[cls_name] = {
                        "sam_score": best_score,
                        "pixel_count": int(np.count_nonzero(best_mask)),
                    }
        return pred_mask, detail


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    intersection = float(np.logical_and(pred_bool, gt_bool).sum())
    pred_area = float(pred_bool.sum())
    gt_area = float(gt_bool.sum())
    union = float(np.logical_or(pred_bool, gt_bool).sum())

    dice = (2.0 * intersection) / (pred_area + gt_area) if (pred_area + gt_area) > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return dice, iou


def compute_frame_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    a_dice, a_iou = binary_metrics(pred_mask == 1, gt_mask == 1)
    v_dice, v_iou = binary_metrics(pred_mask == 2, gt_mask == 2)
    mean_dice = float(np.mean([a_dice, v_dice]))
    miou = float(np.mean([a_iou, v_iou]))
    return {
        "artery_dice": a_dice,
        "artery_iou": a_iou,
        "vein_dice": v_dice,
        "vein_iou": v_iou,
        "mean_dice": mean_dice,
        "miou": miou,
    }


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    canvas = image.copy()
    for cls_name, cls_value in SEGMENTATION_CLASS_VALUES.items():
        region = mask == cls_value
        if not np.any(region):
            continue
        color = np.array(CLASS_COLORS[cls_name], dtype=np.float32)
        src = canvas[region].astype(np.float32)
        blended = src * (1.0 - alpha) + color * alpha
        canvas[region] = blended.astype(np.uint8)
    return canvas


def draw_boxes(image: np.ndarray, boxes: Dict[str, Dict | None]) -> np.ndarray:
    out = image.copy()
    for cls_name in ("artery", "vein"):
        det = boxes.get(cls_name)
        if det is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        color = CLASS_COLORS[cls_name]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tags = [f"{cls_name}:{det['conf']:.2f}"]
        if det.get("inferred"):
            tags.append("inferred")
        if det.get("fixed"):
            tags.append("fixed")
        if det.get("prior_all"):
            tags.append("prior-all")
        cv2.putText(
            out,
            "|".join(tags),
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def build_visualization(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    boxes: Dict[str, Dict | None],
    metrics: Dict[str, float],
) -> np.ndarray:
    raw_panel = draw_boxes(image, boxes)
    gt_panel = overlay_mask(image, gt_mask)
    pred_panel = draw_boxes(overlay_mask(image, pred_mask), boxes)

    cv2.putText(raw_panel, "Image + YOLO Box", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(gt_panel, "Ground Truth Mask", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(pred_panel, "SAM2 Prediction", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    metric_lines = [
        f"Dice={metrics['mean_dice']:.4f}",
        f"mIoU={metrics['miou']:.4f}",
        f"A Dice/IoU={metrics['artery_dice']:.4f}/{metrics['artery_iou']:.4f}",
        f"V Dice/IoU={metrics['vein_dice']:.4f}/{metrics['vein_iou']:.4f}",
    ]
    y = 56
    for line in metric_lines:
        cv2.putText(pred_panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 24

    h = image.shape[0]
    separator = np.full((h, 8, 3), 255, dtype=np.uint8)
    return np.hstack([raw_panel, separator, gt_panel, separator, pred_panel])


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = [
        "artery_dice",
        "artery_iou",
        "vein_dice",
        "vein_iou",
        "mean_dice",
        "miou",
    ]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def resolve_sam2_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def resolve_yolo_device(yolo_device_arg: str, sam2_device: str) -> str | int:
    if yolo_device_arg == "auto":
        return 0 if sam2_device.startswith("cuda") else "cpu"
    if yolo_device_arg.isdigit():
        return int(yolo_device_arg)
    return yolo_device_arg


def run(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    data_root = Path(args.data_root).resolve()
    split_dir = data_root / args.split
    yolo_model_path = Path(args.yolo_model).resolve()
    yolo_prior_path = Path(args.yolo_prior).resolve()
    sam2_ckpt = Path(args.sam2_checkpoint).resolve()

    if not split_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在: {split_dir}")
    if not yolo_model_path.exists():
        raise FileNotFoundError(f"YOLO 权重不存在: {yolo_model_path}")
    if not sam2_ckpt.exists():
        raise FileNotFoundError(f"SAM2 checkpoint 不存在: {sam2_ckpt}")

    output_root = Path(args.output_root).resolve()
    run_name = f"{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_name
    vis_root = run_dir / "visualizations"
    run_dir.mkdir(parents=True, exist_ok=True)
    vis_root.mkdir(parents=True, exist_ok=True)

    logger = Logger(run_dir / "inference_eval.log")
    sam2_device = resolve_sam2_device(args.device)
    yolo_device = resolve_yolo_device(args.yolo_device, sam2_device)

    logger.log("=" * 80)
    logger.log("SAM2 large + YOLO box 推理评估")
    logger.log(f"Split: {args.split}")
    logger.log(f"Data root: {data_root}")
    logger.log(f"YOLO model: {yolo_model_path}")
    logger.log(f"YOLO prior: {yolo_prior_path}")
    logger.log(f"SAM2 config: {args.sam2_config}")
    logger.log(f"SAM2 ckpt: {sam2_ckpt}")
    logger.log(f"SAM2 device: {sam2_device} | YOLO device: {yolo_device}")
    logger.log(f"Output dir: {run_dir}")
    logger.log("=" * 80)

    detector = VesselDetector(
        model_path=yolo_model_path,
        yolo_device=yolo_device,
        prior_path=yolo_prior_path,
        retry_conf=args.retry_conf,
    )
    segmenter = SAM2BoxSegmenter(
        model_cfg=args.sam2_config,
        checkpoint=sam2_ckpt,
        device=sam2_device,
    )

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

        gt_masks = sorted(masks_dir.glob("*.png"))
        if not gt_masks:
            logger.log(f"[Skip case] {case_dir.name}: 无标准掩码")
            continue

        case_metrics: List[Dict[str, float]] = []
        case_vis_dir = vis_root / case_dir.name
        case_vis_dir.mkdir(parents=True, exist_ok=True)

        for mask_path in gt_masks:
            stem = mask_path.stem
            image_path = images_dir / f"{stem}.jpg"
            if not image_path.exists():
                logger.log(f"[Skip frame] {case_dir.name}/{stem}: 图像缺失 {image_path.name}")
                skipped_frames += 1
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or gt_mask is None:
                logger.log(f"[Skip frame] {case_dir.name}/{stem}: 读取失败")
                skipped_frames += 1
                continue

            h, w = image.shape[:2]
            if gt_mask.shape != (h, w):
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                logger.log(f"[Warn] {case_dir.name}/{stem}: GT 尺寸与图像不一致，已重采样到 {w}x{h}")

            boxes = detector.predict(image, conf=args.yolo_conf)
            pred_mask, _ = segmenter.segment(image, boxes)
            metrics = compute_frame_metrics(pred_mask, gt_mask)

            frame_row = {
                "case": case_dir.name,
                "frame": stem,
                **metrics,
            }
            frame_rows.append(frame_row)
            case_metrics.append(metrics)
            processed_frames += 1

            vis = build_visualization(image, gt_mask, pred_mask, boxes, metrics)
            vis_path = case_vis_dir / f"{stem}_viz.jpg"
            cv2.imwrite(str(vis_path), vis)

            logger.log(
                f"[{case_dir.name}/{stem}] "
                f"Dice={metrics['mean_dice']:.4f}, mIoU={metrics['miou']:.4f}, "
                f"A_dice={metrics['artery_dice']:.4f}, V_dice={metrics['vein_dice']:.4f}"
            )

        if case_metrics:
            case_summary = summarize_rows(case_metrics)
            case_row = {"case": case_dir.name, "n_frames": len(case_metrics), **case_summary}
            case_rows.append(case_row)
            logger.log(
                f"[Case Summary] {case_dir.name}: n={len(case_metrics)}, "
                f"Dice={case_summary['mean_dice']:.4f}, mIoU={case_summary['miou']:.4f}"
            )

    if not frame_rows:
        logger.close()
        raise RuntimeError("没有可评估帧（请检查数据路径与标注文件）。")

    global_summary = summarize_rows(frame_rows)
    logger.log("-" * 80)
    logger.log(f"Processed frames: {processed_frames}, Skipped frames: {skipped_frames}")
    logger.log(
        f"[Global] Dice={global_summary['mean_dice']:.4f}, "
        f"mIoU={global_summary['miou']:.4f}, "
        f"A Dice/IoU={global_summary['artery_dice']:.4f}/{global_summary['artery_iou']:.4f}, "
        f"V Dice/IoU={global_summary['vein_dice']:.4f}/{global_summary['vein_iou']:.4f}"
    )
    logger.log("-" * 80)

    frame_csv_fields = [
        "case",
        "frame",
        "artery_dice",
        "artery_iou",
        "vein_dice",
        "vein_iou",
        "mean_dice",
        "miou",
    ]
    case_csv_fields = [
        "case",
        "n_frames",
        "artery_dice",
        "artery_iou",
        "vein_dice",
        "vein_iou",
        "mean_dice",
        "miou",
    ]
    write_csv(run_dir / "frame_metrics.csv", frame_rows, frame_csv_fields)
    write_csv(run_dir / "case_metrics.csv", case_rows, case_csv_fields)

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "processed_frames": processed_frames,
                "skipped_frames": skipped_frames,
                "global_metrics": global_summary,
                "split": args.split,
                "sam2_config": args.sam2_config,
                "sam2_checkpoint": str(sam2_ckpt),
                "yolo_model": str(yolo_model_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.log(f"结果已保存到: {run_dir}")
    logger.close()


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_yolo_model = (
        repo_root
        / "yolo"
        / "runs"
        / "detect"
        / "runs"
        / "detect"
        / "dvt_runs"
        / "aug_step5_speckle_translate_scale"
        / "weights"
        / "best.pt"
    )
    default_yolo_prior = repo_root / "yolo" / "prior_stats.json"
    default_data_root = script_dir / "dataset"
    default_output_root = script_dir / "predictions" / "sam2_large_yolo_box"
    default_sam2_ckpt = script_dir / "checkpoints" / "sam2.1_hiera_large.pt"

    parser = argparse.ArgumentParser(description="SAM2 large + YOLO box 推理评估脚本")
    parser.add_argument("--data-root", type=str, default=str(default_data_root), help="数据集根目录（含 train/val）")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="评估数据划分")
    parser.add_argument("--output-root", type=str, default=str(default_output_root), help="输出目录根路径")
    parser.add_argument("--max-cases", type=int, default=0, help="仅处理前N个case（0表示全部）")

    parser.add_argument("--yolo-model", type=str, default=str(default_yolo_model), help="YOLO权重路径")
    parser.add_argument("--yolo-prior", type=str, default=str(default_yolo_prior), help="YOLO先验统计文件路径")
    parser.add_argument("--yolo-conf", type=float, default=0.1, help="YOLO主推理置信度阈值")
    parser.add_argument("--retry-conf", type=float, default=0.01, help="YOLO补检置信度阈值")
    parser.add_argument(
        "--yolo-device",
        type=str,
        default="auto",
        help="YOLO设备: auto / cpu / cuda:0 / 0",
    )

    parser.add_argument(
        "--sam2-config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2配置文件（large推荐: configs/sam2.1/sam2.1_hiera_l.yaml）",
    )
    parser.add_argument("--sam2-checkpoint", type=str, default=str(default_sam2_ckpt), help="SAM2 checkpoint 路径")
    parser.add_argument("--device", type=str, default="auto", help="SAM2设备: auto / cpu / cuda / cuda:0")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
