#!/usr/bin/env python3
"""
SAM2 large 推理评估脚本（YOLO box prompt）

功能：
1) 使用指定 YOLO 权重在每个样例首帧提供 artery/vein 框（含缺失框补全）
2) 使用 SAM2 Large（sam2_hiera_large）视频预测器，仅首帧提示，后续帧依赖记忆传播
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
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from sam2.build_sam import build_sam2_video_predictor


DETECTION_CLASS_NAMES = {0: "artery", 1: "vein"}
SEGMENTATION_CLASS_VALUES = {"artery": 1, "vein": 2}
OBJECT_ID_TO_CLASS = {1: "artery", 2: "vein"}
CLASS_TO_OBJECT_ID = {"artery": 1, "vein": 2}
CLASS_COLORS = {
    "artery": (0, 0, 255),  # BGR
    "vein": (0, 255, 0),
}
METRIC_KEYS = [
    "artery_dice",
    "artery_miou",
    "vein_dice",
    "vein_miou",
    "mean_dice",
    "miou",
]


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
        prompt_min_conf: float = 0.05,
        prompt_min_area_ratio: float = 0.0005,
        prompt_max_area_ratio: float = 0.6,
        prompt_max_aspect_ratio: float = 8.0,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.device = yolo_device
        self.retry_conf = retry_conf
        self.prompt_min_conf = prompt_min_conf
        self.prompt_min_area_ratio = prompt_min_area_ratio
        self.prompt_max_area_ratio = prompt_max_area_ratio
        self.prompt_max_aspect_ratio = prompt_max_aspect_ratio
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

    def _is_box_reasonable(self, box: List[float], w: int, h: int) -> bool:
        bw = max(0.0, box[2] - box[0])
        bh = max(0.0, box[3] - box[1])
        area = bw * bh
        full = float(max(1, w * h))
        area_ratio = area / full
        if area_ratio < self.prompt_min_area_ratio or area_ratio > self.prompt_max_area_ratio:
            return False
        aspect = max(bw / max(1e-6, bh), bh / max(1e-6, bw))
        if aspect > self.prompt_max_aspect_ratio:
            return False
        return True

    def _apply_quality_gate(self, det: Dict | None, w: int, h: int) -> Dict | None:
        if det is None:
            return None
        if det.get("inferred", False):
            return det
        if det["conf"] < self.prompt_min_conf:
            return None
        if not self._is_box_reasonable(det["box"], w, h):
            return None
        return det

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

        artery = self._apply_quality_gate(artery, w, h)
        vein = self._apply_quality_gate(vein, w, h)

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


class SAM2MemoryVideoSegmenter:
    """
    SAM2 视频预测器：
    - 每个样例仅在首帧添加 box prompt
    - 后续帧仅依赖 memory 传播
    """

    def __init__(
        self,
        model_cfg: str,
        checkpoint: Path,
        device: str,
    ) -> None:
        self.predictor = build_sam2_video_predictor(
            config_file=model_cfg,
            ckpt_path=str(checkpoint),
            device=device,
        )
        self.device = device

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
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

        return {
            "artery": artery_mask,
            "vein": vein_mask,
            "semantic": semantic,
        }

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


def get_bbox_from_mask(mask: np.ndarray, cls_value: int) -> Optional[List[float]]:
    ys, xs = np.where(mask == cls_value)
    if len(xs) == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max())
    y2 = float(ys.max())
    return [x1, y1, x2, y2]


def build_prompt_boxes_from_gt_mask(gt_mask: np.ndarray) -> Optional[Dict[str, Dict]]:
    artery_box = get_bbox_from_mask(gt_mask, SEGMENTATION_CLASS_VALUES["artery"])
    vein_box = get_bbox_from_mask(gt_mask, SEGMENTATION_CLASS_VALUES["vein"])
    if artery_box is None or vein_box is None:
        return None
    return {
        "artery": {"box": artery_box, "conf": 1.0, "from_gt": True},
        "vein": {"box": vein_box, "conf": 1.0, "from_gt": True},
    }


def compute_frame_metrics(pred_by_class: Dict[str, np.ndarray], gt_mask: np.ndarray) -> Dict[str, float]:
    a_dice, a_miou = binary_metrics(pred_by_class["artery"], gt_mask == 1)
    v_dice, v_miou = binary_metrics(pred_by_class["vein"], gt_mask == 2)
    mean_dice = float(np.mean([a_dice, v_dice]))
    miou = float(np.mean([a_miou, v_miou]))
    return {
        "artery_dice": a_dice,
        "artery_miou": a_miou,
        "artery_iou": a_miou,
        "vein_dice": v_dice,
        "vein_miou": v_miou,
        "vein_iou": v_miou,
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
    is_prompt_frame: bool,
) -> np.ndarray:
    raw_panel = draw_boxes(image, boxes)
    gt_panel = overlay_mask(image, gt_mask)
    pred_panel = draw_boxes(overlay_mask(image, pred_mask), boxes)

    cv2.putText(raw_panel, "Image + YOLO Box", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(gt_panel, "Ground Truth Mask", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(pred_panel, "SAM2 Prediction", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    mode_text = "Prompt frame (box)" if is_prompt_frame else "Memory-only frame"
    cv2.putText(pred_panel, mode_text, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    metric_lines = [
        f"Dice={metrics['mean_dice']:.4f}",
        f"mIoU={metrics['miou']:.4f}",
        f"A Dice/mIoU={metrics['artery_dice']:.4f}/{metrics['artery_miou']:.4f}",
        f"V Dice/mIoU={metrics['vein_dice']:.4f}/{metrics['vein_miou']:.4f}",
    ]
    y = 76
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
    def _avg(key: str, fallback: Optional[str] = None) -> float:
        vals = []
        for row in rows:
            if key in row:
                vals.append(row[key])
            elif fallback is not None and fallback in row:
                vals.append(row[fallback])
            else:
                raise KeyError(f"缺少指标字段: {key}")
        return float(np.mean(vals))

    return {
        "artery_dice": _avg("artery_dice"),
        "artery_miou": _avg("artery_miou", "artery_iou"),
        "artery_iou": _avg("artery_iou", "artery_miou"),
        "vein_dice": _avg("vein_dice"),
        "vein_miou": _avg("vein_miou", "vein_iou"),
        "vein_iou": _avg("vein_iou", "vein_miou"),
        "mean_dice": _avg("mean_dice"),
        "miou": _avg("miou"),
    }


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


def str2bool(value: str) -> bool:
    value_lower = value.strip().lower()
    if value_lower in {"true", "1", "yes", "y", "on"}:
        return True
    if value_lower in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


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
    artery_thresh = args.mask_score_thresh if args.artery_mask_thresh is None else args.artery_mask_thresh
    vein_thresh = args.mask_score_thresh if args.vein_mask_thresh is None else args.vein_mask_thresh

    logger.log("=" * 80)
    logger.log("SAM2 large + YOLO首帧box提示 + 记忆传播评估")
    logger.log(f"Split: {args.split}")
    logger.log(f"Data root: {data_root}")
    logger.log(f"YOLO model: {yolo_model_path}")
    logger.log(f"YOLO prior: {yolo_prior_path}")
    logger.log(f"SAM2 config: {args.sam2_config}")
    logger.log(f"SAM2 ckpt: {sam2_ckpt}")
    logger.log("Prompt strategy: 仅首帧使用YOLO box，后续帧仅依赖memory")
    logger.log(
        f"Mask thresh: artery={artery_thresh:.3f}, vein={vein_thresh:.3f}, "
        f"postprocess={args.postprocess_mode}"
    )
    logger.log(f"SAM2 device: {sam2_device} | YOLO device: {yolo_device}")
    logger.log(f"Output dir: {run_dir}")
    logger.log("=" * 80)

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
    segmenter = SAM2MemoryVideoSegmenter(
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

        prompt_frame_idx = 0
        prompt_stem = frame_stems[prompt_frame_idx]
        prompt_image = cv2.imread(str(image_files[prompt_frame_idx]), cv2.IMREAD_COLOR)
        if prompt_image is None:
            logger.log(f"[Skip case] {case_dir.name}: 首帧读取失败 {image_files[prompt_frame_idx].name}")
            continue
        first_frame_boxes = detector.predict(prompt_image, conf=args.yolo_conf)
        logger.log(
            f"[Case Prompt] {case_dir.name}: first_frame={prompt_stem}, "
            f"artery_conf={first_frame_boxes['artery']['conf']:.3f}, "
            f"vein_conf={first_frame_boxes['vein']['conf']:.3f}"
        )

        eval_targets: Dict[int, Path] = {}
        for mask_path in gt_masks:
            stem = mask_path.stem
            frame_idx = stem_to_frame_idx.get(stem)
            if frame_idx is None:
                logger.log(f"[Skip frame] {case_dir.name}/{stem}: 在images中找不到对应帧")
                skipped_frames += 1
                continue
            eval_targets[frame_idx] = mask_path

        if not eval_targets:
            logger.log(f"[Skip case] {case_dir.name}: 无可评估帧")
            continue

        pred_masks_by_idx = segmenter.segment_video_with_first_frame_prompt(
            images_dir=images_dir,
            first_frame_boxes=first_frame_boxes,
            target_frame_indices=set(eval_targets.keys()),
            artery_score_thresh=artery_thresh,
            vein_score_thresh=vein_thresh,
            postprocess_mode=args.postprocess_mode,
            morph_kernel=args.morph_kernel,
        )

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
                logger.log(f"[Skip frame] {case_dir.name}/{stem}: 读取失败")
                skipped_frames += 1
                continue

            h, w = image.shape[:2]
            if gt_mask.shape != (h, w):
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                logger.log(f"[Warn] {case_dir.name}/{stem}: GT 尺寸与图像不一致，已重采样到 {w}x{h}")

            frame_pred = pred_masks_by_idx.get(frame_idx)
            if frame_pred is None:
                logger.log(f"[Skip frame] {case_dir.name}/{stem}: 记忆传播未返回该帧预测")
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

            frame_row = {
                "case": case_dir.name,
                "frame": stem,
                **metrics,
            }
            frame_rows.append(frame_row)
            case_metrics.append(metrics)
            processed_frames += 1

            vis_boxes = first_frame_boxes if frame_idx == prompt_frame_idx else {"artery": None, "vein": None}
            vis = build_visualization(
                image=image,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                boxes=vis_boxes,
                metrics=metrics,
                is_prompt_frame=(frame_idx == prompt_frame_idx),
            )
            vis_path = case_vis_dir / f"{stem}_viz.jpg"
            cv2.imwrite(str(vis_path), vis)

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

    if not frame_rows:
        logger.close()
        raise RuntimeError("没有可评估帧（请检查数据路径与标注文件）。")

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

    frame_csv_fields = [
        "case",
        "frame",
        "artery_dice",
        "artery_miou",
        "artery_iou",
        "vein_dice",
        "vein_miou",
        "vein_iou",
        "mean_dice",
        "miou",
    ]
    case_csv_fields = [
        "case",
        "n_frames",
        "artery_dice",
        "artery_miou",
        "artery_iou",
        "vein_dice",
        "vein_miou",
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
                "global_frame_weighted_metrics": global_summary,
                "global_case_weighted_metrics": global_case_summary,
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
    default_sam2_ckpt = script_dir / "checkpoints" / "sam2_hiera_large.pt"

    parser = argparse.ArgumentParser(description="SAM2 Large + YOLO首帧框 推理评估脚本")
    parser.add_argument("--data-root", type=str, default=str(default_data_root), help="数据集根目录（含 train/val）")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="评估数据划分")
    parser.add_argument("--output-root", type=str, default=str(default_output_root), help="输出目录根路径")
    parser.add_argument("--max-cases", type=int, default=0, help="仅处理前N个case（0表示全部）")

    parser.add_argument("--yolo-model", type=str, default=str(default_yolo_model), help="YOLO权重路径")
    parser.add_argument("--yolo-prior", type=str, default=str(default_yolo_prior), help="YOLO先验统计文件路径")
    parser.add_argument("--yolo-conf", type=float, default=0.1, help="YOLO主推理置信度阈值")
    parser.add_argument("--retry-conf", type=float, default=0.01, help="YOLO补检置信度阈值")
    parser.add_argument("--prompt-min-conf", type=float, default=0.05, help="首帧提示框最低置信度")
    parser.add_argument("--prompt-min-area-ratio", type=float, default=0.0005, help="首帧提示框最小面积占比")
    parser.add_argument("--prompt-max-area-ratio", type=float, default=0.6, help="首帧提示框最大面积占比")
    parser.add_argument("--prompt-max-aspect-ratio", type=float, default=8.0, help="首帧提示框最大长宽比")
    parser.add_argument(
        "--yolo-device",
        type=str,
        default="auto",
        help="YOLO设备: auto / cpu / cuda:0 / 0",
    )
    parser.add_argument(
        "--mask-score-thresh",
        type=float,
        default=0.0,
        help="SAM2传播掩码默认logit阈值（动静脉未单独设置时使用）",
    )
    parser.add_argument("--artery-mask-thresh", type=float, default=None, help="动脉mask logit阈值（可覆盖默认阈值）")
    parser.add_argument("--vein-mask-thresh", type=float, default=None, help="静脉mask logit阈值（可覆盖默认阈值）")
    parser.add_argument(
        "--postprocess-mode",
        type=str,
        choices=["none", "lcc", "lcc_morph"],
        default="lcc",
        help="传播结果后处理方式：none/lcc/lcc_morph",
    )
    parser.add_argument("--morph-kernel", type=int, default=3, help="lcc_morph 模式下形态学核大小")

    parser.add_argument(
        "--sam2-config",
        type=str,
        default="configs/sam2/sam2_hiera_l.yaml",
        help="SAM2配置文件（固定Large推荐: configs/sam2/sam2_hiera_l.yaml）",
    )
    parser.add_argument("--sam2-checkpoint", type=str, default=str(default_sam2_ckpt), help="SAM2 checkpoint 路径")
    parser.add_argument("--device", type=str, default="auto", help="SAM2设备: auto / cpu / cuda / cuda:0")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
