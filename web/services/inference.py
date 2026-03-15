"""
EchoDVT 推理服务层 — 单例封装所有模型的惰性加载与推理 API

用于 Gradio Web 端统一调用 YOLO 检测、SAM2 LoRA 分割、DVT 特征提取。
模型仅在首次调用时加载，后续调用复用缓存实例。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAM2_DIR = PROJECT_ROOT / "sam2"

# ── 默认路径 ──
DEFAULT_SAM2_CONFIG = "configs/sam2/sam2_hiera_l.yaml"
DEFAULT_SAM2_CHECKPOINT = SAM2_DIR / "checkpoints" / "sam2_hiera_large.pt"
DEFAULT_LORA_WEIGHTS = {
    "LoRA r8": SAM2_DIR / "checkpoints" / "lora_runs"
               / "lora_r8_lr0.0003_e25_20260314_153210" / "lora_best.pt",
    "LoRA r4": SAM2_DIR / "checkpoints" / "lora_runs"
               / "lora_r4_lr0.0005_e25_20260314_153134" / "lora_best.pt",
}
DEFAULT_YOLO_MODEL = (
    PROJECT_ROOT / "yolo" / "runs" / "detect" / "runs" / "detect"
    / "dvt_runs" / "aug_step5_speckle_translate_scale" / "weights" / "best.pt"
)
DEFAULT_YOLO_PRIOR = PROJECT_ROOT / "yolo" / "prior_stats.json"


def _ensure_paths():
    """Ensure sam2 and yolo directories are on sys.path."""
    for p in [str(SAM2_DIR), str(PROJECT_ROOT)]:
        if p not in sys.path:
            sys.path.insert(0, p)


class InferenceService:
    """Singleton service wrapping YOLO detection, SAM2 LoRA segmentation, and DVT feature extraction."""

    _instance: Optional["InferenceService"] = None

    @classmethod
    def get(cls) -> "InferenceService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._detector = None
        self._segmenters: Dict[str, object] = {}  # keyed by variant name
        self._prompter = None

    # ─────────────────── lazy model getters ───────────────────

    def get_detector(self):
        """Return cached VesselDetector (YOLO)."""
        if self._detector is not None:
            return self._detector
        _ensure_paths()
        from inference_box_prompt_large import VesselDetector, resolve_yolo_device, resolve_sam2_device

        sam2_device = resolve_sam2_device("auto")
        yolo_device = resolve_yolo_device("auto", sam2_device)
        self._detector = VesselDetector(
            model_path=DEFAULT_YOLO_MODEL,
            yolo_device=yolo_device,
            prior_path=DEFAULT_YOLO_PRIOR,
        )
        return self._detector

    def get_segmenter(self, variant: str = "LoRA r8"):
        """Return cached LoRASAM2VideoSegmenter for the given variant."""
        if variant in self._segmenters:
            return self._segmenters[variant]
        _ensure_paths()
        from inference_lora import LoRASAM2VideoSegmenter
        from inference_box_prompt_large import resolve_sam2_device

        device = resolve_sam2_device("auto")

        lora_r = 8 if "r8" in variant else 4
        lora_weights = DEFAULT_LORA_WEIGHTS.get(variant)
        if lora_weights is None:
            # fallback to r8
            lora_weights = DEFAULT_LORA_WEIGHTS["LoRA r8"]
            lora_r = 8

        seg = LoRASAM2VideoSegmenter(
            model_cfg=DEFAULT_SAM2_CONFIG,
            checkpoint=DEFAULT_SAM2_CHECKPOINT,
            lora_weights=lora_weights,
            device=device,
            lora_r=lora_r,
        )
        self._segmenters[variant] = seg
        return seg

    def get_prompter(self, interval: int = 15) -> object:
        """Return cached MultiFramePrompter."""
        if self._prompter is not None:
            return self._prompter
        _ensure_paths()
        from sam2.postprocess import MultiFramePrompter

        self._prompter = MultiFramePrompter(interval=interval, min_conf=0.3, max_prompts=5)
        return self._prompter

    # ─────────────────── high-level APIs ───────────────────

    def run_detection(self, image_bgr: np.ndarray, conf: float = 0.1) -> dict:
        """Run YOLO vessel detection on a single image (BGR).

        Returns dict with keys "artery" and "vein", each containing
        {"box": [x1,y1,x2,y2], "conf": float, ...} or None.
        """
        detector = self.get_detector()
        return detector.predict(image_bgr, conf=conf)

    def run_segmentation(
        self,
        images_dir: str | Path,
        detections: dict,
        num_frames: int,
        use_mfp: bool = False,
        variant: str = "LoRA r8",
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Run SAM2 LoRA video segmentation.

        Args:
            images_dir: Directory containing frame PNGs named 00000.png, 00001.png, ...
            detections: First-frame detection dict with "artery"/"vein" keys.
            num_frames: Total number of frames.
            use_mfp: If True, use Multi-Frame Prompting.
            variant: Model variant ("LoRA r8", "LoRA r4").

        Returns:
            Dict mapping frame index to {"semantic", "artery", "vein"} mask arrays.
        """
        images_dir = Path(images_dir)
        segmenter = self.get_segmenter(variant)

        # Build first_frame_boxes dict
        first_frame_boxes = {
            "artery": detections["artery"],
            "vein": detections["vein"],
        }

        # Optionally run MFP
        extra_prompt_frames = None
        if use_mfp:
            prompter = self.get_prompter()
            detector = self.get_detector()
            image_files = sorted(images_dir.glob("*.png"), key=lambda p: int(p.stem))
            if not image_files:
                image_files = sorted(images_dir.glob("*.jpg"), key=lambda p: int(p.stem))
            extra_prompt_frames = prompter.select_prompt_frames(
                num_frames=num_frames,
                detector=detector,
                images_dir=images_dir,
                image_files=image_files,
            )

        target_indices = set(range(num_frames))
        pred_masks = segmenter.segment_video_with_first_frame_prompt(
            images_dir=images_dir,
            first_frame_boxes=first_frame_boxes,
            target_frame_indices=target_indices,
            extra_prompt_frames=extra_prompt_frames,
        )
        return pred_masks

    def run_diagnosis(self, masks_list: list) -> dict:
        """Extract full DVT features from a list of semantic mask arrays.

        Args:
            masks_list: List of 2D numpy arrays (semantic masks, 0=bg, 1=artery, 2=vein),
                        one per frame in temporal order.

        Returns:
            Dict with keys:
                "features": dict of feature_name -> value (19 features),
                "is_dvt": bool (based on optimized threshold),
                "confidence": float,
                "diagnosis": str,
                plus all individual feature values for display.
        """
        _ensure_paths()
        sys.path.insert(0, str(PROJECT_ROOT))
        from classify_dvt import extract_features

        features = extract_features(masks_list)

        # Primary classification: use VCR with optimized threshold (0.314 from classify_dvt)
        vcr = features.get("vcr", 0)
        threshold = 0.314
        is_dvt = vcr > threshold
        distance = abs(vcr - threshold)
        confidence = min(1.0, distance / 0.3)

        if is_dvt:
            diagnosis = "DVT 疑似（静脉拒绝塌陷）"
        else:
            diagnosis = "正常（静脉正常塌陷）"

        return {
            "features": features,
            "is_dvt": is_dvt,
            "confidence": confidence,
            "diagnosis": diagnosis,
            "threshold": threshold,
            "vcr": vcr,
        }
