"""
SAM2 Post-Processing Modules for Ultrasound Vessel Segmentation

Design principle: NEVER modify SAM2's internal memory or propagation.
Modules operate on the INPUT side or OUTPUT side, outside of SAM2.

MultiFramePrompter (input side):
    - Adds YOLO detections on multiple frames as conditioning

RelativePositionAnchor (output side):
    - Uses artery position to suppress drifted vein masks
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class MultiFramePrompter:
    """
    Multi-Frame Prompting Strategy

    核心思想: SAM2 原生支持多个 conditioning frame。
    在推理前通过 YOLO 在多个帧上检测并添加 conditioning prompt,
    让 SAM2 自己利用这些 anchor 来减少误差累积。

    使用方法:
        prompter = MultiFramePrompter(interval=15, min_conf=0.3)
        prompt_frames = prompter.select_prompt_frames(
            num_frames=50,
            detector=yolo_detector,
            images_dir=images_dir,
            image_files=image_files,
        )
        # prompt_frames 是一个 dict: {frame_idx: {"artery": box_info, "vein": box_info}}
        # 在 SAM2 init_state 后, 对每个 prompt frame 调用 add_new_points_or_box
    """

    def __init__(
        self,
        interval: int = 15,
        min_conf: float = 0.3,
        max_prompts: int = 5,
    ):
        """
        Args:
            interval: 每隔多少帧尝试添加一个新的 conditioning frame
            min_conf: YOLO 检测置信度阈值 (低于此值不添加)
            max_prompts: 最多添加多少个额外 conditioning frame (不含首帧)
        """
        self.interval = interval
        self.min_conf = min_conf
        self.max_prompts = max_prompts

    def select_prompt_frames(
        self,
        num_frames: int,
        detector,
        images_dir,
        image_files: List,
    ) -> Dict[int, Dict]:
        """
        选择额外的 prompt frame 并运行 YOLO 检测。

        Args:
            num_frames: 总帧数
            detector: VesselDetector 实例
            images_dir: 图像目录
            image_files: 排序后的图像文件列表

        Returns:
            Dict[frame_idx, {"artery": box_info, "vein": box_info}]
            只包含满足置信度要求的帧
        """
        # 候选帧: 每隔 interval 取一帧, 跳过首帧 (首帧已经是 conditioning)
        candidate_indices = list(range(self.interval, num_frames, self.interval))
        if len(candidate_indices) > self.max_prompts:
            # 均匀采样
            step = len(candidate_indices) / self.max_prompts
            candidate_indices = [
                candidate_indices[int(i * step)]
                for i in range(self.max_prompts)
            ]

        prompt_frames = {}
        for frame_idx in candidate_indices:
            if frame_idx >= len(image_files):
                continue
            image = cv2.imread(str(image_files[frame_idx]), cv2.IMREAD_COLOR)
            if image is None:
                continue

            boxes = detector.predict(image, conf=0.1)

            # 检查两个类别的置信度
            artery_conf = boxes.get("artery", {}).get("conf", 0) if boxes.get("artery") else 0
            vein_conf = boxes.get("vein", {}).get("conf", 0) if boxes.get("vein") else 0

            # 两个类别都必须超过阈值才有用
            if artery_conf >= self.min_conf and vein_conf >= self.min_conf:
                prompt_frames[frame_idx] = boxes

        return prompt_frames


class RelativePositionAnchor:
    """
    Relative Position Anchoring (RPA) — 动静脉相对位置锚定

    核心思想: 在整个超声视频中，动脉和静脉的相对空间位置基本不变。
    动脉始终可见且位置稳定，可以作为"锚点"。
    当静脉 mask 的质心偏离预期位置太远时，说明发生了漂移（假静脉），应当抑制。

    使用方法:
        rpa = RelativePositionAnchor(max_drift=0.15)
        masks_by_frame = rpa.anchor(masks_by_frame)
    """

    def __init__(
        self,
        max_drift: float = 0.15,
        min_good_frames: int = 3,
        min_area: int = 100,
    ):
        """
        Args:
            max_drift: 允许的最大偏移（归一化到图像对角线）
            min_good_frames: 建立 baseline offset 所需的最少好帧数
            min_area: 判定目标"可见"的最小 mask 面积（像素）
        """
        self.max_drift = max_drift
        self.min_good_frames = min_good_frames
        self.min_area = min_area

    @staticmethod
    def _centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return None
        return float(coords[:, 0].mean()), float(coords[:, 1].mean())

    def anchor(
        self,
        masks_by_frame: Dict[int, Dict[str, np.ndarray]],
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        对输出 mask 做相对位置锚定，抑制漂移的静脉 mask。

        Args:
            masks_by_frame: {frame_idx: {"artery": mask, "vein": mask, "semantic": mask}}

        Returns:
            修正后的 masks_by_frame
        """
        sorted_frames = sorted(masks_by_frame.keys())
        if len(sorted_frames) < 2:
            return masks_by_frame

        # Step 1: 从前面的好帧中学习 artery→vein 偏移量
        offsets_y, offsets_x = [], []
        for frame_idx in sorted_frames:
            fd = masks_by_frame[frame_idx]
            a_mask, v_mask = fd.get("artery"), fd.get("vein")
            if a_mask is None or v_mask is None:
                continue
            if np.sum(a_mask > 0) < self.min_area or np.sum(v_mask > 0) < self.min_area:
                continue
            a_c = self._centroid(a_mask)
            v_c = self._centroid(v_mask)
            if a_c is None or v_c is None:
                continue
            offsets_y.append(v_c[0] - a_c[0])
            offsets_x.append(v_c[1] - a_c[1])
            if len(offsets_y) >= self.min_good_frames:
                break

        if len(offsets_y) < 1:
            return masks_by_frame  # 无法建立 baseline

        # 鲁棒偏移量: 取中位数
        med_dy = float(np.median(offsets_y))
        med_dx = float(np.median(offsets_x))

        # Step 2: 检查每帧的静脉位置是否一致
        h, w = next(iter(masks_by_frame.values()))["semantic"].shape[:2]
        diag = (h ** 2 + w ** 2) ** 0.5
        corrections = 0

        for frame_idx in sorted_frames:
            fd = masks_by_frame[frame_idx]
            a_mask, v_mask = fd.get("artery"), fd.get("vein")
            if a_mask is None or v_mask is None:
                continue
            a_area = int(np.sum(a_mask > 0))
            v_area = int(np.sum(v_mask > 0))
            if a_area < self.min_area or v_area < self.min_area:
                continue

            a_c = self._centroid(a_mask)
            v_c = self._centroid(v_mask)
            if a_c is None or v_c is None:
                continue

            # 预期静脉位置
            expected_y = a_c[0] + med_dy
            expected_x = a_c[1] + med_dx

            # 实际偏差
            drift = ((v_c[0] - expected_y) ** 2 + (v_c[1] - expected_x) ** 2) ** 0.5

            if drift / diag > self.max_drift:
                # 静脉漂移过大 → 抑制
                fd["vein"] = np.zeros_like(v_mask)
                semantic = fd["semantic"].copy()
                semantic[semantic == 2] = 0
                fd["semantic"] = semantic
                corrections += 1

        return masks_by_frame
