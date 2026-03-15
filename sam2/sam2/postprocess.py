"""
SAM2 Post-Processing Module (V2) for Ultrasound Vessel Segmentation

Design principle: NEVER modify SAM2's internal memory or propagation.
This module operates on the INPUT side, outside of SAM2.

MultiFramePrompter:
    - Adds YOLO detections on multiple frames as conditioning (not just frame 0)
    - Resets error accumulation by periodically re-anchoring with fresh detections
    - Uses SAM2's native multi-conditioning-frame support
"""

from __future__ import annotations

from typing import Dict, List

import cv2


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
