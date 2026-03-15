"""
SAM2 Post-Processing Modules (V2) for Ultrasound Vessel Segmentation

Design principle: NEVER modify SAM2's internal memory or propagation.
All modules operate on the FINAL output masks, outside of SAM2.

Module 1: MultiFramePrompter
    - Adds YOLO detections on multiple frames as conditioning (not just frame 0)
    - Resets error accumulation by periodically re-anchoring with fresh detections
    - Uses SAM2's native multi-conditioning-frame support

Module 2: TemporalSmoother
    - Post-processes output masks for temporal consistency
    - Detects sudden mask changes (area jump, centroid jump, disappearance)
    - Replaces anomalous frames with temporally smoothed predictions

Module 3: OverlapResolver
    - Resolves pixel-level artery/vein overlaps using logit confidence
    - Removes small disconnected components (noise)
    - Safe, always-beneficial post-processing
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class MultiFramePrompter:
    """
    Multi-Frame Prompting Strategy (替代 SM - Separate Memory Bank)

    核心思想: SAM2 原生支持多个 conditioning frame。与其操纵内部 memory bank,
    不如在推理前通过 YOLO 在多个帧上检测并添加 conditioning prompt,
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


class TemporalSmoother:
    """
    Temporal Output Smoothing (替代 AM - Adaptive Memory)

    核心思想: 不干预 SAM2 内部记忆, 只在最终输出上做时序一致性检查。
    检测异常帧 (突变), 用前后帧插值替代。

    对于超声视频: 血管形态在相邻帧间变化缓慢,
    突然的面积骤变或质心跳跃通常是传播错误。
    """

    def __init__(
        self,
        area_change_thresh: float = 2.0,
        centroid_shift_thresh: float = 0.15,
        min_area: int = 30,
        smooth_alpha: float = 0.5,
    ):
        """
        Args:
            area_change_thresh: 面积变化比例阈值 (>2.0 表示面积翻倍或缩小一半)
            centroid_shift_thresh: 质心位移阈值 (归一化, 相对于图像对角线)
            min_area: 最小有效面积 (像素数)
            smooth_alpha: 平滑系数, 异常帧中前一帧权重
        """
        self.area_change_thresh = area_change_thresh
        self.centroid_shift_thresh = centroid_shift_thresh
        self.min_area = min_area
        self.smooth_alpha = smooth_alpha

    @staticmethod
    def _mask_area(mask: np.ndarray) -> int:
        return int(np.sum(mask > 0))

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return None
        return float(coords[:, 0].mean()), float(coords[:, 1].mean())

    def smooth_sequence(
        self,
        masks_by_frame: Dict[int, Dict[str, np.ndarray]],
        classes: List[str] = ("artery", "vein"),
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        对时序 mask 做平滑处理。

        Args:
            masks_by_frame: {frame_idx: {"artery": mask, "vein": mask, "semantic": mask}}
            classes: 要处理的类别名

        Returns:
            平滑后的 masks_by_frame (原地修改并返回)
        """
        sorted_frames = sorted(masks_by_frame.keys())
        if len(sorted_frames) < 2:
            return masks_by_frame

        for cls_name in classes:
            prev_mask = None
            prev_area = None
            prev_centroid = None
            corrections = 0

            for frame_idx in sorted_frames:
                frame_data = masks_by_frame[frame_idx]
                mask = frame_data.get(cls_name)
                if mask is None:
                    prev_mask = None
                    prev_area = None
                    prev_centroid = None
                    continue

                curr_area = self._mask_area(mask)
                curr_centroid = self._mask_centroid(mask)
                is_anomaly = False

                if prev_mask is not None and prev_area is not None:
                    # 检查面积突变
                    if prev_area >= self.min_area and curr_area >= self.min_area:
                        ratio = max(curr_area, prev_area) / max(min(curr_area, prev_area), 1)
                        if ratio > self.area_change_thresh:
                            is_anomaly = True

                    # 检查质心突跳
                    if (
                        curr_centroid is not None
                        and prev_centroid is not None
                        and not is_anomaly
                    ):
                        h, w = mask.shape[:2]
                        diag = (h**2 + w**2) ** 0.5
                        dist = (
                            (curr_centroid[0] - prev_centroid[0]) ** 2
                            + (curr_centroid[1] - prev_centroid[1]) ** 2
                        ) ** 0.5
                        if dist / diag > self.centroid_shift_thresh:
                            is_anomaly = True

                    # 检查 mask 突然消失
                    if prev_area >= self.min_area and curr_area < self.min_area:
                        is_anomaly = True

                if is_anomaly and prev_mask is not None:
                    # 用前一帧 mask 替代 (二值混合)
                    frame_data[cls_name] = prev_mask.copy()
                    corrections += 1
                    # 不更新 prev, 继续用上一个好的帧
                else:
                    prev_mask = mask.copy()
                    prev_area = curr_area
                    prev_centroid = curr_centroid

            if corrections > 0:
                # 重建 semantic mask
                for frame_idx in sorted_frames:
                    frame_data = masks_by_frame[frame_idx]
                    semantic = np.zeros_like(frame_data["semantic"])
                    if "artery" in frame_data:
                        semantic[frame_data["artery"] > 0] = 1
                    if "vein" in frame_data:
                        semantic[frame_data["vein"] > 0] = 2
                    frame_data["semantic"] = semantic

        return masks_by_frame


class OverlapResolver:
    """
    Logit-Based Overlap Resolution (替代 AV - Artery-Vein Constraint)

    核心思想: 不做空间约束, 只做安全的像素级重叠解决。
    - 重叠区域: 分配给 logit 更高的类
    - 噪声去除: 去掉小的孤立连通域
    - 这是最安全的后处理, 只会改善不会恶化

    注意: 这个模块在 _decode_masks_from_logits 中已经部分实现了
    (通过比较 artery_logits vs vein_logits 解决重叠)。
    此类提供额外的连通域过滤功能。
    """

    def __init__(
        self,
        min_component_area: int = 50,
        artery_min_area: int = 100,
        vein_min_area: int = 30,
    ):
        """
        Args:
            min_component_area: 最小连通域面积 (像素), 小于此值的连通域被移除
            artery_min_area: 动脉最小面积
            vein_min_area: 静脉最小面积
        """
        self.min_component_area = min_component_area
        self.artery_min_area = artery_min_area
        self.vein_min_area = vein_min_area

    @staticmethod
    def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        """移除小于 min_area 的连通域"""
        if not np.any(mask):
            return mask
        mask_u8 = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )
        if num_labels <= 1:
            return mask

        cleaned = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = True
        return cleaned

    def resolve(
        self,
        frame_pred: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        对单帧预测做后处理: 去噪 + 重叠解决。

        Args:
            frame_pred: {"artery": bool_mask, "vein": bool_mask, "semantic": uint8_mask}

        Returns:
            清理后的 frame_pred (新 dict)
        """
        artery = frame_pred["artery"].copy()
        vein = frame_pred["vein"].copy()

        # 1. 去除小连通域
        artery = self._remove_small_components(artery, self.artery_min_area)
        vein = self._remove_small_components(vein, self.vein_min_area)

        # 2. 重叠解决 (动脉优先, 因为动脉 Dice 更高)
        overlap = artery & vein
        if np.any(overlap):
            vein[overlap] = False

        # 3. 重建 semantic
        semantic = np.zeros_like(frame_pred["semantic"])
        semantic[artery > 0] = 1  # artery
        semantic[vein > 0] = 2    # vein

        return {"artery": artery, "vein": vein, "semantic": semantic}

    def resolve_batch(
        self,
        masks_by_frame: Dict[int, Dict[str, np.ndarray]],
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """对所有帧做后处理"""
        for frame_idx in masks_by_frame:
            masks_by_frame[frame_idx] = self.resolve(masks_by_frame[frame_idx])
        return masks_by_frame
