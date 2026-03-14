# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Adaptive Memory Module for SAM2 Video Predictor

This module implements adaptive memory filtering mechanisms to improve
segmentation quality, particularly for challenging objects like veins
that are prone to error accumulation during video propagation.

Key features:
1. Frame quality scoring based on IoU consistency, area stability, and centroid shift
2. Configurable thresholds for different object types (artery vs vein)
3. Quality-weighted memory storage to reduce error accumulation
"""

import torch
from typing import Dict, Optional, Tuple


class MemoryQualityScorer:
    """
    Computes quality scores for frames to determine if they should be stored in memory.

    The quality score is computed based on:
    1. SAM's predicted IoU confidence
    2. Frame-to-frame mask IoU consistency
    3. Area change stability
    4. Centroid displacement stability

    Different object types (artery vs vein) can have different threshold configurations.
    """

    # Default configuration for different object types
    # obj_id 1 = artery (larger, more stable)
    # obj_id 2 = vein (smaller, more prone to displacement)
    DEFAULT_CONFIG = {
        "artery": {  # obj_id = 1
            "weights": {
                "iou_pred": 0.4,
                "iou_consistency": 0.3,
                "area_stability": 0.15,
                "centroid_stability": 0.15,
            },
            "threshold": 0.55,
            "min_weight": 0.2,
            "weight_power": 2.0,
            "cross_obj_weight": 0.0,
            "cross_max_time_diff": 3,
            "refresh_interval": 0,
            "refresh_weight": 0.2,
            "time_decay": 0.0,
            "topk_memory": 0,
        },
        "vein": {  # obj_id = 2
            "weights": {
                "iou_pred": 0.3,
                "iou_consistency": 0.3,
                "area_stability": 0.2,
                "centroid_stability": 0.2,
            },
            "threshold": 0.65,  # Higher threshold for vein (more strict filtering)
            "min_weight": 0.05,
            "weight_power": 2.5,
            "cross_obj_weight": 0.2,
            "cross_max_time_diff": 3,
            "refresh_interval": 8,
            "refresh_weight": 0.2,
            "collapse_quality_threshold": 0.55,
            "collapse_area_ratio_threshold": 0.2,
            "collapse_min_area": 30,
            "reopen_weight": 0.3,
            "collapse_reinit": True,
            "time_decay": 0.03,
            "topk_memory": 5,
        },
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the MemoryQualityScorer.

        Args:
            config: Optional custom configuration dictionary. If None, uses DEFAULT_CONFIG.
        """
        self.config = config if config is not None else self.DEFAULT_CONFIG

    def get_config_for_obj(self, obj_id: int) -> Dict:
        """Get configuration for a specific object ID."""
        if obj_id == 1:
            return self.config.get("artery", self.DEFAULT_CONFIG["artery"])
        elif obj_id == 2:
            return self.config.get("vein", self.DEFAULT_CONFIG["vein"])
        else:
            # Default to artery config for unknown objects
            return self.config.get("artery", self.DEFAULT_CONFIG["artery"])

    @staticmethod
    def compute_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
        """
        Compute IoU between two binary masks.

        Args:
            mask1: First mask tensor
            mask2: Second mask tensor

        Returns:
            IoU value between 0 and 1
        """
        binary1 = (mask1 > 0).float()
        binary2 = (mask2 > 0).float()

        intersection = (binary1 * binary2).sum()
        union = (binary1 + binary2).clamp(0, 1).sum()

        if union < 1e-6:
            return 1.0 if intersection < 1e-6 else 0.0

        return (intersection / (union + 1e-6)).item()

    @staticmethod
    def compute_area_stability(current_mask: torch.Tensor, prev_mask: torch.Tensor) -> Tuple[float, float]:
        """
        Compute area change stability between consecutive masks.

        Args:
            current_mask: Current frame mask
            prev_mask: Previous frame mask

        Returns:
            Tuple of (area_stability, area_ratio)
        """
        current_binary = (current_mask > 0).float()
        prev_binary = (prev_mask > 0).float()

        current_area = current_binary.sum().item()
        prev_area = prev_binary.sum().item()

        if prev_area < 1e-6:
            return 0.5, 1.0

        area_ratio = current_area / prev_area
        # Area stability: 1.0 when ratio is 1.0, decreases as ratio deviates from 1.0
        area_stability = 1.0 - min(abs(area_ratio - 1.0), 1.0)

        return area_stability, area_ratio

    @staticmethod
    def compute_centroid_shift(current_mask: torch.Tensor, prev_mask: torch.Tensor) -> Tuple[float, float]:
        """
        Compute normalized centroid displacement between consecutive masks.

        Args:
            current_mask: Current frame mask
            prev_mask: Previous frame mask

        Returns:
            Tuple of (centroid_stability, centroid_shift_normalized)
        """
        def get_centroid(mask: torch.Tensor) -> Optional[torch.Tensor]:
            binary = (mask > 0).float()
            coords = torch.nonzero(binary.squeeze(), as_tuple=False).float()
            if coords.shape[0] == 0:
                return None
            return coords.mean(dim=0)

        curr_centroid = get_centroid(current_mask)
        prev_centroid = get_centroid(prev_mask)

        if curr_centroid is None or prev_centroid is None:
            return 0.5, 0.0

        # Get image dimensions for normalization
        H, W = current_mask.shape[-2:]
        max_dist = (H**2 + W**2) ** 0.5

        # Compute Euclidean distance between centroids
        centroid_dist = ((curr_centroid - prev_centroid) ** 2).sum() ** 0.5
        centroid_shift = (centroid_dist / max_dist).item()

        # Centroid stability: amplify sensitivity to displacement (multiply by 5)
        centroid_stability = 1.0 - min(centroid_shift * 5, 1.0)

        return centroid_stability, centroid_shift

    def compute_quality_score(
        self,
        current_mask: torch.Tensor,
        prev_mask: torch.Tensor,
        iou_pred: float,
        obj_id: int = 1,
    ) -> Dict:
        """
        Compute comprehensive quality score for the current frame.

        Args:
            current_mask: Current frame's predicted mask [1, H, W] or [B, 1, H, W]
            prev_mask: Previous frame's predicted mask [1, H, W] or [B, 1, H, W]
            iou_pred: SAM's predicted IoU confidence score
            obj_id: Object ID (1=artery, 2=vein)

        Returns:
            Dictionary containing:
                - quality_score: Overall quality score [0, 1]
                - should_store: Boolean indicating if frame should be stored in memory
                - iou_consistency: Frame-to-frame IoU
                - area_ratio: Area change ratio
                - area_stability: Area stability score
                - centroid_shift: Normalized centroid displacement
                - centroid_stability: Centroid stability score
        """
        config = self.get_config_for_obj(obj_id)
        weights = config["weights"]
        threshold = config["threshold"]
        min_weight = config.get("min_weight", 0.1)
        weight_power = config.get("weight_power", 2.0)

        # Ensure masks have consistent shape
        if current_mask.dim() == 4:
            current_mask = current_mask.squeeze(0)
        if prev_mask.dim() == 4:
            prev_mask = prev_mask.squeeze(0)

        # 1. Compute IoU consistency with previous frame
        iou_consistency = self.compute_mask_iou(current_mask, prev_mask)

        # 2. Compute area stability
        area_stability, area_ratio = self.compute_area_stability(current_mask, prev_mask)

        # 3. Compute centroid shift stability
        centroid_stability, centroid_shift = self.compute_centroid_shift(current_mask, prev_mask)

        # 4. Compute weighted quality score
        quality_score = (
            weights["iou_pred"] * iou_pred +
            weights["iou_consistency"] * iou_consistency +
            weights["area_stability"] * area_stability +
            weights["centroid_stability"] * centroid_stability
        )
        memory_weight = compute_quality_weight(
            quality_score, min_weight=min_weight, weight_power=weight_power
        )

        return {
            "quality_score": quality_score,
            "should_store": quality_score >= threshold,
            "iou_consistency": iou_consistency,
            "area_ratio": area_ratio,
            "area_stability": area_stability,
            "centroid_shift": centroid_shift,
            "centroid_stability": centroid_stability,
            "memory_weight": memory_weight,
            "threshold": threshold,
        }


class SeparateMemoryBank:
    """
    Manages separate memory banks for different object types (artery and vein).

    This helps prevent feature confusion between different object types
    during video propagation.
    """

    def __init__(self):
        """Initialize separate memory banks."""
        self.memory_banks = {
            "artery": {  # obj_id = 1
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
            "vein": {  # obj_id = 2
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
        }

    def get_obj_type(self, obj_id: int) -> str:
        """Convert object ID to type string."""
        return "artery" if obj_id == 1 else "vein"

    def store_output(
        self,
        obj_id: int,
        frame_idx: int,
        output: Dict,
        is_cond: bool = False,
    ):
        """
        Store frame output in the appropriate memory bank.

        Args:
            obj_id: Object ID (1=artery, 2=vein)
            frame_idx: Frame index
            output: Output dictionary to store
            is_cond: Whether this is a conditioning frame
        """
        obj_type = self.get_obj_type(obj_id)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        self.memory_banks[obj_type][storage_key][frame_idx] = output

    def get_output(
        self,
        obj_id: int,
        frame_idx: int,
        is_cond: Optional[bool] = None,
    ) -> Optional[Dict]:
        """
        Retrieve frame output from the appropriate memory bank.

        Args:
            obj_id: Object ID (1=artery, 2=vein)
            frame_idx: Frame index
            is_cond: If specified, only look in cond or non-cond outputs.
                     If None, look in both (cond first).

        Returns:
            Output dictionary if found, None otherwise
        """
        obj_type = self.get_obj_type(obj_id)
        bank = self.memory_banks[obj_type]

        if is_cond is True:
            return bank["cond_frame_outputs"].get(frame_idx)
        elif is_cond is False:
            return bank["non_cond_frame_outputs"].get(frame_idx)
        else:
            # Look in both, conditioning frames take priority
            out = bank["cond_frame_outputs"].get(frame_idx)
            if out is None:
                out = bank["non_cond_frame_outputs"].get(frame_idx)
            return out

    def get_cross_object_memory(
        self,
        obj_id: int,
        frame_idx: int,
        max_time_diff: int = 3,
    ) -> list:
        """
        Get memory features from the other object type for cross-reference.

        This allows veins to reference artery positions for better localization.

        Args:
            obj_id: Current object ID
            frame_idx: Current frame index
            max_time_diff: Maximum time difference for cross-reference

        Returns:
            List of (frame_idx, output) tuples from the other object type
        """
        # Get the other object type
        other_type = "artery" if self.get_obj_type(obj_id) == "vein" else "vein"
        other_bank = self.memory_banks[other_type]

        cross_memories = []

        # Collect outputs from conditioning frames within time range
        for t, out in other_bank["cond_frame_outputs"].items():
            if abs(t - frame_idx) <= max_time_diff:
                cross_memories.append((t, out))

        return cross_memories

    def clear(self):
        """Clear all memory banks."""
        for obj_type in self.memory_banks:
            self.memory_banks[obj_type]["cond_frame_outputs"].clear()
            self.memory_banks[obj_type]["non_cond_frame_outputs"].clear()


def apply_quality_weight_to_memory(
    maskmem_features: torch.Tensor,
    quality_score: float,
    min_weight: float = 0.1,
    weight_power: float = 2.0,
) -> torch.Tensor:
    """
    Apply quality-based weighting to memory features.

    Lower quality frames get reduced weight to minimize their influence
    on future predictions.

    Args:
        maskmem_features: Memory features tensor
        quality_score: Quality score from MemoryQualityScorer
        min_weight: Minimum weight to apply (prevents complete zeroing)
        weight_power: Exponent for more aggressive weighting

    Returns:
        Weighted memory features
    """
    # Map quality score to weight (higher quality = higher weight)
    weight = compute_quality_weight(
        quality_score, min_weight=min_weight, weight_power=weight_power
    )

    return maskmem_features * weight


def compute_quality_weight(
    quality_score: Optional[float],
    min_weight: float = 0.1,
    weight_power: float = 2.0,
) -> float:
    """
    Compute the final memory weight based on quality score.

    Args:
        quality_score: Quality score from MemoryQualityScorer
        min_weight: Minimum weight to apply (prevents complete zeroing)
        weight_power: Exponent for more aggressive weighting

    Returns:
        Weight in [min_weight, 1.0]
    """
    if quality_score is None:
        return 1.0
    weight = max(min_weight, float(quality_score) ** weight_power)
    return min(weight, 1.0)


class ArteryVeinConstraint:
    """
    基于GT统计的动静脉空间约束模块

    核心约束（来自300个训练视频的GT分析）:
    1. 动静脉距离约束: 归一化距离 < 0.264 (99%分位)
    2. 单血管规则: 如果只检测到一个血管，一定是动脉
    3. 静脉可大幅变化: 正常人静脉面积变化可达200%+
    4. 动脉稳定性: 动脉在所有视频中变化不大
    """

    def __init__(
        self,
        # 基于GT统计的参数
        max_av_distance: float = 0.264,      # 动静脉最大归一化距离 (99%分位)
        max_artery_area_change: float = 0.3, # 动脉面积变化上限 (动脉稳定)
        max_vein_area_change: float = 2.5,   # 静脉面积变化上限 (可大幅变化)
        max_centroid_shift: float = 0.15,    # 单帧质心位移上限
        min_mask_area: int = 50,             # 最小有效面积
    ):
        self.max_av_distance = max_av_distance
        self.max_artery_area_change = max_artery_area_change
        self.max_vein_area_change = max_vein_area_change
        self.max_centroid_shift = max_centroid_shift
        self.min_mask_area = min_mask_area

    def compute_centroid(self, mask: torch.Tensor) -> Optional[torch.Tensor]:
        """计算mask质心"""
        binary = (mask > 0).float().squeeze()
        coords = torch.nonzero(binary, as_tuple=False).float()
        if coords.shape[0] == 0:
            return None
        return coords.mean(dim=0)

    def compute_area(self, mask: torch.Tensor) -> int:
        """计算mask面积"""
        return int((mask > 0).float().sum().item())

    def compute_av_distance(
        self,
        artery_mask: torch.Tensor,
        vein_mask: torch.Tensor,
    ) -> Optional[float]:
        """
        计算动静脉之间的归一化距离

        Returns:
            归一化距离 (相对于图像对角线)，如果无法计算返回None
        """
        artery_centroid = self.compute_centroid(artery_mask)
        vein_centroid = self.compute_centroid(vein_mask)

        if artery_centroid is None or vein_centroid is None:
            return None

        # 计算欧氏距离
        dist = ((artery_centroid - vein_centroid) ** 2).sum().sqrt().item()

        # 归一化
        H, W = artery_mask.shape[-2:]
        diag = (H**2 + W**2) ** 0.5

        return dist / diag

    def check_av_proximity(
        self,
        artery_mask: torch.Tensor,
        vein_mask: torch.Tensor,
    ) -> Dict:
        """
        检查动静脉是否满足距离约束

        Returns:
            Dict with:
                - valid: bool, 是否满足约束
                - av_distance: float, 归一化距离
                - violation: str or None
        """
        result = {
            "valid": True,
            "av_distance": None,
            "violation": None,
        }

        av_dist = self.compute_av_distance(artery_mask, vein_mask)
        result["av_distance"] = av_dist

        if av_dist is not None and av_dist > self.max_av_distance:
            result["valid"] = False
            result["violation"] = f"av_distance_too_large ({av_dist:.3f} > {self.max_av_distance})"

        return result

    def check_temporal_consistency(
        self,
        current_mask: torch.Tensor,
        prev_mask: torch.Tensor,
        obj_type: str = "vein",  # "artery" or "vein"
    ) -> Dict:
        """
        检查时序一致性（基于对象类型使用不同阈值）

        Args:
            current_mask: 当前帧mask
            prev_mask: 上一帧mask
            obj_type: "artery" 或 "vein"

        Returns:
            Dict with constraint check results
        """
        result = {
            "valid": True,
            "area_change": 0.0,
            "centroid_shift": 0.0,
            "violations": [],
        }

        current_area = self.compute_area(current_mask)
        prev_area = self.compute_area(prev_mask)

        # 检查最小面积
        if current_area < self.min_mask_area:
            # 对于静脉，面积很小可能是被压缩了，不一定是错误
            if obj_type == "artery":
                result["valid"] = False
                result["violations"].append("artery_area_too_small")
            # vein可以面积很小（被压缩时）

        # 检查面积变化
        if prev_area > self.min_mask_area:
            area_change = abs(current_area - prev_area) / prev_area
            result["area_change"] = area_change

            # 根据对象类型使用不同阈值
            max_change = self.max_artery_area_change if obj_type == "artery" else self.max_vein_area_change

            if area_change > max_change:
                result["violations"].append(f"{obj_type}_area_change_too_large")

        # 检查质心位移
        curr_centroid = self.compute_centroid(current_mask)
        prev_centroid = self.compute_centroid(prev_mask)

        if curr_centroid is not None and prev_centroid is not None:
            H, W = current_mask.shape[-2:]
            max_dist = (H**2 + W**2) ** 0.5
            shift = ((curr_centroid - prev_centroid) ** 2).sum().sqrt().item()
            centroid_shift = shift / max_dist
            result["centroid_shift"] = centroid_shift

            if centroid_shift > self.max_centroid_shift:
                result["violations"].append(f"{obj_type}_centroid_shift_too_large")

        result["valid"] = len(result["violations"]) == 0
        return result

    def correct_vein_using_artery(
        self,
        vein_mask: torch.Tensor,
        artery_mask: torch.Tensor,
        prev_vein_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        使用动脉位置来修正静脉预测

        核心逻辑：
        1. 如果静脉离动脉太远，将静脉拉向动脉附近
        2. 如果静脉消失但动脉存在，使用上一帧静脉位置

        Args:
            vein_mask: 当前预测的静脉mask
            artery_mask: 当前预测的动脉mask
            prev_vein_mask: 上一帧的静脉mask (可选)

        Returns:
            修正后的静脉mask
        """
        vein_area = self.compute_area(vein_mask)
        artery_area = self.compute_area(artery_mask)

        # 情况1：静脉消失但动脉存在 → 使用上一帧静脉
        if vein_area < self.min_mask_area and artery_area >= self.min_mask_area:
            if prev_vein_mask is not None:
                return prev_vein_mask.clone()

        # 情况2：检查动静脉距离
        av_check = self.check_av_proximity(artery_mask, vein_mask)

        if not av_check["valid"] and prev_vein_mask is not None:
            # 静脉离动脉太远，可能是误检
            # 混合使用当前预测和上一帧
            alpha = 0.7  # 更多依赖上一帧
            return alpha * prev_vein_mask + (1 - alpha) * vein_mask

        return vein_mask

    def apply_single_vessel_rule(
        self,
        artery_mask: torch.Tensor,
        vein_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        单血管规则：如果只检测到一个血管，一定是动脉

        Returns:
            (corrected_artery, corrected_vein, was_corrected)
        """
        artery_area = self.compute_area(artery_mask)
        vein_area = self.compute_area(vein_mask)

        # 如果两个都有效，不需要修正
        if artery_area >= self.min_mask_area and vein_area >= self.min_mask_area:
            return artery_mask, vein_mask, False

        # 如果只有静脉（没有动脉），这是不太可能的情况
        # 动脉更稳定，应该总是存在
        if artery_area < self.min_mask_area and vein_area >= self.min_mask_area:
            # 把"静脉"当作动脉，静脉设为空
            return vein_mask.clone(), torch.zeros_like(vein_mask), True

        return artery_mask, vein_mask, False
    
    def resolve_overlap(
        self,
        artery_mask: torch.Tensor,
        vein_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解决动静脉重叠问题 (Pixel-wise Mutual Exclusion)
        策略：动脉优先 (Artery Priority) - 因为动脉预测通常更准 (Dice 0.90+)
        """
        # 确保是二值 mask
        artery_bin = (artery_mask > 0).float()
        vein_bin = (vein_mask > 0).float()
        
        # 计算重叠
        overlap = (artery_bin * vein_bin) > 0
        
        if overlap.any():
            # 从静脉中移除重叠部分
            vein_mask = vein_mask * (~overlap).float()
            
        return artery_mask, vein_mask
    

class TemporalMemoryAggregator:
    """
    时序记忆聚合器 - 专门为静脉设计

    核心思想：
    1. 维护一个"可靠帧"列表
    2. 在传播时始终参考这些可靠帧
    3. 对不可靠的预测使用时序平滑
    """

    def __init__(
        self,
        max_reliable_frames: int = 5,
        reliability_threshold: float = 0.7,
        temporal_smoothing: float = 0.3,
    ):
        self.max_reliable_frames = max_reliable_frames
        self.reliability_threshold = reliability_threshold
        self.temporal_smoothing = temporal_smoothing

        # 存储可靠帧
        self.reliable_frames = {}  # {obj_id: [(frame_idx, mask, quality_score), ...]}

    def add_frame(
        self,
        obj_id: int,
        frame_idx: int,
        mask: torch.Tensor,
        quality_score: float,
    ):
        """添加一帧到记忆中"""
        if obj_id not in self.reliable_frames:
            self.reliable_frames[obj_id] = []

        frames = self.reliable_frames[obj_id]

        # 如果质量足够高，加入可靠帧列表
        if quality_score >= self.reliability_threshold:
            frames.append((frame_idx, mask.clone(), quality_score))

            # 保持最多max_reliable_frames个可靠帧
            if len(frames) > self.max_reliable_frames:
                # 移除质量最低的
                frames.sort(key=lambda x: x[2], reverse=True)
                self.reliable_frames[obj_id] = frames[:self.max_reliable_frames]

    def get_temporal_prior(
        self,
        obj_id: int,
        current_frame_idx: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        获取基于可靠帧的时序先验

        Returns:
            加权平均的mask作为先验，或None
        """
        if obj_id not in self.reliable_frames:
            return None

        frames = self.reliable_frames[obj_id]
        if len(frames) == 0:
            return None

        # 根据时间距离和质量加权
        weighted_masks = []
        total_weight = 0.0

        for frame_idx, mask, quality in frames:
            # 时间距离权重（越近权重越高）
            time_dist = abs(current_frame_idx - frame_idx)
            time_weight = 1.0 / (1.0 + time_dist * 0.1)

            # 综合权重
            weight = quality * time_weight

            weighted_masks.append(mask.to(device) * weight)
            total_weight += weight

        if total_weight < 1e-6:
            return None

        # 加权平均
        prior = sum(weighted_masks) / total_weight
        return prior

    def apply_temporal_smoothing(
        self,
        current_mask: torch.Tensor,
        temporal_prior: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        应用时序平滑

        将当前预测与时序先验混合
        """
        if temporal_prior is None:
            return current_mask

        alpha = self.temporal_smoothing
        smoothed = (1 - alpha) * current_mask + alpha * temporal_prior
        return smoothed

    def clear(self, obj_id: Optional[int] = None):
        """清除记忆"""
        if obj_id is not None:
            self.reliable_frames.pop(obj_id, None)
        else:
            self.reliable_frames.clear()
