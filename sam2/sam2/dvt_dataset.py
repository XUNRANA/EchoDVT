"""
DVT 超声视频数据集 - 用于 SAM2 LoRA 微调

每个 case 包含:
  images/  -> 00000.jpg, 00001.jpg, ...  (所有帧)
  masks/   -> 00000.png, 00003.png, ...  (稀疏标注, 像素 0=背景, 1=动脉, 2=静脉)

训练策略:
  - 每个 case 视为一段视频
  - 首帧必须有 mask (用于生成 box prompt)
  - 从 mask 中提取动脉/静脉 bounding box 作为 SAM2 prompt
  - 后续有标注的帧用于计算分割损失
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# 类别定义 (与 inference_box_prompt_large.py 一致)
SEGMENTATION_CLASS_VALUES = {"artery": 1, "vein": 2}
CLASS_TO_OBJECT_ID = {"artery": 1, "vein": 2}


def get_bbox_from_mask(mask: np.ndarray, cls_value: int, margin: float = 0.0) -> Optional[List[float]]:
    """从 mask 中提取某类别的 bounding box (xyxy 格式)"""
    ys, xs = np.where(mask == cls_value)
    if len(xs) == 0:
        return None
    x1, y1 = float(xs.min()), float(ys.min())
    x2, y2 = float(xs.max()), float(ys.max())
    if margin > 0:
        h, w = mask.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        dx, dy = bw * margin, bh * margin
        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(w - 1, x2 + dx)
        y2 = min(h - 1, y2 + dy)
    return [x1, y1, x2, y2]


class DVTVideoDataset(Dataset):
    """
    DVT 超声视频数据集

    返回一个 case 的所有信息:
      - case_name: str
      - images_dir: Path (帧图像目录, SAM2 直接用)
      - prompt_boxes: Dict[str, List[float]]  (首帧 artery/vein box)
      - gt_masks: Dict[int, np.ndarray]  (有标注帧的 GT mask)
      - annotated_frame_indices: List[int]  (有标注的帧索引)
      - num_frames: int
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        max_frames: int = 0,
        box_margin: float = 0.05,
        augment: bool = False,
        box_jitter: float = 0.0,
    ):
        """
        Args:
            data_root: 数据集根目录 (包含 train/ val/)
            split: "train" 或 "val"
            max_frames: 每个 case 最多加载多少帧 (0 = 全部)
            box_margin: box prompt 外扩比例
            augment: 是否做数据增强 (训练时)
            box_jitter: box prompt 抖动比例 (训练时模拟 YOLO 误差)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.split_dir = self.data_root / split
        self.max_frames = max_frames
        self.box_margin = box_margin
        self.augment = augment
        self.box_jitter = box_jitter

        # 收集所有有效 case (首帧必须有 mask 且包含动脉+静脉)
        self.cases: List[Dict] = []
        self._collect_cases()

    def _collect_cases(self):
        if not self.split_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.split_dir}")

        for case_dir in sorted(self.split_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            images_dir = case_dir / "images"
            masks_dir = case_dir / "masks"
            if not images_dir.exists() or not masks_dir.exists():
                continue

            # 获取所有帧 (按文件名排序)
            image_files = sorted(images_dir.glob("*.jpg"), key=lambda p: int(p.stem))
            if not image_files:
                continue
            frame_stems = [p.stem for p in image_files]
            stem_to_idx = {stem: idx for idx, stem in enumerate(frame_stems)}

            # 获取所有标注帧
            mask_files = sorted(masks_dir.glob("*.png"), key=lambda p: int(p.stem))
            if not mask_files:
                continue

            # 检查首帧 (00000) 是否有 mask
            first_mask_path = masks_dir / "00000.png"
            if not first_mask_path.exists():
                # 尝试用第一个有 mask 的帧
                first_mask_path = mask_files[0]

            first_mask = cv2.imread(str(first_mask_path), cv2.IMREAD_GRAYSCALE)
            if first_mask is None:
                continue

            # 首帧必须同时包含动脉和静脉
            artery_box = get_bbox_from_mask(first_mask, SEGMENTATION_CLASS_VALUES["artery"])
            vein_box = get_bbox_from_mask(first_mask, SEGMENTATION_CLASS_VALUES["vein"])
            if artery_box is None or vein_box is None:
                continue

            # 收集标注帧索引
            annotated = {}
            for mf in mask_files:
                idx = stem_to_idx.get(mf.stem)
                if idx is not None:
                    annotated[idx] = mf

            self.cases.append({
                "case_name": case_dir.name,
                "images_dir": images_dir,
                "masks_dir": masks_dir,
                "num_frames": len(image_files),
                "frame_stems": frame_stems,
                "annotated_frames": annotated,
                "first_mask_stem": first_mask_path.stem,
            })

    def __len__(self):
        return len(self.cases)

    def _jitter_box(self, box: List[float], h: int, w: int) -> List[float]:
        """对 box 添加随机抖动 (模拟 YOLO 检测框误差)"""
        if self.box_jitter <= 0:
            return box
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        jx = bw * self.box_jitter
        jy = bh * self.box_jitter
        x1 += random.uniform(-jx, jx)
        y1 += random.uniform(-jy, jy)
        x2 += random.uniform(-jx, jx)
        y2 += random.uniform(-jy, jy)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w - 1))
        y2 = max(y1 + 1, min(y2, h - 1))
        return [x1, y1, x2, y2]

    def __getitem__(self, idx: int) -> Dict:
        case = self.cases[idx]
        images_dir = case["images_dir"]
        masks_dir = case["masks_dir"]
        annotated = case["annotated_frames"]
        frame_stems = case["frame_stems"]

        # 加载首帧 mask 并提取 box prompt
        first_mask_path = masks_dir / f"{case['first_mask_stem']}.png"
        first_mask = cv2.imread(str(first_mask_path), cv2.IMREAD_GRAYSCALE)
        h, w = first_mask.shape[:2]

        artery_box = get_bbox_from_mask(first_mask, SEGMENTATION_CLASS_VALUES["artery"], self.box_margin)
        vein_box = get_bbox_from_mask(first_mask, SEGMENTATION_CLASS_VALUES["vein"], self.box_margin)

        if self.augment:
            artery_box = self._jitter_box(artery_box, h, w)
            vein_box = self._jitter_box(vein_box, h, w)

        # 加载所有有标注帧的 GT mask
        gt_masks = {}
        for frame_idx, mask_path in annotated.items():
            gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                gt_masks[frame_idx] = torch.from_numpy(gt.astype(np.int64))

        # 限制帧数
        num_frames = case["num_frames"]
        if self.max_frames > 0 and num_frames > self.max_frames:
            num_frames = self.max_frames
            # 只保留在范围内的标注帧
            gt_masks = {k: v for k, v in gt_masks.items() if k < num_frames}

        # 首帧 mask 索引
        prompt_frame_idx = int(case["first_mask_stem"])

        return {
            "case_name": case["case_name"],
            "images_dir": str(images_dir),
            "num_frames": num_frames,
            "prompt_frame_idx": prompt_frame_idx,
            "artery_box": np.array(artery_box, dtype=np.float32),
            "vein_box": np.array(vein_box, dtype=np.float32),
            "gt_masks": gt_masks,
            "annotated_frame_indices": sorted(gt_masks.keys()),
        }
