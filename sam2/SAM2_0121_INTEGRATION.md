## Source

Integrated from `/data1/ouyangxinglong/2026/0121/sam2_0122`.

## What Was Ported

- `sam2/sam2/build_sam.py`
  - `build_sam2_video_predictor(...)` now accepts:
    - `use_adaptive_memory`
    - `use_separate_memory`
    - `memory_quality_config`
    - `use_av_constraint`
- `sam2/sam2/modeling/sam2_base.py`
  - memory features can be weighted by `quality_score` / `memory_weight`
  - optional cross-object memory is concatenated into memory attention
- `sam2/sam2/sam2_video_predictor.py`
  - adaptive memory quality scoring
  - separate artery/vein memory banks
  - vein collapse / refresh / top-k memory heuristics
  - artery-vein geometric constraints and overlap resolution
- new files:
  - `sam2/sam2/adaptive_memory.py`
  - `sam2/sam2/lora_sam2.py`
  - `sam2/sam2/sam2_video_trainer.py`

## Core Idea

The 0121 branch does not change SAM2 backbone structure. It changes how video memory is stored and reused:

- low-quality frames can be skipped or down-weighted before entering memory
- artery and vein can use separate memory banks
- vein can borrow artery memory with a small cross-object weight
- geometric rules are applied after propagation:
  - single-vessel => artery-first
  - artery-vein distance must stay reasonable
  - artery has priority when masks overlap

## Tuning Findings From 0121

- Baseline without LoRA:
  - artery: `mDice=0.8704`, `mIoU=0.7921`
  - vein: `mDice=0.6820~0.6844`, `mIoU=0.5958~0.5981`
- Final LoRA evaluation:
  - log uses `train_sam2/sam2_lora_20260125_032720/best_vein_dice_lora.pt`
  - artery: `mDice=0.8720`, `mIoU=0.7913`
  - vein: `mDice=0.6914`, `mIoU=0.6063`
- Adaptive-memory search best trial in `tuning/results/tuning_results.csv` is effectively:
  - artery threshold around `0.55`
  - vein threshold around `0.60`
  - vein `cross_obj_weight` around `0.3`
  - vein `min_weight` around `0.1`
  - vein `weight_power` around `2.0`

## Recommended Config Shape

Use this schema, not the flat `iou_weight` style from some experimental scripts:

```python
MEMORY_QUALITY_CONFIG = {
    "artery": {
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
    },
    "vein": {
        "weights": {
            "iou_pred": 0.3,
            "iou_consistency": 0.3,
            "area_stability": 0.2,
            "centroid_stability": 0.2,
        },
        "threshold": 0.60,
        "min_weight": 0.1,
        "weight_power": 2.0,
        "cross_obj_weight": 0.3,
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
```

## How To Use In EchoDVT

```python
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(
    "configs/sam2.1/sam2.1_hiera_l.yaml",
    "checkpoints/sam2.1_hiera_large.pt",
    use_adaptive_memory=True,
    use_separate_memory=True,
    memory_quality_config=MEMORY_QUALITY_CONFIG,
    use_av_constraint=True,
)
```

For training:

```python
from sam2.sam2_video_trainer import build_sam2_video_trainer
from sam2.lora_sam2 import build_lora_sam2_video
```

## Important Caveats

- `use_temporal_aggregation` exists in `sam2_video_predictor.py` but is not actually used in propagation.
- Several 0121 root-level scripts are experimental and were not copied into EchoDVT as-is.
- `train_sam2_with_predictor.py` in 0121 uses a wrong memory config schema and needs cleanup before direct reuse.
- Current EchoDVT main script `sam2/inference_box_prompt_large.py` is untouched. To enable these new features there, pass the new kwargs when building the predictor.
