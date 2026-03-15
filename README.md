# EchoDVT — 基于超声视频的深静脉血栓自动检测

## 1. 项目背景

压缩超声（Compression Ultrasonography, CUS）是深静脉血栓（DVT）的金标准影像诊断之一。医生用超声探头轻压静脉，正常静脉会塌陷消失，而有血栓的静脉拒绝塌陷，始终可见。

本项目基于超声视频，通过 **YOLO 目标检测 + SAM2 视频分割 + 二分类算法**，实现 DVT 的端到端自动检测。

## 2. 整体流程

```
YOLO 首帧检测 (动脉/静脉框)
        ↓
SAM2 + LoRA 视频传播分割
        ↓
V2 后处理 (多帧提示 / 时序平滑 / 重叠解决)
        ↓
二分类判断 (是否患病)
```

## 3. 数据集

### DVT 分割数据集（用于 SAM2）

| 划分 | 样例数 | 构成 |
|------|--------|------|
| train | 300 例 | 全部正常人 |
| val | 76 例 | 38 例正常 + 38 例患者 |

每个 case 包含：
- `images/` — 视频帧 (`00000.jpg` 起递增，每帧都有)
- `masks/` — 稀疏标注 (`00000.png` 一定有，像素值 0=背景, 1=动脉, 2=静脉)

### YOLO 检测数据集

- 包含每帧医生标注的图像 `images/` 及 `labels/`（YOLO 格式）
- train/val 划分与 DVT 数据集一致

## 4. 创新点

### 创新点 1: YOLO 检测框生成与自动补全

如果 YOLO 只检测到动脉或静脉中的一个，利用数据集先验统计（动静脉相对位置分布）自动补齐缺失框，确保每帧都有动脉+静脉两个框作为 SAM2 的 box prompt。

### 创新点 2: SAM2 LoRA 低秩高效微调

| 微调组件 | 方法 | 参数量 |
|----------|------|--------|
| Image Encoder QKV (48 层) | LoRA 注入 (r=8) | ~0.3M |
| Memory Attention self_attn + cross_attn (32 投影) | LoRA 注入 (r=8) | ~0.15M |
| Mask Decoder | 全量微调 | ~0.05M |
| **总计** | | **~0.5M (占总参数 0.2%)** |

### 创新点 3: V2 后处理增强模块（train-free）

三个模块 **不修改 SAM2 内部状态**，只在输入端（更多 prompt）或输出端（mask 后处理）操作：

| 模块 | 缩写 | 作用位置 | 功能 |
|------|------|----------|------|
| 多帧提示 | MFP | SAM2 输入端 | 每隔 N 帧用 YOLO 重新检测并作为 conditioning frame，打断误差累积 |
| 时序平滑 | TS | SAM2 输出端 | 传播完成后检测异常帧（面积骤变/质心跳跃/消失），用前一帧替代 |
| 重叠解决 | OR | SAM2 输出端 | 连通域去噪 + 动脉优先解决像素级重叠 |

### 创新点 4: 二分类算法

分割完成后，通过分析静脉在压缩过程中的形变特征，判断是否存在血栓。

## 5. V2 后处理模块详解

### 5.1 MultiFramePrompter (MFP) — 多帧提示

**解决的问题**：SAM2 仅用首帧 prompt，后续帧靠 memory 传播，随帧数增加误差累积，尤其静脉 mask 容易漂移/消失。

**方法**：
1. 推理前，对视频每隔 N 帧（默认 15）运行 YOLO 检测
2. 若该帧动脉和静脉检测置信度均 ≥ 阈值（默认 0.3），作为额外 conditioning frame
3. SAM2 原生支持多 conditioning frame，这些帧不依赖 memory 传播，而是用 YOLO 框直接 prompt
4. 等于每隔 N 帧"重新锚定"，打断误差累积链

**参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mfp-interval` | 15 | 每隔多少帧尝试添加 prompt |
| `--mfp-min-conf` | 0.3 | YOLO 最低置信度阈值 |
| `--mfp-max-prompts` | 5 | 最多添加多少个额外 prompt 帧 |

### 5.2 TemporalSmoother (TS) — 时序平滑

**解决的问题**：传播过程中某些帧出现突发错误（mask 突然消失、面积骤变、位置跳跃）。

**方法**：
1. SAM2 传播 **完成后**，按时序扫描所有输出 mask
2. 对每帧计算与前帧的变化指标：面积变化比、质心位移、mask 消失
3. 超过阈值判定为异常帧，用前一帧 mask 替代
4. 重建 semantic mask

**参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ts-area-thresh` | 2.0 | 面积变化比阈值（2.0 = 面积翻倍或减半） |
| `--ts-centroid-thresh` | 0.15 | 质心位移阈值（归一化，15% 图像对角线） |

### 5.3 OverlapResolver (OR) — 重叠解决与去噪

**解决的问题**：动脉和静脉分别预测，可能像素级重叠；预测中可能有噪声小块。

**方法**：
1. 连通域分析，移除面积 < 阈值的孤立小块（动脉 min=100px, 静脉 min=30px）
2. 重叠区域分配给动脉（动脉 Dice ~0.85 更可靠，优先级更高）
3. 重建 semantic mask

**参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--or-min-area` | 50 | 最小连通域面积（像素） |

### 5.4 设计原则

| 原则 | 说明 |
|------|------|
| 不修改 SAM2 内部 | 所有模块在 SAM2 外部运行，不动 memory feature、不动 propagation 逻辑 |
| 与 LoRA 互补 | LoRA 优化模型特征，V2 优化输入 (MFP) 和输出 (TS/OR) |
| 安全无害 | 每个模块最坏情况是"不起作用"，不会让结果变差 |
| train-free | 不需要额外训练，纯推理时后处理 |

## 6. 推理命令

> 所有命令在 `sam2/` 目录下运行。

### 6.1 基线推理（YOLO 首帧 box prompt + SAM2 原始模型，无 LoRA）

```bash
cd sam2
python inference_box_prompt_large.py \
  --split val \
  --yolo-model /data1/ouyangxinglong/EchoDVT/yolo/runs/detect/runs/detect/dvt_runs/aug_step5_speckle_translate_scale/weights/best.pt \
  --sam2-config configs/sam2/sam2_hiera_l.yaml \
  --sam2-checkpoint checkpoints/sam2_hiera_large.pt
```

### 6.2 LoRA 微调推理（纯 LoRA，无后处理）

```bash
cd sam2
CUDA_VISIBLE_DEVICES=0 python inference_lora.py \
  --lora-weights checkpoints/lora_runs/lora_r8_lr0.0003_e25_20260314_153210/lora_best.pt \
  --lora-r 8 --split val
```

### 6.3 LoRA + V2 单模块消融

**MFP only（多帧提示）**：

```bash
cd sam2
CUDA_VISIBLE_DEVICES=0 python inference_lora.py \
  --lora-weights checkpoints/lora_runs/lora_r8_lr0.0003_e25_20260314_153210/lora_best.pt \
  --lora-r 8 --split val \
  --multi-frame-prompt True
```

**TS only（时序平滑）**：

```bash
cd sam2
CUDA_VISIBLE_DEVICES=0 python inference_lora.py \
  --lora-weights checkpoints/lora_runs/lora_r8_lr0.0003_e25_20260314_153210/lora_best.pt \
  --lora-r 8 --split val \
  --temporal-smooth True
```

**OR only（重叠解决）**：

```bash
cd sam2
CUDA_VISIBLE_DEVICES=0 python inference_lora.py \
  --lora-weights checkpoints/lora_runs/lora_r8_lr0.0003_e25_20260314_153210/lora_best.pt \
  --lora-r 8 --split val \
  --overlap-resolve True
```

### 6.4 LoRA + V2 全部开启

```bash
cd sam2
CUDA_VISIBLE_DEVICES=0 python inference_lora.py \
  --lora-weights checkpoints/lora_runs/lora_r8_lr0.0003_e25_20260314_153210/lora_best.pt \
  --lora-r 8 --split val \
  --multi-frame-prompt True --temporal-smooth True --overlap-resolve True
```

### 6.5 自定义参数示例

```bash
cd sam2
CUDA_VISIBLE_DEVICES=0 python inference_lora.py \
  --lora-weights checkpoints/lora_runs/lora_r8_lr0.0003_e25_20260314_153210/lora_best.pt \
  --lora-r 8 --split val \
  --multi-frame-prompt True --mfp-interval 10 --mfp-min-conf 0.25 \
  --temporal-smooth True --ts-area-thresh 1.5 \
  --overlap-resolve True --or-min-area 80
```

## 7. 评估说明

- YOLO 在首帧（及 MFP 选择的额外帧）输出 artery/vein 两个框作为 SAM2 box prompt，缺失框自动补全
- 后续帧仅依赖 SAM2 memory 传播分割
- 仅在有 `masks/*.png` 标注的帧上计算动脉/静脉各自 Dice、mIoU
- 提供 frame-weighted 与 case-weighted 两种全局统计口径
- 输出：日志、`frame_metrics.csv`、`case_metrics.csv`、`summary.json`、每帧可视化

## 8. 项目目标

最终目标是构建一个完整的 DVT 辅助诊断系统：
- 本地可部署的 Web 应用
- 图形化界面，支持上传超声视频
- 自动完成检测、分割、诊断全流程
- 提供可视化结果展示
