# SAM2 视频分割模块 — 与官方 SAM2 的区别与创新

本目录基于 Meta 官方 [SAM2 (Segment Anything Model 2)](https://github.com/facebookresearch/sam2) 二次开发，针对超声血管视频分割任务进行了多项定制创新。**核心设计原则：不修改 SAM2 内部记忆和传播机制，所有创新在外部输入端或输出端实现。**

## 与官方 SAM2 的对比总览

| 类别 | 官方 SAM2 | 本项目 EchoDVT |
|------|----------|---------------|
| 微调方式 | 全量微调（全部 224M 参数） | **LoRA 低秩微调（仅 ~0.5M，占 0.2%）** |
| 训练支持 | 仅推理 (`@inference_mode`) | **新增 SAM2VideoTrainer（移除推理装饰器，支持梯度）** |
| 输入端 | 仅首帧 prompt | **多帧提示 MFP（每隔 N 帧用 YOLO 重新锚定）** |
| 输出端 | 无后处理 | **RPA 相对位置锚定（用动脉位置约束静脉漂移）** |
| Prompt 来源 | 手动标注 | **YOLO 自动检测 + 先验补全** |
| 数据集 | 通用视频/图像 | **DVT 超声视频专用 Dataset** |
| 损失函数 | 无 | **Dice + Focal Loss 组合** |
| 应用场景 | 通用分割 | **DVT 超声诊断（动脉/静脉二类分割）** |

## 新增文件清单

以下文件为**本项目原创**，官方 SAM2 仓库中不存在：

```
sam2/
├── 新增推理脚本
│   ├── inference_box_prompt_large.py   # Baseline SAM2 推理 + VesselDetector
│   └── inference_lora.py              # LoRA 推理 + MFP/RPA 集成
│
├── 新增训练脚本
│   └── train_lora.py                  # LoRA 端到端训练流程
│
└── sam2/  (核心库目录)
    ├── lora_sam2.py           # [新增] LoRA 注入实现
    ├── postprocess.py         # [新增] MFP 多帧提示 + RPA 位置锚定
    ├── dvt_dataset.py         # [新增] DVT 超声视频 Dataset
    ├── sam2_video_trainer.py   # [新增] 支持训练的 VideoPredictor
    └── adaptive_memory.py     # [已移除] V1 AM/SM/AV（与 LoRA 冲突）
```

官方文件保持不变：`build_sam.py`、`sam2_video_predictor.py`、`sam2_image_predictor.py` 等。

---

## 创新 1: LoRA 低秩高效微调

**文件**：`sam2/lora_sam2.py`（520 行）

### 问题

SAM2 Large 共 224M 参数，在仅 300 例超声训练集上全量微调会严重过拟合，且需要大量 GPU 显存。

### 方案

LoRA (Low-Rank Adaptation)：冻结原始参数，仅在关键层注入低秩可训练矩阵。

```
原始: y = Wx
LoRA: y = Wx + (B × A)x × (α/r)
      其中 A ∈ R^{r×d}, B ∈ R^{d×r}, r << d
```

### 注入位置

| 组件 | 注入方式 | 参数量 |
|------|---------|--------|
| **Image Encoder** — 48 层 Hiera 的 QKV Attention | `_LoRA_qkv`：对 Q 和 V 注入，K 保持不变 | ~0.3M |
| **Memory Attention** — 4 层的 self_attn + cross_attn | `_LoRA_Linear`：对 q/k/v/out_proj 全部注入 | ~0.15M |
| **Mask Decoder** | 全量微调（参数量本身很小） | ~0.05M |
| **Memory Encoder** | 可选解冻（学习超声纹理特征） | 可选 |
| **总计** | | **~0.5M (占总参数 0.2%)** |

### 关键实现

```python
class _LoRA_qkv(nn.Module):
    """替换 QKV 线性层，仅对 Q 和 V 注入低秩增量"""
    def forward(self, x):
        qkv = self.qkv(x)                         # 原始输出
        new_q = self.B_q(self.A_q(x)) * scaling    # Q 增量
        new_v = self.B_v(self.A_v(x)) * scaling    # V 增量
        q = qkv[..., :dim] + new_q
        k = qkv[..., dim:2*dim]                    # K 不变
        v = qkv[..., 2*dim:] + new_v
        return torch.cat([q, k, v], dim=-1)

class LoRA_SAM2_Video(nn.Module):
    """SAM2 LoRA 封装：冻结 → 注入 → 保存/加载"""
    def save_lora_parameters(self, path)  # 仅保存 LoRA 矩阵 + Decoder
    def load_lora_parameters(self, path)  # 加载到任意 SAM2 基座
```

### 初始化策略

- A 矩阵：Kaiming 均匀初始化
- B 矩阵：零初始化（训练初期 LoRA 增量为零，不破坏预训练权重）

---

## 创新 2: 多帧提示 MFP (Multi-Frame Prompting)

**文件**：`sam2/postprocess.py` — `MultiFramePrompter` 类

### 问题

SAM2 仅在首帧接受 prompt，后续帧完全依赖 Memory 传播。随着帧数增加，误差逐帧累积，尤其静脉 mask 容易漂移或消失。

### 方案

利用 SAM2 **原生支持的多 conditioning frame 能力**，在推理前通过 YOLO 自动在多个帧上添加检测框作为额外 prompt，每隔 N 帧"重新锚定"。

```
Frame 0 ─── YOLO prompt ─── SAM2 init
Frame 1                      memory propagation
...                          memory propagation (误差累积)
Frame 15 ── YOLO prompt ─── SAM2 re-anchor (打断累积链)
...                          memory propagation
Frame 30 ── YOLO prompt ─── SAM2 re-anchor
...
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `interval` | 15 | 每隔多少帧尝试添加 prompt |
| `min_conf` | 0.3 | YOLO 置信度阈值（两类都需达到才添加） |
| `max_prompts` | 5 | 最多添加多少个额外 prompt 帧 |

### 消融结果（val set, frame-weighted）

| 配置 | Mean Dice | Vein Dice | 相对提升 |
|------|-----------|-----------|---------|
| LoRA r8 (仅首帧) | 0.7692 | 0.7029 | — |
| **LoRA r8 + MFP** | **0.7853** | **0.7166** | **+1.6%** |

### 设计原则

- **不修改 SAM2 内部**：不动 memory feature、不动 propagation 逻辑
- **与 LoRA 互补**：LoRA 优化模型特征，MFP 优化输入锚点
- **安全无害**：最坏情况是"不起作用"（YOLO 不够好则不添加 prompt）
- **train-free**：纯推理时增强，不需要额外训练

---

## 创新 3: 相对位置锚定 RPA (Relative Position Anchoring)

**文件**：`sam2/postprocess.py` — `RelativePositionAnchor` 类

### 问题

SAM2 传播过程中，静脉 mask 可能"漂移"到其他区域（如与动脉混淆），产生假阳性。

### 方案

利用动脉位置的稳定性作为"锚点"。在超声视频中，动脉不受探头压力影响，位置始终稳定；而动脉与静脉的相对空间关系也基本不变。

```
Step 1: 学习基线
  从前 N 帧中提取"好帧"（动脉和静脉都检测到）
  计算动脉→静脉的平均偏移量 (baseline_offset)

Step 2: 逐帧检查
  对每一帧：
    expected_vein_pos = artery_centroid + baseline_offset
    actual_vein_pos   = vein_mask_centroid
    drift = distance(expected, actual) / image_diagonal
    if drift > max_drift:
        suppress vein mask (set to zero)  # 判定为漂移假阳性
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_drift` | 0.15 | 允许的最大偏移（归一化到图像对角线） |
| `min_good_frames` | 3 | 学习基线需要的最少好帧数 |
| `min_area` | 100 | 忽略面积太小的 mask |

---

## 创新 4: 支持训练的 SAM2VideoTrainer

**文件**：`sam2/sam2_video_trainer.py`

### 问题

官方 SAM2VideoPredictor 的关键方法（`init_state`、`add_new_points_or_box`、`propagate_in_video`）都使用了 `@torch.inference_mode()` 装饰器，无法计算梯度，不能直接用于训练。

### 方案

继承 `SAM2VideoPredictor`，重写所有关键方法为训练兼容版本：

```python
class SAM2VideoTrainer(SAM2VideoPredictor):
    # 移除 @torch.inference_mode() 的训练版本
    def init_state_train(self, video_path, ...)
    def add_new_points_or_box_train(self, inference_state, ...)
    def propagate_in_video_train(self, inference_state, ...)
```

通过 `from_predictor()` 类方法可从现有 Predictor 创建 Trainer，共享权重无需重新加载。

---

## 创新 5: DVT 超声视频 Dataset

**文件**：`sam2/dvt_dataset.py`

### 功能

专为 SAM2 LoRA 训练设计的 PyTorch Dataset：

- 每个 case 作为一个样本（视频级别），包含 `images/` 和 `masks/`
- 首帧从 GT mask 自动提取 artery/vein 的 bounding box 作为 SAM2 prompt
- 支持 box jitter（随机扰动框坐标，模拟 YOLO 检测误差，增强鲁棒性）
- 支持 margin 扩展（框外扩一定比例，避免目标截断）
- 稀疏标注支持（只有部分帧有 mask，未标注帧不参与 loss 计算）

---

## 创新 6: LoRA 端到端训练流程

**文件**：`train_lora.py`（601 行）

### 训练策略

```
对每个 epoch:
  对每个 case (视频):
    1. 初始化 SAM2 inference_state
    2. 首帧: 用 GT box 作为 prompt → add_new_points_or_box_train
    3. 传播: propagate_in_video_train → 获取每帧预测 logits
    4. 仅在有标注帧上计算 loss (稀疏监督)
    5. 反向传播 → 更新 LoRA 参数 + Mask Decoder
```

### 损失函数

```python
loss = dice_loss(pred_logits, target) + sigmoid_focal_loss(pred_logits, target)
```

- **Dice Loss**：解决类别不平衡（血管面积远小于背景）
- **Focal Loss**：聚焦于难样本（边界像素、小目标）
- 支持动脉/静脉不同权重（默认 artery=1.0, vein=1.5，因静脉更难分割）

### 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| LoRA rank | 4~8 | 低秩维度 |
| 学习率 | 3e-4 | AdamW |
| 调度器 | CosineAnnealing | 配合 warm restarts |
| 梯度累积 | 4 steps | 增大有效 batch size |
| AMP | bfloat16 | 混合精度训练 |
| 最大帧数 | 40 | 单 case 最多处理帧数（显存限制） |

### 可用权重

| 配置 | 路径 |
|------|------|
| **LoRA r8** (推荐) | `checkpoints/lora_runs/lora_r8_lr0.0003_e25_20260314_153210/lora_best.pt` |
| LoRA r4 | `checkpoints/lora_runs/lora_r4_lr0.0005_e25_20260314_153134/lora_best.pt` |

---

## 创新 7: VesselDetector + 推理封装

**文件**：`inference_box_prompt_large.py`（1072 行）

为 SAM2 提供统一的推理入口，包含：

- `VesselDetector`：YOLO 检测封装，含先验补全、重叠修正、质量门控（详见 [yolo/README.md](../yolo/README.md)）
- `SAM2MemoryVideoSegmenter`：Baseline SAM2 推理封装，支持 AM/SM/AV 变体标志
- 完整评估流程：逐帧 Dice/mIoU → CSV 导出 → JSON 汇总 → PNG 可视化

---

## 目录结构

```
sam2/
├── README.md                          # 官方 SAM2 说明
├── README_EchoDVT.md                  # 本文件（EchoDVT 创新说明）
│
├── inference_box_prompt_large.py      # [新增] Baseline 推理 + VesselDetector
├── inference_lora.py                  # [新增] LoRA 推理 + MFP/RPA
├── train_lora.py                      # [新增] LoRA 训练脚本
│
├── checkpoints/                       # 模型权重
│   ├── sam2_hiera_large.pt               # SAM2 Large 官方权重
│   └── lora_runs/                        # LoRA 微调权重
│       ├── lora_r8_.../lora_best.pt
│       └── lora_r4_.../lora_best.pt
│
├── sam2/                              # SAM2 核心库
│   ├── lora_sam2.py                      # [新增] LoRA 注入
│   ├── postprocess.py                    # [新增] MFP + RPA
│   ├── dvt_dataset.py                    # [新增] DVT Dataset
│   ├── sam2_video_trainer.py             # [新增] 训练 Predictor
│   ├── adaptive_memory.py               # [已移除] V1 AM/SM/AV
│   ├── build_sam.py                      # [官方] 模型构建
│   ├── sam2_video_predictor.py           # [官方] 推理 Predictor
│   ├── sam2_image_predictor.py           # [官方] 图像 Predictor
│   └── configs/sam2/sam2_hiera_l.yaml    # [官方] Large 配置
│
├── dataset/                           # 数据集（软链接）
│   ├── train/                            # 300 例正常
│   └── val/                              # 76 例 (38正常 + 38患者)
│
└── results/                           # 推理输出
```

## 使用方法

### Baseline 推理

```bash
cd sam2
python inference_box_prompt_large.py --split val
```

### LoRA 推理

```bash
cd sam2
python inference_lora.py \
  --lora-weights checkpoints/lora_runs/lora_r8_lr0.0003_e25_20260314_153210/lora_best.pt \
  --lora-r 8 --split val
```

### LoRA + MFP

```bash
cd sam2
python inference_lora.py \
  --lora-weights checkpoints/lora_runs/lora_r8_lr0.0003_e25_20260314_153210/lora_best.pt \
  --lora-r 8 --split val \
  --multi-frame-prompt True --mfp-interval 15 --mfp-min-conf 0.3
```

### LoRA 训练

```bash
cd sam2
python train_lora.py --lr 3e-4 --epochs 25 --lora-r 8 --gpu 0
```
