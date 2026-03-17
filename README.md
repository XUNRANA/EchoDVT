# EchoDVT — 基于超声视频的深静脉血栓自动检测

## 1. 项目背景

压缩超声（Compression Ultrasonography, CUS）是深静脉血栓（DVT）的金标准影像诊断之一。医生用超声探头轻压静脉，正常静脉会塌陷消失，而有血栓的静脉拒绝塌陷，始终可见。

本项目基于超声视频，通过 **YOLO 目标检测 + SAM2 视频分割 + 二分类算法**，实现 DVT 的端到端自动检测，并提供基于 Gradio 的可视化 Web 诊断平台。

## 2. 整体流程

```
超声视频输入
    ↓
YOLO 首帧检测 (动脉/静脉框) + 先验补全
    ↓
SAM2 + LoRA 视频传播分割 (+ 多帧提示 MFP)
    ↓
21 维时序特征提取 (VCR, VDR, VARR, MVAR ...)
    ↓
二分类判断 (DVT / 正常)
    ↓
Gradio Web 可视化展示
```

## 3. 项目结构

```
EchoDVT/
├── README.md                       # 项目总说明（本文件）
├── classify_dvt.py                 # DVT 二分类：特征提取 + ML 分类器
├── sam2/                           # SAM2 分割模块（详见 sam2/README_EchoDVT.md）
│   ├── inference_box_prompt_large.py   # Baseline SAM2 推理 + YOLO 检测器
│   ├── inference_lora.py               # LoRA 微调 SAM2 推理 + MFP
│   ├── train_lora.py                   # LoRA 训练脚本
│   ├── checkpoints/                    # 模型权重
│   │   ├── sam2_hiera_large.pt             # SAM2 Large 基础权重
│   │   └── lora_runs/                      # LoRA 微调权重
│   │       ├── lora_r8_.../lora_best.pt
│   │       └── lora_r4_.../lora_best.pt
│   ├── sam2/                           # SAM2 核心库（含新增模块）
│   │   ├── lora_sam2.py                    # [新增] LoRA 注入封装
│   │   ├── postprocess.py                  # [新增] MFP 多帧提示 + RPA 位置锚定
│   │   ├── dvt_dataset.py                  # [新增] DVT 超声视频 Dataset
│   │   ├── sam2_video_trainer.py           # [新增] 支持训练的 Predictor
│   │   └── configs/                        # 模型配置
│   └── dataset/                        # 数据集（软链接）
│       ├── train/                          # 300 例正常
│       └── val/                            # 76 例 (38 正常 + 38 患者)
├── yolo/                           # YOLO 检测模块（详见 yolo/README.md）
│   ├── compute_prior_stats.py          # 先验统计计算
│   ├── train_*_*.py                    # 5 步渐进式训练脚本
│   ├── prior_stats.json                # 动静脉位置先验统计
│   └── runs/detect/.../best.pt         # YOLO 训练权重
├── web/                            # Gradio Web 前端（详见 web/README.md）
│   ├── app.py                          # 应用入口
│   ├── services/                       # 推理服务层（单例封装）
│   ├── tabs/                           # 7 个功能 Tab
│   ├── utils/                          # 可视化 & 指标工具
│   └── assets/                         # CSS 样式
└── results/                        # 推理结果输出目录
```

## 4. 数据集

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

## 5. 创新点

### 创新点 1: YOLO 检测框生成与自动补全

如果 YOLO 只检测到动脉或静脉中的一个，利用数据集先验统计（动静脉相对位置分布）自动补齐缺失框，确保每帧都有动脉+静脉两个框作为 SAM2 的 box prompt。

### 创新点 2: SAM2 LoRA 低秩高效微调

| 微调组件 | 方法 | 参数量 |
|----------|------|--------|
| Image Encoder QKV (48 层) | LoRA 注入 (r=8) | ~0.3M |
| Memory Attention self_attn + cross_attn (32 投影) | LoRA 注入 (r=8) | ~0.15M |
| Mask Decoder | 全量微调 | ~0.05M |
| **总计** | | **~0.5M (占总参数 0.2%)** |

### 创新点 3: 多帧提示 MFP（Multi-Frame Prompting, train-free）

**不修改 SAM2 内部状态**，只在输入端添加更多 conditioning frame：

**解决的问题**：SAM2 仅用首帧 prompt，后续帧靠 memory 传播，随帧数增加误差累积，尤其静脉 mask 容易漂移/消失。

**方法**：
1. 推理前，对视频每隔 N 帧（默认 15）运行 YOLO 检测
2. 若该帧动脉和静脉检测置信度均 >= 阈值（默认 0.3），作为额外 conditioning frame
3. SAM2 原生支持多 conditioning frame，这些帧不依赖 memory 传播，而是用 YOLO 框直接 prompt
4. 等于每隔 N 帧"重新锚定"，打断误差累积链

**参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mfp-interval` | 15 | 每隔多少帧尝试添加 prompt |
| `--mfp-min-conf` | 0.3 | YOLO 最低置信度阈值 |
| `--mfp-max-prompts` | 5 | 最多添加多少个额外 prompt 帧 |

**设计原则**：

| 原则 | 说明 |
|------|------|
| 不修改 SAM2 内部 | 在 SAM2 外部运行，不动 memory feature、不动 propagation 逻辑 |
| 与 LoRA 互补 | LoRA 优化模型特征，MFP 优化输入（更多 anchor） |
| 安全无害 | 最坏情况是"不起作用"，不会让结果变差 |
| train-free | 不需要额外训练，纯推理时增强 |

**消融实验结果**（val set, frame-weighted）：

| 配置 | Dice | V_Dice | 相对 Baseline |
|------|------|--------|--------------|
| Baseline (LoRA only) | 0.7692 | 0.7029 | — |
| **+MFP** | **0.7853** | **0.7166** | **+1.6%** |

### 创新点 4: 多维特征二分类

分割完成后，从静脉面积时序中提取 **21 维手工特征**，使用统一 RF 分类器进行 DVT 判断：

| 特征 | 含义 | 正常人趋势 |
|------|------|-----------|
| **VCR** (Vein Compression Ratio) | min/max 面积比 | ≈ 0（塌陷） |
| **VDR** (Vein Disappearance Rate) | 面积 < 10% max 的帧占比 | 高（消失多） |
| **VARR** (Vein Area Relative Range) | (max-min)/max | 大（变化大） |
| **vein_cv** | 面积变异系数 std/mean | 大（波动大） |
| **MVAR** | 最小静脉/动脉面积比 | 小（被压扁） |
| vein_slope | 面积线性趋势斜率 | 负（下降） |
| vein_autocorr | 面积 lag-1 自相关 | 高（平滑压缩） |
| circ_cv / circ_min | 静脉圆度变化 | 变化大（被压扁） |
| ... | 共 21 维 | |

当前 Web 与离线统一模型默认使用 `RF unified`，固定概率阈值 `prob ≥ 0.05`。
当前元信息：`train_accuracy = 94.33%`，`val_accuracy = 94.74%`。

## 6. Web 诊断平台

基于 Gradio 6.x 的交互式诊断系统，7 个 Tab 覆盖完整分析流程：

```bash
cd web
python app.py
```

| Tab | 功能 |
|-----|------|
| 仪表盘 | 系统状态、综合评估指标（train+val+test 500/50）、误判案例分析 |
| 数据输入 | 选择 val/train，或勾选 test(normal/patient)，也支持本地视频上传 |
| YOLO 检测 | 运行 YOLO 检测，可视化动脉/静脉框 |
| SAM2 分割 | LoRA r4/r8 + 可选 MFP，逐帧分割结果 Gallery |
| DVT 诊断 | 21 维特征提取，面积曲线，DVT/正常判断 |
| 定量评估 | 逐帧 Dice/mIoU 指标，最佳/最差帧标注 |
| 一键分析 | 检测→分割→诊断全流程一键运行并生成报告 |

详见 [web/README.md](web/README.md)。

## 7. 推理命令

> 所有命令在 `sam2/` 目录下运行。

### 7.1 基线推理（YOLO 首帧 box prompt + SAM2 原始模型，无 LoRA）

```bash
cd sam2
python inference_box_prompt_large.py \
  --split val \
  --yolo-model /data1/ouyangxinglong/EchoDVT/yolo/runs/detect/runs/detect/dvt_runs/aug_step5_speckle_translate_scale/weights/best.pt \
  --sam2-config configs/sam2/sam2_hiera_l.yaml \
  --sam2-checkpoint checkpoints/sam2_hiera_large.pt
```

### 7.2 LoRA 微调推理（纯 LoRA，无 MFP）

```bash
cd sam2
CUDA_VISIBLE_DEVICES=0 python inference_lora.py \
  --lora-weights checkpoints/lora_runs/lora_r8_lr0.0003_e25_20260314_153210/lora_best.pt \
  --lora-r 8 --split val
```

### 7.3 LoRA + MFP（多帧提示）

```bash
cd sam2
CUDA_VISIBLE_DEVICES=0 python inference_lora.py \
  --lora-weights checkpoints/lora_runs/lora_r8_lr0.0003_e25_20260314_153210/lora_best.pt \
  --lora-r 8 --split val \
  --multi-frame-prompt True
```

### 7.4 自定义 MFP 参数

```bash
cd sam2
CUDA_VISIBLE_DEVICES=0 python inference_lora.py \
  --lora-weights checkpoints/lora_runs/lora_r8_lr0.0003_e25_20260314_153210/lora_best.pt \
  --lora-r 8 --split val \
  --multi-frame-prompt True --mfp-interval 10 --mfp-min-conf 0.25 --mfp-max-prompts 3
```

### 7.5 端到端 DVT 分类

```bash
# 使用预计算 mask
python classify_dvt.py --pred-dir results/lora_r8_mfp/predictions

# 端到端模式（自动推理 + 分类）
python classify_dvt.py --split val
```

## 8. 评估说明

- YOLO 在首帧（及 MFP 选择的额外帧）输出 artery/vein 两个框作为 SAM2 box prompt，缺失框自动补全
- 后续帧仅依赖 SAM2 memory 传播分割
- 仅在有 `masks/*.png` 标注的帧上计算动脉/静脉各自 Dice、mIoU
- 提供 frame-weighted 与 case-weighted 两种全局统计口径
- 输出：日志、`frame_metrics.csv`、`case_metrics.csv`、`summary.json`、每帧可视化

## 9. 环境依赖

```bash
conda activate echodvt

# 核心依赖
pip install torch torchvision          # PyTorch 2.1+ (CUDA 12.8)
pip install ultralytics                # YOLO
pip install gradio>=6.0                # Web 前端
pip install scikit-learn pandas scipy  # 分类模块
pip install matplotlib opencv-python   # 可视化
```

硬件要求：2× NVIDIA A100 40GB（推荐），单卡 A100 亦可运行。
