# YOLO 血管检测模块

基于 YOLOv8 的超声图像动脉/静脉目标检测，为 SAM2 分割提供首帧 box prompt。

## 当前主线配置

当前 Web 与主线推理固定使用以下 YOLO 配置：

| 项目 | 当前值 |
|------|--------|
| 默认权重 | `runs/detect/runs/detect/dvt_runs/aug_step5_speckle_translate_scale/weights/best.pt` |
| 默认先验 | `prior_stats.json` |
| 首帧检测阈值 | `conf = 0.1` |
| 兜底重试阈值 | `conf = 0.01` |
| 用途 | 为 SAM2 提供首帧 artery / vein box prompt |

> 说明：训练脚本中 `project='runs/detect/dvt_runs'`，当前实际产物落盘路径包含一层重复的 `runs/detect`。README 统一以当前仓库内真实路径为准。

## 当前指标口径

这里有两类指标，不能混写成同一种“准确率”：

- `训练日志指标`：YOLO 标准检测指标，来自 `results.csv`
- `首帧病例成功率`：对每个病例首帧 `00000.jpg` 计算 artery / vein 的 `IoU >= 0.5` 是否成功，包含先验补全和重叠修正后的最终输出

### 1. 训练日志指标

当前最优 run 为 `aug_step5_speckle_translate_scale`。

基于 `runs/detect/runs/detect/dvt_runs/aug_step5_speckle_translate_scale/results.csv`：

| 口径 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| 最终 epoch 50 | `86.1%` | `81.6%` | `85.8%` | `56.0%` |
| 最佳 mAP50 epoch 29 | `82.9%` | `83.1%` | `86.2%` | `54.4%` |

### 2. 首帧病例成功率

基于当前 `inference.py` 配置、`conf=0.1`、先验补全开启，对每个病例首帧做评估：

| 划分 | 病例数 | Artery 成功率 | Vein 成功率 | 两类同时成功率 |
|------|--------|---------------|-------------|----------------|
| train | `300` | `100.0%` | `100.0%` | `100.0%` |
| val | `76` | `90.8%` | `90.8%` | `85.5%` |

对应的首帧平均 IoU 为：

| 划分 | Artery mIoU | Vein mIoU |
|------|-------------|-----------|
| train | `0.9303` | `0.9330` |
| val | `0.7788` | `0.7759` |

在 `val` 首帧上，先验和后处理的触发比例为：

- 动脉由先验补全：`3.9%`
- 静脉由先验补全：`2.6%`
- 重叠修正触发：`6.6%`

这组“首帧病例成功率”才更接近 Web 与 SAM2 实际收到的 prompt 质量。

## 核心创新

### 创新 1: 基于先验统计的检测框自动补全

**问题**：超声图像中动脉和静脉对比度低、形态相似，YOLO 经常漏检其中一类（尤其是静脉），导致 SAM2 缺少 prompt 无法分割。

**方案**：从训练集标签中学习动静脉的**绝对位置分布**和**相对位置关系**，当 YOLO 漏检时自动推断补全。

#### 先验类型

| 先验 | 来源 | 用途 |
|------|------|------|
| **绝对先验** (class_absolute) | 各类别的 cx/cy/w/h 分布 | 两类都漏检时，用均值位置生成框 |
| **相对先验** (artery2vein) | 已知动脉时，静脉的偏移 + 缩放比 | 仅漏检静脉时，从动脉推断 |
| **相对先验** (vein2artery) | 已知静脉时，动脉的偏移 + 缩放比 | 仅漏检动脉时，从静脉推断 |

#### 推断逻辑（VesselDetector）

```
YOLO 检测 (conf=0.1)
    │
    ├── 两类都检到 → 检查重叠 → 若 IoU > 0.3 → 用先验修正低置信度的一方
    │
    ├── 只检到动脉 → artery2vein 相对先验推断静脉
    │                  v_cx = a_cx + offset_cx
    │                  v_cy = a_cy + offset_cy
    │                  v_w  = a_w  × w_ratio
    │                  v_h  = a_h  × h_ratio
    │
    ├── 只检到静脉 → vein2artery 相对先验推断动脉（逆向）
    │
    └── 都没检到 → 降低阈值重试 (conf=0.01)
                    └── 仍失败 → class_absolute 绝对先验生成两个框
```

#### 先验统计数据

来自 `compute_prior_stats.py`，基于训练集 1,423 个动静脉共现样本：

```
动脉 → 推静脉:
  cx_offset = +0.021 (静脉偏右 ~2%)
  cy_offset = +0.168 (静脉偏下 ~17%)
  w_ratio   = 1.117  (静脉宽 ~12%)
  h_ratio   = 0.845  (静脉矮 ~15%)
```

即使 `prior_stats.json` 缺失，代码内置硬编码默认值保证可用。

---

### 创新 2: 面向医学超声的渐进式数据增强策略

**问题**：YOLO 默认增强策略（Mosaic、MixUp、色彩抖动、翻转）针对自然图像设计，直接用于超声图像会**破坏诊断信息**。

**方案**：设计 5 步渐进式增强消融实验，逐步验证每种增强的效果。

#### 增强策略设计原则

| 原则 | 具体做法 | 原因 |
|------|---------|------|
| **禁用色彩增强** | hsv_h/s/v 全设 0 | 超声依赖灰度对比，色彩变换破坏诊断信息 |
| **禁用翻转** | fliplr=0, flipud=0 | 翻转破坏解剖方向（动脉在上、静脉在下） |
| **禁用 Mosaic/MixUp** | mosaic=0, mixup=0 | 拼接产生非真实图像，不符合医学场景 |
| **保守几何增强** | scale=0.1（非默认 0.5） | 过大缩放会让小目标消失 |
| **模拟探头位移** | translate=0.05~0.1 | 对应真实超声中探头位置微调 |

#### 5 步消融实验

| 步骤 | 文件 | 增强配置 | 说明 |
|------|------|---------|------|
| **Step 1** | `train_1_baseline.py` | 无增强 | 纯净基线 |
| **Step 2** | `train_2_translate.py` | translate=0.05 | 模拟探头微移 (±32px) |
| **Step 3** | `train_3_translate_scale.py` | translate=0.05, scale=0.1 | 加入保守缩放 |
| **Step 4** | `train_4_translate0.1_scale0.1.py` | translate=0.1, scale=0.1 | 增大位移范围 |
| **Step 5** | `train_5_speckle_translate_scale.py` | translate=0.05, scale=0.1 + 斑点噪声 | 最优配置 |

---

### 创新 3: 课程式超声斑点噪声注入

**问题**：超声图像的核心噪声是**斑点噪声 (Speckle Noise)**，由超声波相干散射产生，属于乘性噪声。YOLO 内置不支持此类增强。

**方案**：自定义 `SpeckleTrainer`，继承 Ultralytics 的 `DetectionTrainer`，在 `preprocess_batch` 阶段注入物理真实的斑点噪声。

#### 噪声公式

```
I_out = I_in × (1 + N),    N ~ Normal(0, intensity)
```

这符合超声斑点的真实物理成因（相干散射的乘性干扰）。

#### 课程式强度调度

```
训练进度    噪声强度    策略
─────────────────────────────────
0% ~ 40%   0.03       轻度噪声 — 先学会基本结构
40% ~ 80%  0.06       中度噪声 — 强化鲁棒性
80% ~ 100% 0.03       轻度噪声 — 配合 LR 衰减精细化
```

每个 batch 以 50% 概率应用噪声，保留部分干净样本避免过拟合。

#### 实现

```python
class SpeckleTrainer(DetectionTrainer):
    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        progress = self.epoch / self.epochs
        intensity = 0.06 if 0.4 <= progress < 0.8 else 0.03
        if np.random.random() < 0.5:
            batch['img'] = add_speckle_noise(batch['img'], intensity)
        return batch
```

---

## 目录结构

```
yolo/
├── README.md                           # 本文件
├── dataset.yaml                        # YOLO 数据集配置 (2 类: artery, vein)
├── prior_stats.json                    # 动静脉位置先验统计 (1,423 配对样本)
│
├── compute_prior_stats.py              # 从训练标签计算先验统计
├── inference.py                        # 检测推理 + 先验补全 + 评估
├── train_1_baseline.py                 # Step 1: 无增强基线
├── train_2_translate.py                # Step 2: +平移
├── train_3_translate_scale.py          # Step 3: +缩放
├── train_4_translate0.1_scale0.1.py    # Step 4: 增大平移范围
├── train_5_speckle_translate_scale.py  # Step 5: +课程式斑点噪声 (最优)
│
├── dataset/                            # 数据集
│   ├── train/images/ + labels/             # 训练集 (YOLO 格式)
│   └── val/images/ + labels/               # 验证集
│
└── runs/detect/runs/detect/dvt_runs/   # 当前仓库中的实际训练产物目录
    ├── aug_step1_baseline/                 # 各步骤权重 + 日志
    ├── aug_step2_translate/
    ├── aug_step3_translate_scale/
    ├── aug_step4_translate0.1_scale0.1/
    └── aug_step5_speckle_translate_scale/  # 最优模型
        └── weights/best.pt
```

当前仓库内实际最优权重路径为：

```text
yolo/runs/detect/runs/detect/dvt_runs/aug_step5_speckle_translate_scale/weights/best.pt
```

## 使用方法

### 训练

```bash
cd yolo

# Step 1-4: 直接运行
python train_1_baseline.py

# Step 5: 含自定义 SpeckleTrainer
python train_5_speckle_translate_scale.py
```

### 推理

```bash
cd yolo
python inference.py \
  --weights runs/detect/runs/detect/dvt_runs/aug_step5_speckle_translate_scale/weights/best.pt \
  --split val --conf 0.1
```

### 计算先验

```bash
cd yolo
python compute_prior_stats.py
# 输出: prior_stats.json
```

## 输出格式

VesselDetector 检测结果字典：

```python
{
    "artery": {
        "box": [x1, y1, x2, y2],   # 像素坐标
        "conf": 0.85,               # 检测置信度
        "inferred": False,          # 是否由先验推断
        "fixed": False,             # 是否经过重叠修正
        "prior_all": False,         # 是否两框都由先验生成
    },
    "vein": { ... }
}
```

此格式直接作为 SAM2 的 box prompt 输入。
