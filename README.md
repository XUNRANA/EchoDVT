# EchoDVT

EchoDVT 是一个基于超声视频的深静脉血栓辅助诊断项目，主线流程为：

```text
超声视频
→ YOLO 首帧血管检测
→ SAM2 + LoRA 视频分割
→ 21 维时序特征提取
→ RF unified 二分类
→ Web 可视化与 PDF 报告
```

项目当前的目标不是做通用实验平台，而是收敛成一条稳定、可复现、可展示的主线诊断链路。

## 当前默认配置

当前离线评估与 Web 主线统一使用以下配置：

| 模块 | 当前默认配置 |
|------|--------------|
| YOLO | `yolo/runs/detect/runs/detect/dvt_runs/aug_step5_speckle_translate_scale/weights/best.pt` |
| YOLO 阈值 | `conf = 0.1` |
| SAM2 主干 | `sam2_hiera_large.pt` |
| SAM2 微调 | `LoRA r8` |
| 多帧提示 | `MFP` |
| 分类器 | `RF unified` |
| 分类阈值 | `prob >= 0.05` |
| 特征维度 | `21` |

统一模型元信息位于：

```text
results/unified_model/rf_unified.json
```

当前记录的关键指标为：
- `train_accuracy = 94.33%`
- `val_accuracy = 94.74%`

## 目录结构

```text
EchoDVT/
├── README.md
├── classify_dvt.py
├── results/
│   └── unified_model/
├── web/
│   ├── app.py
│   ├── services/
│   ├── tabs/
│   ├── utils/
│   └── assets/
├── yolo/
│   ├── inference.py
│   ├── compute_prior_stats.py
│   ├── train_*.py
│   ├── prior_stats.json
│   └── README.md
└── sam2/
    ├── inference_box_prompt_large.py
    ├── inference_lora.py
    ├── train_lora.py
    ├── checkpoints/
    ├── sam2/
    ├── README_EchoDVT.md
    └── README.md
```

## 数据

### 标注分割数据

当前主线使用的分割数据位于：

```text
sam2/dataset/
├── train/   # 300 例，全部正常
└── val/     # 76 例，38 正常 + 38 患者
```

每个 case 通常包含：
- `images/`
- `masks/`

其中 mask 的语义约定为：
- `0 = 背景`
- `1 = 动脉`
- `2 = 静脉`

### Web 可浏览测试集

Web 的数据输入页还支持浏览：

```text
test/
├── normal/
└── patient/
```

这部分主要用于病例浏览和推理，不等同于带稀疏标注的 train / val 分割集。

## 各模块职责

### 1. `yolo/`

负责首帧动脉/静脉检测，并在漏检时利用位置先验补全缺失框。

重点：
- 渐进式增强训练
- Speckle 噪声增强
- 先验补全与重叠修正

详见 [yolo/README.md](yolo/README.md)。

### 2. `sam2/`

负责视频分割与训练。

当前主线是：
- SAM2 Large
- LoRA 微调
- 多帧提示 MFP
- 相对位置约束 RPA

V1 的 adaptive-memory 支线已经移除，不属于当前支持实现。

详见：
- [sam2/README_EchoDVT.md](sam2/README_EchoDVT.md)
- [sam2/README.md](sam2/README.md)

### 3. `classify_dvt.py`

负责从语义 mask 序列中提取 21 维时序特征，并使用统一 RF 分类器完成 DVT 判断。

输出核心包括：
- `probability`
- `threshold`
- `is_dvt`
- `vcr`
- 全量特征字典

### 4. `web/`

负责将整条主线串成可交互界面。

当前 Web 设计原则：
- 固定最优权重
- 固定稳定参数
- 支持单案例完整分析
- 不对外暴露实验性变体切换

详见 [web/README.md](web/README.md)。

## Web 快速启动

```bash
conda activate echodvt
cd /data1/ouyangxinglong/EchoDVT/web
python app.py --server-name 0.0.0.0 --port 18081
```

浏览器访问：

```text
http://<server-ip>:18081
```

如果通过 SSH 使用，建议本地转发：

```bash
ssh -N -L 7860:127.0.0.1:18081 <user>@<server>
```

## 常用命令

### YOLO 检测推理

```bash
cd yolo
python inference.py \
  --weights runs/detect/runs/detect/dvt_runs/aug_step5_speckle_translate_scale/weights/best.pt \
  --split val \
  --conf 0.1
```

### SAM2 LoRA 推理

```bash
cd sam2
python inference_lora.py \
  --lora-weights checkpoints/lora_runs/lora_r8_lr0.0003_e25_20260314_153210/lora_best.pt \
  --lora-r 8 \
  --split val \
  --multi-frame-prompt True
```

### DVT 离线分类

```bash
python classify_dvt.py --split val
```

## 环境

推荐直接使用项目环境：

```bash
conda activate echodvt
```

核心依赖包括：
- `torch`
- `torchvision`
- `ultralytics`
- `gradio >= 6.0`
- `opencv-python`
- `matplotlib`
- `scikit-learn`
- `pandas`
- `scipy`

## 文档分工

为了减少重复维护，当前文档分工如下：
- 本文件：项目总览、主线配置、目录与入口
- [web/README.md](web/README.md)：当前 Web 结构、状态流和使用方式
- [yolo/README.md](yolo/README.md)：YOLO 训练与检测设计
- [sam2/README_EchoDVT.md](sam2/README_EchoDVT.md)：SAM2 定制点与分割主线

如果代码与 README 不一致，应优先以当前代码实现为准，再回补文档。
