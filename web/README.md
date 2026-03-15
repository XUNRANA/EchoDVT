# EchoDVT Web — 超声深静脉血栓智能诊断系统

基于 Gradio 6.x 的交互式 DVT 辅助诊断系统，将 YOLO 检测、SAM2 LoRA 分割、19 维时序特征提取、DVT 分类的完整流程封装为可视化 Web 应用。

---

## 快速启动

```bash
conda activate echodvt
cd EchoDVT/web
python app.py --port 7860
```

浏览器访问 `http://<server-ip>:7860` 即可使用。

### 启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | 7860 | 服务端口 |
| `--server-name` | 0.0.0.0 | 监听地址 |
| `--share` | False | 启用 Gradio 公网共享链接 |

### 环境依赖

```
gradio >= 6.0
torch >= 2.1
ultralytics
opencv-python
numpy
matplotlib
scikit-learn
scipy
```

> **注意**: 如需使用 `gr.Video` 组件上传视频，需安装 `ffmpeg`。当前版本已改用 `gr.File` 绕过此依赖。

---

## 系统架构

```
用户交互流程:

  📤 数据输入 ──→ 🎯 YOLO 检测 ──→ 🔬 SAM2 分割 ──→ 🩺 DVT 诊断
       │                                                    │
       │              🚀 一键分析（一步完成上述全流程）          │
       │                                                    │
       └──→ 📊 定量评估（Dice/mIoU） ──→ ⚖️ 模型对比（消融实验）
```

### 核心技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 血管检测 | YOLOv8 | 首帧动脉/静脉边界框检测 |
| 视频分割 | SAM2 + LoRA | 基于记忆传播的全视频语义分割 |
| 特征提取 | 19 维时序分析 | VCR/VDR/VARR 等可解释特征 |
| DVT 判断 | 阈值分类 | 高召回优化 (VCR > 0.314) |
| Web 框架 | Gradio 6.x | 暗黑主题交互界面 |

---

## 功能模块（7 个 Tab）

### Tab 1: 📤 数据输入 (`tabs/upload.py`)

**功能**: 加载超声数据到系统中，支持两种输入方式。

#### 方式 A: 从数据集选择
- 支持 `val`（76 例：38 正常 + 38 患者）和 `train`（300 例）数据集
- 下拉搜索案例名，支持筛选
- 加载后自动展示首帧预览（含 GT 标注叠加）
- 显示案例元信息：帧数、标注帧数、分辨率、标注帧索引

#### 方式 B: 上传本地视频
- 支持 MP4 / AVI / MOV / MKV 格式（最大 2GB）
- 使用 `gr.File` 组件上传，无需 ffmpeg
- 自动逐帧提取 (OpenCV)，存入临时目录
- 上传前自动检查磁盘空间和文件大小

#### 输出
- **首帧预览**: 带 GT Mask 半透明叠加的 RGB 图像
- **帧序列 Gallery**: 均匀采样 12 帧缩略图
- **案例信息卡**: Markdown 表格展示元数据

---

### Tab 2: 🚀 一键分析 (`tabs/pipeline.py`)

**功能**: 单击完成从检测到诊断的端到端全流程，生成完整诊断报告。

#### 处理流程

```
[1/4] YOLO 血管检测 (首帧)
  │
  ▼
[2/4] SAM2 视频分割 (全帧)
  │
  ▼
[3/4] 面积和指标计算
  │
  ▼
[4/4] DVT 智能诊断 + 报告生成
```

#### 参数设置
- **分割模型**: LoRA r8 (默认) / LoRA r4 / Baseline (Large)
- **多帧提示 (MFP)**: 每隔 15 帧用 YOLO 重新锚定
- **YOLO 置信度阈值**: 0.01 ~ 0.5，默认 0.1

#### 输出
- **诊断摘要卡片**: DVT 疑似 (红色) / 正常 (绿色)，含 VCR 值和置信度
- **首帧检测结果**: 带 artery/vein 边界框的可视化图
- **分割 Gallery**: 逐帧分割结果采样（最多 24 帧）
- **面积曲线**: 动脉/静脉面积变化 + V/A 比值双图
- **完整报告**: HTML 格式的 6 节诊断报告（含 19 维特征表、检测框信息、分割指标、诊断结论）

---

### Tab 3: 🎯 YOLO 检测 (`tabs/detection.py`)

**功能**: 在首帧上运行 YOLO 血管检测，识别动脉和静脉的边界框。

#### 核心特性
- 支持选择不同训练阶段的 YOLO 权重
- 自动搜索 `yolo/runs/` 下所有 `best.pt` 权重
- 可调置信度阈值 (0.01 ~ 0.95)
- **统计先验补全**: 当检测缺失某类血管时，从 `prior_stats.json` 中加载先验位置补全
- **重叠修正**: 当动脉/静脉框重叠时自动修正

#### 检测框标记
| 标记 | 含义 |
|------|------|
| ✅ 正常检测 | YOLO 直接检测到 |
| 🔮 推断补全 | 通过先验位置推断 |
| 🔧 重叠修正 | 检测后修正了重叠 |
| GT 提取 | 从 GT mask 提取（Demo 模式） |

#### Demo 降级
无 YOLO 权重时自动从 GT Mask 中提取边界框，确保流程可演示。

---

### Tab 4: 🔬 SAM2 分割 (`tabs/segmentation.py`)

**功能**: 使用首帧 YOLO 检测框作为 box prompt，通过 SAM2 记忆传播机制完成全视频分割。

#### 支持的模型变体

| 变体 | 说明 | 参数量 |
|------|------|--------|
| **LoRA r8** (默认) | LoRA 微调 (rank=8) | 5.19M 可训练 (2.3%) |
| LoRA r4 | LoRA 微调 (rank=4) | 更轻量 |
| Baseline (Large) | SAM2 Large 原始模型 | 225M |
| Baseline + AM | 自适应记忆增强 | |
| Baseline + SM | 独立记忆通道 | |
| Baseline + AV | 动脉-静脉约束 | |
| Baseline + AM + SM + AV | 全部增强策略 | |

#### 多帧提示 (MFP)
- 仅对 LoRA 变体生效
- 每隔 15 帧运行 YOLO 检测作为额外 conditioning
- 减少长视频中的分割误差累积

#### 输出
- **分割预览**: 首帧分割结果（红=动脉，绿=静脉）
- **逐帧 Gallery**: 采样展示分割叠加图
- **指标报告**: Mean Dice / mIoU 平均值（基于有 GT 标注的帧）

#### Demo 降级
无 SAM2 权重时使用 GT Mask 模拟记忆传播。

---

### Tab 5: 🩺 DVT 诊断 (`tabs/diagnosis.py`)

**功能**: 基于静脉在压缩超声过程中的面积变化率进行自动 DVT 判断。

#### 诊断原理

```
正常静脉: 探头压迫 → 面积大幅缩小 → VCR ≈ 0 → ✅ 正常
血栓静脉: 探头压迫 → 面积基本不变 → VCR ≈ 1 → ⚠️ DVT 疑似
```

#### 19 维时序特征

| 特征 | 名称 | 说明 |
|------|------|------|
| VCR | 静脉压缩比 | min/max，核心指标 |
| VDR | 消失率 | 面积 < 10% max 的帧占比 |
| VARR | 相对范围 | (max-min)/max |
| vein_cv | 变异系数 | std/mean |
| MVAR | 最小 V/A 比 | min(vein/artery) |
| mean_var | 均值 V/A 比 | mean(vein/artery) |
| vein_slope | 趋势斜率 | 负=下降(正常) |
| vein_min_position | 最小面积位置 | 相对帧位置 |
| artery_stability | 动脉稳定性 | 动脉稳定参考 |
| max_drop_ratio | 最大帧间下降 | 越大=急剧塌陷 |
| vein_p10/p25/p50 | 百分位值 | 归一化面积分布 |
| vein_detect_rate | 静脉检出率 | 检出帧占比 |
| artery_detect_rate | 动脉检出率 | 检出帧占比 |
| vein_jitter | 帧间跳变 | 越大越不稳 |
| vein_autocorr | 自相关 | 高=平滑 |
| circ_cv/min/range | 圆度特征 | 形态变化 |

#### 判断阈值
- **ML 优化阈值**: VCR > 0.314 → DVT 疑似（高召回优化）
- **简单阈值** (可调): min/max > 0.4 → DVT 疑似

#### 输出
- **面积变化曲线**: 动脉/静脉面积 + V/A 比值双图
- **诊断报告**: 含完整 19 维特征表的详细报告
- **诊断卡片**: DVT 疑似 ⚠️ / 正常 ✅

---

### Tab 6: 📊 定量评估 (`tabs/evaluation.py`)

**功能**: 展示分割质量的逐帧 Dice / mIoU 指标和 case 级汇总。

#### 评估指标

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| Dice | 2·\|A∩B\| / (\|A\|+\|B\|) | 分割重叠度 |
| IoU | \|A∩B\| / \|A∪B\| | 交并比 |
| Mean Dice | (Artery Dice + Vein Dice) / 2 | 平均 Dice |
| mIoU | (Artery IoU + Vein IoU) / 2 | 平均 IoU |

#### 输出
- **逐帧曲线图**: 左=Dice 变化，右=mIoU 变化
- **Case 级汇总表**: Artery / Vein / Mean 三列指标表
- **最佳/最差帧标注**: 标记分割质量最高和最低的帧
- **逐帧明细表**: 每帧的 6 项指标 Markdown 表格

> 仅在有 GT 标注的帧上评估。

---

### Tab 7: ⚖️ 模型对比 (`tabs/comparison.py`)

**功能**: 并排展示不同 SAM2 模型变体的分割效果和指标差异，适用于答辩/论文的消融实验展示。

#### 对比方式
- 选择两个模型变体进行对比
- 生成 Dice 柱状图 + 多维雷达图

#### 可对比维度
- Artery Dice / Vein Dice / Mean Dice
- Artery IoU / Vein IoU / mIoU

#### 输出
- **Dice 对比柱状图**: 两个模型的 3 项 Dice 指标并排柱状图
- **多维雷达图**: 6 项指标的雷达图
- **对比报告**: Markdown 表格含 Delta 差异列

> 完整对比需在「SAM2 分割」Tab 中分别以不同模型变体运行后查看。

---

## 目录结构

```
web/
├── app.py                  # 应用入口，构建 Gradio Blocks + 暗黑主题
├── README.md               # 本文件
├── services/               # 推理服务层
│   ├── __init__.py
│   └── inference.py        # InferenceService 单例：惰性加载 YOLO/SAM2/分类器
├── tabs/                   # 7 个功能 Tab
│   ├── __init__.py
│   ├── upload.py           # Tab 1: 数据输入（数据集选择 + 视频上传）
│   ├── pipeline.py         # Tab 2: 一键全流程分析
│   ├── detection.py        # Tab 3: YOLO 血管检测
│   ├── segmentation.py     # Tab 4: SAM2 视频分割（7 种变体）
│   ├── diagnosis.py        # Tab 5: DVT 诊断（19 维特征）
│   ├── evaluation.py       # Tab 6: 定量评估（Dice/mIoU）
│   └── comparison.py       # Tab 7: 模型变体对比（消融实验）
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── visualization.py    # 检测框绘制、Mask 叠加、对比图生成
│   ├── metrics.py          # Dice/IoU 计算、DVT 诊断、Case 汇总
│   └── chart_style.py      # Matplotlib 暗黑主题 + 中文字体自动检测
└── assets/
    └── custom.css          # 自定义暗黑主题样式（渐变、动画、响应式）
```

---

## 核心架构

### InferenceService 单例

`web/services/inference.py` 中的 `InferenceService` 是核心服务层：

- **单例模式**: `InferenceService.get()` 获取全局唯一实例
- **惰性加载**: YOLO / SAM2 LoRA 模型仅在首次调用时加载
- **GPU 自动管理**: 自动检测 CUDA 可用性

```python
service = InferenceService.get()

# YOLO 检测
detections = service.run_detection(image_bgr, conf=0.1)

# SAM2 LoRA 分割
pred_masks = service.run_segmentation(images_dir, detections, num_frames,
                                       use_mfp=True, variant="LoRA r8")

# DVT 特征提取 + 分类
result = service.run_diagnosis(masks_list)
```

### 数据流

```
gr.State (全局状态字典)
    │
    ├── upload.py: 写入 images_dir, masks_dir, frame_files
    │
    ├── detection.py: 写入 detections (artery/vein 检测框)
    │
    ├── segmentation.py: 写入 pred_masks, vein_areas, artery_areas, frame_metrics
    │
    ├── diagnosis.py: 读取 pred_masks + vein_areas → 特征提取 → 诊断结果
    │
    ├── evaluation.py: 读取 frame_metrics → Dice/mIoU 图表
    │
    └── comparison.py: 读取 frame_metrics → 多模型对比图表
```

### Demo 降级机制

当模型权重不可用时，各模块自动降级：

| 模块 | 降级行为 |
|------|---------| 
| YOLO 检测 | 从 GT Mask 中提取边界框 |
| SAM2 分割 | 使用 GT Mask 模拟 Memory 传播 |
| DVT 诊断 | 降级为简单 VCR 阈值判断 |

---

## 模型权重路径

| 模型 | 默认路径 |
|------|---------|
| SAM2 Large | `sam2/checkpoints/sam2_hiera_large.pt` |
| LoRA r8 | `sam2/checkpoints/lora_runs/lora_r8_lr0.0003_e25_*/lora_best.pt` |
| LoRA r4 | `sam2/checkpoints/lora_runs/lora_r4_lr0.0005_e25_*/lora_best.pt` |
| YOLO | `yolo/runs/detect/.../aug_step5_speckle_translate_scale/weights/best.pt` |
| YOLO 先验 | `yolo/prior_stats.json` |

---

## UI 设计

- **暗黑主题**: 基于 `gr.themes.Base` + 自定义 CSS
- **渐变标题卡片**: 每个 Tab 顶部有统一的渐变色标题区
- **响应式布局**: 1024px / 768px 断点自适应
- **微动画**: 按钮悬浮上移、卡片 hover 发光、内容 fade-in
- **配色方案**: 
  - 动脉: `#ef4444` (红色)
  - 静脉: `#22c55e` (绿色)  
  - 主色: `#2563eb` → `#06b6d4` (蓝-青渐变)
