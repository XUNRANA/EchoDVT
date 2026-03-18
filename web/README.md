# EchoDVT Web — 超声深静脉血栓智能诊断系统

基于 Gradio 6.x 的交互式 DVT 辅助诊断系统，将 YOLO 检测、SAM2 LoRA 分割、21 维时序特征提取、DVT 分类的完整流程封装为可视化 Web 应用。

---

## 快速启动

```bash
conda activate echodvt
cd EchoDVT/web
python app.py
```

浏览器访问终端输出的 URL 即可使用（例如 `http://<server-ip>:7860`）。
不指定 `--port` 时会自动选择空闲端口。

### 启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | 自动 | 可选；不指定时由 Gradio 自动选择可用端口 |
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
页面布局:

  ┌─ 顶部栏（标题 + 状态指示 + 北京时间）─────────────────┐
  ├─ 流水线流程图（超声输入→YOLO检测→SAM2分割→特征提取→二分类→可视化）─┤
  ├─ 统计摘要行（准确率 | 特征维度 | 样本量 | 模型状态）────────┤
  ├─ 水平标签导航（7 个 Tab）───────────────────────┤
  └─ Tab 内容区（max-width 1400px 居中）──────────────────┘

功能流程:

  📊 仪表盘（系统概览）
       │
       ├──→ 📤 数据输入 ──→ 🎯 YOLO 检测 ──→ 🔬 SAM2 分割 ──→ 🩺 DVT 诊断
       │
       └──→ 🚀 一键分析（一步完成检测→分割→诊断→报告）──→ 📄 导出报告
```

### 核心技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 血管检测 | YOLOv8 | 首帧动脉/静脉边界框检测 |
| 视频分割 | SAM2 + LoRA | 基于记忆传播的全视频语义分割 |
| 特征提取 | 21 维时序分析 | VCR/VDR/VARR 等可解释特征 |
| DVT 判断 | RF unified | 固定阈值 `prob ≥ 0.05`（train 94.33%，val 94.74%） |
| Web 框架 | Gradio 6.x | 浅色医疗风格交互界面 |

---

## 功能模块（7 个 Tab）

> 顶部水平标签导航，Tab 顺序为：
> **📊 仪表盘 → 📤 数据输入 → 🚀 一键分析 → 🎯 目标检测 → 🔬 视频分割 → 🩺 DVT 诊断 → 📄 导出报告**

### Tab 1: 📊 仪表盘 (`tabs/dashboard.py`)

系统概览：
- GPU/内存/模型状态卡片
- 数据集统计（train/val 数量、正常/患者分布）
- 统一模型准确率概览（train + val）
- 数据分布图表

---

### Tab 2: 📤 数据输入 (`tabs/upload.py`)

支持两种入口：
- 从 `val/train` 数据集选择案例
- 上传本地视频（`gr.File`）

写入全局 `gr.State`：首帧预览、采样帧图库、路径元信息

---

### Tab 3: 🚀 一键分析 (`tabs/pipeline.py`)

一键执行：检测 → 分割 → 特征 → 诊断，输出完整可视化结果。
- 默认参数：`LoRA r8`、`MFP=False`、`conf=0.1`
- 诊断报告紧凑展示，21 维特征表可折叠展开

---

### Tab 4: 🎯 目标检测 (`tabs/detection.py`)

在首帧上运行 YOLO 检测，输出动脉/静脉框，并写入 `state[“detections”]`。
- 固定使用最优 YOLO 权重
- 支持先验补全与重叠修正

---

### Tab 5: 🔬 视频分割 (`tabs/segmentation.py`)

使用首帧检测框作为 prompt，调用 SAM2（LoRA / Baseline）做全视频分割。
- 写入 `pred_masks / vein_areas / artery_areas / frame_metrics`
- 支持 LoRA r4/r8、Baseline 及扩展变体
- 支持多帧提示 MFP

---

### Tab 6: 🩺 DVT 诊断 (`tabs/diagnosis.py`)

基于面积时序与 21 维特征进行 DVT 判断：
- 输出诊断摘要卡、面积曲线和详细报告
- 默认优先走 `InferenceService` 的 `RF unified` 推理（固定阈值 `prob ≥ 0.05`）
- 服务不可用时降级到同阈值对齐的简单 VCR 规则

---

### Tab 7: 📄 导出报告 (`tabs/evaluation.py`)

生成并下载 PDF 诊断报告：
- 汇总检测、分割、诊断全部结果
- 一键生成 PDF 文件并提供下载

---

## 目录结构

```
web/
├── app.py                  # 应用入口（header + 流水线 + 统计行 + 水平 Tabs + footer）
├── README.md               # 本文件
├── services/               # 推理服务层
│   ├── __init__.py
│   └── inference.py        # InferenceService 单例：惰性加载 YOLO/SAM2/分类器
├── tabs/                   # 7 个功能 Tab
│   ├── __init__.py
│   ├── dashboard.py        # Tab 1: 仪表盘（系统概览）
│   ├── upload.py           # Tab 2: 数据输入（数据集 + 视频上传）
│   ├── pipeline.py         # Tab 3: 一键全流程分析
│   ├── detection.py        # Tab 4: YOLO 血管检测
│   ├── segmentation.py     # Tab 5: SAM2 视频分割
│   ├── diagnosis.py        # Tab 6: DVT 诊断（21 维特征）
│   └── evaluation.py       # Tab 7: 导出报告（PDF 生成）
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── visualization.py    # 检测框绘制、Mask 叠加、对比图生成
│   ├── metrics.py          # Dice/IoU 计算、DVT 诊断、Case 汇总
│   └── chart_style.py      # Matplotlib 浅色主题 + 中文字体自动检测
└── assets/
    └── custom.css          # 自定义浅色主题样式（水平标签 + 流水线 + 统计行）
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
| DVT 诊断 | 降级为与统一模型阈值对齐的简单 VCR 判断 |

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

- **浅色医疗主题**: 基于 `gr.themes.Base` + 自定义 CSS，简洁专业
- **流水线流程图**: 页面顶部 6 步静态流程卡片（超声输入→可视化）
- **统计摘要行**: 4 列关键指标（准确率 / 特征维度 / 样本量 / 模型状态），构建时计算
- **水平标签导航**: 居中 pill 样式，sticky 固定在 header 下方
- **内容区居中**: 最大宽度 1400px，自适应
- **配色方案**:
  - 动脉: `#ef4444` (红色)
  - 静脉: `#22c55e` (绿色)
  - 主色: `#2563eb` (蓝)
