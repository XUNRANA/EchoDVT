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
用户交互流程:

  📊 仪表盘（综合指标 + 误判案例分析）
       │
       ├──→ 📤 数据输入 ──→ 🎯 YOLO 检测 ──→ 🔬 SAM2 分割 ──→ 🩺 DVT 诊断 ──→ 📈 定量评估
       │
       └──→ 🚀 一键分析（一步完成检测→分割→诊断）
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

> 当前 UI 使用“侧边栏”单导航，默认入口与顺序为：  
> **📊 仪表盘 → 📤 数据输入 → 🎯 目标检测 → 🔬 视频分割 → 🩺 DVT 诊断 → 📈 定量评估 → 🚀 一键分析**。

### Tab 1: 📊 仪表盘 (`tabs/dashboard.py`)

系统总览与运营入口：
- GPU/内存/模型状态、数据集统计、验证/测试准确率
- 综合评估：`train` 全量 + `val` 全量 + `test(normal=500, patient=50)` 的综合指标
- 当前流程进度（加载/检测/分割/诊断/评估）
- 误判案例分析（含误判原因说明）
- 快速操作：
  - 自动加载下一个验证集案例
  - 运行全流程分析
  - 运行综合本地评估（固定配置：`train + val + test 500/50`）

---

### Tab 2: 📤 数据输入 (`tabs/upload.py`)

支持三种入口：
- 从 `val/train` 选择案例
- 勾选 **“从 test 集加载案例”** 后选择 `test/normal` 或 `test/patient`
- 上传本地视频（`gr.File`）

并写入全局 `gr.State`：
- `images_dir / masks_dir / frame_files`
- 案例元信息、首帧预览与采样帧图库

---

### Tab 3: 🎯 目标检测 (`tabs/detection.py`)

在首帧上运行 YOLO 检测，输出动脉/静脉框，并写入 `state["detections"]`。
- 固定使用最优 YOLO 权重
- 支持先验补全与重叠修正
- 无权重时支持 GT 降级演示

---

### Tab 4: 🔬 视频分割 (`tabs/segmentation.py`)

使用首帧检测框作为 prompt，调用 SAM2（LoRA / Baseline）做全视频分割。
- 写入 `pred_masks / vein_areas / artery_areas / frame_metrics`
- 支持 LoRA r4/r8、Baseline 及扩展变体
- 支持多帧提示 MFP

---

### Tab 5: 🩺 DVT 诊断 (`tabs/diagnosis.py`)

基于面积时序与 21 维特征进行 DVT 判断：
- 输出诊断摘要卡、面积曲线和详细报告
- 默认优先走 `InferenceService` 的 `RF unified` 推理（固定阈值 `prob ≥ 0.05`）
- 服务不可用时降级到同阈值对齐的简单 VCR 规则

---

### Tab 6: 📈 定量评估 (`tabs/evaluation.py`)

对有 GT 的帧计算逐帧与病例级指标：
- Dice / IoU / Mean Dice / mIoU
- 曲线图、汇总表、逐帧明细表
- 仅在存在标注时生效

---

### Tab 7: 🚀 一键分析 (`tabs/pipeline.py`)

一键执行：检测 → 分割 → 特征 → 诊断，输出完整可视化结果与报告。
- 默认参数：`LoRA r8`、`MFP=False`、`conf=0.1`
- 结果直接返回在当前页面，不再额外生成 Dashboard 日志记录

---

## 目录结构

```
web/
├── app.py                  # 应用入口，构建 Gradio Blocks + 浅色医疗主题
├── README.md               # 本文件
├── services/               # 推理服务层
│   ├── __init__.py
│   └── inference.py        # InferenceService 单例：惰性加载 YOLO/SAM2/分类器
├── tabs/                   # 7 个功能 Tab
│   ├── __init__.py
│   ├── dashboard.py        # Tab 1: 仪表盘（总览 + 快速操作）
│   ├── upload.py           # Tab 2: 数据输入（含 test 勾选加载 + 视频上传）
│   ├── detection.py        # Tab 3: YOLO 血管检测
│   ├── segmentation.py     # Tab 4: SAM2 视频分割（7 种变体）
│   ├── diagnosis.py        # Tab 5: DVT 诊断（21 维特征）
│   ├── evaluation.py       # Tab 6: 定量评估（Dice/mIoU）
│   ├── pipeline.py         # Tab 7: 一键全流程分析
│   └── comparison.py       # 扩展模块：模型变体对比（未默认挂载）
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── visualization.py    # 检测框绘制、Mask 叠加、对比图生成
│   ├── metrics.py          # Dice/IoU 计算、DVT 诊断、Case 汇总
│   └── chart_style.py      # Matplotlib 浅色主题 + 中文字体自动检测
└── assets/
    └── custom.css          # 自定义浅色主题样式
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

- **浅色医疗主题**: 基于 `gr.themes.Base` + 自定义 CSS
- **渐变标题卡片**: 每个 Tab 顶部有统一的渐变色标题区
- **分栏 Dashboard 布局**: 顶栏 + 左侧单导航 + 右侧工作区
- **轻量微交互**: 按钮 hover、卡片 hover、渐变欢迎横幅
- **配色方案**: 
  - 动脉: `#ef4444` (红色)
  - 静脉: `#22c55e` (绿色)  
  - 主色: `#2563eb` → `#06b6d4` (蓝-青渐变)
