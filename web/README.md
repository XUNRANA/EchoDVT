# EchoDVT Web — Gradio 超声诊断可视化平台

基于 Gradio 6.x 的交互式 DVT 辅助诊断系统，将 YOLO 检测、SAM2 LoRA 分割、DVT 分类的完整流程封装为可视化 Web 应用。

## 快速启动

```bash
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

## 功能模块（6 个 Tab）

### Tab 1: 视频上传

- 从 `val`（76 例）或 `train`（300 例）数据集中选择案例
- 加载后展示首帧预览（含 GT 标注叠加）
- 显示案例信息：帧数、标注帧数、分辨率

### Tab 2: YOLO 检测

- 在首帧上运行 YOLO 血管检测
- 识别动脉（红框）和静脉（绿框）的边界框
- 支持选择不同训练阶段的 YOLO 权重
- 支持调节置信度阈值（默认 0.1）
- 缺失框通过统计先验自动补全（虚线框标记）
- 无权重时自动降级为 GT Mask Demo 模式

### Tab 3: SAM2 分割

- 使用首帧 YOLO 检测框作为 box prompt
- 通过 SAM2 记忆传播机制完成全视频分割
- 支持多种模型变体：

| 变体 | 说明 |
|------|------|
| **LoRA r8** (默认) | LoRA 微调 (r=8)，效果最佳 |
| LoRA r4 | LoRA 微调 (r=4)，更轻量 |
| Baseline (Large) | SAM2 Large 原始模型 |
| Baseline + AM/SM/AV | 带记忆增强变体 |

- 可选启用 **多帧提示 (MFP)**：在多个帧上运行 YOLO 检测作为额外 conditioning，减少误差累积
- 输出：逐帧分割结果 Gallery，Dice/mIoU 指标汇总
- 无模型权重时自动降级为 GT Mask Demo 模式

### Tab 4: DVT 诊断

- 从分割结果中提取 **19 维时序特征**
- 关键特征：VCR（静脉压缩比）、VDR（消失率）、VARR（相对范围）、MVAR（最小 V/A 比）等
- 面积变化曲线可视化（动脉/静脉面积 + V/A 比值）
- 高召回优化阈值：VCR > 0.314 即判为 DVT 疑似
- 兜底机制：特征提取失败时降级为简单 min/max 阈值判断

### Tab 5: 定量评估

- 逐帧 Dice / mIoU 指标曲线图
- Case 级汇总统计表（动脉 / 静脉 / 平均）
- 最佳帧 & 最差帧标注
- 仅在有 GT 标注的帧上评估

### Tab 6: 模型对比

- 并排展示不同 SAM2 模型变体的指标差异
- Dice 对比柱状图 + 多维雷达图
- 适用于答辩展示消融实验结果

## 目录结构

```
web/
├── app.py                  # 应用入口，构建 Gradio Blocks 界面
├── README.md               # 本文件
├── services/               # 推理服务层
│   ├── __init__.py             # 导出 InferenceService
│   └── inference.py            # 单例服务：惰性加载 YOLO / SAM2 LoRA / 特征提取
├── tabs/                   # 6 个功能 Tab
│   ├── __init__.py
│   ├── upload.py               # Tab 1: 视频上传与案例选择
│   ├── detection.py            # Tab 2: YOLO 血管检测
│   ├── segmentation.py         # Tab 3: SAM2 视频分割
│   ├── diagnosis.py            # Tab 4: DVT 诊断（19 维特征）
│   ├── evaluation.py           # Tab 5: 定量评估（Dice/mIoU）
│   └── comparison.py           # Tab 6: 模型变体对比
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── visualization.py        # 检测框绘制、Mask 叠加、对比图生成
│   └── metrics.py              # Dice/IoU 计算、DVT 诊断、Case 汇总
└── assets/
    └── custom.css              # 自定义主题样式
```

## 架构设计

### InferenceService 单例

`web/services/inference.py` 中的 `InferenceService` 是核心服务层，特点：

- **单例模式**：通过 `InferenceService.get()` 获取全局唯一实例
- **惰性加载**：YOLO / SAM2 LoRA 模型仅在首次调用时加载，后续复用缓存
- **GPU 自动管理**：自动检测 CUDA 可用性，合理分配 YOLO / SAM2 设备

提供三个高层 API：

```python
service = InferenceService.get()

# YOLO 检测
detections = service.run_detection(image_bgr, conf=0.1)

# SAM2 LoRA 分割（可选 MFP）
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

用户按 Tab 顺序操作：上传 → 检测 → 分割 → 诊断 → 评估，每一步的结果通过 `gr.State` 传递给下一步。

### Demo 降级机制

当模型权重不可用时（如首次部署、权重未下载），各模块自动降级：

| 模块 | 降级行为 |
|------|---------|
| YOLO 检测 | 从 GT Mask 中提取边界框 |
| SAM2 分割 | 使用 GT Mask 模拟 Memory 传播 |
| DVT 诊断 | 降级为简单 VCR 阈值判断 |

这确保 Web 应用在无 GPU / 无权重环境下也能完整演示。

## 模型权重路径

| 模型 | 默认路径 |
|------|---------|
| SAM2 Large | `sam2/checkpoints/sam2_hiera_large.pt` |
| LoRA r8 | `sam2/checkpoints/lora_runs/lora_r8_lr0.0003_e25_20260314_153210/lora_best.pt` |
| LoRA r4 | `sam2/checkpoints/lora_runs/lora_r4_lr0.0005_e25_20260314_153134/lora_best.pt` |
| YOLO | `yolo/runs/detect/runs/detect/dvt_runs/aug_step5_speckle_translate_scale/weights/best.pt` |
| YOLO 先验 | `yolo/prior_stats.json` |

路径在 `web/services/inference.py` 顶部统一配置。

## 依赖

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
