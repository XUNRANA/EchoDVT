# EchoDVT Web

EchoDVT Web 是基于 Gradio 6.x 的诊断界面，用来串联数据输入、YOLO 检测、SAM2 分割、DVT 诊断和 PDF 报告导出。

当前 Web 的定位是：
- 固定使用最优权重与稳定参数
- 优先服务单案例诊断与科研展示
- 不在界面里暴露实验性模型切换选项

## 展示亮点

Web 端把算法链路包装成面向演示和病例复查的交互界面，重点突出四件事：

| 亮点 | 说明 |
|------|------|
| 一键串联 | 从病例加载到检测、分割、特征提取、RF 诊断一次完成 |
| 可解释诊断 | 不只给结论，还展示静脉面积曲线、VCR 和完整 21 维特征 |
| 可降级演示 | 权重缺失时可用 GT mask 兜底，保证页面流程仍可展示 |
| 可交付输出 | 支持 PDF 报告导出，包含检测、分割、诊断和图表信息 |

最适合展示的路径是“数据输入 → 一键分析 → 导出报告”。如果需要解释算法细节，可以依次打开目标检测、视频分割和 DVT 诊断页。

## 快速启动

```bash
conda activate echodvt
cd /data1/ouyangxinglong/EchoDVT/web
python app.py --server-name 0.0.0.0 --port 18081
```

浏览器访问：

```text
http://<server-ip>:18081
```

如果通过 SSH 开发，推荐本地端口转发：

```bash
ssh -N -L 7860:127.0.0.1:18081 <user>@<server>
```

然后在本机浏览器打开：

```text
http://127.0.0.1:7860
```

## 当前页面结构

当前版本不是前后端分离 SPA，而是单体式 Gradio 应用。整体结构如下：

```text
顶部栏
  ├─ 品牌名称
  ├─ 模型状态
  └─ 北京时间

全局摘要条
  ├─ 验证准确率
  ├─ 特征维度
  ├─ 样本总量
  └─ 统一模型状态

顶部标签导航
  ├─ 仪表盘
  ├─ 数据输入
  ├─ 一键分析
  ├─ 目标检测
  ├─ 视频分割
  ├─ DVT 诊断
  └─ 导出报告

Tab 内容区
```

与旧版相比，当前实现已经去掉了页面顶部的全局流程 stepper，改成更克制的 header + summary + tab 布局。

## 当前固定配置

Web 默认固定使用当前主线最优配置，不在界面中开放切换。

| 模块 | 当前配置 |
|------|----------|
| YOLO 检测 | `yolo/runs/detect/runs/detect/dvt_runs/aug_step5_speckle_translate_scale/weights/best.pt` |
| YOLO 置信度 | 固定 `conf = 0.1` |
| SAM2 主干 | `sam2_hiera_large.pt` |
| SAM2 LoRA | 固定 `LoRA r8` |
| 多帧提示 MFP | Web 中固定开启，`interval=15`，`min_conf=0.3`，`max_prompts=5` |
| DVT 分类器 | `RF unified` |
| DVT 阈值 | 固定 `prob >= 0.05` |

统一模型元信息当前来自：

```text
artifacts/unified_model/rf_unified.json
```

其中当前记录为：
- `train_accuracy = 94.33%`
- `val_accuracy = 94.74%`
- `val_recall = 97.37%`
- `val_precision = 92.50%`
- `feature_dim = 21`
- `n_train_cases = 376`

## 功能页说明

### 1. 仪表盘

用于展示系统状态和全局概览：
- GPU 与内存状态
- 模型可用性
- train / val 数据概览
- 当前统一模型的 train / val 指标

这个页面面向“系统总览”，不是病例诊断页。

适合展示：
- 当前统一模型指标
- 模型权重是否可用
- 数据集规模和病例分布

### 2. 数据输入

提供两个入口：
- 从数据集选择
- 上传本地视频

数据集选择页目前支持三个并行入口：
- `train`
- `val`
- `test`

其中 `test` 支持子集切换：
- `normal`
- `patient`

本页会写入全局状态：
- `current_case`
- `images_dir`
- `masks_dir`
- `frame_files`
- `mask_files`

上传本地视频时：
- Web 会先抽帧到临时目录
- 上传视频默认没有 GT 标注
- 后续仍可继续检测、分割和诊断

数据集病例适合展示完整指标，因为通常能读取 GT mask；上传视频更接近真实使用，但没有 GT 时不会显示 Dice/mIoU。

### 3. 一键分析

这是当前最接近真实使用流程的页面。

执行链路：

```text
数据加载
→ YOLO 检测
→ SAM2 分割
→ 21 维特征提取
→ RF unified 诊断
→ 面积曲线 / 报告展示
```

输出包括：
- 首帧检测结果
- 分割采样图库
- 面积变化曲线
- 诊断摘要卡
- 完整病例报告

这是最推荐的演示页。它体现 EchoDVT 的完整功能闭环：自动检测血管、跟踪压缩过程、提取时序特征、输出 DVT 风险。

### 4. 目标检测

只负责首帧 YOLO 检测。

特点：
- 固定最优权重
- 固定 `conf=0.1`
- 支持先验补全
- 支持重叠修正

结果会写入：

```text
state["detections"]
```

检测结果会保留 `inferred`、`fixed` 等标记，用来区分普通检测框、先验补全框和重叠修正框。

### 5. 视频分割

使用首帧检测框作为 prompt，执行 SAM2 Large + LoRA r8 分割。

当前 Web 口径：
- 不允许切换变体
- 不允许关闭 MFP
- 固定走当前最优配置
- 不暴露 RPA、OKM、DAM 等实验开关

结果会写入：
- `pred_masks`
- `vein_areas`
- `artery_areas`
- `frame_metrics`

页面展示的采样分割图用于快速判断传播是否稳定；面积序列会继续传给诊断页生成压缩曲线。

### 6. DVT 诊断

使用面积时序与 21 维特征进行 DVT 判断。

当前展示逻辑：
- 主指标优先显示 `RF 概率`
- `VCR` 作为辅助特征展示
- 当统一模型不可用时，回退到与当前阈值对齐的简单 VCR 规则

诊断页最适合解释项目医学逻辑：正常静脉在压迫下面积明显下降，DVT 患者静脉面积更稳定，因此分类器会重点利用面积比例、消失率、变异系数和形状变化特征。

### 7. 导出报告

将当前案例结果导出为 PDF，汇总：
- 案例信息
- 检测结果
- 分割指标
- 面积曲线
- DVT 诊断结果

生成 PDF 前需要已经完成分割和诊断。上传视频没有 GT 时，报告会跳过 GT 相关指标，但仍保留检测、面积曲线和诊断结论。

## 代码结构

```text
web/
├── app.py
├── README.md
├── assets/
│   └── custom.css
├── services/
│   ├── __init__.py
│   └── inference.py
├── tabs/
│   ├── dashboard.py
│   ├── upload.py
│   ├── pipeline.py
│   ├── detection.py
│   ├── segmentation.py
│   ├── diagnosis.py
│   └── evaluation.py
└── utils/
    ├── visualization.py
    ├── metrics.py
    ├── chart_style.py
    └── ui.py
```

## 关键模块

### `app.py`

负责：
- 构建 Gradio `Blocks`
- 生成 header 和全局摘要条
- 注册顶层 Tabs
- 维护全局 `gr.State`

### `services/inference.py`

`InferenceService` 是 Web 的统一推理入口，负责：
- 惰性加载 YOLO
- 惰性加载 SAM2 + LoRA
- 惰性加载分类器
- 统一封装检测、分割、诊断接口

默认权重定义也在这里：
- `DEFAULT_YOLO_MODEL`
- `DEFAULT_SAM2_VARIANT`
- `DEFAULT_LORA_WEIGHTS`
- `DEFAULT_YOLO_PRIOR`

### `utils/ui.py`

当前版本新增的小型 UI helper，用来统一：
- 页头
- 空态
- 说明块
- 诊断摘要卡

### `assets/custom.css`

当前 Web 的主要视觉系统：
- 顶部栏
- 摘要条
- 标签导航
- 卡片与表格风格
- 上传区与图库样式

## 状态流转

Web 使用一个全局 `gr.State` 在各个页面之间共享病例上下文。

```text
upload.py
  └─ 写入 frame_files / masks_dir / case info

detection.py
  └─ 写入 detections

segmentation.py
  └─ 写入 pred_masks / vein_areas / artery_areas / frame_metrics

diagnosis.py
  └─ 读取 pred_masks + vein_areas，输出诊断结果

evaluation.py
  └─ 读取已有结果，导出 PDF
```

## 降级行为

当部分模型或权重缺失时，Web 会自动降级，保证界面仍可演示：

| 模块 | 降级方式 |
|------|----------|
| YOLO 检测 | 从 GT mask 提取框 |
| SAM2 分割 | 用 GT mask 模拟传播结果 |
| DVT 诊断 | 回退到简单 VCR 规则 |

## 依赖

最少依赖包括：

```text
gradio >= 6.0
torch
ultralytics
opencv-python
numpy
matplotlib
scikit-learn
scipy
```

推荐直接使用项目环境：

```bash
conda activate echodvt
```

## 说明

- 当前 Web 是“固定最优配置”的诊断界面，不是实验面板。
- 如果要做消融实验、切换 LoRA rank、测试 MFP/RPA/OKM/DAM 参数，请直接使用 `sam2/` 和 `yolo/` 下的脚本。
- Web 文档只描述当前主线实现；算法细节请分别看：
  - [../README.md](../README.md)
  - [../yolo/README.md](../yolo/README.md)
  - [../sam2/README_EchoDVT.md](../sam2/README_EchoDVT.md)
