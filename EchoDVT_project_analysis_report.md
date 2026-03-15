# EchoDVT 项目详细解析报告

基于仓库当前工作树与已有训练/推理产物整理，时间点为 `2026-03-15`。本报告重点解析两件事：

1. EchoDVT 整体工程是如何把 `YOLO + SAM2 + 二分类` 串成一个 DVT 自动诊断系统的。
2. `SAM2` 相关改造里，哪些是稳定主线，哪些是 `LoRA` 微调核心，哪些是后续叠加的实验模块，以及它们现在到底处于什么状态。

---

## 1. 项目整体定位

从根目录 `README.md` 可以看出，EchoDVT 的目标不是单一分割模型，而是一个完整的超声视频 DVT 辅助诊断系统。主流程是：

`YOLO 首帧检测` -> `SAM2 视频分割` -> `时序特征分类`

对应说明见 `README.md:1-150`，其中最重要的信息是：

- 任务对象是压缩超声视频中的动脉和静脉。
- `YOLO` 负责先给出 artery / vein 的框提示。
- `SAM2` 负责视频级传播分割。
- 最后用静脉在压迫过程中的形变特征做 DVT 二分类。

这意味着 EchoDVT 的核心不是“让 SAM2 从零学会超声分割”，而是“让 YOLO 负责定位，让 SAM2 负责时序传播，让分类器负责诊断解释”，三个模块各司其职。

---

## 2. 仓库结构与职责划分

当前仓库可以分成三层：

### 2.1 业务主线

- `README.md`
  - 定义整体任务、数据集、命令和实验结论。
- `classify_dvt.py`
  - 把分割输出变成 DVT 诊断结果。
- `web/`
  - Gradio Web 应用，做上传、检测、分割、诊断、评估、对比展示。

### 2.2 检测模块

- `yolo/`
  - 训练脚本、先验统计、推理脚本。
  - `yolo/prior_stats.json` 用于在检测缺框时自动补全 artery / vein。

### 2.3 分割模块

- `sam2/`
  - 既包含 Meta 原始 SAM2 代码，也包含 EchoDVT 的定制改造。
  - 真正的核心定制几乎都在这里：
    - `train_lora.py`
    - `inference_lora.py`
    - `sam2/sam2/lora_sam2.py`
    - `sam2/sam2/dvt_dataset.py`
    - `sam2/sam2/sam2_video_trainer.py`
    - `sam2/sam2/postprocess.py`
    - `sam2/sam2/sam2_video_predictor.py`
    - `sam2/sam2/modeling/sam2_base.py`

结论很明确：EchoDVT 的工程创新主要不是 Web 或分类器，而是 `SAM2` 这一层的“任务适配”和“视频记忆控制”。

---

## 3. 端到端数据流

### 3.1 检测阶段

`sam2/inference_box_prompt_large.py` 中的 `VesselDetector` 会在首帧做 YOLO 检测，并在缺框时用先验补框，核心入口见：

- `sam2/inference_box_prompt_large.py:341-367`
- `sam2/inference_box_prompt_large.py:744-762`

这一步的工程意义很大：

- SAM2 在这里不是自由分割，而是被约束成“跟踪两个目标对象”。
- YOLO 的框把问题从“整幅图语义分割”缩成“围绕 artery / vein 的目标传播”。
- 缺框补全让系统鲁棒性更高，避免某一类目标漏检时整个视频传播失效。

### 3.2 分割阶段

分割基线是只在首帧加 box prompt，之后让 SAM2 纯靠 memory 往后传播。这个基本范式没有改变，只是 EchoDVT 在两个方向上做了增强：

1. 训练侧：用 LoRA 对 SAM2 做低成本任务适配。
2. 推理侧：加多帧 prompt 和记忆控制策略，减少时序漂移。

### 3.3 诊断阶段

`classify_dvt.py:45-176` 定义了一系列时序特征，核心都围绕“静脉是否在压迫中塌陷”：

- `vcr` 静脉压缩比
- `vdr` 静脉消失率
- `vein_cv` 静脉面积变异系数
- `varr` 静脉面积相对范围
- `mvar` 最小静脉/动脉面积比
- `vein_slope` 静脉面积趋势
- `max_drop_ratio` 最大下降比
- `circ_cv` / `circ_min` / `circ_range` 圆度变化

这说明 EchoDVT 的最终诊断解释性其实很强：分类器并不是黑箱 CNN，而是建立在分割后的形态动力学特征之上。

---

## 4. SAM2 基线结构到底是什么

当前主模型配置来自 `sam2/sam2/configs/sam2/sam2_hiera_l.yaml:4-117`，关键信息如下：

- 主干是 `Hiera Large`
  - `stages: [2, 6, 36, 4]`
  - 总 block 数 = `2 + 6 + 36 + 4 = 48`
- `memory_attention.num_layers = 4`
- 每层有两套注意力：
  - `self_attn`
  - `cross_attn_image`
- 每套注意力有四个线性投影：
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `out_proj`

因此 EchoDVT 后面做 LoRA 注入时，目标数量是可以直接算出来的：

- Image Encoder：`48` 个 Hiera block
- Memory Attention：`4 layers * 2 attentions * 4 projections = 32` 个投影层

这也解释了为什么 README 中写“Image Encoder QKV (48 层)”和“Memory Attention 32 投影”，这不是拍脑袋，是由配置直接决定的。

---

## 5. SAM2 的 LoRA 微调实现

这一部分是本项目最核心的工程定制。

### 5.1 训练入口

训练主脚本是 `sam2/train_lora.py`。

文件顶部已经写清楚训练策略，见 `sam2/train_lora.py:16-21`：

- 冻结绝大部分 SAM2 参数
- 对 Image Encoder QKV 注入 LoRA
- 对 Memory Attention 注入 LoRA
- 对 Mask Decoder 做全量微调
- 可选解冻 Memory Encoder

这套思路不是纯 LoRA，也不是纯全量微调，而是“LoRA + 小头全量微调”的混合方案。

### 5.2 LoRA 注入点

真正的注入实现位于 `sam2/sam2/lora_sam2.py`。

#### 5.2.1 Image Encoder 的 LoRA

`_LoRA_qkv` 定义见 `sam2/sam2/lora_sam2.py:22-82`。它的策略是：

- 保留原始 `qkv` 线性层
- 只给 `Q` 和 `V` 增加低秩增量
- `K` 保持不变
- 输出形式是：
  - `q = q_orig + delta_q`
  - `k = k_orig`
  - `v = v_orig + delta_v`

注入过程见 `sam2/sam2/lora_sam2.py:190-235`：

- 遍历 `self.predictor.image_encoder.trunk.blocks`
- 对每个 block 的 `blk.attn.qkv` 做替换
- 为每个 block 分别创建 `Q` 和 `V` 的 `A/B` 矩阵

这意味着 Image Encoder 的 LoRA 不是只打一两层，而是覆盖全部 `48` 个 Hiera block。

#### 5.2.2 Memory Attention 的 LoRA

普通线性层 LoRA 包装器 `_LoRA_Linear` 定义在 `sam2/sam2/lora_sam2.py:85-103`。

Memory Attention 注入逻辑见 `sam2/sam2/lora_sam2.py:236-275`：

- 遍历 `memory_attention.layers`
- 对 `self_attn` 和 `cross_attn_image`
- 在 `q_proj/k_proj/v_proj/out_proj` 上全部加 LoRA

这是一个很重要的设计选择：

- Image Encoder LoRA 解决“超声图像特征不一样”的问题。
- Memory Attention LoRA 解决“超声视频传播关系不一样”的问题。

也就是说，这个仓库不是只把 SAM2 当图像编码器改一下，而是连视频记忆交互层都做了适配。

#### 5.2.3 Mask Decoder 全量微调

在 `sam2/sam2/lora_sam2.py:167-182` 中，代码先冻结所有参数，再：

- 应用 LoRA
- 最后把 `self.predictor.sam_mask_decoder.parameters()` 全部 `requires_grad = True`

这是 EchoDVT 训练策略里最关键也最容易被误读的一点：

- 这个项目并不是“只训练 LoRA 参数”
- 它还全量训练了整个 `mask decoder`

这会显著提高任务适配能力，但也意味着：

- 实际 trainable 参数量比 README 里写的更大
- 保存的 checkpoint 也不会是“纯 LoRA 小文件”

### 5.3 实际参数量，不是 README 里的近似值

我直接在当前代码上实例化了模型并统计了参数，结果如下。

#### r = 4

- 总参数：`224.92M`
- 可训练参数：`4.704M`
- LoRA 参数：`0.489M`
- 其中：
  - Image Encoder LoRA：`0.4297M`
  - Memory Attention LoRA：`0.0594M`
  - Mask Decoder：`4.215M`

#### r = 8

- 总参数：`225.41M`
- 可训练参数：`5.193M`
- LoRA 参数：`0.978M`
- 其中：
  - Image Encoder LoRA：`0.8594M`
  - Memory Attention LoRA：`0.1188M`
  - Mask Decoder：`4.215M`

所以需要明确区分两个概念：

- “LoRA 参数量”
  - r=4 时约 `0.489M`
  - r=8 时约 `0.978M`
- “总可训练参数量”
  - 因为还包含完整 `mask decoder`
  - 实际是 `4.7M ~ 5.2M`

结论：

- README 里“总计约 0.5M”的说法，只能近似理解成“LoRA 本体量级”。
- 如果按真实训练口径算，这个项目是“小规模微调”，不是“极限参数高效微调”。

### 5.4 训练数据是怎么喂进来的

自定义数据集是 `sam2/sam2/dvt_dataset.py`。

关键逻辑见：

- `sam2/sam2/dvt_dataset.py:91-146`
- `sam2/sam2/dvt_dataset.py:169-214`

它做了几件事：

1. 按 case 组织视频，而不是按单帧组织样本。
2. 要求首个 prompt 帧必须同时有 artery 和 vein 的 mask。
3. 从首帧 mask 中提取 artery / vein box，作为训练时 prompt。
4. 其余有标注帧保留为监督信号。
5. 可对 box 做 `jitter`，模拟 YOLO 检测误差。

这个数据设计非常符合任务本质：

- 训练目标不是“给一帧分割一帧”，而是“给首帧 prompt，学会整段视频传播”。
- `box_jitter` 很关键，它让训练分布更接近真实推理时的 YOLO 框，而不是理想 GT 框。

### 5.5 为什么要单独造一个可训练版 SAM2VideoTrainer

原版 `SAM2VideoPredictor` 的很多方法带有 `@torch.inference_mode()`，这对推理没问题，但训练时会切断梯度。

所以 EchoDVT 新造了 `sam2/sam2/sam2_video_trainer.py`，核心目标是：

- 去掉推理装饰器
- 保留视频传播流程
- 允许反向传播穿过整条时序路径

关键实现见：

- `init_state_train`：`sam2/sam2/sam2_video_trainer.py:78-141`
- `add_new_points_or_box_train`：`sam2/sam2/sam2_video_trainer.py:143-259`
- `_run_single_frame_inference_train`：`sam2/sam2/sam2_video_trainer.py:261-343`
- `propagate_in_video_train`：`sam2/sam2/sam2_video_trainer.py:345-481`
- `_compute_frame_loss`：`sam2/sam2/sam2_video_trainer.py:568-652`

这里最关键的工程点有三个：

#### 5.5.1 保留 low-res mask 用于监督

`_run_single_frame_inference_train` 明确把 `current_out["pred_masks"]` 当作 low-res mask 保留下来，见 `sam2/sam2/sam2_video_trainer.py:320-341`。

这样训练损失不是在最终 resize 到原图后的 mask 上算，而是在模型原生 decoder 输出尺度上算，更稳定、更接近 SAM2 内部表示。

#### 5.5.2 条件帧只做 prompt，不重复编码记忆

`add_new_points_or_box_train` 在首帧只做“带 prompt 的单帧推理”，不立即运行 memory encoder，见 `sam2/sam2/sam2_video_trainer.py:240-252`。

之后由 `_propagate_in_video_preflight_train` 统一把条件帧结果编码成记忆，见 `sam2/sam2/sam2_video_trainer.py:483-530`。

这样做的好处是流程和原版 predictor 的“先收集交互，再开始传播”非常一致。

#### 5.5.3 `detach_memory=True` 是显存换效果的折中

`propagate_in_video_train` 默认会把写入 memory bank 的特征 `detach`，见 `sam2/sam2/sam2_video_trainer.py:436-455`。

这意味着：

- 梯度不会沿整段视频无限回传
- 显存可控
- 训练更像 truncated BPTT

对于超声视频这种帧数较多、单卡训练资源有限的场景，这是一个非常现实的工程折中。

### 5.6 损失函数设计

训练损失在 `sam2/train_lora.py:55-85` 中定义：

- `Dice Loss`
- `Sigmoid Focal Loss`
- 两者相加为 `combined_loss`

而在 `sam2/sam2/sam2_video_trainer.py:645-650` 中，loss 还会按类别加权：

- artery 权重默认 `1.0`
- vein 权重默认 `1.5`

这个设计非常合理，因为在 EchoDVT 任务里：

- artery 通常更稳定、边界更清楚
- vein 更容易漂移、消失，也是最终诊断更关键的目标

所以训练实际上是“向静脉召回倾斜”的。

### 5.7 单个 case 的训练路径

`train_one_case` 在 `sam2/train_lora.py:108-206`，它完整体现了本项目的训练哲学：

1. 初始化整段视频的 state。
2. 在首帧为 artery / vein 各加一个 box prompt。
3. 让模型向后传播。
4. 只在有标注的帧上算损失。

这不是逐帧 teacher forcing，而是“首帧交互 + 时序传播 + 稀疏监督”。

从任务视角看，这非常贴合真实使用方式：

- 临床上系统也只会从少量提示开始。
- 后面必须靠视频记忆维持稳定性。

### 5.8 训练循环与实验设置

训练主循环在 `sam2/train_lora.py:301-532`：

- 数据集：
  - train split 做增强和 box jitter
  - val split 不增强
- 优化器：
  - `AdamW`
- 学习率：
  - `CosineAnnealingLR`
- 数值策略：
  - `AMP + GradScaler`
- 小 batch 解决方式：
  - 梯度累积 `grad_accum`
- 稳定性：
  - `clip_grad_norm_`
- 保存策略：
  - 周期性保存
  - 按 `val_mean_dice` 保存最佳权重

当前已有两组关键 LoRA 训练产物：

- `sam2/checkpoints/lora_runs/lora_r4_lr0.0005_e25_20260314_153134`
- `sam2/checkpoints/lora_runs/lora_r8_lr0.0003_e25_20260314_153210`

对应训练日志显示：

- r4 最佳 `val_mean_dice = 0.84845`
- r8 最佳 `val_mean_dice = 0.85391`

也就是说，在当前实验里，`r=8` 比 `r=4` 更优，但优势不算特别大。

### 5.9 权重保存格式的真实含义

`save_lora_parameters` 见 `sam2/sam2/lora_sam2.py:327-366`。

从命名看像是在“只保存 LoRA”，但实际保存内容是：

1. 所有 `w_a_xxx` / `w_b_xxx`
2. 全部 `mask_decoder` 参数
3. 若启用 memory attention LoRA，则保存整个 `memory_attention` state_dict
4. 可选保存 `memory_encoder`

我直接读取了 `lora_best.pt`：

- r8 checkpoint 中共有 `558` 个 key
- `w_a` 数量 `128`
- `w_b` 数量 `128`
- 总保存参数量约 `11.23M`

所以这个 checkpoint 不是“纯 LoRA adapter 文件”，而是“LoRA + 全量微调 head + 一部分完整模块状态”的混合权重包。

这带来两个实际影响：

- checkpoint 比预期大很多
- 如果未来要做真正通用的 LoRA 复用，这种保存方式不够干净

---

## 6. 推理侧增强模块

EchoDVT 在推理阶段又叠了三类增强：

1. `MFP`：多帧 prompt
2. `OKM`：关键帧记忆
3. `DAM`：目标消失感知记忆

其中真正稳定且最有效的是 `MFP`。

### 6.1 MFP：Multi-Frame Prompting

实现分两部分：

- 选择额外 prompt 帧：`sam2/sam2/postprocess.py:22-106`
- 在推理时把这些 frame 直接作为 conditioning frame 注入：`sam2/inference_lora.py:183-270`

其机制非常简单，但非常聪明：

1. 每隔 `interval` 帧做一次 YOLO 检测。
2. 只有 artery 和 vein 都达到置信度阈值，才把该帧作为额外 prompt。
3. 在 `predictor.init_state()` 之后，对这些帧额外调用 `add_new_points_or_box()`。
4. 之后仍然使用 SAM2 原生 `propagate_in_video()`。

重点在于：`MFP` 不修改 SAM2 内部 memory bank，它只是向 SAM2 提供更多 conditioning frame。

这在工程上有三个好处：

- 风险低：不碰底层传播逻辑。
- 与 LoRA 正交：LoRA 改参数，MFP 改输入。
- 可解释：本质上是“周期性重新锚定”。

从效果看，MFP 是本项目当前最成功的新增模块。

### 6.2 OKM：Object-Aware Keyframe Memory

这是一个真正修改 SAM2 内部记忆选择的模块，相关改动分布在两个文件：

- 记忆选择注入：`sam2/sam2/modeling/sam2_base.py:545-600`
- 关键帧跟踪与更新：`sam2/sam2/sam2_video_predictor.py:587-646`

逻辑可以概括成两步：

#### 第一步：每个目标维护一个“最佳关键帧”

在 `propagate_in_video` 中，代码会为每个 obj 记录 mask 面积最大的帧作为 keyframe，见 `sam2/sam2/sam2_video_predictor.py:639-646`。

这里的直觉是：

- 对血管而言，面积大的帧通常意味着目标完整、可见、质量更高。
- 尤其对静脉来说，压迫后会塌陷，后续帧可能越来越难分。

#### 第二步：如果关键帧不在正常 stride 选出来的 memory window 里，就强行塞回去

这部分发生在 `sam2_base.py:584-600`：

- 先按原始 SAM2 规则选最近几帧 memory
- 如果 keyframe 不在已选 memory 中
- 就用它替换最老的 memory slot

它的设计目标很明确：

- 保留一个“目标最清楚时刻”的参考记忆
- 防止后续由于静脉塌陷或漂移，memory bank 里全变成低质量帧

### 6.3 DAM：Disappearance-Aware Memory

实现位于 `sam2/sam2/sam2_video_predictor.py:648-662`。

机制也很直接：

- 如果当前预测 mask 面积还够大，说明目标可见，就把这帧记成 `last_good_mem`
- 如果当前目标几乎消失，而且这帧属于非条件传播帧
- 就不要把“空目标记忆”写进 bank，而是复用上一帧好记忆

它想解决的问题是：

- 静脉在压迫过程中真的会接近消失
- 如果 SAM2 把“空 mask”也编码成 memory，后续可能越来越偏向 no-object

这个思路在任务上是合理的，但从当前实验结果看，现版本实现并没有带来收益。

---

## 7. `inference_lora.py` 是如何把这些模块串起来的

推理主线全部在 `sam2/inference_lora.py`。

### 7.1 LoRA 模型构建

`LoRASAM2VideoSegmenter` 见 `sam2/inference_lora.py:70-181`。

流程是：

1. 用 `build_lora_sam2_video(... use_trainer=False)` 构建推理版模型。
2. 加载 `lora_weights`。
3. 拿到内部的 `predictor` 句柄。

### 7.2 MFP 注入

`segment_video_with_first_frame_prompt` 见 `sam2/inference_lora.py:183-270`。

先做：

- frame 0 的 artery / vein prompt

再做：

- 额外 prompt frame 的 artery / vein prompt

最后才调用：

- `predictor.propagate_in_video()`

这说明 MFP 确实完全遵守“先加 conditioning，再统一传播”的 SAM2 原生范式。

### 7.3 OKM / DAM 打开方式

在 `run()` 里，代码只是简单给 predictor 追加两个运行时属性，见 `sam2/inference_lora.py:371-376`：

- `segmenter.predictor.use_okm = True`
- `segmenter.predictor.use_dam = True`

也就是说：

- `OKM/DAM` 不是通过 config 构建时注册的
- 而是通过运行时 flag 触发 predictor 内部的分支逻辑

这是一种很轻量的实验接法，优点是快，缺点是工程规范性一般。

---

## 8. 实验结果解读

当前仓库里已经有多组推理评估结果，主要集中在：

- `sam2/predictions/sam2_lora_yolo_box/*/summary.json`

我只采用完整验证集 `598` 帧的结果进行比较。

### 8.1 LoRA r4 vs r8

完整验证集结果：

| 配置 | Frame-weighted Dice | Artery Dice | Vein Dice |
|------|---------------------|-------------|-----------|
| r4 baseline | 0.7694 | 0.8387 | 0.7001 |
| r8 baseline | 0.7692 | 0.8354 | 0.7029 |

结论：

- `r8` 的静脉 Dice 稍高。
- `r4` 的动脉 Dice 略高。
- 两者整体非常接近。
- 从训练集小样本验证看，`r8` 的 best val 更高，所以项目后续主力使用了 `r8`。

### 8.2 MFP 的收益

| 配置 | Frame-weighted Dice | Vein Dice |
|------|---------------------|-----------|
| r8 baseline | 0.7692 | 0.7029 |
| r8 + MFP | 0.7853 | 0.7166 |

结论：

- 总 Dice 提升约 `+1.61` 个点。
- 静脉 Dice 提升约 `+1.37` 个点。
- 这是当前最明确、最稳定的增益来源。

为什么它有效：

- EchoDVT 的难点本来就集中在静脉时序漂移。
- MFP 相当于每隔一段时间重新给 SAM2 一个可靠锚点。

### 8.3 OKM 的收益

| 配置 | Frame-weighted Dice | Vein Dice |
|------|---------------------|-----------|
| r8 baseline | 0.7692 | 0.7029 |
| r8 + OKM | 0.7691 | 0.7010 |
| r8 + MFP + OKM | 0.7863 | 0.7170 |

结论：

- 单独开 `OKM` 基本没有收益，甚至略退。
- 和 `MFP` 叠加时有极小增益，但幅度非常有限。

这说明：

- “保留最佳关键帧”这个想法不是错的
- 但在当前实现里，收益远小于 MFP
- 它更像辅助修补，而不是主力提升项

### 8.4 DAM 的收益

| 配置 | Frame-weighted Dice | Vein Dice |
|------|---------------------|-----------|
| r8 baseline | 0.7692 | 0.7029 |
| r8 + DAM | 0.7672 | 0.6935 |
| r8 + MFP + DAM | 0.7805 | 0.7059 |

结论：

- `DAM` 单独使用明显退化。
- 和 `MFP` 叠加后仍然比 `MFP` 单独更差。

这意味着当前版本的 DAM 机制虽然“想法合理”，但实现上仍然会破坏有效记忆分布，至少在现有数据和阈值下还不成熟。

### 8.5 当前最优组合

从现有产物看，最值得保留的主线是：

- `LoRA r8 + MFP`

如果只看数值，`MFP + OKM` 略高于 `MFP`，但优势非常小，还没有达到“必须保留”的程度。

---

## 9. 当前代码状态中的关键不一致与风险

这一部分非常重要，因为它关系到“项目现在能不能被稳定复现”。

### 9.1 `adaptive_memory.py` 实际已经被移除

`sam2/sam2/adaptive_memory.py:1-3` 现在只剩下注释：

- 旧版 AM / SM / AV 模块已经移除
- 原因是它们会干扰 LoRA 学到的特征
- 现在推荐的是 `postprocess.py` 中的 MFP

这意味着项目发展路径已经发生了明确转向：

- 早期尝试是“改内部记忆机制”
- 当前稳定方案是“保留 LoRA + 外部 prompt 增强”

### 9.2 但是训练器里还残留了已移除模块的接口

`sam2/sam2/sam2_video_trainer.py:124-135` 仍然会在 `init_state_train` 里访问：

- `self.use_adaptive_memory`
- `self.use_separate_memory`
- `SeparateMemoryBank`

问题在于：

- 当前 predictor 上并没有这些属性
- `sam2/sam2/build_sam.py:100-142` 里的 `build_sam2_video_predictor()` 虽然接受 `**kwargs`，但根本没有把这些参数传进模型实例
- `sam2/sam2/adaptive_memory.py` 也已经没有 `SeparateMemoryBank` 实现

我在当前工作树上实际做了最小复现，结果是：

- 构建 `use_trainer=True` 的 LoRA 模型是可以成功的
- 但一旦调用 `init_state_train(...)`
- 会报错：
  - `AttributeError: 'SAM2VideoTrainer' object has no attribute 'use_adaptive_memory'`

这说明一个非常关键的事实：

- 仓库里现有的 LoRA checkpoint 证明“这套训练曾经跑通过”
- 但按当前工作树直接重跑训练，接口已经出现断裂

换句话说，项目代码和实验产物之间存在版本漂移。

### 9.3 `inference_box_prompt_large.py` 的 AM/SM/AV 开关现在基本是空接线

在 `sam2/inference_box_prompt_large.py:348-366` 和 `:754-762` 中，基线推理脚本仍然会把这些参数传给 `build_sam2_video_predictor()`：

- `use_adaptive_memory`
- `use_separate_memory`
- `use_av_constraint`

CLI 也仍然暴露这些选项，见 `sam2/inference_box_prompt_large.py:1041-1066`。

但结合 `build_sam.py:100-142` 可以看到：

- 构建函数并没有真正使用这些参数
- 当前工作树里这些开关事实上不会生效

因此：

- 这些选项在代码层面是“遗留接口”
- 在当前版本里不应再被当成主功能理解

### 9.4 注释与实现存在轻微不一致

有几处注释已经落后于实现：

- `sam2/sam2/sam2_video_trainer.py:4-8` 还写着“复用所有自适应记忆模块”
- `sam2/sam2/postprocess.py:7-12` 的注释提到 `RelativePositionAnchor`，但当前文件主体只剩 `MultiFramePrompter`

这说明 EchoDVT 的 `SAM2` 分支经历过多轮实验迭代，代码已经进入“主线保留、旧注释未完全清理”的状态。

### 9.5 当前工作树是脏的，OKM/DAM 还处于本地实验态

`git status` 显示以下文件处于修改状态：

- `sam2/inference_lora.py`
- `sam2/sam2/modeling/sam2_base.py`
- `sam2/sam2/sam2_video_predictor.py`

这意味着：

- `OKM/DAM` 相关逻辑很可能是近期加入、尚未完全沉淀的本地实验
- 当前最可信的稳定主线仍然是：
  - `LoRA`
  - `MFP`

---

## 10. 我对这个项目演进路线的判断

结合代码、注释、已有 checkpoint 和评估结果，EchoDVT 的 `SAM2` 路线大概率经历了三阶段：

### 第一阶段：先用原版 SAM2 跑通视频传播

对应基线脚本：

- `sam2/inference_box_prompt_large.py`

目标是先验证：

- YOLO box prompt 是否足够支撑视频分割
- 首帧提示 + memory propagation 的框架是否成立

### 第二阶段：尝试在内部记忆机制上做很多手工增强

对应痕迹：

- `SAM2_0121_INTEGRATION.md`
- `adaptive_memory.py` 的残留注释
- `use_adaptive_memory` / `use_separate_memory` / `use_av_constraint` 的遗留接口

这一阶段的思路是“深入改 SAM2 内部记忆逻辑”，但后来被证明：

- 工程复杂度高
- 与 LoRA 学到的特征可能互相干扰
- 维护成本大

### 第三阶段：收敛到“LoRA + 外部增强”主线

当前最稳定的方案正是：

- LoRA 负责把 SAM2 适配到超声血管域
- MFP 在输入侧给更多锚点
- 必要时再用 OKM 这种轻量内存修补

这条路线之所以更靠谱，是因为它遵守了一个非常好的工程原则：

- 尽量少碰原始 SAM2 内核
- 优先在 adapter 和输入端解决问题

---

## 11. 对 `SAM2 LoRA` 的最终结论

如果只看 `SAM2` 相关部分，这个项目最有价值的不是单个技巧，而是形成了一套完整的、可运行的超声视频微调范式：

### 11.1 它不是“往 SAM2 上套一个 LoRA”那么简单

它实际上做了四层事情：

1. 任务数据重组
   - 把数据从静态 mask 样本变成“视频 case + 稀疏标注帧 + 首帧 prompt”
2. 模型微调重构
   - 让原本只适合推理的 predictor 变成可训练 video trainer
3. 低秩适配
   - 给 image encoder 和 memory attention 注入 LoRA
4. 推理增强
   - 用 MFP/OKM/DAM 等模块去解决长视频传播漂移

### 11.2 其中最成功的两个模块是

- `LoRA_SAM2_Video`
- `MultiFramePrompter`

理由：

- `LoRA_SAM2_Video` 解决了“超声域适配”问题
- `MultiFramePrompter` 解决了“长时漂移”问题
- 两者一个改参数，一个改输入，互补性非常强

### 11.3 当前最需要修的不是算法，而是工程一致性

如果后续要继续在这条线上推进，我认为最优先的工作不是再发明新 heuristics，而是先把下面三件事做干净：

1. 修复 `SAM2VideoTrainer` 中残留的 `adaptive_memory` 断链接口。
2. 明确区分：
   - 纯 LoRA 参数
   - mask decoder 全量微调参数
   - memory attention 完整保存状态
3. 清理无效 CLI 和过期注释，让“当前主线方案”表达得更一致。

---

## 12. 最终总结

一句话概括 EchoDVT：

这是一个以 `YOLO prompt + SAM2 视频传播 + LoRA 域适配 + 时序形态分类` 为核心的 DVT 自动诊断系统，而它最成熟、最可信的创新主线，是 `SAM2 LoRA 微调 + MFP 多帧提示`。

更具体地说：

- `LoRA` 让 SAM2 从通用视频分割模型变成了适配超声血管视频的专用模型。
- `MFP` 用非常小的工程代价显著缓解了静脉时序漂移。
- `OKM` 有一点潜力，但收益很小。
- `DAM` 当前版本还不成熟。
- 老的自适应记忆分支已经基本退出主线，但代码接口还没有完全清理。

所以，如果把这个项目的 SAM2 部分抽象成一句工程判断：

> EchoDVT 已经找到了一条有效的超声视频分割适配路线，但当前代码库还处在“实验主线已明确、工程清理尚未完成”的阶段。

