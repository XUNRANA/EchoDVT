"""
LoRA for SAM2 Video Predictor
=============================
低秩适配器 (Low-Rank Adaptation) 用于SAM2视频分割的高效微调

优势:
1. 参数量从 11.72M 降到 ~0.5M
2. 显存占用大幅减少
3. 训练更快
4. 保留原始模型能力

参考: https://github.com/JamesQFreeman/Sam_LoRA
"""

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import List, Optional, Dict, Any


class _LoRA_qkv(nn.Module):
    """
    LoRA层用于替换QKV线性层

    原始SAM2实现:
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(0)

    LoRA修改:
        qkv = original_qkv(x) + scale * [delta_q, 0, delta_v]
        delta_q = B_q @ A_q(x)  # 低秩分解
        delta_v = B_v @ A_v(x)
        scale = lora_alpha / r  # 标准LoRA缩放
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        lora_alpha: float = 1.0,
        r: int = 4,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        # 注意: dim是每个Q/K/V的输出维度，不是输入维度!
        self.dim = qkv.out_features // 3
        # LoRA 缩放因子: alpha/r 控制LoRA贡献的大小
        self.scaling = lora_alpha / r

    def forward(self, x):
        # 原始 QKV (不要in-place修改!)
        qkv = self.qkv(x)  # [..., 3*dim_out]

        # LoRA 增量 (带缩放)
        new_q = self.linear_b_q(self.linear_a_q(x)) * self.scaling
        new_v = self.linear_b_v(self.linear_a_v(x)) * self.scaling

        # 调试: 检查形状是否匹配
        if qkv.shape[-1] != 3 * self.dim:
            print(f"[LoRA DEBUG] Shape mismatch!")
            print(f"  qkv shape: {qkv.shape}, expected last dim: {3 * self.dim}")
            print(f"  self.dim (out_dim): {self.dim}")
            print(f"  new_q shape: {new_q.shape}")
            print(f"  linear_a_q: in={self.linear_a_q.in_features}, out={self.linear_a_q.out_features}")
            print(f"  linear_b_q: in={self.linear_b_q.in_features}, out={self.linear_b_q.out_features}")

        # 分离 Q, K, V 并添加 LoRA 增量 (避免in-place操作)
        q = qkv[..., :self.dim] + new_q
        k = qkv[..., self.dim:2*self.dim]  # K 保持不变
        v = qkv[..., 2*self.dim:] + new_v

        # 重新拼接
        return torch.cat([q, k, v], dim=-1)


class _LoRA_Linear(nn.Module):
    """
    LoRA层用于普通线性层

    y = W @ x + B @ A @ x
    """
    def __init__(
        self,
        original_linear: nn.Module,
        linear_a: nn.Module,
        linear_b: nn.Module,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.linear_a = linear_a
        self.linear_b = linear_b

    def forward(self, x):
        return self.original_linear(x) + self.linear_b(self.linear_a(x))


class LoRA_SAM2_Video(nn.Module):
    """
    SAM2 Video Predictor 的 LoRA 微调版本

    Args:
        predictor: SAM2VideoPredictor 或 SAM2VideoTrainer 实例
        r: LoRA 秩 (rank), 越小参数越少但表达能力降低
        lora_layer: 要应用LoRA的层索引列表, None表示所有层
        apply_to_image_encoder: 是否对image_encoder应用LoRA
        apply_to_memory_attention: 是否对memory_attention应用LoRA
        apply_to_memory_encoder: 是否对memory_encoder应用LoRA

    Usage:
        predictor = build_sam2_video_predictor(...)
        lora_model = LoRA_SAM2_Video(predictor, r=4)

        # 训练
        for batch in dataloader:
            loss = lora_model.compute_loss(batch)
            loss.backward()
            optimizer.step()

        # 保存
        lora_model.save_lora_parameters("lora_weights.pt")
    """

    def __init__(
        self,
        predictor,
        r: int = 4,
        lora_alpha: float = 1.0,
        lora_layer: Optional[List[int]] = None,
        apply_to_image_encoder: bool = True,
        apply_to_memory_attention: bool = True,
        apply_to_memory_encoder: bool = False,
    ):
        super().__init__()

        assert r > 0, "LoRA rank must be positive"

        self.predictor = predictor
        self.r = r
        self.lora_alpha = lora_alpha  # LoRA缩放参数，通常设为r或1
        self.apply_to_image_encoder = apply_to_image_encoder
        self.apply_to_memory_attention = apply_to_memory_attention
        self.apply_to_memory_encoder = apply_to_memory_encoder

        # 存储 LoRA 层
        self.w_As = nn.ModuleList()
        self.w_Bs = nn.ModuleList()

        # 设置要应用LoRA的层
        if apply_to_image_encoder:
            num_blocks = len(self.predictor.image_encoder.trunk.blocks)
            if lora_layer is None:
                self.lora_layer = list(range(num_blocks))
            else:
                self.lora_layer = lora_layer
        else:
            self.lora_layer = []

        # 先冻结所有参数
        for param in self.predictor.parameters():
            param.requires_grad = False

        # 应用 LoRA
        self._apply_lora_to_image_encoder()

        if apply_to_memory_attention:
            self._apply_lora_to_memory_attention()

        if apply_to_memory_encoder:
            self._apply_lora_to_memory_encoder()

        # 解冻 mask_decoder (全量微调这部分,因为参数量小)
        for param in self.predictor.sam_mask_decoder.parameters():
            param.requires_grad = True

        # 初始化 LoRA 参数
        self.reset_parameters()

        # 统计参数
        self._print_trainable_parameters()

    def _apply_lora_to_image_encoder(self):
        """对 Image Encoder 的注意力层应用 LoRA"""
        if not self.apply_to_image_encoder:
            return

        print(f"Applying LoRA to Image Encoder ({len(self.lora_layer)} layers, r={self.r})")

        for t_layer_i, blk in enumerate(self.predictor.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            # 获取 QKV 线性层
            w_qkv_linear = blk.attn.qkv
            in_dim = w_qkv_linear.in_features  # 输入维度
            out_dim = w_qkv_linear.out_features // 3  # 每个Q/K/V的输出维度

            # 打印第一层的维度信息
            if t_layer_i == 0:
                print(f"  Layer 0: in_dim={in_dim}, out_dim={out_dim}, qkv_out={w_qkv_linear.out_features}")

            # 创建 LoRA 层 for Q
            # A: 降维 (in_dim -> r)
            # B: 升维 (r -> out_dim)
            w_a_linear_q = nn.Linear(in_dim, self.r, bias=False)
            w_b_linear_q = nn.Linear(self.r, out_dim, bias=False)

            # 创建 LoRA 层 for V
            w_a_linear_v = nn.Linear(in_dim, self.r, bias=False)
            w_b_linear_v = nn.Linear(self.r, out_dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            # 替换原始层
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                lora_alpha=self.lora_alpha,
                r=self.r,
            )

    def _apply_lora_to_memory_attention(self):
        """
        Memory Attention LoRA 注入

        SAM2 MemoryAttention 结构:
          memory_attention.layers[i]  (MemoryAttentionLayer)
            ├── self_attn       (RoPEAttention)  → q_proj, k_proj, v_proj, out_proj
            └── cross_attn_image (RoPEAttention) → q_proj, k_proj, v_proj, out_proj

        对 self_attn 和 cross_attn_image 的 Q/K/V/Out 全部注入 LoRA。
        """
        print(f"Applying LoRA to Memory Attention (r={self.r})")

        injected_count = 0

        if not hasattr(self.predictor.memory_attention, 'layers'):
            print("  Warning: memory_attention has no 'layers', skipping")
            return

        proj_names = ['q_proj', 'k_proj', 'v_proj', 'out_proj']
        attn_names = ['self_attn', 'cross_attn_image']

        for i, layer in enumerate(self.predictor.memory_attention.layers):
            for attn_name in attn_names:
                attn = getattr(layer, attn_name, None)
                if attn is None:
                    continue
                for proj_name in proj_names:
                    original_linear = getattr(attn, proj_name, None)
                    if original_linear is None or not isinstance(original_linear, nn.Linear):
                        continue
                    lora_a = nn.Linear(original_linear.in_features, self.r, bias=False)
                    lora_b = nn.Linear(self.r, original_linear.out_features, bias=False)
                    lora_layer = _LoRA_Linear(original_linear, lora_a, lora_b)
                    setattr(attn, proj_name, lora_layer)
                    self.w_As.append(lora_a)
                    self.w_Bs.append(lora_b)
                    injected_count += 1

        print(f"  Injected LoRA into {injected_count} projections in Memory Attention.")

    def _apply_lora_to_memory_encoder(self):
        """对 Memory Encoder 进行微调 (直接解冻，不使用 LoRA，因为参数量小且需要学习纹理)"""
        print(f"Unfreezing Memory Encoder for Texture Learning...")
        # Memory Encoder 负责将 Mask 编码为 Memory Embedding
        # 对于 DVT（血栓）和超声特有的噪声纹理，这一步至关重要
        for param in self.predictor.memory_encoder.parameters():
            param.requires_grad = True

    def reset_parameters(self):
        """初始化 LoRA 参数"""
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def _print_trainable_parameters(self):
        """打印可训练参数统计"""
        total_params = sum(p.numel() for p in self.predictor.parameters())
        trainable_params = sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for p in self.w_As.parameters()) + sum(p.numel() for p in self.w_Bs.parameters())

        print(f"=" * 50)
        print(f"LoRA SAM2 Parameter Summary:")
        print(f"  Total parameters:     {total_params / 1e6:.2f}M")
        print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")
        print(f"  LoRA parameters:      {lora_params / 1e6:.4f}M")
        print(f"=" * 50)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """获取所有可训练参数 (去重)"""
        params = []
        seen = set()
        for p in self.predictor.parameters():
            if p.requires_grad:
                if id(p) in seen:
                    continue
                params.append(p)
                seen.add(id(p))
        for p in self.w_As.parameters():
            if id(p) in seen:
                continue
            params.append(p)
            seen.add(id(p))
        for p in self.w_Bs.parameters():
            if id(p) in seen:
                continue
            params.append(p)
            seen.add(id(p))
        return params

    def save_lora_parameters(self, filename: str):
        """
        保存 LoRA 参数和微调的模块参数

        保存内容:
        - LoRA A/B 矩阵
        - mask_decoder 参数
        - memory_attention 微调参数 (如果有)
        """
        assert filename.endswith(".pt") or filename.endswith('.pth')

        save_dict = {}

        # 保存 LoRA 参数
        for i, w_A in enumerate(self.w_As):
            save_dict[f"w_a_{i:03d}"] = w_A.weight.data
        for i, w_B in enumerate(self.w_Bs):
            save_dict[f"w_b_{i:03d}"] = w_B.weight.data

        # 保存 mask_decoder 参数
        state_dict = self.predictor.state_dict()
        for key, value in state_dict.items():
            if 'sam_mask_decoder' in key or 'mask_decoder' in key:
                save_dict[key] = value
            if self.apply_to_memory_attention and 'memory_attention' in key:
                save_dict[key] = value
            if self.apply_to_memory_encoder and 'memory_encoder' in key:
                save_dict[key] = value

        # 保存元数据
        save_dict['_lora_config'] = {
            'r': self.r,
            'lora_layer': self.lora_layer,
            'apply_to_image_encoder': self.apply_to_image_encoder,
            'apply_to_memory_attention': self.apply_to_memory_attention,
            'apply_to_memory_encoder': self.apply_to_memory_encoder,
        }

        torch.save(save_dict, filename)
        print(f"LoRA parameters saved to {filename}")

    def load_lora_parameters(self, filename: str):
        """加载 LoRA 参数"""
        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename, map_location='cpu', weights_only=True)

        # 加载 LoRA 参数
        for i, w_A in enumerate(self.w_As):
            key = f"w_a_{i:03d}"
            if key in state_dict:
                w_A.weight = Parameter(state_dict[key])

        for i, w_B in enumerate(self.w_Bs):
            key = f"w_b_{i:03d}"
            if key in state_dict:
                w_B.weight = Parameter(state_dict[key])

        # 加载其他参数
        model_dict = self.predictor.state_dict()
        for key in model_dict.keys():
            if key in state_dict:
                model_dict[key] = state_dict[key]

        self.predictor.load_state_dict(model_dict)

        # Move LoRA weights to the same device as predictor
        device = next(self.predictor.parameters()).device
        self.w_As.to(device)
        self.w_Bs.to(device)

        print(f"LoRA parameters loaded from {filename}")

    # === 代理方法,让 LoRA_SAM2_Video 可以像 predictor 一样使用 ===

    def init_state(self, *args, **kwargs):
        return self.predictor.init_state(*args, **kwargs)

    def init_state_train(self, *args, **kwargs):
        if hasattr(self.predictor, 'init_state_train'):
            return self.predictor.init_state_train(*args, **kwargs)
        return self.predictor.init_state(*args, **kwargs)

    def add_new_points_or_box(self, *args, **kwargs):
        return self.predictor.add_new_points_or_box(*args, **kwargs)

    def add_new_points_or_box_train(self, *args, **kwargs):
        if hasattr(self.predictor, 'add_new_points_or_box_train'):
            return self.predictor.add_new_points_or_box_train(*args, **kwargs)
        return self.predictor.add_new_points_or_box(*args, **kwargs)

    def propagate_in_video(self, *args, **kwargs):
        return self.predictor.propagate_in_video(*args, **kwargs)

    def propagate_in_video_train(self, *args, **kwargs):
        if hasattr(self.predictor, 'propagate_in_video_train'):
            return self.predictor.propagate_in_video_train(*args, **kwargs)
        return self.predictor.propagate_in_video(*args, **kwargs)

    def reset_state(self, *args, **kwargs):
        return self.predictor.reset_state(*args, **kwargs)

    def train(self, mode=True):
        self.predictor.train(mode)
        # 同步训练模式到LoRA层
        for m in self.w_As:
            m.train(mode)
        for m in self.w_Bs:
            m.train(mode)
        return self

    def eval(self):
        self.predictor.eval()
        # 同步eval模式到LoRA层
        for m in self.w_As:
            m.eval()
        for m in self.w_Bs:
            m.eval()
        return self

    def to(self, device):
        self.predictor.to(device)
        self.w_As.to(device)
        self.w_Bs.to(device)
        return self

    def parameters(self):
        return self.get_trainable_parameters()

    @property
    def device(self):
        return self.predictor.device


def build_lora_sam2_video(
    config_file: str,
    ckpt_path: str,
    r: int = 4,
    lora_alpha: float = 1.0,
    device: str = "cuda",
    lora_layer: Optional[List[int]] = None,
    apply_to_image_encoder: bool = True,
    apply_to_memory_attention: bool = False,  
    apply_to_memory_encoder: bool = False,
    use_trainer: bool = True,
):
    """
    构建 LoRA SAM2 Video 模型

    Args:
        config_file: SAM2 配置文件路径
        ckpt_path: SAM2 checkpoint 路径
        r: LoRA 秩 (推荐 4-8)
        lora_alpha: LoRA 缩放因子 (通常设为 r 或 1)
        device: 设备
        lora_layer: 要应用LoRA的层 (None=全部)
        apply_to_image_encoder: 是否对image_encoder应用LoRA
        apply_to_memory_attention: 是否微调memory_attention (会增加参数量)
        apply_to_memory_encoder: 是否微调memory_encoder
        use_trainer: 是否使用SAM2VideoTrainer (支持训练模式)

    Returns:
        LoRA_SAM2_Video 实例
    """
    if use_trainer:
        from sam2.sam2_video_trainer import build_sam2_video_trainer
        predictor = build_sam2_video_trainer(
            config_file=config_file,
            ckpt_path=ckpt_path,
            device=device,
        )
    else:
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=ckpt_path,
            device=device,
        )

    lora_model = LoRA_SAM2_Video(
        predictor=predictor,
        r=r,
        lora_alpha=lora_alpha,
        lora_layer=lora_layer,
        apply_to_image_encoder=apply_to_image_encoder,
        apply_to_memory_attention=apply_to_memory_attention,
        apply_to_memory_encoder=apply_to_memory_encoder,
    )

    # Move LoRA layers to same device as predictor
    lora_model.to(device)

    return lora_model
