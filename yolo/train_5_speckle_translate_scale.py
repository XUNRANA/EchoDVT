import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


# ─────────────────────────────────────────────
# 1. 斑点噪声函数（逻辑与之前完全一致）
# ─────────────────────────────────────────────
def add_speckle_noise(img_tensor: torch.Tensor, intensity: float) -> torch.Tensor:
    """
    对单张已归一化的图像tensor注入乘性斑点噪声
    
    输入：CHW格式，值域[0,1]，已在GPU上
    输出：同格式，同设备
    
    乘性噪声公式：I_out = I_in × (1 + N)
    N ~ Normal(0, intensity)
    这符合超声斑点的真实物理成因（相干散射）
    """
    noise = torch.randn_like(img_tensor) * intensity
    noisy = img_tensor * (1.0 + noise)
    return torch.clamp(noisy, 0.0, 1.0)


# ─────────────────────────────────────────────
# 2. 自定义Trainer：重写preprocess_batch
#    这是Ultralytics官方推荐的自定义预处理入口
#    执行时机：batch加载完毕、模型前向传播之前
#    此时self.batch已经存在，不会报AttributeError
# ─────────────────────────────────────────────
class SpeckleTrainer(DetectionTrainer):
    
    def preprocess_batch(self, batch):
        # 先调用父类的标准预处理（归一化、移至GPU等）
        batch = super().preprocess_batch(batch)
        
        # 课程式噪声强度调度：
        # 前40%轮次用轻度噪声，让模型先学会基本结构
        # 中间40%轮次用中度噪声，强化鲁棒性
        # 最后20%轮次回到轻度，配合学习率衰减做精细化
        progress = self.epoch / self.epochs
        if progress < 0.4:
            intensity = 0.03
        elif progress < 0.8:
            intensity = 0.06
        else:
            intensity = 0.03
        
        # 以50%概率对整个batch应用增强
        # 保留部分干净样本，避免模型只见到噪声版本
        if np.random.random() < 0.5:
            imgs = batch['img']  # (B, C, H, W)，已在GPU，值域[0,1]
            augmented = torch.stack([
                add_speckle_noise(imgs[i], intensity)
                for i in range(imgs.shape[0])
            ])
            batch['img'] = augmented
        
        return batch


# ─────────────────────────────────────────────
# 3. 将自定义Trainer注入YOLO的训练流程
# ─────────────────────────────────────────────
if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    
    # 关键：通过trainer参数告诉YOLO用我们的子类，而不是默认的DetectionTrainer
    model.train(
        data='dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        
        # ——— 保留Step3验证最优的几何增强 ———
        translate=0.05,
        scale=0.1,
        
        # ——— 其余内置增强保持关闭 ———
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=0.0,
        mixup=0.0,
        degrees=0.0,
        fliplr=0.0,
        flipud=0.0,
        
        # 注入自定义Trainer
        trainer=SpeckleTrainer,
        
        project='runs/detect/dvt_runs',
        name='aug_step5_speckle_translate_scale',
        device=0,
        workers=4
    )

