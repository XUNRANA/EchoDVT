# """
# 超声血管检测 YOLO 训练脚本
# 运行: python train.py
# """

# from ultralytics import YOLO

# # ============== 配置（直接改这里） ==============

# # 模型路径
# MODEL_PATH = '/data1/ouyangxinglong/2026/yolo/checkpoints/yolo11m.pt'

# # 数据集配置
# DATA_YAML = '/data1/ouyangxinglong/2026/yolo/yaml/dvt.yaml'

# # 输出目录
# PROJECT = '/data1/ouyangxinglong/2026/yolo/runs'
# NAME = 'vessel_yolo11m_0114_32--'

# # GPU设备
# DEVICE = 1  

# # 训练参数
# EPOCHS = 200
# BATCH = 32  # A100 40G 可以开32
# IMGSZ = 640
# PATIENCE = 50  # 早停

# # ============== 训练 ==============

# if __name__ == '__main__':
    
#     # 加载预训练模型
#     model = YOLO(MODEL_PATH)
    
#     # 开始训练
#     results = model.train(
#         # 数据
#         data=DATA_YAML,
        
#         # 基础参数
#         epochs=EPOCHS,
#         batch=BATCH,
#         imgsz=IMGSZ,
#         patience=PATIENCE,
        
#         # 优化器
#         optimizer='AdamW',
#         lr0=0.001,
#         lrf=0.01,
#         weight_decay=0.0005,
        
#         # 数据增强（超声图像适用）
#         degrees=10.0,       # 旋转
#         translate=0.1,      # 平移
#         scale=0.3,          # 缩放
#         fliplr=0.5,         # 水平翻转
#         flipud=0.0,         # 禁止垂直翻转
#         mosaic=0.5,
#         mixup=0.1,
#         hsv_h=0.01,
#         hsv_s=0.3,
#         hsv_v=0.3,
        
#         # 输出
#         project=PROJECT,
#         name=NAME,
#         exist_ok=True,
        
#         # 其他
#         device=DEVICE,
#         workers=8,
#         save=True,
#         save_period=10,
#         val=True,
#         plots=True,
#     )
    
#     print(f"\n训练完成！")
#     print(f"最佳模型: {PROJECT}/{NAME}/weights/best.pt")







"""
超声血管检测 YOLO 训练脚本
运行: python train.py
"""

from ultralytics import YOLO

# ============== 配置（直接改这里） ==============

# 模型路径
MODEL_PATH = '/data1/ouyangxinglong/2026/yolo/checkpoints/yolo26l.pt'

# 数据集配置
DATA_YAML = '/data1/ouyangxinglong/2026/yolo/yaml/dvt.yaml'

# 输出目录
PROJECT = '/data1/ouyangxinglong/2026/yolo/runs'
NAME = 'vessel_yolo26l'

# GPU设备
DEVICE = 1  

# 训练参数
EPOCHS = 200
BATCH = 16  # A100 40G 可以开32
IMGSZ = 640
PATIENCE = 50  # 早停

if __name__ == '__main__':
    model = YOLO(MODEL_PATH)
    
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,     # 如果爆显存，改成 16
        imgsz=IMGSZ,
        patience=PATIENCE,
        
        # --- 优化器 ---
        optimizer='auto', # 建议改为 auto，让 YOLO 根据数据集大小自动选，或者保持 AdamW
        lr0=0.001,
        lrf=0.01,
        
        # --- 医学超声增强微调 ---
        degrees=15.0,     # 稍微加大旋转角度，血管走向很随意
        translate=0.1,
        scale=0.5,        # 加大缩放范围，血管粗细变化大
        fliplr=0.5,
        flipud=0.5,       # 【建议开启】超声通常上下翻转也成立
        
        # --- 关键：防止伪影 ---
        mosaic=0.5,       # 保持现状，但注意观察
        mixup=0.1,        # 保持低值，防止两根血管叠在一起变成"鬼影"
        copy_paste=0.0,   # 确保关闭，医学图像不适合把一个病灶剪切粘贴到别处
        
        # --- 颜色控制 ---
        hsv_h=0.01,       # 极低色调变化
        hsv_s=0.1,        # 【降低】饱和度变化，超声几乎无色
        hsv_v=0.4,        # 【增加】亮度变化，模拟不同增益(Gain)下的超声图
        
        # --- 训练策略 ---
        close_mosaic=20,  # 最后20轮关闭马赛克，让模型适应真实尺寸
        
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        device=DEVICE,
        workers=8,        # 注意：如果CPU内存吃紧，改小这个值
        save=True,
        plots=True,
        
        # --- 针对 DVT 任务 ---
        # 如果这是为 SAM2 做提示词 (Prompting)，你需要极高的 Recall (召回率)
        # 也就是宁可错杀(检测大一点)，不可漏掉血管。
        # 这里不需要特殊参数，但在看结果时，重点关注 R (Recall) 指标。
    )