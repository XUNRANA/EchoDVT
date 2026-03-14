from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载最轻量的 Nano 模型（跑得最快，最适合用来验证流程）
    model = YOLO('yolov8n.pt') 

    # 2. 开始最基础的训练
    model.train(
        # --- 核心必填项 ---
        data='dataset.yaml',  # 你的数据集配置文件
        epochs=50,            # 先跑 50 轮看看情况
        imgsz=640,            # 图像输入大小
        batch=16,             # 批次大小（如果报错 Out of Memory，改成 8）
        
        # --- 强制关闭所有默认数据增强（纯净版训练） ---
        mosaic=0.0,           # 关闭马赛克
        mixup=0.0,            # 关闭图像混合
        degrees=0.0,          # 关闭旋转
        translate=0.0,        # 关闭平移
        scale=0.0,            # 关闭缩放
        fliplr=0.0,           # 关闭左右翻转
        flipud=0.0,           # 关闭上下翻转
        hsv_h=0.0,            # 关闭色调变化
        hsv_s=0.0,            # 关闭饱和度变化
        hsv_v=0.0,            # 关闭亮度变化
        
        # --- 基础工程配置 ---
        project='runs/detect/dvt_runs',   # 训练结果保存的主文件夹
        name='aug_step1_baseline',   # 这次极简实验的名称
        device=0,             # 默认使用第0块GPU（如果没有GPU，框架会自动用CPU）
        workers=4             # 读取数据的线程数
    )