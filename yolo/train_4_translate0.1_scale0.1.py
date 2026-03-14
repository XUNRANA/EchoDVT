from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    
    model.train(
        data='dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        
        # ——— 核心变量：放大平移幅度，从0.05→0.1 ———
        # 0.1是YOLO官方默认值，对640px图像意味着最多64px偏移
        # 这模拟了探头在皮肤上大幅滑动的情况
        translate=0.1,
        
        # ——— 缩放保留Step3中的保守值 ———
        # Step3已经证明scale=0.1没有害处，带入继续
        scale=0.1,
        
        # ——— 其余配置与Step3保持一致 ———
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,   # 已证明有害，关闭
        mosaic=0.0,
        mixup=0.0,
        degrees=0.0,
        fliplr=0.0,
        flipud=0.0,
        
        project='runs/detect/dvt_runs',
        # 名字里记录关键变化，方便回溯
        name='aug_step4_translate0.1_scale0.1',
        device=0,
        workers=4
    )