from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    
    model.train(
        data='dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        
        # ——— 亮度全部关闭，已证明有害 ———
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,  # 明确写出来，提醒自己这是刻意的决定
        
        # ——— 本轮唯一的变量：轻微平移 ———
        # 5% 的平移幅度，对640px图像意味着最多32px的偏移
        # 这在超声检查中是完全真实的探头位移范围
        translate=0.05,
        
        # ——— 其余几何增强保持关闭 ———
        mosaic=0.0,
        mixup=0.0,
        degrees=0.0,
        scale=0.0,
        fliplr=0.0,   # 翻转破坏解剖方向，坚决不开
        flipud=0.0,
        
        project='runs/detect/dvt_runs',
        name='aug_step2_translate_only',  # 名字里明确写only，强调单变量
        device=0,
        workers=4
    )