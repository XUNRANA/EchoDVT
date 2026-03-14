from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    
    model.train(
        data='dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        
        # ——— 保留Step2中已证明有效的平移增强 ———
        translate=0.05,
        
        # ——— Step3新增：轻微缩放 ———
        # scale=0.1 表示图像随机缩放 ±10%
        # 这模拟了超声医生调整深度刻度时血管的视觉大小变化
        # 注意：YOLO默认的scale=0.5非常激进，对小目标有害
        # 这里使用保守的0.1，是安全范围的下限
        scale=0.1,
        
        # ——— 亮度已证明有害，继续关闭 ———
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        
        # ——— 其余增强保持关闭 ———
        mosaic=0.0,
        mixup=0.0,
        degrees=0.0,
        fliplr=0.0,
        flipud=0.0,
        
        project='runs/detect/dvt_runs',
        name='aug_step3_translate_scale',
        device=0,
        workers=4
    )