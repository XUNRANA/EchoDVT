import os
import argparse
from ultralytics import YOLO

def train_yolo(data_yaml='dataset.yaml', model_name='yolov8x.pt', epochs=300, imgsz=640, batch=128, device='0,1', workers=16):
    """
    Train a YOLO model on the DVT dataset.
    Optimized for high-performance servers (e.g., 2x A100 40GB).
    """
    print(f"Starting YOLO training using config: {data_yaml}")
    
    # Load a model
    # Upgrading to YOLOv8x (or YOLOv11x) to fully utilize the A100 compute capability
    model = YOLO(model_name)  # load a pretrained model

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        cache=True,    # Cache data into RAM for maximum data loading speed
        project='EchoDVT_YOLO',
        name='dvt_detection',
        exist_ok=True, # Overwrite previous if exists
        patience=50    # Early stopping (increased patience)
    )
    
    print("Training customized YOLO model completed.")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLO for EchoDVT")
    parser.add_argument('--data', type=str, default='dataset.yaml', help='Path to dataset.yaml')
    parser.add_argument('--model', type=str, default='yolov8x.pt', help='Pretrained model weights (e.g., yolov8x.pt or yolo11x.pt)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=128, help='Batch size (use e.g. 128 for 2xA100, or -1 for AutoBatch)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image resolution (can increase to 1024 for A100)')
    parser.add_argument('--device', type=str, default='0,1', help='GPU devices, e.g., "0,1" for two GPUs')
    parser.add_argument('--workers', type=int, default=16, help='Number of CPU workers for dataloading')
    
    args = parser.parse_args()
    
    # Ensure current working directory is the one containing dataset.yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    train_yolo(
        data_yaml=args.data, 
        model_name=args.model,
        epochs=args.epochs, 
        batch=args.batch, 
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers
    )
