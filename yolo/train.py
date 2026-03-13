import os
import argparse
from ultralytics import YOLO

def train_yolo(data_yaml='dataset.yaml', epochs=100, imgsz=640, batch=16, device='0'):
    """
    Train a YOLO model on the DVT dataset.
    We start with a pre-trained YOLOv8/11 weights to speed up convergence.
    """
    print(f"Starting YOLO training using config: {data_yaml}")
    
    # Load a model
    # We use YOLOv8 nano or small as a starting point. Can be upgraded to 'yolov8m.pt' for better accuracy.
    model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project='EchoDVT_YOLO',
        name='dvt_detection',
        exist_ok=True, # Overwrite previous if exists
        patience=20    # Early stopping
    )
    
    print("Training customized YOLO model completed.")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLO for EchoDVT")
    parser.add_argument('--data', type=str, default='dataset.yaml', help='Path to dataset.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image resolution')
    
    args = parser.parse_args()
    
    # Ensure current working directory is the one containing dataset.yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    train_yolo(data_yaml=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
