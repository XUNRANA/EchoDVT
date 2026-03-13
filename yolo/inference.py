import os
import numpy as np
from ultralytics import YOLO

class DVTPromptGenerator:
    """
    This class handles the YOLO inference and ensures exactly two bounding boxes
    (one for artery, one for vein) are generated for the first frame as SAM2 prompts.
    
    Implements Innovation 1:
    - If missing one box, automatically infer its position based on anatomical priors.
    - If multiple are detected, filter by top confidence.
    """
    def __init__(self, model_path='EchoDVT_YOLO/dvt_detection/weights/best.pt', conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf
        # Adjust these ratios based on statistical analysis of your specific dataset
        # In typical lower extremity DVT ultrasound, the vein is usually slightly superficial (above) 
        # or alongside the artery, and often larger. 
        # Here we assume a simple heuristic: Vein is typically to the right or slightly above/below the artery.
        # This is a placeholder heuristic that needs tuning based on your specific probe orientation!
        self.heuristic_vein_offset_x_ratio = 1.1 # Vein is ~10% to the right of artery's width
        self.heuristic_vein_offset_y_ratio = 0.0 # Same vertical level

    def get_prompts(self, image_path):
        """
        Run inference and return guaranteed [artery_box], [vein_box]
        Boxes are format: [x1, y1, x2, y2]
        """
        results = self.model(image_path, conf=self.conf)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy() # [N, 4]
        confs = results.boxes.conf.cpu().numpy() # [N]
        classes = results.boxes.cls.cpu().numpy() # [N]
        
        artery_box = None
        vein_box = None
        
        # 1. Extract best artery
        artery_indices = np.where(classes == 0)[0]
        if len(artery_indices) > 0:
            # Get the one with highest confidence
            best_idx = artery_indices[np.argmax(confs[artery_indices])]
            artery_box = boxes[best_idx]
            
        # 2. Extract best vein
        vein_indices = np.where(classes == 1)[0]
        if len(vein_indices) > 0:
            # Get the one with highest confidence
            best_idx = vein_indices[np.argmax(confs[vein_indices])]
            vein_box = boxes[best_idx]
            
        # 3. Apply Heuristics (Innovation 1: Auto-completion)
        if artery_box is not None and vein_box is None:
            print(f"Missing vein detection in {image_path}. Applying heuristic constraint.")
            vein_box = self._infer_vein_from_artery(artery_box)
            
        elif vein_box is not None and artery_box is None:
            print(f"Missing artery detection in {image_path}. Applying heuristic constraint.")
            artery_box = self._infer_artery_from_vein(vein_box)
            
        elif artery_box is None and vein_box is None:
            print(f"CRITICAL WARNING: No vessels detected in {image_path}. Returning fallback.")
            # Fallback to center of image or raise Exception, depends on pipeline strictness
            # For now, return dummy boxes in the center
            # Ideally we'd need image dimensions here.
            artery_box = [100, 100, 150, 150]
            vein_box = [160, 100, 210, 150]

        return {
            "artery": artery_box.tolist(),
            "vein": vein_box.tolist()
        }
        
    def _infer_vein_from_artery(self, artery_box):
        """Heuristically generate a vein box based on artery position."""
        ax1, ay1, ax2, ay2 = artery_box
        w = ax2 - ax1
        h = ay2 - ay1
        
        # Assume vein is slightly to the right, slightly larger
        dx = w * self.heuristic_vein_offset_x_ratio
        dy = h * self.heuristic_vein_offset_y_ratio
        
        vx1 = ax1 + dx
        vy1 = ay1 + dy
        # Veins are usually larger than arteries when not compressed
        vx2 = vx1 + (w * 1.5)
        vy2 = vy1 + (h * 1.5)
        
        return np.array([vx1, vy1, vx2, vy2])

    def _infer_artery_from_vein(self, vein_box):
        """Heuristically generate an artery box based on vein position."""
        vx1, vy1, vx2, vy2 = vein_box
        w = vx2 - vx1
        h = vy2 - vy1
        
        # Assume artery is slightly to the left, slightly smaller
        dx = - (w / 1.5) * self.heuristic_vein_offset_x_ratio
        dy = - h * self.heuristic_vein_offset_y_ratio
        
        ax1 = vx1 + dx
        ay1 = vy1 + dy
        # Arteries are usually smaller
        ax2 = ax1 + (w / 1.5)
        ay2 = ay1 + (h / 1.5)
        
        return np.array([ax1, ay1, ax2, ay2])

if __name__ == "__main__":
    # Test stub
    generator = DVTPromptGenerator('yolov8n.pt') # Using dummy model for syntax test
    # generator.get_prompts("path_to_test_image.jpg")
