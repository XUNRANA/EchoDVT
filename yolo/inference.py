"""
超声血管检测 YOLO 推理脚本 (终极完整版)
功能：
1. 只检测第一帧（00000.jpg）
2. 强制双输出逻辑（缺失时推断，重叠时修正）- 核心逻辑完全保留
3. 精度评估：mIoU、中心误差、成功率
4. 可视化升级：左图GT(虚线) vs 右图Pred(实线)
5. 日志系统：自动保存 eval_results.log

运行: python inference_final.py
"""

from ultralytics import YOLO
import cv2
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# ============== 配置（直接改这里） ==============

mmm = "yolo11m_0113_32"

# 训练好的模型路径
MODEL_PATH = '/data1/ouyangxinglong/2026/yolo/runs/vessel_'+mmm+'/weights/best.pt'

# 输入：图像文件夹
INPUT_DIR = '/data1/ouyangxinglong/datasets/yolo/val/images'

# 标签文件夹（GT）
LABEL_DIR = '/data1/ouyangxinglong/datasets/yolo/val/labels'

# 输出：保存结果
OUTPUT_BASE = '/data1/ouyangxinglong/2026/yolo/predictions/'+mmm+"_final"

# GPU设备
DEVICE = 0

# 置信度阈值
CONF_THRESHOLD = 0.1

# 评估标准：IoU 大于多少算检测成功
IOU_THRESHOLD = 0.5

# 类别名称
CLASS_NAMES = {0: 'artery', 1: 'vein'}

# ============== 1. 日志工具 (新增) ==============

class Logger:
    """
    双向日志：同时打印到控制台并写入文件
    """
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def log(self, message):
        # 打印到屏幕
        print(message)
        # 写入文件
        self.log_file.write(message + '\n')
        self.log_file.flush() # 确保实时写入

    def close(self):
        self.log_file.close()

# ============== 2. 评估工具类 (适配Logger) ==============

class MetricTracker:
    """
    用于追踪并计算所有图像的平均指标
    """
    def __init__(self):
        self.stats = {
            'artery': {'iou': [], 'dist': [], 'success': []},
            'vein':   {'iou': [], 'dist': [], 'success': []}
        }

    def update(self, cls_name, box_pred, box_gt):
        if box_pred is None or box_gt is None:
            return 0.0

        # 1. 计算 IoU
        iou = self._compute_iou(box_pred, box_gt)
        
        # 2. 计算中心点距离 (像素)
        pred_cx = (box_pred[0] + box_pred[2]) / 2
        pred_cy = (box_pred[1] + box_pred[3]) / 2
        gt_cx = (box_gt[0] + box_gt[2]) / 2
        gt_cy = (box_gt[1] + box_gt[3]) / 2
        
        dist = np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)
        
        # 3. 判定是否成功
        is_success = 1 if iou >= IOU_THRESHOLD else 0

        # 记录数据
        self.stats[cls_name]['iou'].append(iou)
        self.stats[cls_name]['dist'].append(dist)
        self.stats[cls_name]['success'].append(is_success)
        
        return iou

    def print_summary(self, logger):
        """打印最终汇总报告到日志"""
        logger.log(f"\n{'='*30} 最终评估报告 {'='*30}")
        logger.log(f"{'类别':<10} | {'样本数':<8} | {'mIoU':<8} | {'中心误差(px)':<12} | {'成功率(>0.5)':<15}")
        logger.log(f"{'-'*85}")
        
        for name in ['artery', 'vein']:
            data = self.stats[name]
            count = len(data['iou'])
            
            if count == 0:
                logger.log(f"{name:<10} | {0:<8} | {'N/A':<8} | {'N/A':<12} | {'N/A':<15}")
                continue
                
            avg_iou = np.mean(data['iou'])
            avg_dist = np.mean(data['dist'])
            success_rate = np.mean(data['success']) * 100
            
            logger.log(f"{name:<10} | {count:<8} | {avg_iou:.4f}   | {avg_dist:.2f}         | {success_rate:.1f}%")
        logger.log(f"{'='*82}\n")

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area
        
        if union_area == 0: return 0
        return inter_area / union_area

# ============== 3. 核心检测类 (逻辑完全保留) ==============

class VesselDetector:
    """
    血管检测器：包含之前的全部逻辑（推断、重叠修正）
    """
    def __init__(self, model_path, device=0):
        self.model = YOLO(model_path)
        self.device = device
    
    def predict(self, image, conf=0.1):
        """预测并强制返回两个框"""
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image
        
        h, w = img.shape[:2]
        
        # 第一次预测
        results = self.model(img, conf=conf, device=self.device, verbose=False)[0]
        
        artery_list = []
        vein_list = []
        
        for box in results.boxes:
            cls_id = int(box.cls)
            conf_score = float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            
            item = {'box': xyxy, 'conf': conf_score}
            if cls_id == 0:
                artery_list.append(item)
            else:
                vein_list.append(item)
        
        # 选置信度最高的
        artery = max(artery_list, key=lambda x: x['conf']) if artery_list else None
        vein = max(vein_list, key=lambda x: x['conf']) if vein_list else None
        
        # 如果缺失，降低阈值重试
        if artery is None or vein is None:
            artery, vein = self._retry_lower_conf(img, artery, vein)
        
        # 如果还缺失，进行几何推断
        if artery is not None and vein is None:
            vein = self._infer_vein(artery['box'], w, h)
        elif vein is not None and artery is None:
            artery = self._infer_artery(vein['box'], w, h)
        
        # 检查重叠并修正
        if artery is not None and vein is not None:
            artery, vein = self._check_and_fix_overlap(artery, vein, w, h)
        
        return {'artery': artery, 'vein': vein}
    
    def _retry_lower_conf(self, img, artery, vein):
        """低阈值重试逻辑"""
        results = self.model(img, conf=0.01, device=self.device, verbose=False)[0]
        for box in results.boxes:
            cls_id = int(box.cls)
            conf_score = float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            
            if cls_id == 0 and artery is None:
                artery = {'box': xyxy, 'conf': conf_score}
            elif cls_id == 1 and vein is None:
                vein = {'box': xyxy, 'conf': conf_score}
        return artery, vein

    def _infer_vein(self, a_box, w, h):
        """根据动脉推断静脉"""
        a_cx = (a_box[0] + a_box[2]) / 2
        a_cy = (a_box[1] + a_box[3]) / 2
        a_w = a_box[2] - a_box[0]
        a_h = a_box[3] - a_box[1]
        
        offset = a_w * 1.2
        v_cx = a_cx + offset if a_cx < w/2 else a_cx - offset
        
        return {
            'box': [v_cx - a_w/2*0.9, a_cy - a_h/2*0.9, v_cx + a_w/2*0.9, a_cy + a_h/2*0.9],
            'conf': 0.0,
            'inferred': True
        }
    
    def _infer_artery(self, v_box, w, h):
        """根据静脉推断动脉"""
        v_cx = (v_box[0] + v_box[2]) / 2
        v_cy = (v_box[1] + v_box[3]) / 2
        v_w = v_box[2] - v_box[0]
        v_h = v_box[3] - v_box[1]
        
        offset = v_w * 1.2
        a_cx = v_cx + offset if v_cx < w/2 else v_cx - offset
        
        return {
            'box': [a_cx - v_w/2*1.1, v_cy - v_h/2*1.1, a_cx + v_w/2*1.1, v_cy + v_h/2*1.1],
            'conf': 0.0,
            'inferred': True
        }

    def _compute_iou_boxes(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - inter_area
        return inter_area / union if union > 0 else 0

    def _check_and_fix_overlap(self, artery, vein, w, h, max_iou=0.3):
        """重叠修正逻辑"""
        iou = self._compute_iou_boxes(artery['box'], vein['box'])
        
        if iou <= max_iou:
            return artery, vein
        
        if artery['conf'] >= vein['conf']:
            vein = self._infer_vein(artery['box'], w, h)
            vein['fixed'] = True
        else:
            artery = self._infer_artery(vein['box'], w, h)
            artery['fixed'] = True
            
        return artery, vein

# ============== 4. 可视化函数 (升级：左右分栏) ==============

def visualize_side_by_side(img, pred_result, gt_result, save_path=None, iou_info=None):
    """
    可视化：左右对比模式
    Left: GT (虚线)
    Right: Pred (实线)
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    
    # 复制两份画布
    img_gt = img.copy()
    img_pred = img.copy()
    
    colors = {
        'artery': (0, 0, 255), # 红
        'vein': (0, 255, 0)    # 绿
    }
    
    # --- 画左图：GT (虚线) ---
    cv2.putText(img_gt, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    for name, box in gt_result.items():
        if box is not None:
            box = [int(x) for x in box]
            color = colors[name]
            draw_dashed_rect(img_gt, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(img_gt, f"{name}", (box[0], box[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # --- 画右图：Pred (实线) ---
    cv2.putText(img_pred, "Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    for name, det in pred_result.items():
        if det is not None:
            box = [int(x) for x in det['box']]
            conf = det['conf']
            color = colors[name]
            
            # 实线框
            cv2.rectangle(img_pred, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # 构建标签信息
            lines = [f"{name}: {conf:.2f}"]
            if det.get('inferred'): lines.append("(inferred)")
            if det.get('fixed'): lines.append("(fixed)")
            if iou_info and name in iou_info: lines.append(f"IoU: {iou_info[name]:.2f}")
            
            # 绘制多行标签
            y_text = box[3] + 20
            for line in lines:
                cv2.putText(img_pred, line, (box[0], y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_text += 20 # 换行

    # --- 拼接 ---
    h, w = img.shape[:2]
    # 中间加一个黑色分割线
    separator = np.zeros((h, 10, 3), dtype=np.uint8)
    
    # 水平拼接: GT | 分割线 | Pred
    combined_img = np.hstack((img_gt, separator, img_pred))
            
    if save_path:
        cv2.imwrite(str(save_path), combined_img)

def draw_dashed_rect(img, pt1, pt2, color, thickness):
    """手动画虚线矩形"""
    x1, y1 = pt1
    x2, y2 = pt2
    draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness)
    draw_dashed_line(img, (x1, y2), (x2, y2), color, thickness)
    draw_dashed_line(img, (x1, y1), (x1, y2), color, thickness)
    draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness)

def draw_dashed_line(img, pt1, pt2, color, thickness, dash_len=10, gap_len=5):
    """手动画虚线段"""
    dist = np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
    if dist == 0: return
    
    dx = (pt2[0]-pt1[0]) / dist
    dy = (pt2[1]-pt1[1]) / dist
    
    pos = 0
    while pos < dist:
        sx = int(pt1[0] + dx * pos)
        sy = int(pt1[1] + dy * pos)
        
        end_pos = min(pos + dash_len, dist)
        ex = int(pt1[0] + dx * end_pos)
        ey = int(pt1[1] + dy * end_pos)
        
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)
        pos += dash_len + gap_len

# ============== 5. 主函数 (流程整合) ==============

def load_gt_labels(label_path, img_w, img_h):
    """加载GT标签 (复用)"""
    gt = {'artery': None, 'vein': None}
    if not os.path.exists(label_path):
        return gt
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1])*img_w, float(parts[2])*img_h, float(parts[3])*img_w, float(parts[4])*img_h
            cls_name = CLASS_NAMES.get(cls_id, 'unknown')
            gt[cls_name] = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
    return gt

def is_first_frame(filename):
    return filename.endswith('00000.jpg') or filename.endswith('_00000.jpg')

def batch_predict_and_eval():
    """主程序"""
    
    # 1. 准备输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(OUTPUT_BASE) / f'val_{timestamp}'
    mmm
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 初始化日志
    log_path = output_dir / "eval_results.log"
    logger = Logger(log_path)
    
    # 3. 初始化检测器
    detector = VesselDetector(MODEL_PATH, device=DEVICE)
    tracker = MetricTracker()
    
    input_dir = Path(INPUT_DIR)
    label_dir = Path(LABEL_DIR)
    
    # 筛选文件
    all_images = list(input_dir.glob('*.jpg'))
    first_frame_images = sorted([f for f in all_images if is_first_frame(f.name)])
    
    logger.log(f"\n{'='*60}")
    logger.log(f"超声血管检测 - 推理与评估 (Side-by-Side + Log)")
    logger.log(f"输出目录: {output_dir}")
    logger.log(f"日志文件: {log_path}")
    logger.log(f"待处理图像: {len(first_frame_images)} 张")
    logger.log(f"{'='*60}\n")
    
    for img_path in first_frame_images:
        try:
            # 读取
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            # 加载GT
            label_path = label_dir / f"{img_path.stem}.txt"
            gt_result = load_gt_labels(str(label_path), w, h)
            
            # 预测 (核心逻辑)
            pred_result = detector.predict(img, conf=CONF_THRESHOLD)
            
            # 计算指标
            current_iou_info = {}
            for name in ['artery', 'vein']:
                if gt_result[name] is not None and pred_result[name] is not None:
                    iou = tracker.update(name, pred_result[name]['box'], gt_result[name])
                    current_iou_info[name] = iou
            
            # 可视化 (左右分栏)
            save_path = output_dir / f"{img_path.stem}_compare.jpg"
            visualize_side_by_side(img, pred_result, gt_result, save_path, iou_info=current_iou_info)
            
            logger.log(f"Processed: {img_path.name}")
            
        except Exception as e:
            logger.log(f"Error: {img_path.name} - {e}")
            import traceback
            traceback.print_exc()

    # 4. 打印最终报告
    tracker.print_summary(logger)
    logger.log(f"结果已保存: {output_dir}")
    logger.close()

if __name__ == '__main__':
    batch_predict_and_eval()