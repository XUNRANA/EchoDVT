"""
超声血管检测 YOLO 推理脚本 (统计先验补框版)
改动：
1) _infer_vein / _infer_artery 从硬编码 → 加载 prior_stats.json
2) 当动脉/静脉双缺失时，基于训练集绝对先验自动补全两框
"""

from ultralytics import YOLO
import cv2
import os
import json
import numpy as np
from copy import deepcopy
from pathlib import Path
from datetime import datetime
import sys

# ============== 配置（直接改这里） ==============

mmm = "aug_step5_speckle_translate_scale"

MODEL_PATH     = 'runs/detect/runs/detect/dvt_runs/' + mmm + '/weights/best.pt'
INPUT_DIR      = 'dataset/val/images'
LABEL_DIR      = 'dataset/val/labels'
OUTPUT_BASE    = 'predictions/' + mmm
DEVICE         = 0
CONF_THRESHOLD = 0.1
IOU_THRESHOLD  = 0.5
CLASS_NAMES    = {0: 'artery', 1: 'vein'}

# ★ 新增：统计先验文件路径（compute_prior_stats.py 的输出）
PRIOR_PATH = "prior_stats.json"

# ============== 1. 日志工具（不变） ==============

class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def log(self, message):
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def close(self):
        self.log_file.close()

# ============== 2. MetricTracker（不变） ==============

class MetricTracker:
    def __init__(self):
        self.stats = {
            'artery': {'iou': [], 'dist': [], 'success': []},
            'vein':   {'iou': [], 'dist': [], 'success': []}
        }

    def update(self, cls_name, box_pred, box_gt):
        if box_pred is None or box_gt is None:
            return 0.0
        iou = self._compute_iou(box_pred, box_gt)
        pred_cx = (box_pred[0] + box_pred[2]) / 2
        pred_cy = (box_pred[1] + box_pred[3]) / 2
        gt_cx   = (box_gt[0]   + box_gt[2])   / 2
        gt_cy   = (box_gt[1]   + box_gt[3])   / 2
        dist = np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)
        is_success = 1 if iou >= IOU_THRESHOLD else 0
        self.stats[cls_name]['iou'].append(iou)
        self.stats[cls_name]['dist'].append(dist)
        self.stats[cls_name]['success'].append(is_success)
        return iou

    def print_summary(self, logger):
        logger.log(f"\n{'='*30} 最终评估报告 {'='*30}")
        logger.log(f"{'类别':<10} | {'样本数':<8} | {'mIoU':<8} | {'中心误差(px)':<12} | {'成功率(>0.5)':<15}")
        logger.log(f"{'-'*85}")
        for name in ['artery', 'vein']:
            data  = self.stats[name]
            count = len(data['iou'])
            if count == 0:
                logger.log(f"{name:<10} | {0:<8} | {'N/A':<8} | {'N/A':<12} | {'N/A':<15}")
                continue
            avg_iou      = np.mean(data['iou'])
            avg_dist     = np.mean(data['dist'])
            success_rate = np.mean(data['success']) * 100
            logger.log(f"{name:<10} | {count:<8} | {avg_iou:.4f}   | {avg_dist:.2f}         | {success_rate:.1f}%")
        logger.log(f"{'='*82}\n")

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
        inter_area = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union_area = area1 + area2 - inter_area
        if union_area == 0: return 0
        return inter_area / union_area

# ============== 3. VesselDetector（核心改动在这里） ==============

class VesselDetector:

    # ★ 当 prior_stats.json 不存在或字段缺失时的兜底默认值
    DEFAULT_PRIOR = {
        "class_absolute": {
            "artery": {
                "cx": {"mean": 0.34},
                "cy": {"mean": 0.53},
                "w": {"mean": 0.21},
                "h": {"mean": 0.24},
            },
            "vein": {
                "cx": {"mean": 0.36},
                "cy": {"mean": 0.70},
                "w": {"mean": 0.23},
                "h": {"mean": 0.20},
            },
        },
        "artery2vein": {
            "cx_offset": {"mean": 0.021},
            "cy_offset": {"mean": 0.168},
            "w_ratio": {"mean": 1.117},
            "h_ratio": {"mean": 0.845},
        },
        "vein2artery": {
            "cx_offset": {"mean": -0.021},
            "cy_offset": {"mean": -0.168},
            "w_ratio": {"mean": 0.979},
            "h_ratio": {"mean": 1.625},
        },
    }

    def __init__(self, model_path, device=0, prior_path="prior_stats.json"):
        self.model = YOLO(model_path)
        self.device = device
        self.prior = self._load_prior(prior_path)

    def _load_prior(self, prior_path: str) -> dict:
        path = Path(prior_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data, patched = self._normalize_prior_schema(data)
            n = data.get("meta", {}).get("n_samples", data.get("meta", {}).get("n_pair_samples", "?"))
            print(f"[VesselDetector] 已加载统计先验: {prior_path}（基于 {n} 个训练样本）")
            if patched:
                print("[VesselDetector] 先验字段不完整，已回退部分默认值。建议重新运行 compute_prior_stats.py。")
            return data
        print(f"[VesselDetector] 未找到 {prior_path}，使用默认经验值")
        print("  → 运行 python compute_prior_stats.py 可生成更精准的先验")
        return deepcopy(self.DEFAULT_PRIOR)

    def _normalize_prior_schema(self, prior: dict):
        data = deepcopy(prior) if isinstance(prior, dict) else {}
        patched = False

        for cls_name in ["artery", "vein"]:
            default_cls = self.DEFAULT_PRIOR["class_absolute"][cls_name]
            if "class_absolute" not in data or not isinstance(data["class_absolute"], dict):
                data["class_absolute"] = {}
                patched = True
            if cls_name not in data["class_absolute"] or not isinstance(data["class_absolute"][cls_name], dict):
                data["class_absolute"][cls_name] = {}
                patched = True
            for key, default_val in default_cls.items():
                item = data["class_absolute"][cls_name].get(key)
                if not isinstance(item, dict):
                    data["class_absolute"][cls_name][key] = deepcopy(default_val)
                    patched = True
                elif "mean" not in item:
                    data["class_absolute"][cls_name][key]["mean"] = default_val["mean"]
                    patched = True

        for direction in ["artery2vein", "vein2artery"]:
            default_dir = self.DEFAULT_PRIOR[direction]
            if direction not in data or not isinstance(data[direction], dict):
                data[direction] = {}
                patched = True
            for key, default_val in default_dir.items():
                item = data[direction].get(key)
                if not isinstance(item, dict):
                    data[direction][key] = deepcopy(default_val)
                    patched = True
                elif "mean" not in item:
                    data[direction][key]["mean"] = default_val["mean"]
                    patched = True

        if "meta" not in data or not isinstance(data["meta"], dict):
            data["meta"] = {}
            patched = True
        if "n_samples" not in data["meta"]:
            data["meta"]["n_samples"] = data["meta"].get("n_pair_samples", "?")
            patched = True

        return data, patched

    def _get_prior(self, direction: str, key: str) -> float:
        return float(self.prior[direction][key]["mean"])

    def _get_class_prior(self, cls_name: str, key: str) -> float:
        return float(self.prior["class_absolute"][cls_name][key]["mean"])

    def _norm_box_to_xyxy(self, cx, cy, bw, bh, w, h):
        bw = float(np.clip(bw, 1e-4, 1.0))
        bh = float(np.clip(bh, 1e-4, 1.0))
        cx = float(np.clip(cx, bw / 2, 1.0 - bw / 2))
        cy = float(np.clip(cy, bh / 2, 1.0 - bh / 2))
        return [
            (cx - bw / 2) * w,
            (cy - bh / 2) * h,
            (cx + bw / 2) * w,
            (cy + bh / 2) * h,
        ]

    def _infer_class_box(self, cls_name: str, w, h):
        box = self._norm_box_to_xyxy(
            self._get_class_prior(cls_name, "cx"),
            self._get_class_prior(cls_name, "cy"),
            self._get_class_prior(cls_name, "w"),
            self._get_class_prior(cls_name, "h"),
            w,
            h,
        )
        return {"box": box, "conf": 0.0, "inferred": True, "prior_all": True}

    def _infer_both_from_prior(self, w, h):
        artery = self._infer_class_box("artery", w, h)
        vein = self._infer_vein(artery["box"], w, h)
        vein["prior_all"] = True
        return artery, vein

    def predict(self, image, conf=0.1):
        """预测并强制返回两个框（单缺失与双缺失都补齐）"""
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        h, w = img.shape[:2]

        results = self.model(img, conf=conf, device=self.device, verbose=False)[0]

        artery_list, vein_list = [], []
        for box in results.boxes:
            cls_id     = int(box.cls)
            conf_score = float(box.conf)
            xyxy       = box.xyxy[0].cpu().numpy().tolist()
            item = {'box': xyxy, 'conf': conf_score}
            if cls_id == 0:
                artery_list.append(item)
            else:
                vein_list.append(item)

        artery = max(artery_list, key=lambda x: x['conf']) if artery_list else None
        vein   = max(vein_list,   key=lambda x: x['conf']) if vein_list   else None

        if artery is None or vein is None:
            artery, vein = self._retry_lower_conf(img, artery, vein)

        if artery is None and vein is None:
            artery, vein = self._infer_both_from_prior(w, h)
        elif artery is not None and vein is None:
            vein = self._infer_vein(artery['box'], w, h)
        elif vein is not None and artery is None:
            artery = self._infer_artery(vein['box'], w, h)

        if artery is not None and vein is not None:
            artery, vein = self._check_and_fix_overlap(artery, vein, w, h)

        return {'artery': artery, 'vein': vein}

    def _retry_lower_conf(self, img, artery, vein):
        """低阈值重试逻辑（不变）"""
        results = self.model(img, conf=0.01, device=self.device, verbose=False)[0]
        for box in results.boxes:
            cls_id     = int(box.cls)
            conf_score = float(box.conf)
            xyxy       = box.xyxy[0].cpu().numpy().tolist()
            if cls_id == 0 and artery is None:
                artery = {'box': xyxy, 'conf': conf_score}
            elif cls_id == 1 and vein is None:
                vein   = {'box': xyxy, 'conf': conf_score}
        return artery, vein

    # ★ 核心改动：_infer_vein 从硬编码 → 读统计先验
    def _infer_vein(self, a_box, w, h):
        """根据动脉框推断静脉框（相对先验）"""
        a_cx = (a_box[0] + a_box[2]) / 2 / w
        a_cy = (a_box[1] + a_box[3]) / 2 / h
        a_w  = (a_box[2] - a_box[0])     / w
        a_h  = (a_box[3] - a_box[1])     / h

        cx_offset = self._get_prior("artery2vein", "cx_offset")
        cy_offset = self._get_prior("artery2vein", "cy_offset")
        w_ratio   = self._get_prior("artery2vein", "w_ratio")
        h_ratio   = self._get_prior("artery2vein", "h_ratio")

        v_cx = a_cx + cx_offset
        v_cy = a_cy + cy_offset
        v_w  = a_w  * w_ratio
        v_h  = a_h  * h_ratio

        return {
            'box': self._norm_box_to_xyxy(v_cx, v_cy, v_w, v_h, w, h),
            'conf': 0.0,
            'inferred': True,
        }

    def _infer_artery(self, v_box, w, h):
        """根据静脉框推断动脉框（相对先验）"""
        v_cx = (v_box[0] + v_box[2]) / 2 / w
        v_cy = (v_box[1] + v_box[3]) / 2 / h
        v_w  = (v_box[2] - v_box[0])     / w
        v_h  = (v_box[3] - v_box[1])     / h

        cx_offset = self._get_prior("vein2artery", "cx_offset")
        cy_offset = self._get_prior("vein2artery", "cy_offset")
        w_ratio   = self._get_prior("vein2artery", "w_ratio")
        h_ratio   = self._get_prior("vein2artery", "h_ratio")

        a_cx = v_cx + cx_offset
        a_cy = v_cy + cy_offset
        a_w  = v_w  * w_ratio
        a_h  = v_h  * h_ratio

        return {
            'box': self._norm_box_to_xyxy(a_cx, a_cy, a_w, a_h, w, h),
            'conf': 0.0,
            'inferred': True,
        }

    def _compute_iou_boxes(self, box1, box2):
        x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
        inter_area = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - inter_area
        return inter_area / union if union > 0 else 0

    def _check_and_fix_overlap(self, artery, vein, w, h, max_iou=0.3):
        """重叠修正逻辑（不变）"""
        iou = self._compute_iou_boxes(artery['box'], vein['box'])
        if iou <= max_iou:
            return artery, vein
        if artery['conf'] >= vein['conf']:
            vein           = self._infer_vein(artery['box'], w, h)
            vein['fixed']  = True
        else:
            artery          = self._infer_artery(vein['box'], w, h)
            artery['fixed'] = True
        return artery, vein

# ============== 4. 可视化函数（支持先验补全标记） ==============

def visualize_side_by_side(img, pred_result, gt_result, save_path=None, iou_info=None):
    if isinstance(img, str):
        img = cv2.imread(img)
    img_gt   = img.copy()
    img_pred = img.copy()
    colors   = {'artery': (0, 0, 255), 'vein': (0, 255, 0)}

    cv2.putText(img_gt,   "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    for name, box in gt_result.items():
        if box is not None:
            box   = [int(x) for x in box]
            color = colors[name]
            draw_dashed_rect(img_gt, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(img_gt, name, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    cv2.putText(img_pred, "Prediction",   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    for name, det in pred_result.items():
        if det is not None:
            box   = [int(x) for x in det['box']]
            conf  = det['conf']
            color = colors[name]
            cv2.rectangle(img_pred, (box[0], box[1]), (box[2], box[3]), color, 2)
            lines = [f"{name}: {conf:.2f}"]
            if det.get('inferred'): lines.append("(inferred)")
            if det.get('fixed'):    lines.append("(fixed)")
            if det.get('prior_all'): lines.append("(prior-all)")
            if iou_info and name in iou_info: lines.append(f"IoU: {iou_info[name]:.2f}")
            y_text = box[3] + 20
            for line in lines:
                cv2.putText(img_pred, line, (box[0], y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_text += 20

    h, w      = img.shape[:2]
    separator = np.zeros((h, 10, 3), dtype=np.uint8)
    combined  = np.hstack((img_gt, separator, img_pred))
    if save_path:
        cv2.imwrite(str(save_path), combined)

def draw_dashed_rect(img, pt1, pt2, color, thickness):
    x1, y1 = pt1;  x2, y2 = pt2
    for p, q in [((x1,y1),(x2,y1)), ((x1,y2),(x2,y2)),
                 ((x1,y1),(x1,y2)), ((x2,y1),(x2,y2))]:
        draw_dashed_line(img, p, q, color, thickness)

def draw_dashed_line(img, pt1, pt2, color, thickness, dash_len=10, gap_len=5):
    dist = np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
    if dist == 0: return
    dx, dy = (pt2[0]-pt1[0])/dist, (pt2[1]-pt1[1])/dist
    pos = 0
    while pos < dist:
        sx, sy   = int(pt1[0]+dx*pos),              int(pt1[1]+dy*pos)
        ep       = min(pos+dash_len, dist)
        ex, ey   = int(pt1[0]+dx*ep),               int(pt1[1]+dy*ep)
        cv2.line(img, (sx,sy), (ex,ey), color, thickness)
        pos += dash_len + gap_len

# ============== 5. 辅助函数（不变） ==============

def load_gt_labels(label_path, img_w, img_h):
    gt = {'artery': None, 'vein': None}
    if not os.path.exists(label_path):
        return gt
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls_id = int(parts[0])
            cx = float(parts[1])*img_w;  cy = float(parts[2])*img_h
            w  = float(parts[3])*img_w;  h  = float(parts[4])*img_h
            cls_name = CLASS_NAMES.get(cls_id, 'unknown')
            gt[cls_name] = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
    return gt

def is_first_frame(filename):
    return filename.endswith('00000.jpg') or filename.endswith('_00000.jpg')

# ============== 6. 主函数（先验补框流程） ==============

def batch_predict_and_eval():
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(OUTPUT_BASE) / f'val_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "eval_results.log"
    logger   = Logger(log_path)

    # 初始化检测器与评估器
    detector = VesselDetector(MODEL_PATH, device=DEVICE, prior_path=PRIOR_PATH)
    tracker  = MetricTracker()

    input_dir = Path(INPUT_DIR)
    label_dir = Path(LABEL_DIR)

    all_images         = list(input_dir.glob('*.jpg'))
    first_frame_images = sorted([f for f in all_images if is_first_frame(f.name)])

    logger.log(f"\n{'='*60}")
    logger.log(f"超声血管检测 - 推理与评估")
    logger.log(f"模型: {MODEL_PATH}")
    logger.log(f"先验: {PRIOR_PATH}")
    logger.log(f"输出: {output_dir}")
    logger.log(f"待处理: {len(first_frame_images)} 张")
    logger.log(f"{'='*60}\n")

    for img_path in first_frame_images:
        try:
            img        = cv2.imread(str(img_path))
            h, w       = img.shape[:2]
            label_path = label_dir / f"{img_path.stem}.txt"
            gt_result  = load_gt_labels(str(label_path), w, h)
            pred_result = detector.predict(img, conf=CONF_THRESHOLD)

            current_iou_info = {}
            for name in ['artery', 'vein']:
                if gt_result[name] is not None and pred_result[name] is not None:
                    iou = tracker.update(name, pred_result[name]['box'], gt_result[name])
                    current_iou_info[name] = iou

                    # ★ 新增：把失败案例（IoU < 0.5）单独打印出来
                    if iou < 0.5:
                        inferred = pred_result[name].get('inferred', False)
                        fixed    = pred_result[name].get('fixed', False)
                        logger.log(
                            f"  [FAIL] {img_path.name} | {name} | "
                            f"IoU={iou:.3f} | "
                            f"{'推断框' if inferred else '检测框'}"
                            f"{'(重叠修正)' if fixed else ''}"
                        )

            save_path = output_dir / f"{img_path.stem}_compare.jpg"
            visualize_side_by_side(img, pred_result, gt_result, save_path, iou_info=current_iou_info)
            logger.log(f"Processed: {img_path.name}")



        except Exception as e:
            logger.log(f"Error: {img_path.name} - {e}")
            import traceback
            traceback.print_exc()

    tracker.print_summary(logger)
    logger.log(f"结果已保存: {output_dir}")
    logger.close()


if __name__ == '__main__':
    batch_predict_and_eval()
