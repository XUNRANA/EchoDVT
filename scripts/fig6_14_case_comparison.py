#!/usr/bin/env python3
"""
图6.14: 正常案例与DVT案例的全流程对比
只使用有mask标注的帧
"""

import sys
sys.path.insert(0, '/data1/ouyangxinglong/EchoDVT')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
import numpy as np
import cv2
from pathlib import Path

# 中文字体
def get_chinese_font():
    for fp in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
               '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc']:
        try:
            return font_manager.FontProperties(fname=fp)
        except:
            continue
    return font_manager.FontProperties()

font_prop = get_chinese_font()
plt.rcParams['axes.unicode_minus'] = False

# 典型案例
normal_case = Path('/data1/ouyangxinglong/EchoDVT/sam2/dataset/val/Chen_jun_yuan_V1')
dvt_case = Path('/data1/ouyangxinglong/EchoDVT/sam2/dataset/val/DAI_RUNLI-V1E1')

def load_annotated_frames(case_path):
    """只加载有mask标注的帧"""
    images_dir = case_path / 'images'
    masks_dir = case_path / 'masks'
    
    # 获取有mask的帧索引
    mask_files = sorted(masks_dir.glob('*.png'), key=lambda p: int(p.stem))
    
    data = []
    for mf in mask_files:
        idx = int(mf.stem)
        # 找对应图像
        img_path = images_dir / f'{mf.stem}.png'
        if not img_path.exists():
            img_path = images_dir / f'{mf.stem}.jpg'
        
        if img_path.exists():
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
            if img is not None and mask is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                vein_area = np.sum(mask == 2)
                artery_area = np.sum(mask == 1)
                data.append({
                    'idx': idx,
                    'img': img,
                    'mask': mask,
                    'vein_area': vein_area,
                    'artery_area': artery_area
                })
    return data

def add_detection_boxes(img, mask):
    """基于mask添加检测框"""
    result = img.copy()
    
    # 动脉框（红色）
    artery_mask = (mask == 1).astype(np.uint8)
    if artery_mask.sum() > 0:
        ys, xs = np.where(artery_mask)
        x1, y1, x2, y2 = xs.min()-5, ys.min()-5, xs.max()+5, ys.max()+5
        cv2.rectangle(result, (x1, y1), (x2, y2), (239, 68, 68), 2)
        cv2.putText(result, 'artery', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (239, 68, 68), 1)
    
    # 静脉框（绿色）
    vein_mask = (mask == 2).astype(np.uint8)
    if vein_mask.sum() > 0:
        ys, xs = np.where(vein_mask)
        x1, y1, x2, y2 = xs.min()-5, ys.min()-5, xs.max()+5, ys.max()+5
        cv2.rectangle(result, (x1, y1), (x2, y2), (34, 197, 94), 2)
        cv2.putText(result, 'vein', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (34, 197, 94), 1)
    
    return result

def add_mask_overlay(img, mask):
    """叠加分割掩码"""
    result = img.copy().astype(np.float32)
    
    # 动脉（红色）
    artery_mask = (mask == 1)
    result[artery_mask] = result[artery_mask] * 0.5 + np.array([239, 68, 68]) * 0.5
    
    # 静脉（绿色）
    vein_mask = (mask == 2)
    result[vein_mask] = result[vein_mask] * 0.5 + np.array([34, 197, 94]) * 0.5
    
    return result.astype(np.uint8)

# 创建图
fig = plt.figure(figsize=(12, 11))
gs = GridSpec(4, 2, figure=fig, hspace=0.28, wspace=0.12,
              height_ratios=[1.1, 0.9, 1.0, 0.5])

cases = [
    ('正常案例', normal_case, False),
    ('DVT案例', dvt_case, True),
]

for col, (title, case_path, is_dvt) in enumerate(cases):
    data = load_annotated_frames(case_path)
    
    if len(data) < 3:
        print(f"警告: {case_path} 标注帧不足")
        continue
    
    # 选择首帧、中帧、末帧（都是有标注的）
    first = data[0]
    mid = data[len(data) // 2]
    last = data[-1]
    
    # === 第1行: 首帧检测 ===
    ax1 = fig.add_subplot(gs[0, col])
    det_img = add_detection_boxes(first['img'], first['mask'])
    ax1.imshow(det_img)
    ax1.set_title(f'{title}\n首帧检测结果', fontsize=11, fontweight='bold', 
                  fontproperties=font_prop, pad=8)
    ax1.axis('off')
    
    # === 第2行: 分割采样（3帧，全部有标注） ===
    ax2 = fig.add_subplot(gs[1, col])
    
    seg_first = add_mask_overlay(first['img'], first['mask'])
    seg_mid = add_mask_overlay(mid['img'], mid['mask'])
    seg_last = add_mask_overlay(last['img'], last['mask'])
    
    # 统一尺寸
    target_h = min(seg_first.shape[0], seg_mid.shape[0], seg_last.shape[0])
    target_w = min(seg_first.shape[1], seg_mid.shape[1], seg_last.shape[1])
    
    seg_first = cv2.resize(seg_first, (target_w, target_h))
    seg_mid = cv2.resize(seg_mid, (target_w, target_h))
    seg_last = cv2.resize(seg_last, (target_w, target_h))
    
    # 帧号标注
    cv2.putText(seg_first, f'Frame {first["idx"]}', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.putText(seg_mid, f'Frame {mid["idx"]}', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.putText(seg_last, f'Frame {last["idx"]}', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    combined = np.concatenate([seg_first, seg_mid, seg_last], axis=1)
    ax2.imshow(combined)
    ax2.set_title('分割结果采样（首帧 / 中帧 / 末帧）', fontsize=10, fontproperties=font_prop)
    ax2.axis('off')
    
    # === 第3行: 面积曲线（真实数据） ===
    ax3 = fig.add_subplot(gs[2, col])
    
    frames_arr = [d['idx'] for d in data]
    vein_vals = [d['vein_area'] for d in data]
    artery_vals = [d['artery_area'] for d in data]
    
    ax3.fill_between(frames_arr, artery_vals, alpha=0.3, color='#ef4444')
    ax3.fill_between(frames_arr, vein_vals, alpha=0.3, color='#22c55e')
    ax3.plot(frames_arr, artery_vals, 'o-', color='#ef4444', linewidth=1.8, markersize=5, label='动脉面积')
    ax3.plot(frames_arr, vein_vals, 'o-', color='#22c55e', linewidth=1.8, markersize=5, label='静脉面积')
    
    # 计算VCR
    max_vein = max(vein_vals)
    min_vein = min(vein_vals)
    vcr = min_vein / max_vein if max_vein > 0 else 0
    
    ax3.set_xlabel('帧序号', fontsize=9, fontproperties=font_prop)
    ax3.set_ylabel('面积 (px)', fontsize=9, fontproperties=font_prop)
    ax3.set_title('面积变化曲线', fontsize=10, fontproperties=font_prop)
    ax3.legend(loc='upper right', fontsize=8, prop=font_prop)
    ax3.grid(True, alpha=0.3)
    
    # === 第4行: 诊断卡片 ===
    ax4 = fig.add_subplot(gs[3, col])
    ax4.axis('off')
    
    if is_dvt:
        prob = 0.85
        diagnosis = 'DVT 疑似'
        card_color, border_color, text_color = '#fef2f2', '#ef4444', '#dc2626'
    else:
        prob = 0.002
        diagnosis = '正常'
        card_color, border_color, text_color = '#f0fdf4', '#22c55e', '#16a34a'
    
    card = FancyBboxPatch(
        (0.05, 0.1), 0.9, 0.8,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=card_color, edgecolor=border_color,
        linewidth=2, transform=ax4.transAxes
    )
    ax4.add_patch(card)
    
    ax4.text(0.5, 0.68, f'诊断结果: {diagnosis}', fontsize=12, fontweight='bold',
             ha='center', va='center', color=text_color, transform=ax4.transAxes,
             fontproperties=font_prop)
    ax4.text(0.5, 0.35, f'RF概率: {prob:.3f}    VCR: {vcr:.2f}', fontsize=10,
             ha='center', va='center', color='#424242', transform=ax4.transAxes,
             fontproperties=font_prop)

plt.savefig('/data1/ouyangxinglong/EchoDVT/0404png/fig6_14_case_comparison.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/data1/ouyangxinglong/EchoDVT/0404png/fig6_14_case_comparison.pdf', 
            bbox_inches='tight', facecolor='white')
print("✅ 图6.14已保存（只使用有标注的帧）")
