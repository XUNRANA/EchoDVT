#!/usr/bin/env python3
"""
图6.13: Web系统全局状态流转图（简洁版）
- 减少颜色
- 箭头只有上下左右四个方向
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import font_manager

# 查找中文字体
def get_chinese_font():
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]
    for fp in font_paths:
        try:
            return font_manager.FontProperties(fname=fp)
        except:
            continue
    return font_manager.FontProperties()

font_prop = get_chinese_font()
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(14, 5.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5.5)
ax.axis('off')

# 简化颜色：只用蓝色和灰色
module_color = '#e3f2fd'  # 浅蓝 - 模块
module_edge = '#1976d2'   # 深蓝边框
data_color = '#fafafa'    # 浅灰 - 数据
data_edge = '#9e9e9e'     # 灰色边框
arrow_color = '#424242'   # 深灰箭头

# 模块位置 (x, y, width, height)
modules = [
    (0.3, 3.8, 2.0, 1.0, 'upload.py', '数据输入'),
    (3.0, 3.8, 2.2, 1.0, 'detection.py', '目标检测'),
    (5.9, 3.8, 2.4, 1.0, 'segmentation.py', '视频分割'),
    (9.0, 3.8, 2.2, 1.0, 'diagnosis.py', 'DVT诊断'),
    (11.9, 3.8, 1.8, 1.0, 'evaluation.py', '导出报告'),
]

# 数据字段位置（在模块正下方）
data_fields = [
    (0.3, 1.2, 2.0, 1.8, 'current_case\nimages_dir\nframe_files'),
    (3.0, 1.2, 2.2, 1.8, 'detections\n(artery_box,\nvein_box)'),
    (5.9, 1.2, 2.4, 1.8, 'pred_masks\nvein_areas\nartery_areas'),
    (9.0, 1.2, 2.2, 1.8, 'probability\nis_dvt\nfeatures'),
    (11.9, 1.2, 1.8, 1.8, 'PDF\n报告'),
]

# 绘制 gr.State 容器背景
state_rect = Rectangle((0.1, 1.0), 13.8, 2.2, 
                        facecolor='#f5f5f5', edgecolor='#bdbdbd',
                        linewidth=1.5, linestyle='--')
ax.add_patch(state_rect)
ax.text(7.0, 0.6, 'gr.State (会话级共享状态)', fontsize=10, ha='center', 
        color='#757575', fontproperties=font_prop)

# 绘制模块框
for x, y, w, h, name, label in modules:
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=module_color, edgecolor=module_edge,
        linewidth=2
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h*0.62, name, fontsize=9, ha='center', 
            fontweight='bold', color='#1565c0')
    ax.text(x + w/2, y + h*0.25, f'({label})', fontsize=8, ha='center', 
            color='#616161', fontproperties=font_prop)

# 绘制数据字段框
for i, (x, y, w, h, text) in enumerate(data_fields):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=data_color, edgecolor=data_edge,
        linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, fontsize=8, ha='center', va='center',
            color='#424242', linespacing=1.2, fontproperties=font_prop)

# 绘制水平箭头（模块之间，从左到右）
arrow_kw = dict(arrowstyle='->', color=arrow_color, lw=1.5, 
                mutation_scale=15)

horizontal_arrows = [
    ((2.3, 4.3), (3.0, 4.3)),    # upload -> detection
    ((5.2, 4.3), (5.9, 4.3)),    # detection -> segmentation
    ((8.3, 4.3), (9.0, 4.3)),    # segmentation -> diagnosis
    ((11.2, 4.3), (11.9, 4.3)),  # diagnosis -> evaluation
]

for (x1, y1), (x2, y2) in horizontal_arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

# 绘制垂直箭头（模块到数据，向下写入）
for x, y, w, h, _, _ in modules[:-1]:  # 最后一个模块不写入，只读取
    cx = x + w/2
    ax.annotate('', xy=(cx, 3.0), xytext=(cx, 3.8),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.2))

# evaluation 模块的向下箭头
eval_cx = 11.9 + 1.8/2
ax.annotate('', xy=(eval_cx, 3.0), xytext=(eval_cx, 3.8),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.2))

# 在数据框下方添加水平连接线，表示状态共享
# 用一条长横线连接所有数据框底部
ax.plot([1.3, 12.8], [1.1, 1.1], color='#bdbdbd', lw=1, linestyle='-')

# 标注"写入"和"读取"
ax.text(1.8, 3.4, '写入', fontsize=7, color='#757575', ha='center', fontproperties=font_prop)
ax.text(4.5, 3.4, '写入', fontsize=7, color='#757575', ha='center', fontproperties=font_prop)
ax.text(7.5, 3.4, '写入', fontsize=7, color='#757575', ha='center', fontproperties=font_prop)
ax.text(10.5, 3.4, '写入', fontsize=7, color='#757575', ha='center', fontproperties=font_prop)
ax.text(eval_cx + 0.4, 3.4, '读取', fontsize=7, color='#757575', ha='center', fontproperties=font_prop)

# 图例
legend_elements = [
    mpatches.Patch(facecolor=module_color, edgecolor=module_edge, linewidth=1.5, label='页面模块'),
    mpatches.Patch(facecolor=data_color, edgecolor=data_edge, linewidth=1.5, label='状态字段'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95, prop=font_prop)

plt.tight_layout()
output_dir = Path(__file__).resolve().parent.parent / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'fig6_13_state_flow.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig(output_dir / 'fig6_13_state_flow.pdf',
            bbox_inches='tight', facecolor='white')
print("✅ 图6.13已保存")
