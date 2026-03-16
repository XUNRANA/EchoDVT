"""
EchoDVT 图表样式工具
- Matplotlib 全局暗黑主题配置
- 中文字体自动检测与回退
- 统一的坐标轴/图例/网格样式
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import warnings

# ─── 中文字体自动检测 ───

_CHINESE_FONT = None


def _find_chinese_font() -> str | None:
    """在系统中查找可用的中文字体"""
    candidates = [
        "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
        "Noto Sans CJK SC", "Source Han Sans SC", "Droid Sans Fallback",
        "AR PL UKai CN", "FZHei-B01S",
    ]
    for name in candidates:
        try:
            path = fm.findfont(name, fallback_to_default=False)
            if path and "LastResort" not in path:
                return name
        except Exception:
            continue
    return None


def get_chinese_font():
    """获取中文字体名（有缓存），找不到返回 None"""
    global _CHINESE_FONT
    if _CHINESE_FONT is None:
        _CHINESE_FONT = _find_chinese_font() or ""
    return _CHINESE_FONT if _CHINESE_FONT else None


def setup_matplotlib():
    """配置 Matplotlib 全局样式 (用于极简风格大屏)"""
    # 基础风格
    plt.style.use('dark_background')
    
    # 覆盖关键颜色
    plt.rcParams.update({
        'figure.facecolor': 'none',     # 完全透明，配合 glassmorphism
        'axes.facecolor': 'none',       # 完全透明
        'axes.edgecolor': 'none',       # 隐藏外部边框
        'axes.grid': True,
        'grid.color': (1.0, 1.0, 1.0, 0.05), # 极淡的网格线
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'text.color': '#e2e8f0',        # 柔和主文本颜色 (slate-200)
        "xtick.color": "#64748b",
        "ytick.color": "#64748b",
        "legend.facecolor": "#1e293b",
        "legend.edgecolor": "#334155",
        "legend.labelcolor": "#e2e8f0",
    })

    # 中文字体自动检测与回退
    zh_font = get_chinese_font()
    if zh_font:
        plt.rcParams["font.sans-serif"] = [zh_font, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    else:
        # 没有中文字体时，关掉字体警告并使用英文
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def style_axis(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """统一的坐标轴样式"""
    ax.set_facecolor("#0f172a")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color="#e2e8f0")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color="#94a3b8")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color="#94a3b8")
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, alpha=0.2, color="#475569")


# ─── 初始化 ───
setup_matplotlib()
