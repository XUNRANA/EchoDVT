import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, font_manager
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
FEATURES_CSV = ROOT / "results/e2e_classify_v3/features.csv"
NORMAL_LIST = ROOT / "sam2/dataset/val_normal.txt"
DVT_LIST = ROOT / "sam2/dataset/val_abnormal.txt"
OUTPUT_PATH = ROOT / "results/fig5_3.png"

TEXT = "#253041"
SUBTEXT = "#5B6574"
EDGE = "#D1D5DB"
PANEL = "#F8FAFC"
BAR_LOW = "#CFE1FF"
BAR_HIGH = "#2F6FED"
VCR_HIGHLIGHT = "#D84C54"

FEATURE_LABELS = {
    "vcr": "VCR  静脉压缩比",
    "vdr": "VDR  静脉消失率",
    "vein_cv": "vein_cv  静脉面积变异系数",
    "varr": "VARR  面积相对变化幅度",
    "mvar": "MVAR  最小静脉/动脉面积比",
    "mean_var": "mean_var  平均静脉/动脉面积比",
    "vein_slope": "vein_slope  静脉面积线性斜率",
    "vein_min_position": "vein_min_pos  最小面积出现位置",
    "artery_stability": "artery_stability  动脉稳定性",
    "max_drop_ratio": "max_drop_ratio  最大单帧下降比",
    "vein_p10": "vein_p10  静脉面积 P10/最大值",
    "vein_p25": "vein_p25  静脉面积 P25/最大值",
    "vein_p50": "vein_p50  静脉面积 P50/最大值",
    "vein_detect_rate": "vein_detect_rate  静脉检出率",
    "vein_zero_rate": "vein_zero_rate  静脉零面积占比",
    "artery_detect_rate": "artery_detect_rate  动脉检出率",
    "vein_jitter": "vein_jitter  帧间面积抖动",
    "vein_autocorr": "vein_autocorr  一阶自相关",
    "circ_cv": "circ_cv  圆度变异系数",
    "circ_min": "circ_min  最小圆度",
    "circ_range": "circ_range  圆度变化范围",
}


def lerp_color(hex_a: str, hex_b: str, t: float):
    a = np.asarray(colors.to_rgb(hex_a))
    b = np.asarray(colors.to_rgb(hex_b))
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * a + t * b


def build_figure():
    font = font_manager.FontProperties(fname=str(FONT_PATH))

    df = pd.read_csv(FEATURES_CSV)
    normal_cases = set(NORMAL_LIST.read_text().splitlines())
    dvt_cases = set(DVT_LIST.read_text().splitlines())
    df["label"] = df["case_id"].map(lambda x: 0 if x in normal_cases else (1 if x in dvt_cases else None))

    feature_cols = [c for c in df.columns if c not in {"case_id", "n_frames", "label"}]
    X = df[feature_cols].values
    y = df["label"].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_scaled, y)

    importance = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    labels = [FEATURE_LABELS.get(name, name) for name in importance.index]
    values = importance.values
    vmax = float(values.max())

    fig, ax = plt.subplots(figsize=(12.6, 9.8))
    fig.patch.set_facecolor("white")

    note = FancyBboxPatch(
        (0.06, 0.92),
        0.88,
        0.055,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.0,
        edgecolor=EDGE,
        facecolor=PANEL,
        transform=fig.transFigure,
    )
    fig.patches.append(note)
    fig.text(
        0.08,
        0.947,
        "基于验证集 76 例特征在随机森林中的基尼不纯度减少量计算重要性；当前模型下 vein_cv、MVAR、VARR、VCR 排名前四。",
        ha="left",
        va="center",
        fontsize=11.0,
        color=TEXT,
        fontproperties=font,
    )

    y_pos = np.arange(len(labels))
    bar_colors = [lerp_color(BAR_LOW, BAR_HIGH, value / vmax) for value in values]
    edge_colors = [VCR_HIGHLIGHT if name == "vcr" else "none" for name in importance.index]
    line_widths = [1.4 if name == "vcr" else 0.0 for name in importance.index]

    bars = ax.barh(
        y_pos,
        values,
        color=bar_colors,
        edgecolor=edge_colors,
        linewidth=line_widths,
        height=0.72,
    )
    ax.invert_yaxis()

    for idx, (bar, value, feat_name) in enumerate(zip(bars, values, importance.index)):
        label_color = VCR_HIGHLIGHT if feat_name == "vcr" else TEXT
        ax.text(
            value + vmax * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=10.5,
            color=label_color,
            fontproperties=font,
        )

    vcr_rank = int(np.where(importance.index == "vcr")[0][0])
    ax.annotate(
        "VCR 为临床核心特征",
        xy=(values[vcr_rank], vcr_rank),
        xytext=(vmax * 0.80, vcr_rank + 1.2),
        fontsize=10.8,
        color=VCR_HIGHLIGHT,
        fontproperties=font,
        arrowprops=dict(arrowstyle="->", color=VCR_HIGHLIGHT, lw=1.4),
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.6, color=TEXT, fontproperties=font)
    ax.set_xlabel("特征重要性评分", fontsize=12.5, color=TEXT, fontproperties=font)
    ax.set_ylabel("特征名称", fontsize=12.5, color=TEXT, fontproperties=font)
    ax.set_xlim(0, vmax * 1.28)
    ax.tick_params(axis="x", labelsize=10.8, colors=TEXT)
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font)

    ax.grid(axis="x", linestyle="--", linewidth=0.9, alpha=0.45, color="#CBD5E1")
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    fig.subplots_adjust(left=0.33, right=0.98, top=0.88, bottom=0.08)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
