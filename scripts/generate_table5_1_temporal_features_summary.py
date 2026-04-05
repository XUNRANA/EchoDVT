from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
RTF_OUTPUT = ROOT / "results" / "table5_1_temporal_features_summary.rtf"
TSV_OUTPUT = ROOT / "results" / "table5_1_temporal_features_summary.tsv"


def rtf_escape(text: str) -> str:
    parts = []
    for ch in text:
        code = ord(ch)
        if ch == "\\":
            parts.append(r"\\")
        elif ch == "{":
            parts.append(r"\{")
        elif ch == "}":
            parts.append(r"\}")
        elif ch == "\n":
            parts.append(r"\line ")
        elif 32 <= code <= 126:
            parts.append(ch)
        else:
            if code > 32767:
                code -= 65536
            parts.append(f"\\u{code}?")
    return "".join(parts)


def build_rows():
    return [
        ("类别", "特征名称", "计算方式简述", "临床含义"),
        ("面积变化率", "vcr（静脉压缩比）", "静脉面积 P5 / P95", "反映整体可压缩性，数值越大越提示静脉不易压缩"),
        ("面积变化率", "vdr（静脉消失率）", "面积 < 0.1×最大面积 的帧占比", "反映静脉被压至近乎消失的频率，占比高多见于正常可压缩"),
        ("统计分布", "vein_cv（静脉面积变异系数）", "std(静脉面积) / mean(静脉面积)", "反映面积波动离散程度，越大说明压迫变化越明显"),
        ("面积变化率", "varr（面积相对范围）", "(P95 - P5) / P95", "反映压迫前后面积变化幅度，越大说明压缩更充分"),
        ("面积比率", "mvar（最小静脉/动脉面积比）", "静脉/动脉面积比的 P5 或最小值", "以动脉为参考评估最小压缩程度，越小越接近正常"),
        ("面积比率", "mean_var（平均静脉/动脉面积比）", "mean(静脉面积 / 动脉面积)", "反映全程平均相对面积，偏高提示静脉持续偏大"),
        ("时序动态", "vein_slope（静脉面积斜率）", "归一化静脉面积对帧序号做线性拟合斜率", "反映全程增减趋势，负斜率提示压迫中面积下降"),
        ("时序动态", "vein_min_position（最小面积位置）", "argmin(静脉面积) / (帧数-1)", "反映最小面积出现时刻，中后段更符合压迫过程"),
        ("动脉参考", "artery_stability（动脉稳定性）", "1 - std(动脉面积) / mean(动脉面积)", "反映动脉面积是否稳定，用于验证序列与分割质量"),
        ("面积变化率", "max_drop_ratio（最大下降比）", "-min(diff(静脉面积)) / 最大面积", "反映相邻帧最剧烈的塌陷程度，越大说明压缩更明显"),
        ("统计分布", "vein_p10", "静脉面积 P10 / 最大面积", "反映低位面积水平，越低说明可压到更小面积"),
        ("统计分布", "vein_p25", "静脉面积 P25 / 最大面积", "反映下四分位面积水平，描述持续压缩程度"),
        ("统计分布", "vein_p50", "静脉面积 P50 / 最大面积", "反映中位面积水平，描述整体面积分布中心"),
        ("时序动态", "vein_detect_rate（静脉检出率）", "面积 > 10 像素的帧占比", "反映静脉在视频中的持续可见性"),
        ("时序动态", "vein_zero_rate（静脉零面积率）", "面积 = 0 的帧占比", "反映静脉完全消失频率，可表征极端压缩或检测失败"),
        ("动脉参考", "artery_detect_rate（动脉检出率）", "动脉面积 > 10 像素的帧占比", "反映动脉持续检出情况，为静脉判断提供稳定参照"),
        ("时序动态", "vein_jitter（静脉抖动）", "mean(|diff(静脉面积)|) / 最大面积", "反映帧间波动幅度，过大提示不稳定或快速塌陷"),
        ("时序动态", "vein_autocorr（一阶自相关）", "静脉面积序列 lag=1 自相关", "反映相邻帧连续性，高值提示压迫过程更平滑"),
        ("统计分布", "circ_cv（圆度变异系数）", "std(圆度) / mean(圆度)", "反映静脉形状波动程度，压迫明显时形状变化更大"),
        ("统计分布", "circ_min（最小圆度）", "min(圆度)", "反映最扁平形态，越低提示静脉被压扁更明显"),
        ("统计分布", "circ_range（圆度变化范围）", "max(圆度) - min(圆度)", "反映形状可变性，越大说明压迫导致形态改变更明显"),
    ]


def build_rtf(rows):
    cellx = [1800, 4900, 9200, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表5.1 21维时序特征分类汇总") + r"\b0\par",
        r"\pard\par",
    ]

    for idx, row in enumerate(rows):
        is_header = idx == 0
        weight = r"\b " if is_header else ""
        weight_end = r"\b0 " if is_header else ""
        aligns = [r"\qc", r"\ql", r"\ql", r"\ql"]
        lines.append(r"\trowd\trgaph108\trqc")
        lines.append("".join(rf"\cellx{x}" for x in cellx))
        for col, align in zip(row, aligns):
            lines.append(
                r"\pard\intbl" + align + r"\f0\fs21 " + weight + rtf_escape(col) + weight_end + r"\cell"
            )
        lines.append(r"\row")

    lines.append("}")
    return "\n".join(lines)


def build_tsv(rows):
    return "\n".join("\t".join(row) for row in rows) + "\n"


def main():
    rows = build_rows()
    RTF_OUTPUT.write_text(build_rtf(rows), encoding="utf-8")
    TSV_OUTPUT.write_text(build_tsv(rows), encoding="utf-8")
    print(f"Saved: {RTF_OUTPUT}")
    print(f"Saved: {TSV_OUTPUT}")


if __name__ == "__main__":
    main()
