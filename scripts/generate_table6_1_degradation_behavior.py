from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
RTF_OUTPUT = ROOT / "results" / "table6_1_degradation_behavior.rtf"
TSV_OUTPUT = ROOT / "results" / "table6_1_degradation_behavior.tsv"


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
        ("模块", "正常模式", "降级模式", "降级条件", "降级限制"),
        (
            "YOLO",
            "首帧 YOLO 检测并输出动脉/静脉框",
            "从首帧 GT mask 自动提取包围框",
            "YOLO 权重缺失、依赖导入失败，或未同时得到两类血管框",
            "依赖首帧 GT mask，仅对带标注病例有效；本地无标注上传视频无法降级检测",
        ),
        (
            "SAM2",
            "SAM2 Large + LoRA 执行全视频分割传播",
            "直接使用 GT mask 模拟分割结果",
            "SAM2/LoRA 权重缺失、依赖异常，或分割推理失败",
            "依赖逐帧 GT mask，仅对带标注病例有效；无标注视频无法替代真实分割",
        ),
        (
            "DVT分类",
            "统一 RF 分类器基于 21 维特征输出 DVT 概率",
            "回退到 VCR 阈值规则（0.05）",
            "统一分类器模型缺失，或 ML 诊断不可用/超时",
            "不依赖额外模型和标注，但需要有效分割结果与面积序列；判别能力低于 RF",
        ),
    ]


def build_rtf(rows):
    cellx = [1400, 4700, 8000, 11000, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表6.1 降级行为汇总") + r"\b0\par",
        r"\pard\par",
    ]

    for idx, row in enumerate(rows):
        is_header = idx == 0
        weight = r"\b " if is_header else ""
        weight_end = r"\b0 " if is_header else ""
        aligns = [r"\qc", r"\ql", r"\ql", r"\ql", r"\ql"]
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
