from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
RTF_OUTPUT = ROOT / "results" / "table3_1_ultrasound_augmentation_principles.rtf"
TSV_OUTPUT = ROOT / "results" / "table3_1_ultrasound_augmentation_principles.tsv"


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
        ("增强类型", "自然图像做法", "超声图像做法", "原因说明"),
        ("色彩", "常用 HSV、亮度扰动", "禁用", "超声为灰度图，颜色无诊断意义"),
        ("翻转", "常用水平、垂直翻转", "禁用", "会破坏动静脉方向与解剖位置关系"),
        ("Mosaic/MixUp", "常用拼接、混合增强", "禁用", "易生成不真实结构与伪影"),
        ("几何", "缩放、平移、裁剪", "仅保留小幅缩放和平移", "可模拟探头轻微移动，过强变换会失真"),
        ("噪声", "少用或加入通用噪声", "引入斑点噪声", "更贴合超声成像特性"),
    ]


def build_rtf(rows):
    cellx = [1800, 5200, 8600, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表3.1 超声增强与自然图像增强设计原则对比") + r"\b0\par",
        r"\pard\par",
    ]

    for idx, row in enumerate(rows):
        is_header = idx == 0
        weight = r"\b " if is_header else ""
        weight_end = r"\b0 " if is_header else ""
        aligns = [r"\qc", r"\qc", r"\qc", r"\ql"]
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
