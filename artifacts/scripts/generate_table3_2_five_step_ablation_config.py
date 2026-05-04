from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
RTF_OUTPUT = ROOT / "artifacts" / "tables" / "table3_2_five_step_ablation_config.rtf"
TSV_OUTPUT = ROOT / "artifacts" / "tables" / "table3_2_five_step_ablation_config.tsv"


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
        ("步骤", "增强配置", "训练脚本", "说明"),
        ("Step 1", "无增强", "train_1_baseline.py", "纯净基线，用于后续对比"),
        ("Step 2", "translate=0.05", "train_2_translate.py", "模拟探头轻微位移"),
        ("Step 3", "translate=0.05 + scale=0.1", "train_3_translate_scale.py", "加入保守缩放，避免小目标消失"),
        ("Step 4", "translate=0.1 + scale=0.1", "train_4_translate0.1_scale0.1.py", "测试更大平移是否继续带来增益"),
        (
            "Step 5",
            "translate=0.05 + scale=0.1 + 课程式斑点噪声",
            "train_5_speckle_translate_scale.py",
            "最优配置，通过 SpeckleTrainer 动态注入噪声",
        ),
    ]


def build_rtf(rows):
    cellx = [1500, 5200, 9400, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表3.2 五步渐进消融实验配置") + r"\b0\par",
        r"\pard\par",
    ]

    for idx, row in enumerate(rows):
        is_header = idx == 0
        weight = r"\b " if is_header else ""
        weight_end = r"\b0 " if is_header else ""
        aligns = [r"\qc", r"\ql", r"\qc", r"\ql"]
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
