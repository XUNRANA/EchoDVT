from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
RTF_OUTPUT = ROOT / "artifacts" / "tables" / "table4_7_sam2_comparison.rtf"
TSV_OUTPUT = ROOT / "artifacts" / "tables" / "table4_7_sam2_comparison.tsv"


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
        ("类别", "官方SAM2", "本文EchoDVT"),
        ("微调方式", "全量微调（全部224M参数）", "LoRA低秩微调（仅~0.5M，占0.2%）"),
        ("训练支持", "仅支持推理（@inference_mode）", "新增SAM2VideoTrainer，支持梯度计算"),
        ("输入端", "仅首帧单次提示", "多帧提示MFP（每隔N帧YOLO重新锚定）"),
        ("输出端", "无后处理", "RPA相对位置锚定（约束静脉漂移）"),
        ("Prompt来源", "手动标注", "YOLO自动检测+先验补全"),
        ("数据集", "通用视频/图像", "DVT超声视频专用Dataset"),
        ("损失函数", "无训练损失设计", "Dice+Focal Loss组合并支持类别加权"),
        ("应用场景", "通用分割", "DVT超声诊断（动脉/静脉二类分割）"),
    ]


def build_rtf(rows):
    cellx = [2500, 7200, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表4.7 本文SAM2实现与官方SAM2的对比") + r"\b0\par",
        r"\pard\par",
    ]

    for idx, row in enumerate(rows):
        is_header = idx == 0
        weight = r"\b " if is_header else ""
        weight_end = r"\b0 " if is_header else ""
        aligns = [r"\qc", r"\ql", r"\ql"]
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
