import json
from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
RF_META = ROOT / "results" / "unified_model" / "rf_unified.json"
MFP_SUMMARY = ROOT / "sam2" / "predictions" / "sam2_lora_yolo_box" / "val_20260315_094056_lora_r8_mfp15" / "summary.json"
RTF_OUTPUT = ROOT / "results" / "table7_6_end_to_end_summary.rtf"
TSV_OUTPUT = ROOT / "results" / "table7_6_end_to_end_summary.tsv"


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


def to_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_rows():
    rf_meta = json.loads(RF_META.read_text(encoding="utf-8"))
    mfp_metrics = json.loads(MFP_SUMMARY.read_text(encoding="utf-8"))["global_frame_weighted_metrics"]

    return [
        ("模块", "关键指标名称", "指标值"),
        ("YOLO检测", "验证集首帧两类同时成功率", "85.5%"),
        ("SAM2分割", "验证集 Mean Dice（LoRA r8+MFP）", f"{mfp_metrics['mean_dice']:.4f}"),
        ("DVT分类", "验证集准确率（阈值 0.05）", to_percent(rf_meta["val_accuracy"])),
        ("端到端", "验证集系统级准确率", to_percent(rf_meta["val_accuracy"])),
    ]


def build_rtf(rows):
    cellx = [2800, 9600, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表7.6 端到端系统性能汇总") + r"\b0\par",
        r"\pard\par",
    ]

    for idx, row in enumerate(rows):
        is_header = idx == 0
        weight = r"\b " if is_header else ""
        weight_end = r"\b0 " if is_header else ""
        aligns = [r"\qc", r"\ql", r"\qc"]
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
