import json
from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
RTF_OUTPUT = ROOT / "results" / "table7_4_sam2_scheme_comparison.rtf"
TSV_OUTPUT = ROOT / "results" / "table7_4_sam2_scheme_comparison.tsv"

BASELINE_SUMMARY = ROOT / "sam2" / "predictions" / "sam2_large_yolo_box" / "val_20260314_122603_am0_sm0_av0" / "summary.json"
R4_SUMMARY = ROOT / "sam2" / "predictions" / "sam2_lora_yolo_box" / "val_20260315_034048_lora_r4" / "summary.json"
R8_SUMMARY = ROOT / "sam2" / "predictions" / "sam2_lora_yolo_box" / "val_20260315_091610_lora_r8_baseline" / "summary.json"
MFP_SUMMARY = ROOT / "sam2" / "predictions" / "sam2_lora_yolo_box" / "val_20260315_094056_lora_r8_mfp15" / "summary.json"


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


def load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["global_frame_weighted_metrics"]


def fmt(x: float) -> str:
    return f"{x:.4f}"


def gain_fmt(current: float, baseline: float) -> str:
    return f"+{(current - baseline) * 100:.1f}%"


def build_rows():
    baseline = load_metrics(BASELINE_SUMMARY)
    r4 = load_metrics(R4_SUMMARY)
    r8 = load_metrics(R8_SUMMARY)
    mfp = load_metrics(MFP_SUMMARY)

    base_mean = baseline["mean_dice"]

    return [
        ("配置", "可训练参数量", "Mean Dice", "Artery Dice", "Vein Dice", "相对于Baseline的提升"),
        ("Baseline", "0（无微调）", fmt(baseline["mean_dice"]), fmt(baseline["artery_dice"]), fmt(baseline["vein_dice"]), "-"),
        ("LoRA r4", "~0.25M", fmt(r4["mean_dice"]), fmt(r4["artery_dice"]), fmt(r4["vein_dice"]), gain_fmt(r4["mean_dice"], base_mean)),
        ("LoRA r8", "~0.50M", fmt(r8["mean_dice"]), fmt(r8["artery_dice"]), fmt(r8["vein_dice"]), gain_fmt(r8["mean_dice"], base_mean)),
        ("LoRA r8+MFP", "~0.50M", fmt(mfp["mean_dice"]), fmt(mfp["artery_dice"]), fmt(mfp["vein_dice"]), gain_fmt(mfp["mean_dice"], base_mean)),
    ]


def build_rtf(rows):
    cellx = [2900, 5000, 7600, 9800, 11700, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表7.4 SAM2分割方案总体对比") + r"\b0\par",
        r"\pard\par",
    ]

    for idx, row in enumerate(rows):
        is_header = idx == 0
        use_bold = is_header or row[0] == "LoRA r8+MFP"
        weight = r"\b " if use_bold else ""
        weight_end = r"\b0 " if use_bold else ""
        aligns = [r"\qc", r"\qc", r"\qc", r"\qc", r"\qc", r"\qc"]
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
