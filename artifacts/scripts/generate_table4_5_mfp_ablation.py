import json
from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
RTF_OUTPUT = ROOT / "artifacts" / "tables" / "table4_5_mfp_ablation.rtf"
TSV_OUTPUT = ROOT / "artifacts" / "tables" / "table4_5_mfp_ablation.tsv"

BASELINE_SUMMARY = ROOT / "sam2" / "predictions" / "sam2_lora_yolo_box" / "val_20260315_091610_lora_r8_baseline" / "summary.json"
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


def build_rows():
    baseline = load_metrics(BASELINE_SUMMARY)
    mfp = load_metrics(MFP_SUMMARY)
    gain = (mfp["mean_dice"] - baseline["mean_dice"]) * 100.0
    return [
        ("配置", "Mean Dice", "Artery Dice", "Vein Dice", "相对提升"),
        ("LoRA r8仅首帧", fmt(baseline["mean_dice"]), fmt(baseline["artery_dice"]), fmt(baseline["vein_dice"]), "-"),
        ("LoRA r8+MFP", fmt(mfp["mean_dice"]), fmt(mfp["artery_dice"]), fmt(mfp["vein_dice"]), f"+{gain:.1f}%"),
    ]


def build_rtf(rows):
    cellx = [3600, 6100, 8600, 11100, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表4.5 MFP消融实验结果") + r"\b0\par",
        r"\pard\par",
    ]

    for idx, row in enumerate(rows):
        is_header = idx == 0
        weight = r"\b " if is_header else ""
        weight_end = r"\b0 " if is_header else ""
        aligns = [r"\qc", r"\qc", r"\qc", r"\qc", r"\qc"]
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
