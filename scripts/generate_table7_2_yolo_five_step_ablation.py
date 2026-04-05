import csv
from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
RUN_ROOT = ROOT / "yolo" / "runs" / "detect" / "runs" / "detect" / "dvt_runs"
RTF_OUTPUT = ROOT / "results" / "table7_2_yolo_five_step_ablation.rtf"
TSV_OUTPUT = ROOT / "results" / "table7_2_yolo_five_step_ablation.tsv"

STEP_SPECS = [
    (
        "Step 1",
        "无增强基线",
        RUN_ROOT / "aug_step1_baseline" / "results.csv",
    ),
    (
        "Step 2",
        "+平移 0.05",
        RUN_ROOT / "aug_step2_translate" / "results.csv",
    ),
    (
        "Step 3",
        "+平移 0.05 + 缩放 0.1",
        RUN_ROOT / "aug_step3_translate_scale" / "results.csv",
    ),
    (
        "Step 4",
        "+平移 0.10 + 缩放 0.1",
        RUN_ROOT / "aug_step4_translate0.1_scale0.1" / "results.csv",
    ),
    (
        "Step 5",
        "+平移 0.05 + 缩放 0.1 + 课程式斑点噪声",
        RUN_ROOT / "aug_step5_speckle_translate_scale" / "results.csv",
    ),
]


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


def load_best_row(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = (row.get("epoch") or "").strip()
            if not epoch.isdigit():
                continue
            rows.append({key: float(value) for key, value in row.items()})
    if not rows:
        raise RuntimeError(f"No numeric training rows found in {path}")
    return max(rows, key=lambda x: x["metrics/mAP50(B)"])


def to_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_rows():
    rows = [("步骤", "增强配置摘要", "Precision(%)", "Recall(%)", "mAP50(%)", "mAP50-95(%)")]
    for step_name, summary, csv_path in STEP_SPECS:
        best_row = load_best_row(csv_path)
        rows.append(
            (
                step_name,
                summary,
                to_percent(best_row["metrics/precision(B)"]),
                to_percent(best_row["metrics/recall(B)"]),
                to_percent(best_row["metrics/mAP50(B)"]),
                to_percent(best_row["metrics/mAP50-95(B)"]),
            )
        )
    return rows


def build_rtf(rows):
    cellx = [1500, 5600, 7600, 9600, 11600, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表7.2 YOLO五步渐进消融实验完整结果") + r"\b0\par",
        r"\pard\par",
    ]

    for idx, row in enumerate(rows):
        is_header = idx == 0
        is_step5 = (not is_header) and row[0] == "Step 5"
        use_bold = is_header or is_step5
        weight = r"\b " if use_bold else ""
        weight_end = r"\b0 " if use_bold else ""
        aligns = [r"\qc", r"\ql", r"\qc", r"\qc", r"\qc", r"\qc"]
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
