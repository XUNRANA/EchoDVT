import csv
from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
RUN_ROOT = ROOT / "yolo" / "runs" / "detect" / "runs" / "detect" / "dvt_runs"
RTF_OUTPUT = ROOT / "results" / "table3_4_yolo_ablation_results.rtf"
TSV_OUTPUT = ROOT / "results" / "table3_4_yolo_ablation_results.tsv"

STEP_FILES = {
    "Step 1": RUN_ROOT / "aug_step1_baseline" / "results.csv",
    "Step 2": RUN_ROOT / "aug_step2_translate" / "results.csv",
    "Step 3": RUN_ROOT / "aug_step3_translate_scale" / "results.csv",
    "Step 4": RUN_ROOT / "aug_step4_translate0.1_scale0.1" / "results.csv",
    "Step 5": RUN_ROOT / "aug_step5_speckle_translate_scale" / "results.csv",
}


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


def load_numeric_rows(path: Path):
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
    return sorted(rows, key=lambda x: x["epoch"])


def to_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_rows():
    rows = [("实验项", "Precision", "Recall", "mAP50", "mAP50-95")]

    step5_best = None
    step5_final = None

    for step_name, csv_path in STEP_FILES.items():
        step_rows = load_numeric_rows(csv_path)
        best_row = max(step_rows, key=lambda x: x["metrics/mAP50(B)"])
        final_row = step_rows[-1]

        rows.append(
            (
                step_name,
                to_percent(best_row["metrics/precision(B)"]),
                to_percent(best_row["metrics/recall(B)"]),
                to_percent(best_row["metrics/mAP50(B)"]),
                to_percent(best_row["metrics/mAP50-95(B)"]),
            )
        )

        if step_name == "Step 5":
            step5_best = best_row
            step5_final = final_row

    if step5_final is None or step5_best is None:
        raise RuntimeError("Failed to extract Step 5 summary rows")

    rows.append(
        (
            "Step 5 最终 epoch 50",
            to_percent(step5_final["metrics/precision(B)"]),
            to_percent(step5_final["metrics/recall(B)"]),
            to_percent(step5_final["metrics/mAP50(B)"]),
            to_percent(step5_final["metrics/mAP50-95(B)"]),
        )
    )
    rows.append(
        (
            f"Step 5 最佳 epoch {int(step5_best['epoch'])}",
            to_percent(step5_best["metrics/precision(B)"]),
            to_percent(step5_best["metrics/recall(B)"]),
            to_percent(step5_best["metrics/mAP50(B)"]),
            to_percent(step5_best["metrics/mAP50-95(B)"]),
        )
    )
    return rows


def build_rtf(rows):
    cellx = [4600, 6800, 9000, 11200, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表3.4 五步消融实验完整结果") + r"\b0\par",
        r"\pard\par",
    ]

    for idx, row in enumerate(rows):
        is_header = idx == 0
        weight = r"\b " if is_header else ""
        weight_end = r"\b0 " if is_header else ""
        aligns = [r"\ql", r"\qc", r"\qc", r"\qc", r"\qc"]
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
