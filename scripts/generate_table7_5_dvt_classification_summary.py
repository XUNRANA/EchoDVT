import json
from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
META_JSON = ROOT / "results" / "unified_model" / "rf_unified.json"
VAL_NORMAL_TXT = ROOT / "sam2" / "dataset" / "val_normal.txt"
VAL_ABNORMAL_TXT = ROOT / "sam2" / "dataset" / "val_abnormal.txt"
RTF_OUTPUT = ROOT / "results" / "table7_5_dvt_classification_summary.rtf"
TSV_OUTPUT = ROOT / "results" / "table7_5_dvt_classification_summary.tsv"


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


def read_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def build_rows():
    meta = json.loads(META_JSON.read_text(encoding="utf-8"))

    if abs(meta.get("threshold", 0.0) - 0.05) > 1e-12:
        raise RuntimeError(f"Expected threshold 0.05, got {meta.get('threshold')}")

    n_val_normal = read_count(VAL_NORMAL_TXT)
    n_val_dvt = read_count(VAL_ABNORMAL_TXT)
    fp = int(meta["val_fp"])
    fn = int(meta["val_fn"])
    tn = n_val_normal - fp
    tp = n_val_dvt - fn

    specificity = tn / n_val_normal
    precision = meta["val_precision"]
    recall = meta["val_recall"]
    f1 = (2 * precision * recall) / (precision + recall)

    return [
        ("指标", "数值"),
        ("训练集准确率", to_percent(meta["train_accuracy"])),
        ("验证集准确率", to_percent(meta["val_accuracy"])),
        ("验证集灵敏度", to_percent(recall)),
        ("验证集特异度", to_percent(specificity)),
        ("验证集F1分数", to_percent(f1)),
    ]


def build_rtf(rows):
    cellx = [6300, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表7.5 DVT分类性能汇总") + r"\b0\par",
        r"\pard\par",
    ]

    for idx, row in enumerate(rows):
        is_header = idx == 0
        weight = r"\b " if is_header else ""
        weight_end = r"\b0 " if is_header else ""
        aligns = [r"\qc", r"\qc"]
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
