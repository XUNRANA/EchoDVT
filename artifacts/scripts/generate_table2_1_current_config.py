import json
from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
META_PATH = ROOT / "artifacts" / "unified_model" / "rf_unified.json"
RTF_OUTPUT = ROOT / "artifacts" / "tables" / "table2_1_echoDVT_system_current_config.rtf"
TSV_OUTPUT = ROOT / "artifacts" / "tables" / "table2_1_echoDVT_system_current_config.tsv"

YOLO_WEIGHT = "yolo/runs/detect/runs/detect/dvt_runs/aug_step5_speckle_translate_scale/weights/best.pt"
SAM2_BACKBONE = "Hiera Large (sam2_hiera_large.pt)"
SAM2_LORA = "LoRA r8"


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
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    feature_dim = len(meta.get("feature_cols", []))
    threshold = float(meta.get("threshold", 0.05))

    return [
        ("模块名称", "当前配置"),
        ("YOLO 检测权重", YOLO_WEIGHT),
        ("YOLO 置信度阈值", "0.1"),
        ("SAM2 主干网络", SAM2_BACKBONE),
        ("SAM2 LoRA 微调", SAM2_LORA),
        ("多帧提示 MFP", "开启（固定）"),
        ("分类模型", "RF unified"),
        ("分类阈值", f"prob >= {threshold:.2f}"),
        ("特征维度", f"{feature_dim} 维"),
    ]


def build_rtf(rows):
    left = 3200
    right = 12500
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表2.1 EchoDVT系统当前主线配置") + r"\b0\par",
        r"\pard\par",
    ]

    for idx, (col1, col2) in enumerate(rows):
        weight = r"\b " if idx == 0 else ""
        weight_end = r"\b0 " if idx == 0 else ""
        lines.extend(
            [
                r"\trowd\trgaph108\trqc",
                rf"\cellx{left}\cellx{right}",
                r"\pard\intbl\qc\f0\fs21 " + weight + rtf_escape(col1) + weight_end + r"\cell",
                r"\pard\intbl\ql\f0\fs21 " + weight + rtf_escape(col2) + weight_end + r"\cell",
                r"\row",
            ]
        )

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
