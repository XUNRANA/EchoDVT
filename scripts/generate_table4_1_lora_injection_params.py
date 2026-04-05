from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
RTF_OUTPUT = ROOT / "results" / "table4_1_lora_injection_params.rtf"
TSV_OUTPUT = ROOT / "results" / "table4_1_lora_injection_params.tsv"


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
        ("组件", "具体位置", "注入方式", "参数量", "占总参数比"),
        ("Image Encoder", "48层 Hiera 注意力", "LoRA QKV（Q/V）", "~0.30M", "~0.13%"),
        ("Memory Attention", "4层 self/cross attn", "LoRA Linear（Q/K/V/Out）", "~0.15M", "~0.07%"),
        ("Mask Decoder", "SAM 解码头", "全量微调", "~0.05M", "~0.02%"),
        ("Memory Encoder", "mask->memory embedding", "可选解冻", "可选", "可选"),
        ("合计", "-", "LoRA + 部分解冻", "~0.50M", "~0.22%"),
    ]


def build_rtf(rows):
    cellx = [2500, 5200, 9000, 11200, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表4.1 LoRA注入位置与参数量统计") + r"\b0\par",
        r"\pard\par",
    ]

    for idx, row in enumerate(rows):
        is_header = idx == 0
        weight = r"\b " if is_header else ""
        weight_end = r"\b0 " if is_header else ""
        aligns = [r"\qc", r"\qc", r"\ql", r"\qc", r"\qc"]
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
