import json
from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
PRIOR_STATS = ROOT / "yolo" / "prior_stats.json"
RTF_OUTPUT = ROOT / "results" / "table3_3_artery_to_vein_prior_stats.rtf"
TSV_OUTPUT = ROOT / "results" / "table3_3_artery_to_vein_prior_stats.tsv"


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


def load_means():
    data = json.loads(PRIOR_STATS.read_text(encoding="utf-8"))
    stats = data["artery2vein"]
    return {
        "cx_offset": stats["cx_offset"]["mean"],
        "cy_offset": stats["cy_offset"]["mean"],
        "w_ratio": stats["w_ratio"]["mean"],
        "h_ratio": stats["h_ratio"]["mean"],
    }


def format_value(name: str, value: float) -> str:
    if name.endswith("offset"):
        return f"{value:+.3f}"
    return f"{value:.3f}"


def build_rows():
    means = load_means()
    return [
        ("参数名", "数值", "含义"),
        ("cx_offset", format_value("cx_offset", means["cx_offset"]), "静脉中心相对动脉偏右约2%"),
        ("cy_offset", format_value("cy_offset", means["cy_offset"]), "静脉中心相对动脉偏下约17%"),
        ("w_ratio", format_value("w_ratio", means["w_ratio"]), "静脉宽度约为动脉的112%"),
        ("h_ratio", format_value("h_ratio", means["h_ratio"]), "静脉高度约为动脉的85%"),
    ]


def build_rtf(rows):
    cellx = [2600, 4300, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表3.3 动脉推断静脉的相对先验统计参数") + r"\b0\par",
        r"\pard\par",
    ]

    for idx, row in enumerate(rows):
        is_header = idx == 0
        weight = r"\b " if is_header else ""
        weight_end = r"\b0 " if is_header else ""
        aligns = [r"\qc", r"\qc", r"\ql"]
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
