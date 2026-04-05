from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
RTF_OUTPUT = ROOT / "results" / "table7_1_experiment_environment.rtf"
TSV_OUTPUT = ROOT / "results" / "table7_1_experiment_environment.tsv"


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
        ("类别", "项目", "配置详情"),
        ("硬件", "GPU", "NVIDIA A100-SXM4-40GB，40 GB/卡（实验主流程通常使用单卡完成训练与推理）"),
        ("硬件", "CPU", "AMD EPYC 7R13 48-Core Processor，64 vCPU"),
        ("硬件", "内存", "251 GiB 系统内存"),
        ("软件", "OS", "Ubuntu 22.04.5 LTS（Linux 5.15.0-173-generic）"),
        ("软件", "Python", "Python 3.10.20（Conda 环境：echodvt）"),
        ("软件", "PyTorch", "PyTorch 2.10.0+cu128（CUDA 12.8）"),
        ("软件", "Ultralytics", "Ultralytics 8.4.21（YOLOv8）"),
        ("软件", "SAM2", "Meta 官方 SAM2 1.0 代码库（基于官方实现进行 LoRA 注入与训练/推理二次开发）"),
        ("软件", "scikit-learn", "scikit-learn 1.7.2（随机森林分类器实现）"),
        ("软件", "Gradio", "Gradio 6.9.0（Web 可视化与交互界面）"),
    ]


def build_rtf(rows):
    cellx = [1800, 4300, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表7.1 实验环境配置") + r"\b0\par",
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
