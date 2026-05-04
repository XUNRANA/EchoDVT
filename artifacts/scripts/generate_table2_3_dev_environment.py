from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
RTF_OUTPUT = ROOT / "artifacts" / "tables" / "table2_3_dev_environment.rtf"
TSV_OUTPUT = ROOT / "artifacts" / "tables" / "table2_3_dev_environment.tsv"


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
        ("类别", "项目", "配置"),
        ("硬件", "GPU", "NVIDIA A100-SXM4-40GB，40 GB ×4（当前主机实测；实验与推理通常使用单卡）"),
        ("软件", "操作系统", "Ubuntu 22.04.5 LTS（Linux 5.15.0-173-generic）"),
        ("软件", "环境管理", "Conda，项目环境名为 echodvt"),
        ("软件", "开发语言", "Python 3.10.20"),
        ("软件", "深度学习框架", "PyTorch 2.10.0+cu128"),
        ("软件", "CUDA", "12.8（PyTorch 构建版本）"),
        ("软件", "目标检测库", "Ultralytics 8.4.21（YOLOv8）"),
        ("软件", "视频分割库", "SAM2 1.0（Meta 官方代码库，基于其进行 LoRA、训练器和后处理二次开发）"),
        ("软件", "Web 框架", "Gradio 6.9.0"),
        ("软件", "图像处理库", "OpenCV 4.13.0"),
        ("软件", "可视化库", "Matplotlib 3.10.8"),
        ("软件", "分类库", "scikit-learn 1.7.2"),
        ("软件", "数据分析库", "pandas 2.3.3"),
        ("软件", "科学计算库", "SciPy 1.15.3"),
    ]


def build_rtf(rows):
    cellx = [1800, 4200, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表2.3 开发环境配置表") + r"\b0\par",
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
