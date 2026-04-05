from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
RTF_OUTPUT = ROOT / "results" / "table4_2_lora_training_hparams.rtf"
TSV_OUTPUT = ROOT / "results" / "table4_2_lora_training_hparams.tsv"


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
        ("参数名", "默认值", "说明"),
        ("LoRA rank", "8", "在参数量与表达能力之间取得较好平衡"),
        ("学习率", "3e-4", "AdamW 优化器的初始学习率"),
        ("优化器", "AdamW", "适用于权重衰减与微调训练"),
        ("调度器", "CosineAnnealing", "余弦退火调度，配合 warm restarts"),
        ("梯度累积", "4 steps", "等效批量大小为 4，降低显存压力"),
        ("混合精度", "bfloat16", "减少显存占用并利用 A100 硬件加速"),
        ("最大帧数", "40", "单个视频最多处理 40 帧，超出部分截断"),
        ("训练轮数", "25", "总训练 epoch 数"),
        ("静脉权重", "1.5", "提高静脉分割损失权重，缓解难样本问题"),
    ]


def build_rtf(rows):
    cellx = [2800, 4700, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表4.2 LoRA训练超参数配置") + r"\b0\par",
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
