from pathlib import Path


ROOT = Path("/data1/ouyangxinglong/EchoDVT")
SAM2_DATASET = ROOT / "sam2" / "dataset"
TEST_ROOT = ROOT / "test"

RTF_OUTPUT = ROOT / "results" / "table2_2_dataset_split_statistics.rtf"
TSV_OUTPUT = ROOT / "results" / "table2_2_dataset_split_statistics.tsv"


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


def read_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def has_masks(case_dir: Path) -> bool:
    return (case_dir / "masks").is_dir()


def count_cases(root: Path) -> int:
    return sum(1 for p in root.iterdir() if p.is_dir())


def build_rows():
    train_root = SAM2_DATASET / "train"
    val_root = SAM2_DATASET / "val"
    test_normal_root = TEST_ROOT / "normal"
    test_patient_root = TEST_ROOT / "patient"

    train_cases = count_cases(train_root)
    val_cases = count_cases(val_root)
    test_normal = len(read_list(TEST_ROOT / "normal.txt"))
    test_patient = len(read_list(TEST_ROOT / "patient.txt"))
    val_normal = len(read_list(SAM2_DATASET / "val_normal.txt"))
    val_patient = len(read_list(SAM2_DATASET / "val_abnormal.txt"))

    train_with_masks = all(has_masks(p) for p in train_root.iterdir() if p.is_dir())
    val_with_masks = all(has_masks(p) for p in val_root.iterdir() if p.is_dir())
    test_with_masks = any(has_masks(p) for p in test_normal_root.iterdir() if p.is_dir()) or any(
        has_masks(p) for p in test_patient_root.iterdir() if p.is_dir()
    )

    train_mask_text = "是（全部带 masks）" if train_with_masks else "部分带标注"
    val_mask_text = "是（全部带 masks）" if val_with_masks else "部分带标注"
    test_mask_text = "否（仅 images）" if not test_with_masks else "部分带标注"

    return [
        ("划分", "病例数", "类别分布（正常/患者）", "是否带分割标注", "主要用途"),
        ("train", str(train_cases), f"{train_cases} / 0", train_mask_text, "YOLO 检测训练、SAM2 分割训练、位置先验统计与正常参考特征提取"),
        ("val", str(val_cases), f"{val_normal} / {val_patient}", val_mask_text, "消融实验、模型选型、分割指标评估与分类验证"),
        ("test", str(test_normal + test_patient), f"{test_normal} / {test_patient}", test_mask_text, "Web 病例浏览、完整流水线推理与批量测试"),
    ]


def build_rtf(rows):
    cellx = [1500, 3000, 6100, 8600, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表2.2 数据集划分统计") + r"\b0\par",
        r"\pard\par",
    ]

    for idx, row in enumerate(rows):
        is_header = idx == 0
        weight = r"\b " if is_header else ""
        weight_end = r"\b0 " if is_header else ""
        aligns = [r"\qc", r"\qc", r"\qc", r"\qc", r"\ql"]
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
