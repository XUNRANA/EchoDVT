import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path("/data1/ouyangxinglong/EchoDVT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from classify_dvt import FEATURE_COLS, UNIFIED_RF_PARAMS, load_labels_from_split_files


SAM2_DIR = ROOT / "sam2"
VAL_FEATURES_CSV = ROOT / "results" / "e2e_classify_v3" / "features.csv"
TRAIN_FEATURES_CSV = ROOT / "results" / "train_eval_v3" / "features.csv"
META_JSON = ROOT / "results" / "unified_model" / "rf_unified.json"
RTF_OUTPUT = ROOT / "results" / "table5_2_threshold_performance.rtf"
TSV_OUTPUT = ROOT / "results" / "table5_2_threshold_performance.tsv"
THRESHOLDS = [0.01, 0.05, 0.10, 0.20, 0.50]
DEFAULT_THRESHOLD = 0.05


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


def load_feature_sets():
    val_labels = load_labels_from_split_files(
        str(SAM2_DIR / "dataset" / "val_normal.txt"),
        str(SAM2_DIR / "dataset" / "val_abnormal.txt"),
    )
    train_labels = load_labels_from_split_files(
        str(SAM2_DIR / "dataset" / "train_normal.txt"),
        "/dev/null",
    )

    val_df = pd.read_csv(VAL_FEATURES_CSV, index_col=0)
    train_df = pd.read_csv(TRAIN_FEATURES_CSV, index_col=0)

    val_y = pd.Series({case_id: val_labels[case_id] for case_id in val_df.index if case_id in val_labels})
    train_y = pd.Series({case_id: train_labels[case_id] for case_id in train_df.index if case_id in train_labels})

    val_df = val_df.loc[val_y.index]
    train_df = train_df.loc[train_y.index]

    feature_cols = [col for col in FEATURE_COLS if col in val_df.columns]
    combined_x = pd.concat([val_df[feature_cols], train_df[feature_cols]])
    combined_y = pd.concat([val_y, train_y])
    return combined_x, combined_y, len(val_df)


def compute_val_probs_loo():
    combined_x, combined_y, n_val = load_feature_sets()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(combined_x.values)

    val_x_scaled = x_scaled[:n_val]
    val_y = combined_y.values[:n_val]

    loo = LeaveOneOut()
    val_probs_loo = np.zeros(n_val, dtype=np.float64)

    for train_idx, test_idx in loo.split(val_x_scaled):
        loo_train_x = np.concatenate([x_scaled[train_idx], x_scaled[n_val:]])
        loo_train_y = np.concatenate([combined_y.values[train_idx], combined_y.values[n_val:]])
        clf = RandomForestClassifier(**UNIFIED_RF_PARAMS)
        clf.fit(loo_train_x, loo_train_y)
        val_probs_loo[test_idx] = clf.predict_proba(val_x_scaled[test_idx])[:, 1]

    return val_y, val_probs_loo


def verify_default_threshold(metrics):
    with META_JSON.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    current = next(row for row in metrics if abs(row["threshold"] - DEFAULT_THRESHOLD) < 1e-9)
    if abs(current["accuracy"] - meta["val_accuracy"]) > 1e-12:
        raise RuntimeError("Threshold 0.05 accuracy does not match rf_unified.json")
    if abs(current["sensitivity"] - meta["val_recall"]) > 1e-12:
        raise RuntimeError("Threshold 0.05 sensitivity does not match rf_unified.json")


def build_metrics():
    y_true, y_prob = compute_val_probs_loo()
    metrics = []

    for threshold in THRESHOLDS:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        metrics.append(
            {
                "threshold": threshold,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "accuracy": accuracy,
                "f1": f1,
            }
        )

    verify_default_threshold(metrics)
    return metrics


def build_rows():
    rows = [("阈值", "灵敏度", "特异度", "准确率", "F1分数")]
    for row in build_metrics():
        rows.append(
            (
                f"{row['threshold']:.2f}",
                to_percent(row["sensitivity"]),
                to_percent(row["specificity"]),
                to_percent(row["accuracy"]),
                to_percent(row["f1"]),
            )
        )
    return rows


def build_rtf(rows):
    cellx = [2200, 5100, 8000, 10900, 13800]
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}{\f1\fnil Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\pard\qc\f0\fs21\b " + rtf_escape("表5.2 不同阈值下的分类性能") + r"\b0\par",
        r"\pard\par",
    ]

    for idx, row in enumerate(rows):
        is_header = idx == 0
        weight = r"\b " if is_header else ""
        weight_end = r"\b0 " if is_header else ""
        aligns = [r"\qc", r"\qc", r"\qc", r"\qc", r"\qc"]
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
