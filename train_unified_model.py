#!/usr/bin/env python3
"""
训练统一 RF 分类模型（val+train 联合），保存供 web 和离线评估使用。

用法:
    python train_unified_model.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from classify_dvt import (
    train_unified_model,
    load_labels_from_split_files,
    UNIFIED_MODEL_PATH,
    UNIFIED_RF_PARAMS,
    UNIFIED_THRESHOLD,
)

SAM2_DIR = ROOT / "sam2"

# Val labels
val_labels = load_labels_from_split_files(
    str(SAM2_DIR / "dataset" / "val_normal.txt"),
    str(SAM2_DIR / "dataset" / "val_abnormal.txt"),
)

# Train labels — all normal (300 cases)
train_normal_file = SAM2_DIR / "dataset" / "train_normal.txt"
if not train_normal_file.exists():
    # Generate from directory listing
    train_dir = SAM2_DIR / "dataset" / "train"
    cases = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    train_normal_file.write_text("\n".join(cases) + "\n")
    print(f"Generated {train_normal_file} with {len(cases)} cases")

train_labels = load_labels_from_split_files(str(train_normal_file), "/dev/null")

print(f"Val labels: {len(val_labels)} ({sum(1 for v in val_labels.values() if v==0)} normal, "
      f"{sum(1 for v in val_labels.values() if v==1)} DVT)")
print(f"Train labels: {len(train_labels)} (all normal)")

val_csv = str(ROOT / "results" / "e2e_classify_v3" / "features.csv")
train_csv = str(ROOT / "results" / "train_eval_v3" / "features.csv")

meta = train_unified_model(
    val_features_csv=val_csv,
    train_features_csv=train_csv,
    val_labels=val_labels,
    train_labels=train_labels,
)

print(f"\n=== Results ===")
print(f"Val:   Acc={meta['val_accuracy']:.1%}  Recall={meta['val_recall']:.1%}  "
      f"FP={meta['val_fp']}  FN={meta['val_fn']}")
print(f"Train: Acc={meta['train_accuracy']:.1%}  FP={meta['train_fp']}/300")

val_ok = meta["val_accuracy"] >= 0.90
train_ok = meta["train_accuracy"] >= 0.90
recall_ok = meta["val_recall"] >= 0.95

status = "PASS" if (val_ok and train_ok and recall_ok) else "FAIL"
print(f"\nVal>=90%: {'YES' if val_ok else 'NO'}  "
      f"Train>=90%: {'YES' if train_ok else 'NO'}  "
      f"Recall>=95%: {'YES' if recall_ok else 'NO'}  → {status}")
