#!/usr/bin/env python3
"""Quick sweep of RF params to find unified model that passes all criteria."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from classify_dvt import load_labels_from_split_files, FEATURE_COLS

SAM2_DIR = ROOT / "sam2"

val_labels = load_labels_from_split_files(
    str(SAM2_DIR / "dataset" / "val_normal.txt"),
    str(SAM2_DIR / "dataset" / "val_abnormal.txt"),
)
train_labels = load_labels_from_split_files(
    str(SAM2_DIR / "dataset" / "train_normal.txt"), "/dev/null"
)

val_df = pd.read_csv(ROOT / "results" / "e2e_classify_v3" / "features.csv", index_col=0)
train_df = pd.read_csv(ROOT / "results" / "train_eval_v3" / "features.csv", index_col=0)

val_y = pd.Series({c: val_labels[c] for c in val_df.index if c in val_labels})
train_y = pd.Series({c: train_labels[c] for c in train_df.index if c in train_labels})
val_df = val_df.loc[val_y.index]
train_df = train_df.loc[train_y.index]

feature_cols = [c for c in FEATURE_COLS if c in val_df.columns]
combined_X = pd.concat([val_df[feature_cols], train_df[feature_cols]]).values
combined_y = pd.concat([val_y, train_y]).values
n_val = len(val_df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(combined_X)

print(f"{'w':>4} {'d':>4} {'t':>6} | {'v_acc':>6} {'v_rec':>6} {'v_FP':>4} {'v_FN':>4} | {'t_acc':>6} {'t_FP':>4} | {'pass':>4}")
print("-" * 75)

results = []
for w in [10, 12, 15, 18, 20]:
    for d in [3, 4, 5, 6, 7, None]:
        # LOO-CV on val
        loo = LeaveOneOut()
        val_probs = np.zeros(n_val)
        for train_idx, test_idx in loo.split(X_scaled[:n_val]):
            loo_X = np.concatenate([X_scaled[train_idx], X_scaled[n_val:]])
            loo_y = np.concatenate([combined_y[train_idx], combined_y[n_val:]])
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=d,
                class_weight={0: 1, 1: w}, random_state=42,
            )
            clf.fit(loo_X, loo_y)
            val_probs[test_idx] = clf.predict_proba(X_scaled[test_idx])[:, 1]

        # Final model on all data
        final_clf = RandomForestClassifier(
            n_estimators=100, max_depth=d,
            class_weight={0: 1, 1: w}, random_state=42,
        )
        final_clf.fit(X_scaled, combined_y)
        train_probs = final_clf.predict_proba(X_scaled[n_val:])[:, 1]

        for t in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
            v_preds = (val_probs >= t).astype(int)
            v_acc = accuracy_score(combined_y[:n_val], v_preds)
            v_rec = recall_score(combined_y[:n_val], v_preds, zero_division=0)
            v_cm = confusion_matrix(combined_y[:n_val], v_preds)
            v_fp, v_fn = v_cm[0, 1], v_cm[1, 0]

            t_preds = (train_probs >= t).astype(int)
            t_acc = accuracy_score(combined_y[n_val:], t_preds)
            t_fp = int(t_preds.sum())  # all train are normal

            ok = v_acc >= 0.90 and t_acc >= 0.90 and v_rec >= 0.95
            mark = "***" if ok else ""

            if v_acc >= 0.85 and t_acc >= 0.85:  # only print reasonable results
                d_str = str(d) if d else "None"
                print(f"{w:4d} {d_str:>4} {t:6.2f} | {v_acc:6.3f} {v_rec:6.3f} {v_fp:4d} {v_fn:4d} | {t_acc:6.3f} {t_fp:4d} | {mark}")
                results.append({
                    "w": w, "d": d, "t": t,
                    "v_acc": v_acc, "v_rec": v_rec, "v_fp": v_fp, "v_fn": v_fn,
                    "t_acc": t_acc, "t_fp": t_fp, "pass": ok,
                })

# Show all passing configs
passing = [r for r in results if r["pass"]]
if passing:
    print(f"\n=== {len(passing)} PASSING CONFIGS ===")
    # Sort by val_recall desc, then train_acc desc
    passing.sort(key=lambda r: (-r["v_rec"], -r["t_acc"]))
    for r in passing:
        d_str = str(r['d']) if r['d'] else 'None'
        print(f"  w={r['w']:2d} d={d_str:>4} t={r['t']:.2f} | "
              f"val={r['v_acc']:.1%} rec={r['v_rec']:.1%} FP={r['v_fp']} FN={r['v_fn']} | "
              f"train={r['t_acc']:.1%} FP={r['t_fp']}")
else:
    print("\nNo configs pass all criteria. Showing top by (val_acc + train_acc):")
    results.sort(key=lambda r: -(r["v_acc"] + r["t_acc"]))
    for r in results[:10]:
        d_str = str(r['d']) if r['d'] else 'None'
        print(f"  w={r['w']:2d} d={d_str:>4} t={r['t']:.2f} | "
              f"val={r['v_acc']:.1%} rec={r['v_rec']:.1%} FP={r['v_fp']} FN={r['v_fn']} | "
              f"train={r['t_acc']:.1%} FP={r['t_fp']}")
