"""
compute_prior_stats.py

从 YOLO 训练标签统计两类先验：
1) class_absolute: 动脉/静脉各自的绝对框分布（cx, cy, w, h）
2) artery2vein / vein2artery: 已知一类时推另一类的相对关系（offset + ratio）
"""

import json
from pathlib import Path

import numpy as np

# ── 配置 ──────────────────────────────────────
LABEL_DIR = "dataset/train/labels"
OUTPUT_PATH = "prior_stats.json"
# ─────────────────────────────────────────────


def summarize(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("空数组无法统计")
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def parse_yolo_label(label_path: Path):
    """
    解析单个标签文件，返回 (artery_box, vein_box)
    box 格式: (cx, cy, w, h)，均为 [0,1] 归一化坐标。
    """
    artery, vein = None, None
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            if cls_id == 0:
                artery = (cx, cy, w, h)
            elif cls_id == 1:
                vein = (cx, cy, w, h)
    return artery, vein


def compute_stats(label_dir: str, output_path: str):
    label_dir = Path(label_dir)
    label_files = sorted(label_dir.glob("*.txt"))
    print(f"扫描标签文件: {len(label_files)} 个")

    class_samples = {
        "artery": {"cx": [], "cy": [], "w": [], "h": []},
        "vein": {"cx": [], "cy": [], "w": [], "h": []},
    }
    pair_samples = {
        "artery2vein": {"cx_offset": [], "cy_offset": [], "w_ratio": [], "h_ratio": []},
        "vein2artery": {"cx_offset": [], "cy_offset": [], "w_ratio": [], "h_ratio": []},
    }

    pair_skipped = 0
    for label_path in label_files:
        artery, vein = parse_yolo_label(label_path)

        if artery is not None:
            a_cx, a_cy, a_w, a_h = artery
            class_samples["artery"]["cx"].append(a_cx)
            class_samples["artery"]["cy"].append(a_cy)
            class_samples["artery"]["w"].append(a_w)
            class_samples["artery"]["h"].append(a_h)

        if vein is not None:
            v_cx, v_cy, v_w, v_h = vein
            class_samples["vein"]["cx"].append(v_cx)
            class_samples["vein"]["cy"].append(v_cy)
            class_samples["vein"]["w"].append(v_w)
            class_samples["vein"]["h"].append(v_h)

        if artery is None or vein is None:
            pair_skipped += 1
            continue

        a_cx, a_cy, a_w, a_h = artery
        v_cx, v_cy, v_w, v_h = vein

        pair_samples["artery2vein"]["cx_offset"].append(v_cx - a_cx)
        pair_samples["artery2vein"]["cy_offset"].append(v_cy - a_cy)
        pair_samples["artery2vein"]["w_ratio"].append(v_w / a_w if a_w > 0 else 1.0)
        pair_samples["artery2vein"]["h_ratio"].append(v_h / a_h if a_h > 0 else 1.0)

        pair_samples["vein2artery"]["cx_offset"].append(a_cx - v_cx)
        pair_samples["vein2artery"]["cy_offset"].append(a_cy - v_cy)
        pair_samples["vein2artery"]["w_ratio"].append(a_w / v_w if v_w > 0 else 1.0)
        pair_samples["vein2artery"]["h_ratio"].append(a_h / v_h if v_h > 0 else 1.0)

    n_pair = len(pair_samples["artery2vein"]["cx_offset"])
    n_artery = len(class_samples["artery"]["cx"])
    n_vein = len(class_samples["vein"]["cx"])

    print(f"动脉样本: {n_artery} | 静脉样本: {n_vein} | 共现样本: {n_pair}")

    if n_artery == 0 or n_vein == 0:
        print("错误：动脉或静脉样本为空，请检查标签")
        return
    if n_pair == 0:
        print("错误：没有找到动静脉共现样本，无法统计相对先验")
        return

    stats = {
        "class_absolute": {
            cls_name: {key: summarize(values) for key, values in cls_values.items()}
            for cls_name, cls_values in class_samples.items()
        }
    }
    for direction, direction_values in pair_samples.items():
        stats[direction] = {key: summarize(values) for key, values in direction_values.items()}

    stats["meta"] = {
        "n_samples": n_pair,  # 兼容旧版读取逻辑
        "n_pair_samples": n_pair,
        "n_artery_samples": n_artery,
        "n_vein_samples": n_vein,
        "n_skipped_pair": pair_skipped,
        "n_label_files": len(label_files),
        "label_dir": str(label_dir),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("统计结果（归一化坐标）")
    print(f"{'=' * 60}")

    print("\n[ 绝对先验 ]")
    for cls_name in ["artery", "vein"]:
        print(f"  - {cls_name}")
        for key, v in stats["class_absolute"][cls_name].items():
            print(f"    {key:<3} mean={v['mean']:.4f} std={v['std']:.4f} median={v['median']:.4f}")

    for direction in ["artery2vein", "vein2artery"]:
        title = "动脉 → 推静脉" if direction == "artery2vein" else "静脉 → 推动脉"
        print(f"\n[ {title} ]")
        for key, v in stats[direction].items():
            print(f"  {key:<10} mean={v['mean']:.4f} std={v['std']:.4f} median={v['median']:.4f}")

    print(f"\n已保存至: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    compute_stats(LABEL_DIR, OUTPUT_PATH)
