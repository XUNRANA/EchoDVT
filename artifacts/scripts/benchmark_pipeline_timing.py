import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np

ROOT = Path("/data1/ouyangxinglong/EchoDVT")
sys.path.insert(0, str(ROOT))

from classify_dvt import extract_features, predict_dvt
from web.services.inference import DEFAULT_SAM2_VARIANT, InferenceService

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


OUTPUT_PATH = ROOT / "artifacts/metadata/pipeline_timing_benchmark.json"
TARGET_FRAMES = 50
WARMUP_RUNS = 1
TIMED_RUNS = 3


def sync_cuda():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def list_case_dirs():
    val_root = ROOT / "sam2/dataset/val"
    case_dirs = []
    for case_dir in sorted([p for p in val_root.iterdir() if p.is_dir()]):
        images_dir = case_dir / "images"
        if not images_dir.exists():
            continue
        frame_files = sorted(images_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        if not frame_files:
            frame_files = sorted(images_dir.glob("*.png"), key=lambda p: int(p.stem))
        if len(frame_files) < 2:
            continue
        case_dirs.append((case_dir, frame_files))
    return case_dirs


def select_typical_case():
    candidates = []
    for case_dir, frame_files in list_case_dirs():
        n_frames = len(frame_files)
        candidates.append((abs(n_frames - TARGET_FRAMES), n_frames, case_dir.name, case_dir, frame_files))
    if not candidates:
        raise RuntimeError("No valid validation case found for timing benchmark.")
    _, n_frames, _, case_dir, frame_files = sorted(candidates)[0]
    return case_dir, frame_files, n_frames


def build_masks_list(pred_masks_by_idx, num_frames, shape):
    h, w = shape[:2]
    masks_list = []
    for i in range(num_frames):
        pred = pred_masks_by_idx.get(i)
        if pred is not None and "semantic" in pred:
            semantic = pred["semantic"]
        else:
            semantic = np.zeros((h, w), dtype=np.uint8)
        masks_list.append(semantic)
    return masks_list


def time_single_run(service, images_dir, frame_files):
    num_frames = len(frame_files)
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame: {frame_files[0]}")

    sync_cuda()
    t0 = time.perf_counter()
    detections = service.run_detection(first_frame, conf=0.1)
    sync_cuda()
    t1 = time.perf_counter()

    sync_cuda()
    pred_masks_by_idx = service.run_segmentation(
        images_dir=images_dir,
        detections=detections,
        num_frames=num_frames,
        use_mfp=True,
        variant=DEFAULT_SAM2_VARIANT,
    )
    sync_cuda()
    t2 = time.perf_counter()

    masks_list = build_masks_list(pred_masks_by_idx, num_frames, first_frame.shape)
    sync_cuda()
    _features = extract_features(masks_list)
    _result = predict_dvt(_features)
    sync_cuda()
    t3 = time.perf_counter()

    return {
        "detection_s": t1 - t0,
        "segmentation_s": t2 - t1,
        "diagnosis_s": t3 - t2,
        "total_s": t3 - t0,
    }


def summarize_runs(runs):
    keys = ["detection_s", "segmentation_s", "diagnosis_s", "total_s"]
    summary = {}
    for key in keys:
        values = [float(run[key]) for run in runs]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def get_env_info():
    info = {}
    if torch is not None:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_count"] = int(torch.cuda.device_count())
            info["cuda_device_0"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    return info


def main():
    case_dir, frame_files, n_frames = select_typical_case()
    images_dir = case_dir / "images"

    print(f"Selected case: {case_dir.name} ({n_frames} frames)")
    service = InferenceService.get()

    print("Warmup...")
    for _ in range(WARMUP_RUNS):
        _ = time_single_run(service, images_dir, frame_files)

    print("Timed runs...")
    timed_runs = []
    for idx in range(TIMED_RUNS):
        result = time_single_run(service, images_dir, frame_files)
        timed_runs.append(result)
        print(
            f"Run {idx + 1}: "
            f"detection={result['detection_s']:.4f}s, "
            f"segmentation={result['segmentation_s']:.4f}s, "
            f"diagnosis={result['diagnosis_s']:.4f}s, "
            f"total={result['total_s']:.4f}s"
        )

    payload = {
        "benchmark_type": "steady_state_single_case",
        "case_name": case_dir.name,
        "num_frames": n_frames,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "use_mfp": True,
        "sam2_variant": DEFAULT_SAM2_VARIANT,
        "environment": get_env_info(),
        "runs": timed_runs,
        "summary": summarize_runs(timed_runs),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print(f"Saved benchmark to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
