import argparse
import csv
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scripts.arkit_dataset import ARKitUpsampleDataset

MARIGOLD_REPO_URL = "https://github.com/prs-eth/Marigold-DC.git"
MARIGOLD_REPO_DIR = Path(__file__).resolve().parents[1] / "third_party" / "Marigold-DC"


def ensure_marigold_installed():
    try:
        return importlib.import_module("marigold")
    except ImportError:
        MARIGOLD_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)

        if not MARIGOLD_REPO_DIR.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", MARIGOLD_REPO_URL, str(MARIGOLD_REPO_DIR)],
                check=True,
            )

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(MARIGOLD_REPO_DIR)],
            check=True,
        )
        return importlib.import_module("marigold")


MarigoldDepthCompletionPipeline = ensure_marigold_installed().MarigoldDepthCompletionPipeline

from eval.metrics import METRIC_KEYS, compute_metrics, summarize_metric_rows


CSV_FIELDNAMES = [
    "sample_id",
    "rmse",
    "mae",
    "absrel",
    "d1",
    "d2",
    "d3",
    "valid_pixels",
    "runtime_s",
    "runtime_ms",
    "height",
    "width",
]


def synchronize_if_needed(device: torch.device):
    """Synchronize GPU before/after timing so runtime measurements are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_marigold_dc(
    split_path: Path,
    data_root: Path,
    output_csv_path: Path,
    summary_json_path: Path | None = None,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    max_samples: int | None = None,
    warmup_samples: int = 1,
    target_hw: tuple[int, int] = (768, 1024),
    confidence_threshold: int = 2,
):
    """
    Run Marigold-DC over a split and save per-sample metrics/runtime.

    Args:
        split_path: path to a split txt file, one sample/scene ID per line
        data_root: dataset root directory
        output_csv_path: where to save per-sample metrics
        summary_json_path: where to save aggregate metrics/runtime summary
        min_depth: lower valid-depth threshold in meters
        max_depth: upper valid-depth threshold in meters
        max_samples: optional limit for smoke tests
        warmup_samples: number of initial samples excluded from mean runtime
        target_hw: target resolution shared with the CNN pipeline
        confidence_threshold: low-res ARKit confidence threshold to keep
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Marigold-DC on {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pipe = MarigoldDepthCompletionPipeline.from_pretrained(
        "prs-eth/marigold-depth-completion-v1-0"
    ).to(device)

    dataset = ARKitUpsampleDataset(
        root=data_root,
        split_path=split_path,
        target_hw=target_hw,
        crop_hw=None,
        augment=False,
        confidence_threshold=confidence_threshold,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    sample_indices = list(range(len(dataset)))
    if max_samples is not None:
        sample_indices = sample_indices[:max_samples]

    print(f"Evaluating {len(sample_indices)} samples from {split_path}")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_json_path is None:
        summary_json_path = output_csv_path.with_suffix(".summary.json")
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total_start = time.perf_counter()

    for idx, dataset_idx in enumerate(sample_indices):
        sample_id, rgb, sparse_depth, gt_depth = dataset.get_marigold_sample(dataset_idx)

        synchronize_if_needed(device)
        start = time.perf_counter()

        # Marigold-DC takes RGB + sparse depth and returns dense predicted depth.
        with torch.no_grad():
            output = pipe(
                image=rgb,
                sparse_depth=sparse_depth,
            )

        synchronize_if_needed(device)
        runtime_s = time.perf_counter() - start

        pred = output.depth_np  # expected shape: (H, W), in meters
        metrics = compute_metrics(pred, gt_depth, min_depth=min_depth, max_depth=max_depth)

        height, width = gt_depth.shape[-2:]
        row = {
            "sample_id": sample_id,
            **metrics,
            "runtime_s": float(runtime_s),
            "runtime_ms": float(runtime_s * 1000.0),
            "height": int(height),
            "width": int(width),
        }
        rows.append(row)

        print(
            f"[{idx + 1}/{len(sample_indices)}] {sample_id}: "
            f"absrel={row['absrel']:.4f}, rmse={row['rmse']:.4f}, "
            f"d1={row['d1']:.4f}, runtime={row['runtime_ms']:.1f} ms"
        )

    total_runtime_s = time.perf_counter() - total_start

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    metric_summary = summarize_metric_rows(rows, metric_keys=METRIC_KEYS)

    timed_rows = rows[warmup_samples:] if len(rows) > warmup_samples else rows
    runtime_values = np.array([row["runtime_s"] for row in timed_rows], dtype=np.float64)

    if len(runtime_values) > 0:
        runtime_summary = {
            "runtime_mean_s": float(runtime_values.mean()),
            "runtime_median_s": float(np.median(runtime_values)),
            "runtime_std_s": float(runtime_values.std()),
            "runtime_mean_ms": float(runtime_values.mean() * 1000.0),
            "fps_mean": float(1.0 / runtime_values.mean()),
            "warmup_samples_excluded": int(min(warmup_samples, len(rows))),
        }
    else:
        runtime_summary = {
            "runtime_mean_s": np.nan,
            "runtime_median_s": np.nan,
            "runtime_std_s": np.nan,
            "runtime_mean_ms": np.nan,
            "fps_mean": np.nan,
            "warmup_samples_excluded": 0,
        }

    summary = {
        "model": "Marigold-DC",
        "split_path": str(split_path),
        "data_root": str(data_root),
        "output_csv_path": str(output_csv_path),
        "num_requested_samples": len(sample_indices),
        "num_evaluated_samples": len(rows),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "target_height": int(target_hw[0]),
        "target_width": int(target_hw[1]),
        "confidence_threshold": int(confidence_threshold),
        "total_wall_time_s": float(total_runtime_s),
        **metric_summary,
        **runtime_summary,
    }

    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved per-sample results to {output_csv_path}")
    print(f"Saved summary to {summary_json_path}")

    print("\n--- mean metrics ---")
    for key in METRIC_KEYS:
        print(f"  {key}: {summary[key]:.4f}")

    print("\n--- runtime ---")
    print(f"  mean runtime: {summary['runtime_mean_ms']:.2f} ms/sample")
    print(f"  mean FPS: {summary['fps_mean']:.2f}")
    print(f"  total wall time: {summary['total_wall_time_s']:.2f} s")

    return rows, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Marigold-DC on a depth-completion split.")
    parser.add_argument("--split-path", type=Path, default=Path("splits/test.txt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/arkitscenes"))
    parser.add_argument("--output-csv", type=Path, default=Path("eval/marigold_dc_test_results.csv"))
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=1,
        help="Number of initial evaluated samples to exclude from mean runtime.",
    )
    parser.add_argument("--target-height", type=int, default=768)
    parser.add_argument("--target-width", type=int, default=1024)
    parser.add_argument("--confidence-threshold", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_marigold_dc(
        split_path=args.split_path,
        data_root=args.data_root,
        output_csv_path=args.output_csv,
        summary_json_path=args.summary_json,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
        warmup_samples=args.warmup_samples,
        target_hw=(args.target_height, args.target_width),
        confidence_threshold=args.confidence_threshold,
    )
