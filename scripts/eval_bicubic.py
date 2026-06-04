"""Evaluate bicubic upsampling baseline on a split.

Bicubic interpolates the low-resolution LiDAR depth to the target resolution
without any learned guidance. Output format matches eval_cnn.py exactly so
results can be compared directly.

Usage:
    python -m scripts.eval_bicubic --split-path splits/test.txt --data-root data/arkitscenes
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import METRIC_KEYS, compute_metrics, summarize_metric_rows
from scripts.arkit_dataset import ARKitUpsampleDataset

DEPTH_MAX = 10.0

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


def run_eval_bicubic(
    split_path: Path,
    data_root: Path,
    output_csv_path: Path,
    summary_json_path: Optional[Path] = None,
    min_depth: float = 0.1,
    max_depth: float = DEPTH_MAX,
    max_samples: Optional[int] = None,
    warmup_samples: int = 1,
    target_hw: tuple = (768, 1024),
    confidence_threshold: int = 2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running bicubic eval on {device}")

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

    indices = list(range(len(dataset)))
    if max_samples is not None:
        indices = indices[:max_samples]
    print(f"Evaluating {len(indices)} samples from {split_path}")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_json_path is None:
        summary_json_path = output_csv_path.with_suffix(".summary.json")

    rows = []
    total_start = time.perf_counter()

    for i, idx in enumerate(indices):
        batch = dataset[idx]
        lowres = batch["lowres"].unsqueeze(0).to(device)
        gt_np = batch["gt"][0].numpy()

        hw = (target_hw[0], target_hw[1])

        t0 = time.perf_counter()
        bicubic = F.interpolate(lowres.float(), size=hw, mode="bicubic", align_corners=False)
        bicubic = bicubic.clamp(min_depth, max_depth)
        if device.type == "cuda":
            torch.cuda.synchronize()
        runtime_s = time.perf_counter() - t0

        pred_m = bicubic[0, 0].float().cpu().numpy()
        metrics = compute_metrics(pred_m, gt_np, min_depth=min_depth, max_depth=max_depth)

        h, w = gt_np.shape[-2:]
        row = {
            "sample_id": batch["identifier"],
            **metrics,
            "runtime_s": float(runtime_s),
            "runtime_ms": float(runtime_s * 1000.0),
            "height": int(h),
            "width": int(w),
        }
        rows.append(row)

        if (i + 1) % 500 == 0 or i == 0:
            print(
                f"[{i + 1}/{len(indices)}] {batch['identifier']}: "
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
    runtime_values = np.array([r["runtime_s"] for r in timed_rows], dtype=np.float64)

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
            "runtime_mean_s": float("nan"),
            "runtime_median_s": float("nan"),
            "runtime_std_s": float("nan"),
            "runtime_mean_ms": float("nan"),
            "fps_mean": float("nan"),
            "warmup_samples_excluded": 0,
        }

    summary = {
        "model": "BicubicUpsampling",
        "checkpoint": None,
        "split_path": str(split_path),
        "data_root": str(data_root),
        "output_csv_path": str(output_csv_path),
        "num_requested_samples": len(indices),
        "num_evaluated_samples": len(rows),
        "device": str(device),
        "min_depth": min_depth,
        "max_depth": max_depth,
        "target_height": int(target_hw[0]),
        "target_width": int(target_hw[1]),
        "confidence_threshold": int(confidence_threshold),
        "total_wall_time_s": float(total_runtime_s),
        "total_params": 0,
        "trainable_params": 0,
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
    print(f"  total wall time: {summary['total_wall_time_s']:.2f} s")

    return rows, summary


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate bicubic upsampling baseline.")
    p.add_argument("--split-path", type=Path, default=Path("splits/test.txt"))
    p.add_argument("--data-root", type=Path, default=Path("data/arkitscenes"))
    p.add_argument("--output-csv", type=Path, default=Path("eval/bicubic_results.csv"))
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--min-depth", type=float, default=0.1)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--warmup-samples", type=int, default=1)
    p.add_argument("--target-height", type=int, default=768)
    p.add_argument("--target-width", type=int, default=1024)
    p.add_argument("--confidence-threshold", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_eval_bicubic(
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
