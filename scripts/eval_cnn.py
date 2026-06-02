"""Evaluate a trained RGBGuidedDepthUpsampler checkpoint on a split.

Output format matches run_marigold_dc.py for direct comparison.

Usage:
    python -m scripts.eval_cnn --checkpoint checkpoints/baseline_*/best.pth
    python -m scripts.eval_cnn --checkpoint checkpoints/baseline_*/best.pth --max-samples 5
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import METRIC_KEYS, compute_metrics, summarize_metric_rows
from scripts.arkit_dataset import ARKitUpsampleDataset
from scripts.model import RGBGuidedDepthUpsampler, count_parameters

DEPTH_MAX = 10.0

# Must match run_marigold_dc.py exactly so CSVs can be compared directly
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
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[RGBGuidedDepthUpsampler, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_model = ckpt["config"]["model"]

    model = RGBGuidedDepthUpsampler(
        pretrained_rgb=cfg_model.get("pretrained_rgb", True),
        residual_scale=cfg_model.get("residual_scale", 0.2),
        depth_encoder_levels=cfg_model.get("depth_encoder_levels", 4),
        depth_filter_size=cfg_model.get("depth_filter_size", 3),
        depth_channels=tuple(cfg_model.get("depth_channels", [32, 64, 128, 256])),
        fusion_decoder_channels=tuple(cfg_model.get("fusion_decoder_channels", [256, 128, 64, 32])),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.freeze_rgb()
    model.to(device)
    return model, ckpt["config"]


def run_eval_cnn(
    checkpoint_path: Path,
    split_path: Path,
    data_root: Path,
    output_csv_path: Path,
    summary_json_path: Optional[Path] = None,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    max_samples: Optional[int] = None,
    warmup_samples: int = 1,
    target_hw: tuple = (768, 1024),
    confidence_threshold: int = 2,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running CNN eval on {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, full_cfg = load_model_from_checkpoint(checkpoint_path, device)
    cfg_model = full_cfg["model"]
    total_p, trainable_p = count_parameters(model)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Parameters: total={total_p:,}, trainable={trainable_p:,}")

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
        rgb = batch["rgb_norm"].unsqueeze(0).to(device)
        bicubic_norm = batch["bicubic_norm"].unsqueeze(0).to(device)
        conf_hi = batch["conf_hi"].unsqueeze(0).to(device)
        gt_np = batch["gt"][0].numpy()

        synchronize_if_needed(device)
        t0 = time.perf_counter()

        with torch.no_grad():
            pred_norm = model(rgb, bicubic_norm, conf_hi)

        synchronize_if_needed(device)
        runtime_s = time.perf_counter() - t0

        pred_m = (pred_norm[0, 0] * DEPTH_MAX).clamp(min_depth, max_depth).cpu().numpy()
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
        "model": "RGBGuidedDepthUpsampler",
        "checkpoint": str(checkpoint_path),
        "split_path": str(split_path),
        "data_root": str(data_root),
        "output_csv_path": str(output_csv_path),
        "num_requested_samples": len(indices),
        "num_evaluated_samples": len(rows),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "target_height": int(target_hw[0]),
        "target_width": int(target_hw[1]),
        "confidence_threshold": int(confidence_threshold),
        "total_wall_time_s": float(total_runtime_s),
        "depth_encoder_levels": cfg_model.get("depth_encoder_levels", 4),
        "depth_filter_size": cfg_model.get("depth_filter_size", 3),
        "depth_channels": list(cfg_model.get("depth_channels", [32, 64, 128, 256])),
        "fusion_decoder_channels": list(cfg_model.get("fusion_decoder_channels", [256, 128, 64, 32])),
        "total_params": total_p,
        "trainable_params": trainable_p,
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
    p = argparse.ArgumentParser(description="Evaluate CNN depth completion checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split-path", type=Path, default=Path("splits/test.txt"))
    p.add_argument("--data-root", type=Path, default=Path("data/arkitscenes"))
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--min-depth", type=float, default=0.1)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--warmup-samples", type=int, default=1)
    p.add_argument("--target-height", type=int, default=768)
    p.add_argument("--target-width", type=int, default=1024)
    p.add_argument("--confidence-threshold", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device(args.device) if args.device else None

    if args.output_csv is None:
        run_name = args.checkpoint.parent.name
        output_csv = Path("eval") / f"cnn_{run_name}_results.csv"
    else:
        output_csv = args.output_csv

    run_eval_cnn(
        checkpoint_path=args.checkpoint,
        split_path=args.split_path,
        data_root=args.data_root,
        output_csv_path=output_csv,
        summary_json_path=args.summary_json,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
        warmup_samples=args.warmup_samples,
        target_hw=(args.target_height, args.target_width),
        confidence_threshold=args.confidence_threshold,
        device=device,
    )
