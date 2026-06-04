"""Generate training curves and comparison plots from completed runs.

Reads:
  - checkpoints/<run_name>/train_log.csv  (per-epoch train/val metrics)
  - eval/<run_name>_results.summary.json  (final test metrics)
  - eval/marigold_dc_test_results.summary.json (Marigold baseline)

Writes PNG files to results/plots/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

PLOTS_DIR = Path("results/plots")

METRIC_LABELS = {
    "absrel": "AbsRel (↓)",
    "rmse": "RMSE (↓)",
    "mae": "MAE (↓)",
    "d1": "δ₁ (↑)",
    "d2": "δ₂ (↑)",
    "d3": "δ₃ (↑)",
}

RUN_DISPLAY_NAMES = {
    "baseline": "Baseline (filter=3, levels=4, ch=default)",
    "ablation_filter_1x1": "Filter 1×1",
    "ablation_filter_3x3": "Filter 3×3",
    "ablation_filter_5x5": "Filter 5×5",
    "ablation_levels_2": "Levels=2",
    "ablation_levels_3": "Levels=3",
    "ablation_levels_4": "Levels=4",
    "ablation_channels_narrow": "Channels narrow",
    "ablation_channels_default": "Channels default",
    "ablation_channels_wide": "Channels wide",
}


def _label(run_name: str) -> str:
    return RUN_DISPLAY_NAMES.get(run_name, run_name)


def load_train_log(run_name: str, ckpt_root: Path = Path("checkpoints")) -> Optional[pd.DataFrame]:
    path = ckpt_root / run_name / "train_log.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_eval_summary(run_name: str, eval_dir: Path = Path("eval")) -> Optional[dict]:
    path = eval_dir / f"cnn_{run_name}_results.summary.json"
    if not path.exists():
        # try glob for timestamped run names
        matches = list(eval_dir.glob(f"cnn_{run_name}_*results.summary.json"))
        if not matches:
            return None
        path = sorted(matches)[-1]
    with open(path) as f:
        return json.load(f)


def load_marigold_summary(eval_dir: Path = Path("eval")) -> Optional[dict]:
    path = eval_dir / "marigold_dc_test_results.summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_bicubic_summary(eval_dir: Path = Path("eval")) -> Optional[dict]:
    path = eval_dir / "bicubic_results.summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-run training curves
# ---------------------------------------------------------------------------

def plot_training_curves(run_name: str, df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Training curves — {_label(run_name)}", fontsize=13)

    # train loss
    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], "b-o", markersize=3, label="train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train Loss")
    ax.grid(True, alpha=0.3)

    # val metrics
    metric_axes = [
        ("val_absrel", axes[0, 1], "AbsRel (↓)", "r"),
        ("val_rmse", axes[0, 2], "RMSE (↓)", "m"),
        ("val_d1", axes[1, 0], "δ₁ (↑)", "g"),
        ("val_d2", axes[1, 1], "δ₂ (↑)", "c"),
        ("val_d3", axes[1, 2], "δ₃ (↑)", "orange"),
    ]
    for col, ax, title, color in metric_axes:
        if col in df.columns:
            ax.plot(df["epoch"], df[col], "-o", color=color, markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"{run_name}_training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Cross-run comparison
# ---------------------------------------------------------------------------

def plot_comparison(
    run_summaries: Dict[str, dict],
    marigold_summary: Optional[dict],
    out_dir: Path,
):
    all_names = list(run_summaries.keys())
    if marigold_summary:
        all_names = ["Marigold-DC"] + all_names

    display = ["Marigold-DC" if n == "Marigold-DC" else _label(n) for n in all_names]

    metrics = ["absrel", "rmse", "mae", "d1", "d2", "d3"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Model Comparison — Test Set", fontsize=14)

    for ax, metric in zip(axes.flat, metrics):
        values = []
        for name in all_names:
            if name == "Marigold-DC":
                v = marigold_summary.get(metric, float("nan"))
            else:
                v = run_summaries[name].get(metric, float("nan"))
            values.append(v)

        x = np.arange(len(all_names))
        colors = ["steelblue"] + ["salmon"] * len(run_summaries) if marigold_summary else ["salmon"] * len(run_summaries)
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(display, rotation=30, ha="right", fontsize=7)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

    plt.tight_layout()
    out_path = out_dir / "comparison_test_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_val_curves_comparison(
    run_logs: Dict[str, pd.DataFrame],
    metric: str = "val_absrel",
    out_dir: Path = PLOTS_DIR,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    for run_name, df in run_logs.items():
        if metric in df.columns:
            ax.plot(df["epoch"], df[metric], "-o", markersize=3, label=_label(run_name))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(METRIC_LABELS.get(metric.replace("val_", ""), metric))
    ax.set_title(f"Validation {metric.replace('val_', '').upper()} over epochs — all runs")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f"all_runs_{metric}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Per-category ablation grouped bar charts
# ---------------------------------------------------------------------------

ABLATION_GROUPS = [
    {
        "title": "Filter Size Ablation",
        "filename": "ablation_filter_size.png",
        "runs": ["ablation_filter_1x1", "baseline", "ablation_filter_5x5"],
        "labels": ["1×1", "3×3 (baseline)", "5×5"],
    },
    {
        "title": "Encoder Levels Ablation",
        "filename": "ablation_encoder_levels.png",
        "runs": ["ablation_levels_2", "ablation_levels_3", "baseline"],
        "labels": ["2 levels", "3 levels", "4 levels (baseline)"],
    },
    {
        "title": "Channel Width Ablation",
        "filename": "ablation_channel_width.png",
        "runs": ["ablation_channels_narrow", "baseline", "ablation_channels_wide"],
        "labels": ["Narrow\n(16,32,64,128)", "Default\n(32,64,128,256)", "Wide\n(64,128,256,512)"],
    },
]

HIGHLIGHT_METRICS = ["absrel", "rmse", "d1"]


def plot_ablation_group(
    group: dict,
    run_summaries: Dict[str, dict],
    out_dir: Path,
):
    runs = group["runs"]
    labels = group["labels"]
    available = [(r, l) for r, l in zip(runs, labels) if r in run_summaries]
    if len(available) < 2:
        print(f"  Skipping {group['title']}: fewer than 2 configs available.")
        return

    avail_runs, avail_labels = zip(*available)
    fig, axes = plt.subplots(1, len(HIGHLIGHT_METRICS), figsize=(4 * len(HIGHLIGHT_METRICS), 4))
    fig.suptitle(group["title"], fontsize=13, fontweight="bold")

    x = np.arange(len(avail_runs))
    colors = ["#4C9BE8" if "baseline" in r else "#E8A74C" for r in avail_runs]

    for ax, metric in zip(axes, HIGHLIGHT_METRICS):
        values = [run_summaries[r].get(metric, float("nan")) for r in avail_runs]
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(avail_labels, fontsize=8)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = out_dir / group["filename"]
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_full_comparison(
    run_summaries: Dict[str, dict],
    marigold_summary: Optional[dict],
    bicubic_summary: Optional[dict],
    out_dir: Path,
):
    """Extended comparison chart including bicubic and Marigold baselines."""
    named = {}
    if bicubic_summary:
        named["Bicubic"] = bicubic_summary
    if marigold_summary:
        named["Marigold-DC"] = marigold_summary
    named.update(run_summaries)

    if len(named) < 2:
        return

    display = list(named.keys())
    colors = []
    for n in display:
        if n == "Marigold-DC":
            colors.append("steelblue")
        elif n == "Bicubic":
            colors.append("gray")
        else:
            colors.append("salmon")

    metrics = ["absrel", "rmse", "d1"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Full Model Comparison — Test Set (AbsRel/RMSE: lower is better, δ₁: higher is better)", fontsize=11)

    for ax, metric in zip(axes, metrics):
        values = [named[n].get(metric, float("nan")) for n in display]
        x = np.arange(len(display))
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(display, rotation=35, ha="right", fontsize=7)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color="gray", label="Bicubic"), Patch(color="steelblue", label="Marigold-DC"), Patch(color="salmon", label="CNN models")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    out_path = out_dir / "full_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_plots(
    run_names: List[str],
    ckpt_root: Path = Path("checkpoints"),
    eval_dir: Path = Path("eval"),
    out_dir: Path = PLOTS_DIR,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    run_logs: Dict[str, pd.DataFrame] = {}
    run_summaries: Dict[str, dict] = {}

    for run_name in run_names:
        df = load_train_log(run_name, ckpt_root)
        summary = load_eval_summary(run_name, eval_dir)

        if df is not None:
            run_logs[run_name] = df
            plot_training_curves(run_name, df, out_dir)
        else:
            print(f"  No train log found for {run_name}, skipping curves.")

        if summary is not None:
            run_summaries[run_name] = summary
        else:
            print(f"  No eval summary found for {run_name}.")

    marigold_summary = load_marigold_summary(eval_dir)
    if marigold_summary is None:
        print("  No Marigold-DC summary found.")

    bicubic_summary = load_bicubic_summary(eval_dir)
    if bicubic_summary is None:
        print("  No bicubic summary found.")

    if run_summaries or marigold_summary:
        plot_comparison(run_summaries, marigold_summary, out_dir)

    # Extended comparison (bicubic + Marigold + all CNNs)
    plot_full_comparison(run_summaries, marigold_summary, bicubic_summary, out_dir)

    # Per-category ablation grouped bar charts
    for group in ABLATION_GROUPS:
        plot_ablation_group(group, run_summaries, out_dir)

    for metric in ["val_absrel", "val_rmse", "val_d1"]:
        if run_logs:
            plot_val_curves_comparison(run_logs, metric=metric, out_dir=out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run-names", nargs="+", default=None, help="Run names to include.")
    p.add_argument("--ckpt-root", type=Path, default=Path("checkpoints"))
    p.add_argument("--eval-dir", type=Path, default=Path("eval"))
    p.add_argument("--out-dir", type=Path, default=PLOTS_DIR)
    args = p.parse_args()

    if args.run_names is None:
        # auto-discover from checkpoint dirs
        args.run_names = [d.name for d in args.ckpt_root.iterdir() if d.is_dir()] if args.ckpt_root.exists() else []
        print(f"Auto-discovered runs: {args.run_names}")

    generate_all_plots(args.run_names, args.ckpt_root, args.eval_dir, args.out_dir)
