"""Statistical significance analysis for depth completion results.

Computes:
  - Bootstrap 95% CI (10,000 resamples, seed=42) for AbsRel and RMSE per model
  - Paired Wilcoxon signed-rank test (CNN baseline vs Marigold-DC on overlapping samples)
  - Per-scene AbsRel breakdown bar chart

Outputs:
  - results/stats_analysis.json
  - results/plots/per_scene_absrel.png

Usage:
    python -m scripts.stats_analysis \
        --cnn-csv eval/cnn_baseline_results.csv \
        --marigold-csv eval/marigold_dc_test_results.csv \
        --bicubic-csv eval/bicubic_results.csv \
        --out-dir results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
CI_ALPHA = 0.05  # 95% CI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["absrel", "rmse"])
    df["scene"] = df["sample_id"].str.split("/").str[0]
    return df


def bootstrap_ci(values: np.ndarray, stat_fn=np.mean, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Return (mean, ci_low, ci_high) using percentile bootstrap."""
    rng = np.random.default_rng(seed)
    estimates = np.array(
        [stat_fn(rng.choice(values, size=len(values), replace=True)) for _ in range(n)]
    )
    ci_low = np.percentile(estimates, 100 * CI_ALPHA / 2)
    ci_high = np.percentile(estimates, 100 * (1 - CI_ALPHA / 2))
    return float(stat_fn(values)), float(ci_low), float(ci_high)


def model_stats(df: pd.DataFrame, name: str) -> dict:
    result = {"model": name, "n_samples": len(df)}
    for metric in ["absrel", "rmse", "d1"]:
        vals = df[metric].to_numpy(dtype=np.float64)
        mean, lo, hi = bootstrap_ci(vals)
        result[metric] = mean
        result[f"{metric}_ci95_low"] = lo
        result[f"{metric}_ci95_high"] = hi
    return result


def wilcoxon_test(cnn_df: pd.DataFrame, other_df: pd.DataFrame) -> dict:
    """Paired Wilcoxon on samples present in both CSVs, matched by sample_id."""
    merged = cnn_df.set_index("sample_id")[["absrel", "rmse"]].join(
        other_df.set_index("sample_id")[["absrel", "rmse"]],
        lsuffix="_cnn",
        rsuffix="_other",
        how="inner",
    )
    n = len(merged)
    if n < 20:
        return {"n_overlapping": n, "note": "too few overlapping samples for Wilcoxon"}

    out = {"n_overlapping": n}
    for metric in ["absrel", "rmse"]:
        diffs = merged[f"{metric}_cnn"].to_numpy() - merged[f"{metric}_other"].to_numpy()
        if np.all(diffs == 0):
            out[f"wilcoxon_{metric}_stat"] = 0.0
            out[f"wilcoxon_{metric}_pvalue"] = 1.0
        else:
            stat, pval = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
            out[f"wilcoxon_{metric}_stat"] = float(stat)
            out[f"wilcoxon_{metric}_pvalue"] = float(pval)
    return out


# ---------------------------------------------------------------------------
# Per-scene plot
# ---------------------------------------------------------------------------

def plot_per_scene(dfs: dict[str, pd.DataFrame], out_path: Path):
    """Bar chart of per-scene mean AbsRel for CNN baseline, Marigold, and bicubic."""
    # Collect scenes present in any df
    all_scenes = sorted(
        set().union(*[set(df["scene"].unique()) for df in dfs.values()])
    )

    colors = {
        "CNN Baseline": "#2196F3",
        "Marigold-DC": "#FF5722",
        "Bicubic": "#9E9E9E",
    }

    scene_data: dict[str, dict[str, float]] = {s: {} for s in all_scenes}
    for label, df in dfs.items():
        per_scene = df.groupby("scene")["absrel"].mean()
        for scene in all_scenes:
            if scene in per_scene.index:
                scene_data[scene][label] = float(per_scene[scene])

    # Sort scenes by CNN baseline mean AbsRel (ascending)
    sorted_scenes = sorted(
        all_scenes,
        key=lambda s: scene_data[s].get("CNN Baseline", float("inf")),
    )

    labels_present = list(dfs.keys())
    n_labels = len(labels_present)
    n_scenes = len(sorted_scenes)
    x = np.arange(n_scenes)
    bar_w = 0.8 / max(n_labels, 1)

    fig, ax = plt.subplots(figsize=(max(14, n_scenes * 0.35), 5))

    for i, label in enumerate(labels_present):
        heights = [scene_data[s].get(label, np.nan) for s in sorted_scenes]
        offset = (i - n_labels / 2 + 0.5) * bar_w
        ax.bar(
            x + offset,
            heights,
            width=bar_w * 0.9,
            label=label,
            color=colors.get(label, f"C{i}"),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_scenes, rotation=90, fontsize=6)
    ax.set_ylabel("Mean AbsRel ↓")
    ax.set_title("Per-scene AbsRel — CNN Baseline vs Marigold-DC vs Bicubic")
    ax.legend()
    ax.grid(axis="y", linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved per-scene plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_stats(
    cnn_csv: Path,
    marigold_csv: Optional[Path],
    bicubic_csv: Optional[Path],
    out_dir: Path,
    all_cnn_csvs: Optional[list[Path]] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cnn_df = load_csv(cnn_csv)
    if cnn_df is None:
        raise FileNotFoundError(f"CNN baseline CSV not found: {cnn_csv}")

    marigold_df = load_csv(marigold_csv) if marigold_csv else None
    bicubic_df = load_csv(bicubic_csv) if bicubic_csv else None

    print(f"CNN baseline: {len(cnn_df)} samples")
    if marigold_df is not None:
        print(f"Marigold-DC:  {len(marigold_df)} samples")
    if bicubic_df is not None:
        print(f"Bicubic:      {len(bicubic_df)} samples")

    # --- Bootstrap CI per model ---
    bootstrap_results = [model_stats(cnn_df, "CNN Baseline")]
    if marigold_df is not None:
        bootstrap_results.append(model_stats(marigold_df, "Marigold-DC"))
    if bicubic_df is not None:
        bootstrap_results.append(model_stats(bicubic_df, "Bicubic"))

    if all_cnn_csvs:
        for p in all_cnn_csvs:
            df = load_csv(p)
            if df is not None:
                model_name = p.stem.replace("cnn_", "").replace("_results", "")
                bootstrap_results.append(model_stats(df, model_name))

    print("\n--- Bootstrap 95% CI (AbsRel) ---")
    for r in bootstrap_results:
        print(
            f"  {r['model']:<30} absrel={r['absrel']:.4f} "
            f"[{r['absrel_ci95_low']:.4f}, {r['absrel_ci95_high']:.4f}]  "
            f"n={r['n_samples']}"
        )

    # --- Paired Wilcoxon: CNN vs Marigold ---
    wilcoxon_cnn_vs_marigold = None
    if marigold_df is not None:
        wilcoxon_cnn_vs_marigold = wilcoxon_test(cnn_df, marigold_df)
        n_overlap = wilcoxon_cnn_vs_marigold["n_overlapping"]
        print(f"\n--- Paired Wilcoxon (CNN baseline vs Marigold-DC, n={n_overlap} overlapping) ---")
        for metric in ["absrel", "rmse"]:
            k_stat = f"wilcoxon_{metric}_stat"
            k_pval = f"wilcoxon_{metric}_pvalue"
            if k_pval in wilcoxon_cnn_vs_marigold:
                print(
                    f"  {metric}: stat={wilcoxon_cnn_vs_marigold[k_stat]:.1f}, "
                    f"p={wilcoxon_cnn_vs_marigold[k_pval]:.4e}"
                )

    # --- Wilcoxon: CNN vs Bicubic ---
    wilcoxon_cnn_vs_bicubic = None
    if bicubic_df is not None:
        wilcoxon_cnn_vs_bicubic = wilcoxon_test(cnn_df, bicubic_df)
        n_overlap = wilcoxon_cnn_vs_bicubic["n_overlapping"]
        print(f"\n--- Paired Wilcoxon (CNN baseline vs Bicubic, n={n_overlap} overlapping) ---")
        for metric in ["absrel", "rmse"]:
            k_stat = f"wilcoxon_{metric}_stat"
            k_pval = f"wilcoxon_{metric}_pvalue"
            if k_pval in wilcoxon_cnn_vs_bicubic:
                print(
                    f"  {metric}: stat={wilcoxon_cnn_vs_bicubic[k_stat]:.1f}, "
                    f"p={wilcoxon_cnn_vs_bicubic[k_pval]:.4e}"
                )

    # --- Per-scene plot ---
    plot_dfs: dict[str, pd.DataFrame] = {"CNN Baseline": cnn_df}
    if marigold_df is not None:
        plot_dfs["Marigold-DC"] = marigold_df
    if bicubic_df is not None:
        plot_dfs["Bicubic"] = bicubic_df

    plot_per_scene(plot_dfs, plots_dir / "per_scene_absrel.png")

    # --- Save JSON ---
    output = {
        "bootstrap_ci": bootstrap_results,
        "wilcoxon_cnn_vs_marigold": wilcoxon_cnn_vs_marigold,
        "wilcoxon_cnn_vs_bicubic": wilcoxon_cnn_vs_bicubic,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "ci_alpha": CI_ALPHA,
    }
    out_json = out_dir / "stats_analysis.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved stats → {out_json}")

    return output


def parse_args():
    p = argparse.ArgumentParser(description="Bootstrap CI + Wilcoxon tests for depth completion.")
    p.add_argument("--cnn-csv", type=Path, default=Path("eval/cnn_baseline_results.csv"))
    p.add_argument("--marigold-csv", type=Path, default=None)
    p.add_argument("--bicubic-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument(
        "--all-cnn-csvs",
        type=Path,
        nargs="*",
        default=None,
        help="Additional CNN ablation CSVs for bootstrap CI (optional)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stats(
        cnn_csv=args.cnn_csv,
        marigold_csv=args.marigold_csv,
        bicubic_csv=args.bicubic_csv,
        out_dir=args.out_dir,
        all_cnn_csvs=args.all_cnn_csvs,
    )
