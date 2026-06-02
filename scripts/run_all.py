"""Master orchestration script: eval Marigold-DC → train all CNNs → eval all → plot → push.

Run on the VM:
    python -m scripts.run_all --data-root /path/to/arkitscenes
    python -m scripts.run_all --data-root /path/to/arkitscenes --skip-marigold
    python -m scripts.run_all --data-root /path/to/arkitscenes --smoke-test  # 10 steps/epoch

After completion, results/, eval/, and checkpoints/ are committed and pushed to origin/main.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Repo root is one level above this script
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CNN_CONFIGS = [
    "baseline",
    "ablation_filter_1x1",
    "ablation_filter_5x5",
    "ablation_levels_2",
    "ablation_levels_3",
    "ablation_channels_narrow",
    "ablation_channels_wide",
]


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
    return result


def git_push(message: str):
    run(["git", "-C", str(REPO_ROOT), "add", "eval/", "results/", "checkpoints/", "output.json"])
    run(["git", "-C", str(REPO_ROOT), "commit", "-m", message, "--allow-empty"])
    run(["git", "-C", str(REPO_ROOT), "push"])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/arkitscenes"))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--skip-marigold", action="store_true", help="Skip Marigold-DC evaluation.")
    p.add_argument("--skip-push", action="store_true", help="Skip git push at the end.")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Limit to 10 steps/epoch and 5 eval samples for fast end-to-end testing.",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=CNN_CONFIGS,
        help="Subset of CNN config names to run.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    start_time = time.perf_counter()

    device_flags = ["--device", args.device] if args.device else []
    smoke_flags = ["--max-steps", "10"] if args.smoke_test else []
    smoke_eval_flags = ["--max-samples", "5"] if args.smoke_test else []
    data_flags = ["--data-root", str(args.data_root)]

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    completed_runs: dict[str, dict] = {}
    marigold_summary: dict | None = None

    # ------------------------------------------------------------------
    # 1. Marigold-DC evaluation
    # ------------------------------------------------------------------
    if not args.skip_marigold:
        print("\n" + "=" * 60)
        print("STEP 1: Marigold-DC evaluation")
        print("=" * 60)
        r = run(
            [sys.executable, "-m", "scripts.run_marigold_dc",
             "--split-path", "splits/test.txt",
             "--output-csv", "eval/marigold_dc_test_results.csv",
             *data_flags,
             *smoke_eval_flags],
            cwd=REPO_ROOT,
        )
        if r.returncode == 0:
            summary_path = REPO_ROOT / "eval" / "marigold_dc_test_results.summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    marigold_summary = json.load(f)
                print(f"Marigold AbsRel: {marigold_summary.get('absrel', 'N/A'):.4f}")
    else:
        print("\nSkipping Marigold-DC evaluation (--skip-marigold).")
        summary_path = REPO_ROOT / "eval" / "marigold_dc_test_results.summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                marigold_summary = json.load(f)

    # ------------------------------------------------------------------
    # 2. Train all CNN configs
    # ------------------------------------------------------------------
    for cfg_name in args.configs:
        print("\n" + "=" * 60)
        print(f"STEP 2: Training {cfg_name}")
        print("=" * 60)
        cfg_path = REPO_ROOT / "configs" / f"{cfg_name}.yaml"
        if not cfg_path.exists():
            print(f"[WARN] Config not found: {cfg_path}, skipping.")
            continue

        r = run(
            [sys.executable, "-m", "scripts.train",
             "--config", str(cfg_path),
             "--run-name", cfg_name,
             *device_flags,
             *smoke_flags],
            cwd=REPO_ROOT,
        )

        if r.returncode != 0:
            print(f"[WARN] Training failed for {cfg_name}, continuing.")
            continue

        # push checkpoint after each run so progress is saved incrementally
        if not args.skip_push:
            git_push(f"results: training complete for {cfg_name}")

    # ------------------------------------------------------------------
    # 3. Evaluate all trained CNN models
    # ------------------------------------------------------------------
    for cfg_name in args.configs:
        ckpt = REPO_ROOT / "checkpoints" / cfg_name / "best.pth"
        if not ckpt.exists():
            print(f"[WARN] No checkpoint found for {cfg_name}, skipping eval.")
            continue

        print("\n" + "=" * 60)
        print(f"STEP 3: Evaluating {cfg_name}")
        print("=" * 60)
        output_csv = REPO_ROOT / "eval" / f"cnn_{cfg_name}_results.csv"
        r = run(
            [sys.executable, "-m", "scripts.eval_cnn",
             "--checkpoint", str(ckpt),
             "--output-csv", str(output_csv),
             "--split-path", "splits/test.txt",
             *data_flags,
             *device_flags,
             *smoke_eval_flags],
            cwd=REPO_ROOT,
        )

        summary_path = output_csv.with_suffix(".summary.json")
        if r.returncode == 0 and summary_path.exists():
            with open(summary_path) as f:
                completed_runs[cfg_name] = json.load(f)
            m = completed_runs[cfg_name]
            print(
                f"  absrel={m.get('absrel', float('nan')):.4f}  "
                f"rmse={m.get('rmse', float('nan')):.4f}  "
                f"d1={m.get('d1', float('nan')):.4f}  "
                f"fps={m.get('fps_mean', float('nan')):.1f}"
            )

    # ------------------------------------------------------------------
    # 4. Generate plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Generating plots")
    print("=" * 60)
    run(
        [sys.executable, "-m", "scripts.plot_results",
         "--run-names", *args.configs,
         "--ckpt-root", "checkpoints",
         "--eval-dir", "eval",
         "--out-dir", "results/plots"],
        cwd=REPO_ROOT,
    )

    # ------------------------------------------------------------------
    # 5. Write output.json
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Writing output.json")
    print("=" * 60)

    run_details = {}
    for cfg_name in args.configs:
        train_log_path = REPO_ROOT / "checkpoints" / cfg_name / "train_log.csv"
        training_curves: dict = {}
        if train_log_path.exists():
            df = pd.read_csv(train_log_path)
            training_curves = {col: df[col].tolist() for col in df.columns}

        run_details[cfg_name] = {
            "config": cfg_name,
            "checkpoint": str(REPO_ROOT / "checkpoints" / cfg_name / "best.pth"),
            "training_curves": training_curves,
            "eval": completed_runs.get(cfg_name, {}),
        }

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "smoke_test": args.smoke_test,
        "total_wall_time_s": time.perf_counter() - start_time,
        "marigold_dc": marigold_summary,
        "cnn_runs": run_details,
    }

    output_path = REPO_ROOT / "output.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Written: {output_path}")

    # ------------------------------------------------------------------
    # 6. Push all results
    # ------------------------------------------------------------------
    if not args.skip_push:
        print("\n" + "=" * 60)
        print("STEP 6: Pushing results to GitHub")
        print("=" * 60)
        git_push("results: all runs complete, plots and output.json added")

    total = time.perf_counter() - start_time
    print(f"\nAll done in {total / 3600:.2f} hours.")

    # Print quick summary table
    print("\n--- Final Test Metrics Summary ---")
    header = f"{'Model':<35} {'AbsRel':>8} {'RMSE':>8} {'d1':>8} {'FPS':>8}"
    print(header)
    print("-" * len(header))
    if marigold_summary:
        print(
            f"{'Marigold-DC':<35} "
            f"{marigold_summary.get('absrel', float('nan')):>8.4f} "
            f"{marigold_summary.get('rmse', float('nan')):>8.4f} "
            f"{marigold_summary.get('d1', float('nan')):>8.4f} "
            f"{marigold_summary.get('fps_mean', float('nan')):>8.2f}"
        )
    for cfg_name, s in completed_runs.items():
        print(
            f"{cfg_name:<35} "
            f"{s.get('absrel', float('nan')):>8.4f} "
            f"{s.get('rmse', float('nan')):>8.4f} "
            f"{s.get('d1', float('nan')):>8.4f} "
            f"{s.get('fps_mean', float('nan')):>8.2f}"
        )


if __name__ == "__main__":
    main()
