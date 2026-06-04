"""
Modal entrypoint for depth completion experiments on A100.

Storage strategy: each ARKitScenes scene is stored as a single .tar file on the
Modal Volume (1 inode/scene vs ~31K inodes for individual PNGs). 80 scenes = 85
inodes total — well within Modal Volume's 500K inode limit. At training start,
tars are extracted to the container's ephemeral SSD (512 GB, free with A100).

Run these in order:

  # 1. One-time: download 80 scenes as tars (~15-20 min)
  modal run modal_run.py::download_data

  # 2. One-time: build train/val/test splits
  modal run modal_run.py::build_splits

  # 3. Smoke test — fast end-to-end check (~5 min, ~$0.30)
  modal run modal_run.py::run_experiments --smoke-test

  # 4. Full run (~4-5 hrs, ~$18-22)
  modal run modal_run.py::run_experiments

  # 5. Download results locally
  modal volume get depth-completion-results / ./modal_results/

Check download progress (safely resumable — skips existing tars):
  modal volume ls arkitscenes-data /scenes/Training/
  modal volume ls arkitscenes-data /scenes/Validation/

Re-run a single config:
  modal run modal_run.py::run_experiments --configs baseline
"""

import modal
from pathlib import Path

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "curl", "unzip", "tar", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy", "pandas", "pillow", "pyyaml", "matplotlib", "scipy",
        "opencv-python-headless",
        "diffusers>=0.31.0,<0.39.0", "transformers>=4.40.0,<5.0.0", "accelerate",
    )
    .add_local_dir(
        ".",
        remote_path="/repo",
        copy=True,
        ignore=[".git", "__pycache__", "*.pyc", ".vscode", "modal_results"],
    )
)

app = modal.App("dc-ablation-run", image=image)

# ---------------------------------------------------------------------------
# Volumes
# Inode math (verified):
#   80 tar files + 5 misc files = 85 inodes (0.017% of 500K limit)
#   80 GB data = 0.78% of 10 TB volume max
# ---------------------------------------------------------------------------
dataset_volume = modal.Volume.from_name("arkitscenes-data", create_if_missing=True)
results_volume = modal.Volume.from_name("depth-completion-results", create_if_missing=True)

# Paths inside containers
DATASET_DIR  = "/data/arkitscenes"        # volume mount point
SCENES_DIR   = "/data/arkitscenes/scenes" # tar files live here: scenes/{split}/{vid}.tar
META_PATH    = "/data/arkitscenes/metadata.csv"
LOCAL_DATA   = "/local/arkitscenes"       # ephemeral SSD: tars extracted here for training
RESULTS_DIR  = "/results"
REPO_DIR     = "/repo"

BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
ZIP_ASSETS = ["lowres_wide", "lowres_wide_intrinsics", "confidence", "highres_depth", "lowres_depth"]

CNN_CONFIGS = [
    "baseline",
    "ablation_filter_1x1",
    "ablation_filter_5x5",
    "ablation_levels_2",
    "ablation_levels_3",
    "ablation_channels_narrow",
    "ablation_channels_wide",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _run(cmd: list, cwd: str) -> int:
    import subprocess
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, cwd=cwd).returncode


def _patch_configs(repo_dir: str, data_root: str):
    import yaml
    for cfg_path in Path(f"{repo_dir}/configs").glob("*.yaml"):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg["dataset"]["root"] = data_root
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)


def _sync_to_results(src_dir: str, results_dir: str, names: list):
    import shutil
    for name in names:
        src = Path(f"{src_dir}/{name}")
        if not src.exists():
            continue
        dst = Path(f"{results_dir}/{name}")
        if src.is_dir():
            shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))


# ---------------------------------------------------------------------------
# Step 0: Purge old PNG-based data (run once if migrating from old approach)
# ---------------------------------------------------------------------------
@app.function(
    gpu=None, cpu=2, memory=4096, timeout=600,
    volumes={DATASET_DIR: dataset_volume},
)
def purge_volume():
    """Delete everything except tar files and metadata — cleans up old PNG-based data."""
    import shutil

    deleted = 0
    dataset_path = Path(DATASET_DIR)
    for item in dataset_path.rglob("*"):
        if item.is_dir() and item.name in ("raw", "Training", "Validation"):
            print(f"Removing old directory: {item}")
            shutil.rmtree(str(item))
            deleted += 1

    dataset_volume.commit()
    print(f"Purged {deleted} old items. Volume is clean.")


# ---------------------------------------------------------------------------
# Step 0b: Purge Training/ tars only — fixes data leakage from rebalance_scenes
# ---------------------------------------------------------------------------
@app.function(gpu=None, cpu=2, memory=4096, timeout=300,
              volumes={DATASET_DIR: dataset_volume})
def purge_training():
    """Remove only Training/ tars — fixes data leakage from rebalance step."""
    import shutil

    train_dir = Path(f"{SCENES_DIR}/Training")
    if train_dir.exists():
        shutil.rmtree(str(train_dir))
        dataset_volume.commit()
        print("Training/ purged. Validation/ untouched.")
    else:
        print("Training/ does not exist — nothing to purge.")


# ---------------------------------------------------------------------------
# Step 0c: Rebalance scenes to 70 Training / 10 Validation
# Run once after download_data completes to improve train/test ratio.
# ---------------------------------------------------------------------------
@app.function(
    gpu=None, cpu=2, memory=4096, timeout=300,
    volumes={DATASET_DIR: dataset_volume},
)
def rebalance_scenes(keep_val: int = 10):
    """Move excess Validation tars into Training so we get ~70 train / 10 test.

    After download_data runs (40 Training + 40 Validation), this moves
    (40 - keep_val) Validation tars → Training directory.
    Result: ~70 Training tars, keep_val Validation tars.

    Args:
        keep_val: How many Validation tars to keep (default 10).
    """
    import shutil

    train_dir = Path(f"{SCENES_DIR}/Training")
    val_dir   = Path(f"{SCENES_DIR}/Validation")

    val_tars = sorted(val_dir.glob("*.tar"))
    to_move  = val_tars[keep_val:]   # keep first keep_val in place, move the rest

    print(f"Validation tars found: {len(val_tars)}")
    print(f"Keeping {keep_val} in Validation, moving {len(to_move)} → Training")

    for tar in to_move:
        dst = train_dir / tar.name
        shutil.move(str(tar), str(dst))
        print(f"  Moved {tar.name} → Training/")

    dataset_volume.commit()

    final_train = len(list(train_dir.glob("*.tar")))
    final_val   = len(list(val_dir.glob("*.tar")))
    print(f"\nDone. Training: {final_train} tars, Validation: {final_val} tars")


# ---------------------------------------------------------------------------
# Step 1: Download scenes as tar files (~15-20 min for 80 scenes)
# ---------------------------------------------------------------------------
@app.function(
    gpu=None,
    cpu=4,
    memory=8192,
    timeout=21600,
    volumes={DATASET_DIR: dataset_volume},
)
def download_data(num_scenes: int = 40, num_train: int = -1, num_val: int = -1):
    """Download ARKitScenes depth-upsampling scenes, storing each as a single tar.

    Storage approach:
      - Zips downloaded to /tmp (ephemeral, not on volume) one at a time
      - Extracted to /tmp, renamed (lowres_wide → wide), packed into one .tar
      - Tar written to volume: scenes/{split}/{video_id}.tar
      - /tmp cleaned after each scene
    This keeps inodes at 1/scene instead of ~31K/scene.

    Progress is printed after EVERY scene. Already-downloaded tars are skipped,
    so this function is safely resumable.

    Args:
        num_scenes: Default scenes per split when num_train/num_val not set.
        num_train:  Override scene count for Training split (-1 = use num_scenes).
        num_val:    Override scene count for Validation split (-1 = use num_scenes).
    """
    import subprocess, csv as csv_mod, shutil, os, time

    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/apple/ARKitScenes.git", "/tmp/ARKitScenes"],
        check=True,
    )

    full_csv = "/tmp/ARKitScenes/depth_upsampling/upsampling_train_val_splits.csv"
    with open(full_csv) as f:
        all_rows = list(csv_mod.DictReader(f))

    # Download metadata.csv once (needed by build_splits and dataset loader)
    Path(SCENES_DIR).mkdir(parents=True, exist_ok=True)
    if not Path(META_PATH).exists():
        print("Downloading metadata.csv ...")
        subprocess.run(
            ["curl", "-fsSL", f"{BASE_URL}/metadata.csv", "-o", META_PATH],
            check=True,
        )
        dataset_volume.commit()
        print("metadata.csv saved.")

    overall_start = time.perf_counter()
    total_done = 0
    total_skipped = 0

    split_counts = {
        "Training":   num_train if num_train >= 0 else num_scenes,
        "Validation": num_val   if num_val   >= 0 else num_scenes,
    }

    for split in ["Training", "Validation"]:
        rows = [r for r in all_rows if r["fold"] == split]
        n = split_counts[split]

        tar_dir = Path(f"{SCENES_DIR}/{split}")
        tar_dir.mkdir(parents=True, exist_ok=True)

        done = 0       # newly downloaded this run
        skipped = 0    # already existed, counted as success
        successes = 0  # total valid tars (existing + new) — stop at n

        print(f"\n{'='*60}")
        print(f"{split}: need {n} valid scenes (will scan up to {len(rows)} available)")
        print(f"{'='*60}")

        for i, row in enumerate(rows):
            if successes >= n:
                break

            vid = row["video_id"]
            tar_path = tar_dir / f"{vid}.tar"

            # --- Already downloaded — counts as a success ---
            if tar_path.exists():
                skipped += 1
                successes += 1
                print(f"  [ok={successes}/{n}] {vid} — SKIP (tar exists)")
                continue

            scene_start = time.perf_counter()
            tmp_scene = Path(f"/tmp/scene_{vid}")
            tmp_scene.mkdir(parents=True, exist_ok=True)

            # --- Download each zip to /tmp, extract, delete zip ---
            assets_ok = []
            for asset in ZIP_ASSETS:
                tmp_zip = tmp_scene / f"{asset}.zip"
                url = f"{BASE_URL}/{split}/{vid}/{asset}.zip"
                r = subprocess.run(
                    ["curl", "-fsSL", "--retry", "3", "--retry-delay", "2",
                     url, "-o", str(tmp_zip)],
                    capture_output=True,
                )
                if r.returncode != 0:
                    tmp_zip.unlink(missing_ok=True)
                    continue
                subprocess.run(
                    ["unzip", "-oq", str(tmp_zip), "-d", str(tmp_scene)],
                    capture_output=True,
                )
                tmp_zip.unlink(missing_ok=True)
                assets_ok.append(asset)

            # trajectory file (plain text)
            traj = tmp_scene / "lowres_wide.traj"
            subprocess.run(
                ["curl", "-fsSL", "--retry", "3",
                 f"{BASE_URL}/{split}/{vid}/lowres_wide.traj", "-o", str(traj)],
                capture_output=True,
            )

            # --- Validate: must have highres_depth (download failure, not missing data) ---
            if not (tmp_scene / "highres_depth").exists():
                print(f"  [ok={successes}/{n}] {vid} — INVALID (no highres_depth, assets={assets_ok}), trying next scene")
                shutil.rmtree(str(tmp_scene))
                continue  # does NOT increment successes — keep going

            # --- Rename lowres_wide → wide ---
            for src_n, dst_n in [("lowres_wide", "wide"), ("lowres_wide_intrinsics", "wide_intrinsics")]:
                src = tmp_scene / src_n
                dst = tmp_scene / dst_n
                if src.exists():
                    os.rename(str(src), str(dst))

            # --- Pack into a single tar ---
            tmp_wrapper = Path(f"/tmp/wrap_{vid}")
            tmp_wrapper.mkdir(parents=True, exist_ok=True)
            os.rename(str(tmp_scene), str(tmp_wrapper / vid))

            tmp_tar = Path(f"/tmp/{vid}.tar")
            subprocess.run(
                ["tar", "-cf", str(tmp_tar), "-C", str(tmp_wrapper), vid],
                check=True,
            )
            shutil.rmtree(str(tmp_wrapper))

            # --- Write tar to volume ---
            shutil.copy2(str(tmp_tar), str(tar_path))
            tmp_tar.unlink()

            elapsed = time.perf_counter() - scene_start
            done += 1
            total_done += 1
            successes += 1

            # Commit every 5 scenes so progress survives a container crash
            if done % 5 == 0:
                dataset_volume.commit()

            wall = time.perf_counter() - overall_start
            print(
                f"  [ok={successes}/{n}] {vid} — OK "
                f"(assets={assets_ok}, {elapsed:.0f}s/scene, wall={wall/60:.1f}m)"
            )

        dataset_volume.commit()
        if successes < n:
            print(f"\n[WARN] {split}: only {successes}/{n} valid scenes found after scanning all {len(rows)} rows.")
        else:
            print(f"\n{split} summary: {done} downloaded, {skipped} already existed. Total: {successes}/{n} ✓")
        total_skipped += skipped

    dataset_volume.commit()
    wall = time.perf_counter() - overall_start
    print(f"\nDone. {total_done} new scenes, {total_skipped} skipped. Wall time: {wall/60:.1f} min")
    print("Check progress: modal volume ls arkitscenes-data /scenes/Training/")
    print("Check progress: modal volume ls arkitscenes-data /scenes/Validation/")


# ---------------------------------------------------------------------------
# Step 2: Build splits
# ---------------------------------------------------------------------------
@app.function(
    gpu=None, cpu=2, memory=8192, timeout=1800,
    ephemeral_disk=512 * 1024,  # 512 GB SSD — extract tars to build splits
    volumes={
        DATASET_DIR: dataset_volume,
        RESULTS_DIR: results_volume,
    },
)
def build_splits(require_train: int = 300, require_val: int = 44):
    """Extract all tars to ephemeral SSD, then build train/val/test splits.

    Aborts early if the expected number of scene tars are not yet on the volume,
    so this function is safe to call only after download_data completes.
    """
    import sys, shutil, subprocess

    train_tars = list(Path(f"{SCENES_DIR}/Training").glob("*.tar")) if Path(f"{SCENES_DIR}/Training").exists() else []
    val_tars   = list(Path(f"{SCENES_DIR}/Validation").glob("*.tar")) if Path(f"{SCENES_DIR}/Validation").exists() else []
    print(f"Volume check: {len(train_tars)} Training tars, {len(val_tars)} Validation tars")
    if len(train_tars) < require_train or len(val_tars) < require_val:
        raise RuntimeError(
            f"Download incomplete: need {require_train} Training + {require_val} Validation tars, "
            f"got {len(train_tars)} + {len(val_tars)}. Run download_data first."
        )
    print("Download verified. Proceeding to extract and build splits.")

    _extract_tars_to_local()

    splits_out = f"{REPO_DIR}/splits"
    rc = _run(
        [sys.executable, "-m", "scripts.build_arkitscenes_splits",
         "--data-root", LOCAL_DATA,
         "--output-dir", splits_out],
        cwd=REPO_DIR,
    )
    if rc != 0:
        raise RuntimeError("build_splits failed — check logs above.")

    shutil.copytree(splits_out, f"{RESULTS_DIR}/splits", dirs_exist_ok=True)
    results_volume.commit()
    print("Splits saved to results volume.")
    for p in Path(splits_out).glob("*.txt"):
        lines = [l for l in open(p).read().splitlines() if l.strip()]
        print(f"  {p.name}: {len(lines)} samples")


def _extract_tars_to_local():
    """Extract all .tar files from the volume to the local ephemeral SSD."""
    import subprocess, time

    scenes_path = Path(SCENES_DIR)
    local_path = Path(LOCAL_DATA)
    local_path.mkdir(parents=True, exist_ok=True)

    # Also copy metadata.csv
    import shutil
    if Path(META_PATH).exists():
        shutil.copy2(META_PATH, str(local_path / "metadata.csv"))

    total = 0
    start = time.perf_counter()
    for split in ("Training", "Validation"):
        split_tar_dir = scenes_path / split
        if not split_tar_dir.exists():
            print(f"  No tars found for {split}")
            continue
        local_split = local_path / split
        local_split.mkdir(exist_ok=True)
        tars = sorted(split_tar_dir.glob("*.tar"))
        print(f"  Extracting {len(tars)} {split} tars to local SSD ...")
        for tar in tars:
            subprocess.run(
                ["tar", "-xf", str(tar), "-C", str(local_split)],
                check=True, capture_output=True,
            )
            total += 1
        print(f"  {split}: {len(tars)} scenes extracted.")

    elapsed = time.perf_counter() - start
    print(f"Extraction complete: {total} scenes in {elapsed:.0f}s.")


# ---------------------------------------------------------------------------
# Step 3: Run experiments
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100-40GB",
    cpu=12,
    memory=65536,
    ephemeral_disk=512 * 1024,  # 512 GB SSD — free with A100 hourly rate
    timeout=12 * 3600,
    volumes={
        DATASET_DIR: dataset_volume,
        RESULTS_DIR: results_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_experiments(
    smoke_test: bool = False,
    skip_marigold: bool = False,
    configs: str = "",
):
    """Run the full experiment suite on an A100.

    Startup: extracts all scene tars from volume to local ephemeral SSD (~2-3 min).
    Then trains all 7 CNN configs and evaluates each. Results saved to results
    volume after EVERY config — a crash loses at most one config's work.

    Args:
        smoke_test: 10 steps/epoch, 5 eval samples. Verifies pipeline (~5 min).
        skip_marigold: Skip Marigold-DC baseline evaluation.
        configs: Comma-separated config names to run. Empty = all 7.
    """
    import sys, shutil, json, time, subprocess
    import pandas as pd
    from datetime import datetime

    # Expand /dev/shm for PyTorch multi-worker DataLoader IPC.
    # Modal containers default to 64 MB; 8 workers × ~170 MB/batch = ~1.4 GB needed.
    r = subprocess.run(
        ["mount", "-t", "tmpfs", "-o", "size=8g", "tmpfs", "/dev/shm"],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        print("Remounted /dev/shm as 8 GB tmpfs")
    else:
        print(f"[WARN] /dev/shm remount failed (may be OK): {r.stderr.strip()}")

    # --- Extract dataset tars to local SSD ---
    print("Extracting dataset to local SSD ...")
    _extract_tars_to_local()

    # --- Copy repo to writable workspace ---
    subprocess.run(["cp", "-r", REPO_DIR, "/workspace"], check=True)
    work_dir = "/workspace"

    # --- Load splits from results volume ---
    splits_src = Path(f"{RESULTS_DIR}/splits")
    splits_dst = Path(f"{work_dir}/splits")
    if splits_src.exists():
        shutil.copytree(str(splits_src), str(splits_dst), dirs_exist_ok=True)
        for p in splits_dst.glob("*.txt"):
            lines = [l for l in open(p).read().splitlines() if l.strip()]
            print(f"Loaded {p.name}: {len(lines)} samples")
    else:
        raise RuntimeError("No splits found. Run build_splits first.")

    # --- Load prior results for resumability ---
    prev_output = Path(f"{RESULTS_DIR}/output.json")
    completed_runs: dict = {}
    marigold_summary = None
    if prev_output.exists():
        with open(prev_output) as f:
            prev = json.load(f)
        completed_runs = {
            k: v.get("eval") or {}
            for k, v in prev.get("cnn_runs", {}).items()
            if (v.get("metrics") or {}).get("rmse") is not None
        }
        m_prev = prev.get("marigold_dc") or {}
        m_prev_metrics = m_prev.get("metrics") or {}
        if m_prev_metrics.get("rmse") is not None:
            marigold_summary = m_prev.get("eval")
        if completed_runs:
            print(f"Resuming — already done: {list(completed_runs.keys())}")

    _patch_configs(work_dir, LOCAL_DATA)

    configs_to_run = [c.strip() for c in configs.split(",")] if configs else CNN_CONFIGS
    smoke_flags      = ["--max-steps", "10"] if smoke_test else []
    smoke_eval_flags = ["--max-samples", "5"] if smoke_test else []
    eval_dir = Path(f"{work_dir}/eval")
    eval_dir.mkdir(exist_ok=True)
    total_start = time.perf_counter()

    # --- Marigold-DC baseline ---
    if not skip_marigold and marigold_summary is None:
        print("\n" + "=" * 60)
        print("STEP 1: Marigold-DC evaluation")
        print("=" * 60)
        marigold_max = ["--max-samples", "5"] if smoke_test else ["--max-samples", "500"]
        rc = _run(
            [sys.executable, "-m", "scripts.run_marigold_dc",
             "--split-path", f"{work_dir}/splits/test.txt",
             "--output-csv", str(eval_dir / "marigold_dc_test_results.csv"),
             "--data-root", LOCAL_DATA,
             *marigold_max],
            cwd=work_dir,
        )
        if rc == 0:
            sp = eval_dir / "marigold_dc_test_results.summary.json"
            if sp.exists():
                with open(sp) as f:
                    marigold_summary = json.load(f)
                print(f"Marigold AbsRel: {marigold_summary.get('absrel', 'N/A'):.4f}")
            _sync_to_results(work_dir, RESULTS_DIR, ["eval"])
            results_volume.commit()
        else:
            print("[WARN] Marigold eval failed — continuing.")
    elif skip_marigold:
        print("Skipping Marigold-DC (--skip-marigold).")
    else:
        print("Marigold-DC already evaluated — skipping.")

    # --- Train + eval each CNN config ---
    for cfg_name in configs_to_run:
        if cfg_name in completed_runs:
            print(f"\nSkipping {cfg_name} — already completed.")
            continue

        cfg_path = Path(f"{work_dir}/configs/{cfg_name}.yaml")
        if not cfg_path.exists():
            print(f"[WARN] Config not found: {cfg_path}")
            continue

        print("\n" + "=" * 60)
        print(f"TRAIN: {cfg_name}")
        print("=" * 60)
        rc = _run(
            [sys.executable, "-m", "scripts.train",
             "--config", str(cfg_path),
             "--run-name", cfg_name,
             *smoke_flags],
            cwd=work_dir,
        )
        if rc != 0:
            print(f"[WARN] Training failed for {cfg_name} — skipping eval.")
            _sync_to_results(work_dir, RESULTS_DIR, ["checkpoints"])
            results_volume.commit()
            continue

        print("\n" + "=" * 60)
        print(f"EVAL: {cfg_name}")
        print("=" * 60)
        ckpt = Path(f"{work_dir}/checkpoints/{cfg_name}/best.pth")
        output_csv = eval_dir / f"cnn_{cfg_name}_results.csv"
        rc = _run(
            [sys.executable, "-m", "scripts.eval_cnn",
             "--checkpoint", str(ckpt),
             "--output-csv", str(output_csv),
             "--split-path", f"{work_dir}/splits/test.txt",
             "--data-root", LOCAL_DATA,
             *smoke_eval_flags],
            cwd=work_dir,
        )
        sp = output_csv.with_suffix(".summary.json")
        if rc == 0 and sp.exists():
            with open(sp) as f:
                completed_runs[cfg_name] = json.load(f)
            m = completed_runs[cfg_name]
            print(f"  absrel={m.get('absrel',float('nan')):.4f}  rmse={m.get('rmse',float('nan')):.4f}  d1={m.get('d1',float('nan')):.4f}")

        _sync_to_results(work_dir, RESULTS_DIR, ["checkpoints", "eval"])
        results_volume.commit()
        print(f"Results for {cfg_name} saved to volume.")

    # --- Plots ---
    print("\n" + "=" * 60)
    print("STEP 4: Generating plots")
    print("=" * 60)
    _run(
        [sys.executable, "-m", "scripts.plot_results",
         "--run-names", *configs_to_run,
         "--ckpt-root", "checkpoints",
         "--eval-dir", "eval",
         "--out-dir", "results/plots"],
        cwd=work_dir,
    )

    # --- output.json ---
    print("\n" + "=" * 60)
    print("STEP 5: Writing output.json")
    print("=" * 60)

    def _metrics(s):
        return {k: s.get(k) for k in ["rmse", "mae", "absrel", "runtime_mean_ms", "total_params"]}

    run_details = {}
    for cfg_name in configs_to_run:
        log = Path(f"{work_dir}/checkpoints/{cfg_name}/train_log.csv")
        curves = {}
        if log.exists():
            df = pd.read_csv(log)
            curves = {c: df[c].tolist() for c in df.columns}
        s = completed_runs.get(cfg_name, {})
        run_details[cfg_name] = {
            "config": cfg_name,
            "checkpoint": f"checkpoints/{cfg_name}/best.pth",
            "metrics": _metrics(s),
            "training_curves": curves,
            "eval": s,
        }

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "smoke_test": smoke_test,
        "total_wall_time_s": time.perf_counter() - total_start,
        "marigold_dc": {
            "metrics": _metrics(marigold_summary) if marigold_summary else None,
            "eval": marigold_summary,
        },
        "cnn_runs": run_details,
    }
    out_path = Path(f"{work_dir}/output.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    _sync_to_results(work_dir, RESULTS_DIR, ["eval", "checkpoints", "results", "output.json"])
    results_volume.commit()

    wall = time.perf_counter() - total_start
    print(f"\nAll done in {wall/3600:.2f} hours.")
    print("Download results: modal volume get depth-completion-results / ./modal_results/")

    print("\n--- Final Test Metrics ---")
    hdr = f"{'Model':<35} {'AbsRel':>8} {'RMSE':>8} {'d1':>8}"
    print(hdr)
    print("-" * len(hdr))
    if marigold_summary:
        print(f"{'Marigold-DC':<35} {marigold_summary.get('absrel',float('nan')):>8.4f} {marigold_summary.get('rmse',float('nan')):>8.4f} {marigold_summary.get('d1',float('nan')):>8.4f}")
    for name, s in completed_runs.items():
        print(f"{name:<35} {s.get('absrel',float('nan')):>8.4f} {s.get('rmse',float('nan')):>8.4f} {s.get('d1',float('nan')):>8.4f}")


# ---------------------------------------------------------------------------
# Step 4a: Bicubic upsampling baseline evaluation
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100-40GB",
    cpu=8,
    memory=32768,
    ephemeral_disk=512 * 1024,
    timeout=4 * 3600,
    volumes={
        DATASET_DIR: dataset_volume,
        RESULTS_DIR: results_volume,
    },
)
def run_bicubic_eval(smoke_test: bool = False):
    """Evaluate bicubic depth upsampling baseline on the test split.

    Saves eval/bicubic_results.csv + eval/bicubic_results.summary.json to the
    results volume in the same format as eval_cnn.py for direct comparison.
    """
    import sys, shutil, subprocess

    _extract_tars_to_local()
    subprocess.run(["cp", "-r", REPO_DIR, "/workspace"], check=True)
    work_dir = "/workspace"

    splits_src = Path(f"{RESULTS_DIR}/splits")
    splits_dst = Path(f"{work_dir}/splits")
    if splits_src.exists():
        shutil.copytree(str(splits_src), str(splits_dst), dirs_exist_ok=True)
    else:
        raise RuntimeError("No splits found. Run build_splits first.")

    smoke_flags = ["--max-samples", "20"] if smoke_test else []
    rc = _run(
        [sys.executable, "-m", "scripts.eval_bicubic",
         "--split-path", f"{work_dir}/splits/test.txt",
         "--output-csv", f"{work_dir}/eval/bicubic_results.csv",
         "--data-root", LOCAL_DATA,
         *smoke_flags],
        cwd=work_dir,
    )
    if rc != 0:
        raise RuntimeError("Bicubic eval failed — check logs above.")

    _sync_to_results(work_dir, RESULTS_DIR, ["eval"])
    results_volume.commit()
    print("Bicubic eval saved to volume.")


# ---------------------------------------------------------------------------
# Step 4b: Sample visualization grids
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100-40GB",
    cpu=8,
    memory=32768,
    ephemeral_disk=512 * 1024,
    timeout=2 * 3600,
    volumes={
        DATASET_DIR: dataset_volume,
        RESULTS_DIR: results_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_sample_viz(n_samples: int = 8, no_marigold: bool = False):
    """Generate side-by-side depth sample grids (RGB / Sparse / Bicubic / CNN / Marigold / GT).

    Requires baseline checkpoint to exist in the results volume.
    Saves PNG grids to results/sample_viz/ on the results volume.
    """
    import sys, shutil, subprocess

    _extract_tars_to_local()
    subprocess.run(["cp", "-r", REPO_DIR, "/workspace"], check=True)
    work_dir = "/workspace"

    splits_src = Path(f"{RESULTS_DIR}/splits")
    splits_dst = Path(f"{work_dir}/splits")
    if splits_src.exists():
        shutil.copytree(str(splits_src), str(splits_dst), dirs_exist_ok=True)
    else:
        raise RuntimeError("No splits found.")

    # Copy checkpoints from results volume
    ckpt_src = Path(f"{RESULTS_DIR}/checkpoints")
    ckpt_dst = Path(f"{work_dir}/checkpoints")
    if ckpt_src.exists():
        shutil.copytree(str(ckpt_src), str(ckpt_dst), dirs_exist_ok=True)
    else:
        raise RuntimeError("No checkpoints found on results volume. Run run_experiments first.")

    baseline_ckpt = Path(f"{work_dir}/checkpoints/baseline/best.pth")
    if not baseline_ckpt.exists():
        raise RuntimeError(f"Baseline checkpoint not found: {baseline_ckpt}")

    no_marigold_flags = ["--no-marigold"] if no_marigold else []
    rc = _run(
        [sys.executable, "-m", "scripts.visualize_samples",
         "--checkpoint", str(baseline_ckpt),
         "--split-path", f"{work_dir}/splits/test.txt",
         "--data-root", LOCAL_DATA,
         "--out-dir", f"{work_dir}/results/sample_viz",
         "--n-samples", str(n_samples),
         *no_marigold_flags],
        cwd=work_dir,
    )
    if rc != 0:
        raise RuntimeError("Sample visualization failed — check logs above.")

    _sync_to_results(work_dir, RESULTS_DIR, ["results"])
    results_volume.commit()
    print(f"Sample viz ({n_samples} grids) saved to volume at results/sample_viz/")
