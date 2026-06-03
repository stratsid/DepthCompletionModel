# VM Agent Instructions

You are running on a VM to execute depth completion experiments. Follow these steps in order. Do not skip steps. If any step fails, stop and report the error clearly.

---

## Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2: Download ARKitScenes (4 modalities only)

Clone the official ARKitScenes repo and download only the modalities this project needs. Do NOT download meshes or 3D annotations — storage is limited.

```bash
git clone https://github.com/apple/ARKitScenes.git /tmp/ARKitScenes
cd /tmp/ARKitScenes
python download_data.py \
  --data_type raw_frames depth_annotations \
  --split Training Validation \
  --output_dir /path/to/DepthCompletionModel/data/arkitscenes
cd /path/to/DepthCompletionModel
```

Replace `/path/to/DepthCompletionModel` with the actual repo path on this VM.

---

## Step 3: Build splits

Run once. Seed=0 ensures train/val/test splits are identical across all ablation runs.

```bash
python -m scripts.build_arkitscenes_splits \
  --data-root data/arkitscenes \
  --output-dir splits/
```

Expected output:
```
train samples: <N>
val samples: <N>
test samples: <N>
```

All three numbers must be greater than 0. If any is 0, the data download in Step 2 did not complete correctly — stop and report.

---

## Step 4: Smoke test

Run a fast end-to-end check before committing to the full multi-hour run. This uses 10 steps per epoch and 5 eval samples.

```bash
python -m scripts.run_all \
  --data-root data/arkitscenes \
  --smoke-test \
  --skip-push
```

Every step must complete without a Python exception. If anything fails, stop and report the full traceback.

---

## Step 5: Full run

Once the smoke test passes, run everything. This will:
1. Evaluate Marigold-DC on the test split
2. Train all 7 CNN configs (baseline + 6 ablations) for 20 epochs each
3. Evaluate each trained model on the test split
4. Generate matplotlib plots of training curves and cross-model comparison
5. Write `output.json` with all metrics
6. Commit and push results to the GitHub repo after each training run

```bash
python -m scripts.run_all --data-root data/arkitscenes
```

Results are pushed incrementally — after each training run completes, a commit goes to `origin/main`. If the VM shuts down mid-run, completed runs are already saved.

---

## What gets pushed back to the repo

- `eval/` — per-sample CSV and summary JSON for each model
- `checkpoints/<run_name>/train_log.csv` — per-epoch loss and val metrics
- `results/plots/` — PNG training curves and comparison charts
- `output.json` — all run summaries, training curves, and the 5 key metrics (RMSE, MAE, AbsRel, Runtime, Params) for every model

---

## If a single config fails and you need to re-run it

```bash
# Re-train one config
python -m scripts.train --config configs/baseline.yaml --run-name baseline

# Re-evaluate one config
python -m scripts.eval_cnn \
  --checkpoint checkpoints/baseline/best.pth \
  --split-path splits/test.txt \
  --data-root data/arkitscenes

# Regenerate plots and output.json after re-runs
python -m scripts.plot_results \
  --run-names baseline ablation_filter_1x1 ablation_filter_5x5 \
              ablation_levels_2 ablation_levels_3 \
              ablation_channels_narrow ablation_channels_wide
```

---

## Available configs

| Config | Ablation | Filter | Levels | Channels |
|---|---|---|---|---|
| `baseline` | — | 3×3 | 4 | (32,64,128,256) |
| `ablation_filter_1x1` | Filter size | 1×1 | 4 | (32,64,128,256) |
| `ablation_filter_5x5` | Filter size | 5×5 | 4 | (32,64,128,256) |
| `ablation_levels_2` | Encoder levels | 3×3 | 2 | (32,64) |
| `ablation_levels_3` | Encoder levels | 3×3 | 3 | (32,64,128) |
| `ablation_channels_narrow` | Channel width | 3×3 | 4 | (16,32,64,128) |
| `ablation_channels_wide` | Channel width | 3×3 | 4 | (64,128,256,512) |
