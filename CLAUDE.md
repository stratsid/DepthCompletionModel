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

**Before downloading the full dataset**, download 10 scenes first and check storage use:
```bash
du -sh data/arkitscenes
```
Extrapolate: if 10 scenes = ~1 GB, then 400 scenes ≈ 40 GB. Stop downloading if you're on track to exceed 150 GB total.

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

Run a fast end-to-end check before committing to the full run. This uses 10 steps per epoch and 5 eval samples.

```bash
python -m scripts.run_all \
  --data-root data/arkitscenes \
  --smoke-test \
  --skip-push
```

Every step must complete without a Python exception. If anything fails, stop and report the full traceback. **Do not proceed to Step 5 until the smoke test passes cleanly.**

---

## Step 5: Full run

Once the smoke test passes, run everything. This will:
1. Evaluate Marigold-DC on the test split (capped at 500 samples)
2. Train all 7 CNN configs (baseline + 6 ablations) for **10 epochs** each
3. Evaluate each trained model on the test split
4. Generate matplotlib plots of training curves and cross-model comparison
5. Write `output.json` with all metrics (RMSE, MAE, AbsRel, Runtime, Params)
6. Commit and push results to the GitHub repo after each training run

**Expected total runtime on A100: ~3.5–5 hours.**

```bash
python -m scripts.run_all --data-root data/arkitscenes
```

Results are pushed incrementally — after each training run completes, a commit goes to `origin/main`. If the VM shuts down mid-run, completed runs are already saved.

---

## What gets pushed back to the repo

- `eval/` — per-sample CSV and summary JSON for each model
- `checkpoints/<run_name>/train_log.csv` — per-epoch loss and val metrics
- `results/plots/` — PNG training curves and comparison charts
- `output.json` — all run summaries, training curves, and the 5 key metrics for every model

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

---

## Troubleshooting

### Training is very slow (>40ms/step)
**Cause**: Disk I/O is the bottleneck — the DataLoader can't keep up with the GPU.

**Fix 1** — Copy the dataset to RAM disk (fastest, requires ~100 GB free RAM):
```bash
cp -r data/arkitscenes /dev/shm/arkitscenes
python -m scripts.run_all --data-root /dev/shm/arkitscenes
```

**Fix 2** — Increase DataLoader workers in all configs:
```bash
# Edit configs/baseline.yaml (and all ablation configs) and set:
#   num_workers: 12
python -c "
from pathlib import Path, re
for p in Path('configs').glob('*.yaml'):
    p.write_text(p.read_text().replace('num_workers: 8', 'num_workers: 12'))
"
```

**Fix 3** — Check if storage is a slow network volume:
```bash
dd if=/dev/zero of=/tmp/testfile bs=1G count=1 oflag=direct
# If write speed < 500 MB/s, disk is the bottleneck — use Fix 1.
```

---

### CUDA out of memory
**Cause**: Batch size 16 at 768×1024 exceeds GPU VRAM (unlikely on A100 40GB, but possible if other processes are running).

**Fix** — Reduce batch size across all configs:
```bash
python -c "
from pathlib import Path
for p in Path('configs').glob('*.yaml'):
    p.write_text(p.read_text().replace('batch_size: 16', 'batch_size: 8'))
"
```
Also halve the learning rate back to 0.0003 if you drop to batch_size 8:
```bash
python -c "
from pathlib import Path
for p in Path('configs').glob('*.yaml'):
    p.write_text(p.read_text().replace('learning_rate: 0.0006', 'learning_rate: 0.0003'))
"
```

---

### `split files are empty` / `0 train samples`
**Cause**: The data download didn't complete or the directory structure is wrong.

**Check**:
```bash
ls data/arkitscenes/Training/ | head -5          # should show video_id folders
ls data/arkitscenes/Training/<video_id>/          # should show: wide/ highres_depth/ lowres_depth/ confidence/
ls data/arkitscenes/metadata.csv                  # must exist
```
If `metadata.csv` is missing, re-run the download. If folder structure is wrong, check the `--output_dir` path used in Step 2.

---

### `ModuleNotFoundError: No module named 'marigold'`
**Cause**: Marigold-DC wasn't installed. The script installs it automatically on first run, but may fail if there's no internet access.

**Fix**:
```bash
git clone --depth 1 https://github.com/prs-eth/Marigold-DC.git third_party/Marigold-DC
pip install -e third_party/Marigold-DC
```
Or skip Marigold entirely:
```bash
python -m scripts.run_all --data-root data/arkitscenes --skip-marigold
```

---

### `git push` fails during run_all
**Cause**: The VM doesn't have GitHub credentials configured.

**Fix** — Set up a personal access token before starting the run:
```bash
git config credential.helper store
git push  # enter username + PAT when prompted — it will be saved
```
Or run with `--skip-push` and push manually at the end:
```bash
python -m scripts.run_all --data-root data/arkitscenes --skip-push
# After completion:
git add eval/ results/ checkpoints/ output.json
git commit -m "results: all runs complete"
git push
```

---

### A training run crashed mid-epoch and left a partial checkpoint
**Fix** — Delete the partial checkpoint directory and re-run just that config:
```bash
rm -rf checkpoints/<config_name>
python -m scripts.train \
  --config configs/<config_name>.yaml \
  --run-name <config_name>
```

---

### Smoke test passes but full run crashes on epoch 1
**Cause**: Usually a memory issue that only appears at full batch size / full dataset size.

**Fix** — Run with a reduced batch size and check for OOM errors:
```bash
python -m scripts.train \
  --config configs/baseline.yaml \
  --run-name baseline_debug \
  --max-steps 50
```
If this passes, the issue is intermittent. If it fails with `CUDA out of memory`, apply the batch size fix above.
