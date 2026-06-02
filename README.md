# DepthCompletionModel

Lightweight RGB-guided CNN for indoor depth completion. Converts sparse ARKit LiDAR output into dense metric depth, targeting on-device inference. Evaluated against Marigold-DC with ablations over depth encoder filter size, levels, and channel width.

---

## VM Setup & Run Instructions

### 1. Clone the repo and install dependencies

```bash
git clone git@github.com:stratsid/depth-completion-model.git
cd depth-completion-model
pip install -r requirements.txt
```

### 2. Download ARKitScenes

Download the ARKitScenes dataset (Training + Validation splits) from Apple's official source and place it at:

```
data/arkitscenes/
  Training/
    <video_id>/
      wide/
      highres_depth/
      lowres_depth/
      confidence/
  Validation/
    ...
  metadata.csv
```

### 3. Build splits

```bash
python -m scripts.build_arkitscenes_splits \
  --data-root data/arkitscenes \
  --output-dir splits/
```

### 4. Run everything

```bash
# Full run (Marigold eval + all CNN configs + eval + plots + push)
python -m scripts.run_all --data-root data/arkitscenes

# Skip Marigold-DC (faster, runs CNN only)
python -m scripts.run_all --data-root data/arkitscenes --skip-marigold

# Smoke test (10 steps/epoch, 5 eval samples) to verify setup before full run
python -m scripts.run_all --data-root data/arkitscenes --smoke-test --skip-push
```

Results are committed and pushed to this repo automatically after each training run completes.

### 5. View results

After the run completes, pull on your laptop:

```bash
git pull
```

- **`output.json`** — all run summaries, training curves, and test metrics
- **`results/plots/`** — PNG training curves per model + cross-model comparison charts
- **`eval/`** — per-sample CSV and summary JSON for each model

---

## Running individual steps

```bash
# Train one config
python -m scripts.train --config configs/baseline.yaml

# Evaluate a checkpoint
python -m scripts.eval_cnn --checkpoint checkpoints/baseline/best.pth

# Evaluate Marigold-DC
python -m scripts.run_marigold_dc --split-path splits/test.txt --data-root data/arkitscenes

# Regenerate plots
python -m scripts.plot_results --run-names baseline ablation_filter_1x1 ...
```

---

## Configs

| Config | Filter | Levels | Channels |
|---|---|---|---|
| `baseline` | 3×3 | 4 | (32,64,128,256) |
| `ablation_filter_1x1` | 1×1 | 4 | (32,64,128,256) |
| `ablation_filter_5x5` | 5×5 | 4 | (32,64,128,256) |
| `ablation_levels_2` | 3×3 | 2 | (32,64) |
| `ablation_levels_3` | 3×3 | 3 | (32,64,128) |
| `ablation_channels_narrow` | 3×3 | 4 | (16,32,64,128) |
| `ablation_channels_wide` | 3×3 | 4 | (64,128,256,512) |
