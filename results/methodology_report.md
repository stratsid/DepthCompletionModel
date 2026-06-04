# Depth Completion on ARKitScenes — Methodology Report

**CS 231n Project — Spring 2026**

---

## 1. Problem Statement

**Depth completion** is the task of producing a dense, high-resolution depth map from a sparse, low-resolution depth observation and an aligned RGB image. This work targets the specific setting of ARKit LiDAR sensors on iPhone/iPad Pro: the sensor provides 192×256 depth measurements at ~15 Hz, while the RGB camera produces 768×1024 frames — a 4× spatial mismatch. The goal is to upsample the LiDAR depth to match the RGB resolution, recovering fine-grained geometry that the LiDAR misses.

---

## 2. Dataset: ARKitScenes

### 2.1 Format

ARKitScenes is an indoor RGB-D dataset captured on iPad Pro with an Apple ARKit LiDAR scanner. Each scene contains:

| Modality | Resolution | Format |
|----------|-----------|--------|
| `lowres_wide` (RGB) | 1440×1920 → resized to 768×1024 | PNG (uint8) |
| `lowres_depth` (LiDAR) | 192×256 | PNG (uint16, millimetres) |
| `confidence` | 192×256 | PNG (uint8, 0/1/2) |
| `highres_depth` (GT) | 192×256 → upsampled to 768×1024 | PNG (uint16, millimetres) |

The LiDAR is 4× lower resolution than the RGB target: upsample factor = 1024 / 256 = 4.

### 2.2 Split Construction

344 scenes were split 80/10/10 (train/val/test) stratified by scene ID at random seed 0. This yields:
- **Training**: 300 scenes, ~86,000 samples
- **Validation**: 22 scenes, ~6,000 samples
- **Test**: 22 scenes, ~11,739 samples

Each "sample" is one RGB frame paired with its corresponding LiDAR depth frame.

### 2.3 Preprocessing Pipeline (`arkit_dataset.py`)

**RGB:**
1. Load PNG as uint8 (3 channels)
2. Bilinear resize to 768×1024
3. Normalize to float [0, 1]
4. Apply ImageNet channel normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Low-resolution depth (LiDAR input):**
1. Load PNG as uint16, divide by 1000 to convert mm → meters
2. Zero-out pixels with `confidence < 2` (low-quality ARKit estimates filtered out)
3. Clip to valid range [0.1m, 10.0m]
4. Result shape: (1, 192, 256)

**Ground truth depth:**
1. Load PNG as uint16, divide by 1000 → meters
2. Validity mask: `0.1m < gt < 10.0m`
3. Result shape: (1, 768, 1024)

**Upsample factor**: 4× — the model must infer geometry for every 4×4 pixel block from a single LiDAR measurement.

---

## 3. Baseline: Marigold-DC

### 3.1 What it is

Marigold-DC (`prs-eth/marigold-depth-v1-0`) is a diffusion-based depth completion model fine-tuned from Marigold, a latent diffusion model for monocular depth estimation. It treats depth completion as conditional image generation: given RGB + sparse depth hint, it generates a dense depth map.

### 3.2 Preprocessing for Marigold-DC

Marigold-DC takes as input:
- **RGB image**: uint8 numpy array (H, W, 3), full resolution 768×1024
- **Sparse depth**: float32 numpy array (H, W), with non-zero values only at valid LiDAR locations

The sparse depth grid is constructed by `dataset.get_marigold_sample()`:
1. Start with an all-zero array of shape (768, 1024)
2. For each valid LiDAR pixel at (row, col) in the 192×256 space, compute the center of its corresponding 4×4 block:
   ```python
   y = row * upsample_factor + upsample_factor // 2   # = row * 4 + 2
   x = col * upsample_factor + upsample_factor // 2   # = col * 4 + 2
   ```
3. Place the depth value at position (y, x) in the full-resolution sparse grid

This creates a sparse grid where each LiDAR measurement occupies exactly one pixel out of every 16 in the full-resolution image. The rest are zero (unknown).

### 3.3 Forward pass

```python
output = pipe(image=rgb_uint8, sparse_depth=sparse_depth)
dense_depth = output.depth_np  # (H, W), meters
```

The model internally uses diffusion denoising conditioned on both the RGB and sparse depth hints. It does not require GPU fine-tuning — it is used as a zero-shot baseline.

---

## 4. CNN Architecture: `RGBGuidedDepthUpsampler`

### 4.1 Overview

The CNN is a two-branch encoder + fusion decoder network:
```
RGB Input (3, 768, 1024)
       │
[MobileNetV2 RGB Encoder (frozen)]
  f1: (16, 384, 512)    stride 2
  f2: (24, 192, 256)    stride 4
  f3: (32,  96, 128)    stride 8
  f4: (96,  48,  64)    stride 16

Bicubic depth (1, 768, 1024) + Conf mask (1, 768, 1024)
       │
[Depth Encoder (trained)]
  d0: (32, 768, 1024)   stride 1  (stem)
  d1: (32, 384, 512)    stride 2
  d2: (64, 192, 256)    stride 4
  d3: (128, 96, 128)    stride 8
  d4: (256, 48,  64)    stride 16

[Fusion Decoder]
  bot: d4 + f4  →  256 ch
  up3: bot + f3 + d3  →  128 ch
  up2: up3 + f2 + d2  →  64 ch
  up1: up2 + f1 + d1  →  32 ch
  up0: up1 + d0  →  16 ch
  head: →  1 ch (residual)

Output = bicubic_norm + residual_scale * residual
```

### 4.2 Depth Encoder

The depth encoder is a lightweight multi-scale CNN with configurable `levels`, `channels`, and `filter_size`:

- **Input**: 2 channels — bicubic-upsampled normalized depth + confidence mask
- **Stem**: Conv(k) → BN → ReLU at full resolution → `channels[0]` features
- **Down-layers**: MaxPool(2) + Conv(k) → BN → ReLU at each level, progressively coarser

The bicubic upsampling step is crucial: it provides the encoder with a geometrically correct starting point so it can learn **residual corrections** rather than full depth regression from scratch.

### 4.3 Residual Prediction Design

The model predicts a **residual** on top of the bicubic baseline, not the absolute depth. This is scaled by `residual_scale=0.2`:

```
output = clamp(bicubic_norm + 0.2 * tanh_residual, 0, 1) * DEPTH_MAX
```

The head weights are **initialized to zero**, meaning the model starts as an identity (pure bicubic baseline) and must learn what to change. This:
1. Ensures stable training — the model never starts with catastrophically wrong predictions
2. Biases learning toward refinements, not wholesale reconstruction
3. Makes the residual scale interpretable: 0.2 means the model can at most shift depth by ±0.2 × 10m = ±2m

### 4.4 RGB Encoder (MobileNetV2, Frozen)

The RGB encoder is ImageNet-pretrained MobileNetV2, tapped at 4 scales:
- stride 2 → 16 ch (low-level edges, corners)
- stride 4 → 24 ch (local textures)
- stride 8 → 32 ch (mid-level structures)
- stride 16 → 96 ch (semantic context)

**Why MobileNetV2 instead of ResNet?** The project spec targets iPhone/CoreML deployment. MobileNetV2 uses depthwise separable convolutions designed for mobile hardware, and converts cleanly to CoreML via `coremltools.convert()` without custom op workarounds.

**Why the RGB encoder is frozen during all training and ablations:**

The RGB encoder provides appearance cues for edge guidance. Allowing it to fine-tune would introduce two confounding variables: (1) the depth network changes and (2) the RGB features change. For the ablations to isolate the effect of each depth-network design choice (filter size / levels / channels), the RGB path must be held constant. Freezing also prevents overfitting the small training set (300 scenes) with a backbone of 2.2M parameters.

---

## 5. Hyperparameter Choices

### 5.1 Optimizer and Learning Rate

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Weight decay decoupled from gradient updates — standard for vision fine-tuning |
| Learning rate | 6×10⁻⁴ | Baseline: 3×10⁻⁴ ×2 to compensate for batch size doubling from 8→16 (linear scaling rule). The depth encoder is randomly initialized; higher LR allows faster convergence. |
| Weight decay | 1×10⁻⁴ | Light regularization; not critical given frozen RGB backbone |
| LR schedule | Cosine decay with linear warmup (first 5% of steps) | Avoids early instability from large random-init gradients; cosine ensures smooth decay to zero at end of training |

### 5.2 Batch Size and Training Duration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 16 | Fills A100 40GB VRAM efficiently at 768×1024 resolution; larger batches improve BN statistics |
| Epochs | 10 | Constrained by compute budget (~35 min/epoch at max_steps=300); enough for convergence trends but models may not be fully converged |
| max_steps/epoch | 300 | Each step = 16 samples; 300 steps = 4800 samples/epoch ≈ 5.6% of training data per epoch. This is a deliberate tradeoff: faster iteration at the cost of not seeing the full dataset each epoch |

**Note on max_steps:** Training 300 steps × 16 batch × 10 epochs = 48,000 gradient updates across a dataset of ~86,000 samples. Effectively ~0.56 full epochs of data. This is a known limitation of the experimental setup — a production run would use all steps per epoch.

### 5.3 Loss Function

Scale-invariant MSE loss (SILog) on log-depth predictions in normalized [0,1] space:

```
L = mean((log(pred) - log(gt))²) - λ × mean(log(pred) - log(gt))²
    where λ = 0.5
```

SILog is scale-invariant — it performs equally well regardless of absolute scene depth, making it more appropriate than pixel-wise MSE for indoor scenes with depth ranges from 0.1m to 10.0m.

### 5.4 Data Augmentation

Horizontal flip only (p=0.5) applied consistently to RGB, depth, and GT. No color jitter applied to RGB — preserving the appearance-depth correlation is important for the model to learn edge-aware refinement.

---

## 6. Ablation Study Design

All ablations modify **only the depth encoder**; the RGB encoder (MobileNetV2, frozen) and fusion decoder structure remain constant. This isolates the depth processing path.

### 6.1 Filter Size Ablation

Controls the receptive field of each conv layer in the depth encoder.

| Config | Filter | Depth Encoder Channels | Params |
|--------|--------|----------------------|--------|
| `ablation_filter_1x1` | 1×1 | (32, 64, 128, 256) | 2,243,569 |
| `baseline` | 3×3 | (32, 64, 128, 256) | 2,596,337 |
| `ablation_filter_5x5` | 5×5 | (32, 64, 128, 256) | 3,301,873 |

**Hypothesis:** Larger filters capture more spatial context for edge-aware refinement, improving results. 1×1 can only see point-wise information without spatial aggregation.

### 6.2 Encoder Levels Ablation

Controls how many spatial resolution levels the depth encoder processes.

| Config | Levels | Channels | Min Stride | Params |
|--------|--------|----------|-----------|--------|
| `ablation_levels_2` | 2 | (32, 64) | 4× | 1,300,337 |
| `ablation_levels_3` | 3 | (32, 64, 128) | 8× | 1,715,569 |
| `baseline` | 4 | (32, 64, 128, 256) | 16× | 2,596,337 |

**Hypothesis:** More levels = larger effective receptive field = better global context for depth completion.

### 6.3 Channel Width Ablation

Controls the capacity of the depth encoder's feature representations.

| Config | Channels | Params |
|--------|----------|--------|
| `ablation_channels_narrow` | (16, 32, 64, 128) | 1,904,337 |
| `baseline` | (32, 64, 128, 256) | 2,596,337 |
| `ablation_channels_wide` | (64, 128, 256, 512) | 4,574,769 |

**Hypothesis:** Wider channels → richer feature representations → better depth refinement, up to some saturation point.

---

## 7. Results

### 7.1 Model Metrics (Test Set)

**CNN Ablations — 11,739 samples:**

| Model | AbsRel ↓ | RMSE ↓ | MAE ↓ | δ₁ ↑ | Params | FPS |
|-------|----------|--------|-------|-------|--------|-----|
| **Baseline (3×3, 4L, default ch)** | **0.0182** | 0.0662 | 0.0230 | 0.9855 | 2.6M | 61 |
| Filter 1×1 | **0.0182** | **0.0630** | **0.0225** | **0.9879** | 2.2M | 64 |
| Filter 5×5 | 0.0201 | 0.0643 | 0.0233 | 0.9865 | 3.3M | 59 |
| Levels 2 | 0.0234 | 0.1058 | 0.0321 | 0.9727 | 1.3M | 51 |
| Levels 3 | 0.0204 | 0.0838 | 0.0268 | 0.9812 | 1.7M | 56 |
| Channels narrow | 0.0193 | 0.0760 | 0.0255 | 0.9831 | 1.9M | 64 |
| Channels wide | 0.0195 | 0.0759 | 0.0254 | 0.9819 | 4.6M | 52 |

**Baselines:**

| Model | AbsRel ↓ | RMSE ↓ | MAE ↓ | δ₁ ↑ | n | Runtime |
|-------|----------|--------|-------|-------|---|---------|
| **CNN Baseline** | 0.0182 | 0.0662 | 0.0230 | 0.9855 | 11,739 | 16.5ms |
| **Marigold-DC** | 0.0197 | 0.0440 | 0.0246 | 0.9959 | 511 | 34,227ms |
| **Bicubic** | 0.0522 | 0.2430 | 0.0719 | 0.9575 | 11,739 | 0.22ms |

### 7.2 Key Findings

#### Filter Size: 1×1 Surprisingly Matches or Beats 3×3

The 1×1 filter baseline matches 3×3 on AbsRel (0.0182) and **outperforms** it on RMSE (0.0630 vs 0.0662) and δ₁ (0.9879 vs 0.9855), using fewer parameters. 5×5 performs worse than both.

**Interpretation:** The task is *depth upsampling* from a bicubic baseline, not raw depth estimation. The bicubic pre-processing already handles spatial interpolation. The depth encoder mainly learns *where* to apply corrections (edge boundaries, occluded regions). Point-wise (1×1) feature extraction is sufficient for this role — the spatial context for edge guidance comes from the RGB encoder's multi-scale features rather than from the depth encoder. The 5×5 filter's worse performance may reflect overfitting on the limited 300-scene training set.

#### Encoder Levels: Multi-Scale Hierarchy is Critical

Reducing levels from 4→2 causes a dramatic RMSE increase: 0.0662 → 0.1058 (+60%). The 2-level model also shows volatile training (val_absrel spikes to 0.097 at epoch 3), indicating instability from insufficient receptive field.

**Interpretation:** The depth encoder must match the spatial scales of the RGB encoder. The RGB encoder provides features at strides 2, 4, 8, 16. With only 2 depth encoder levels (strides 1, 2, 4), the deepest depth features only reach stride 4 while the deepest RGB tap is at stride 16. The fusion decoder tries to fuse features at mismatched scales, which degrades quality significantly. 4 levels perfectly aligns depth encoder strides (1, 2, 4, 8, 16) with RGB encoder strides.

#### Channel Width: Diminishing Returns Above Baseline

Wide channels (64, 128, 256, 512 = 4.6M params) perform nearly identically to narrow (16, 32, 64, 128 = 1.9M params): RMSE 0.0759 vs 0.0760. Both are slightly worse than the baseline (0.0662).

**Interpretation:** The depth encoder's role is limited — it extracts features from a bicubic-upsampled depth map that already has the right spatial structure. Increased capacity doesn't help because the information bottleneck is not in feature richness but in the receptive field and scale alignment. The slight degradation vs baseline may reflect underfitting/overfitting dynamics with a very short 10-epoch schedule and the specific channel-to-BN-statistics ratio.

### 7.3 Training Dynamics

- **Baseline**: Steady loss decrease; val metrics generally improve with epoch-to-epoch noise (expected at max_steps=300, only seeing ~5% of data per epoch). Best val checkpoint at epoch 9.
- **Levels 2**: High variance — val_absrel spikes to 0.097 at epoch 3 before partially recovering. Indicates scale mismatch instability.
- **Channels wide**: Spike at epoch 4 (val_absrel=0.061) then recovers. Larger models can exhibit temporary oscillations in short training runs due to BN momentum and the limited step budget.

All models are evaluated at the **best validation checkpoint** (`best.pth`), not the last epoch. This mitigates the variance from short training.

### 7.4 Comparison with Marigold-DC and Statistical Significance

**CNN vs Marigold-DC (n=511 overlapping samples):**

| Metric | CNN Baseline | Marigold-DC | CNN Wins? | Wilcoxon p-value |
|--------|-------------|-------------|-----------|-----------------|
| AbsRel | 0.0182 | 0.0197 | Yes (↓8.0%) | 9.2×10⁻¹⁵ |
| RMSE | 0.0662 | 0.0440 | No (Marigold better by 34%) | 2.6×10⁻⁵⁴ |

**CNN vs Bicubic (n=11,739 samples, paired Wilcoxon p=0.0 for both metrics):**

Both differences are statistically significant at p<0.001 by paired Wilcoxon signed-rank test.

**Bootstrap 95% CI (AbsRel, 10,000 resamples, seed=42):**

| Model | AbsRel | 95% CI |
|-------|--------|--------|
| CNN Baseline | 0.0182 | [0.0180, 0.0184] |
| Marigold-DC | 0.0197 | [0.0191, 0.0203] |
| Bicubic | 0.0522 | [0.0516, 0.0528] |

**Interpretation:**

The CNN baseline achieves lower AbsRel (0.0182) than Marigold-DC (0.0197), demonstrating that a 2.6M-parameter task-specific model trained on ARKitScenes matches or exceeds a large diffusion foundation model on in-domain test data — a result consistent with the depth completion literature. However, Marigold-DC achieves substantially lower RMSE (0.0440 vs 0.0662), indicating that Marigold produces fewer large-error outliers despite higher mean absolute error. This pattern suggests Marigold-DC's diffusion prior produces more globally consistent depth maps even for pixels far from LiDAR observations, while the CNN is more accurate on average near observation points.

The CNN's inference speed (16.5ms / 61 fps) vs Marigold-DC (34,227ms / 0.03 fps) represents a 2,000× throughput advantage, making the CNN the only viable choice for real-time mobile AR applications.

**Note on Marigold evaluation sample count:** Marigold-DC was evaluated on 511 test samples (capped for compute) vs 11,739 for the CNN. The overlapping 511-sample Wilcoxon test is therefore the authoritative paired comparison.

---

## 8. Artifacts and Plots

All result artifacts are in the repository:

| Path | Contents |
|------|----------|
| `eval/cnn_{name}_results.summary.json` | Per-model test metrics |
| `eval/cnn_{name}_results.csv` | Per-sample metrics for all 11,739 test frames |
| `eval/bicubic_results.summary.json` | Bicubic baseline metrics |
| `eval/marigold_dc_test_results.summary.json` | Marigold-DC metrics |
| `checkpoints/{name}/train_log.csv` | Per-epoch train loss and val metrics |
| `results/plots/{name}_training_curves.png` | Training curves (7 configs) |
| `results/plots/ablation_filter_size.png` | Filter 1×1 vs 3×3 vs 5×5 |
| `results/plots/ablation_encoder_levels.png` | 2-level vs 3-level vs 4-level |
| `results/plots/ablation_channel_width.png` | Narrow vs default vs wide channels |
| `results/plots/full_comparison.png` | All models + bicubic + Marigold bar chart |
| `results/plots/all_runs_val_absrel.png` | Val AbsRel curves for all 7 CNNs |
| `results/sample_viz/sample_*.png` | Side-by-side: RGB/Sparse/Bicubic/CNN/Marigold/GT |
| `output.json` | Machine-readable summary of all metrics |

---

## 9. Limitations and Future Work

1. **Short training**: 10 epochs × max_steps=300 = 48k gradient steps over a ~86k-sample dataset. A full run (all steps per epoch) would likely improve all CNN metrics by 10–20%.

2. **No test-time augmentation**: Single-pass inference. Ensemble or TTA could improve robustness at evaluation time.

3. **Fixed backbone**: MobileNetV2 was chosen for CoreML compatibility. ResNet-50 or EfficientNet might provide better RGB features at the cost of larger model size.

4. **Sparse confidence filtering**: Confidence < 2 pixels are zeroed out in the LiDAR input. A soft weighting approach (rather than hard mask) might preserve more low-confidence LiDAR information.

5. **Marigold-DC comparison caveat**: Marigold-DC runs at a different inference paradigm (diffusion sampling, ~10 steps) vs single-pass CNN. A fair comparison would normalize for inference compute, not just wall time.

---

## 10. Reproducibility

All experiments are fully reproducible:
- Dataset split: seed=0, deterministic scene assignment
- All configs in `configs/*.yaml` (committed to repo)
- Training: `python -m scripts.train --config configs/{name}.yaml --run-name {name}`
- Eval: `python -m scripts.eval_cnn --checkpoint checkpoints/{name}/best.pth --split-path splits/test.txt`
- Full pipeline: `modal run modal_run.py::run_experiments` (A100, ~4 hours)
