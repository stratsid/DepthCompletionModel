"""Generate side-by-side depth completion sample grids.

For each selected test sample, produces a row with 6 panels:
  [Input RGB]  [Sparse Depth In]  [Bicubic]  [Baseline CNN]  [Marigold-DC]  [GT Dense]

Each depth panel is annotated with AbsRel and RMSE vs GT.
Samples are selected deterministically from fixed indices.

Usage (on machine with GPU + dataset):
    python -m scripts.visualize_samples \\
      --checkpoint checkpoints/baseline/best.pth \\
      --marigold-eval-csv eval/marigold_dc_test_results.csv \\
      --split-path splits/test.txt \\
      --data-root data/arkitscenes \\
      --n-samples 8 \\
      --out-dir results/sample_viz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import compute_metrics
from scripts.arkit_dataset import ARKitUpsampleDataset
from scripts.model import RGBGuidedDepthUpsampler

DEPTH_MAX = 10.0
DEPTH_MIN = 0.1

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

FIXED_INDICES = [0, 150, 450, 900, 1500, 2500, 4000, 6000]


def _denorm_rgb(t: torch.Tensor) -> np.ndarray:
    img = t.permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def _depth_to_color(depth_m: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return cmap(norm(depth_m))[..., :3]


def _sparse_to_color(lowres_m: np.ndarray, vmin: float, vmax: float, hw: tuple) -> np.ndarray:
    """Bicubic upsample sparse map just for display (not for metrics)."""
    t = torch.from_numpy(lowres_m).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=hw, mode="nearest").squeeze().numpy()
    img = _depth_to_color(up, vmin, vmax)
    # Darken zero (no-data) pixels
    mask_zero = (up < DEPTH_MIN)
    img[mask_zero] = 0.05
    return img


def _annotate(ax, title: str, subtitle: str = ""):
    ax.set_title(title, fontsize=8, fontweight="bold")
    if subtitle:
        ax.text(0.5, -0.04, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color="#333")
    ax.axis("off")


def load_cnn_model(checkpoint_path: Path, device: torch.device):
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
    return model.to(device)


def load_marigold_model(device: torch.device):
    try:
        from diffusers import MarigoldDepthCompletionPipeline
        return MarigoldDepthCompletionPipeline.from_pretrained(
            "prs-eth/marigold-depth-v1-0"
        ).to(device)
    except Exception as e:
        print(f"[WARN] Could not load Marigold-DC: {e}")
        return None


def run_marigold_on_sample(pipe, rgb_uint8: np.ndarray, sparse_depth: np.ndarray) -> Optional[np.ndarray]:
    """Returns dense depth (H, W) in meters, or None if inference fails."""
    try:
        output = pipe(image=rgb_uint8, sparse_depth=sparse_depth)
        return output if isinstance(output, np.ndarray) else output.depth_np
    except Exception as e:
        print(f"[WARN] Marigold inference failed: {e}")
        return None


def make_sample_grid(
    idx: int,
    batch: dict,
    cnn_pred_m: np.ndarray,
    bicubic_m: np.ndarray,
    marigold_pred_m: Optional[np.ndarray],
    out_path: Path,
):
    gt_m = batch["gt"][0].numpy()          # (H, W)
    lowres_m = batch["lowres"][0].numpy()  # (h, w)
    rgb_img = _denorm_rgb(batch["rgb"])    # (H, W, 3)
    hw = gt_m.shape

    valid = (gt_m > DEPTH_MIN) & (gt_m < DEPTH_MAX)
    vmin = float(gt_m[valid].min()) if valid.any() else DEPTH_MIN
    vmax = float(gt_m[valid].max()) if valid.any() else DEPTH_MAX

    def _score(pred):
        if pred is None:
            return "N/A"
        m = compute_metrics(pred, gt_m, DEPTH_MIN, DEPTH_MAX)
        return f"AbsRel={m['absrel']:.3f}\nRMSE={m['rmse']:.3f}"

    n_cols = 6 if marigold_pred_m is not None else 5
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.5))
    fig.subplots_adjust(hspace=0.05, wspace=0.03)

    col = 0
    axes[col].imshow(np.clip(rgb_img, 0, 1))
    _annotate(axes[col], "Input RGB", f"sample {idx}\n{batch['identifier']}")

    col += 1
    axes[col].imshow(_sparse_to_color(lowres_m, vmin, vmax, hw))
    _annotate(axes[col], "Sparse LiDAR Input", "(4× lower res)")

    col += 1
    axes[col].imshow(_depth_to_color(bicubic_m, vmin, vmax))
    _annotate(axes[col], "Bicubic", _score(bicubic_m))

    col += 1
    axes[col].imshow(_depth_to_color(cnn_pred_m, vmin, vmax))
    _annotate(axes[col], "Baseline CNN", _score(cnn_pred_m))

    if marigold_pred_m is not None:
        col += 1
        axes[col].imshow(_depth_to_color(marigold_pred_m, vmin, vmax))
        _annotate(axes[col], "Marigold-DC", _score(marigold_pred_m))

    col += 1
    axes[col].imshow(_depth_to_color(gt_m, vmin, vmax))
    _annotate(axes[col], "GT Dense Depth", "(ARKit ground truth)")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.01, pad=0.01)
    cbar.set_label("Depth (m)", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def run_visualization(
    checkpoint_path: Path,
    split_path: Path,
    data_root: Path,
    out_dir: Path,
    n_samples: int = 8,
    target_hw: tuple = (768, 1024),
    confidence_threshold: int = 2,
    run_marigold: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Visualizing on {device}")

    dataset = ARKitUpsampleDataset(
        root=data_root,
        split_path=split_path,
        target_hw=target_hw,
        crop_hw=None,
        augment=False,
        confidence_threshold=confidence_threshold,
    )

    indices = FIXED_INDICES[:n_samples]
    indices = [i % len(dataset) for i in indices]

    cnn_model = load_cnn_model(checkpoint_path, device)

    marigold_pipe = None
    if run_marigold:
        marigold_pipe = load_marigold_model(device)

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    for rank, idx in enumerate(indices):
        batch = dataset[idx]
        rgb_t  = batch["rgb"].unsqueeze(0).to(device)
        lowres = batch["lowres"].unsqueeze(0).to(device)
        lconf  = batch["lowres_conf"].unsqueeze(0).to(device)
        hw = rgb_t.shape[-2:]

        bicubic = F.interpolate(lowres.float(), size=hw, mode="bicubic", align_corners=False)
        bicubic = bicubic.clamp(0.0, DEPTH_MAX)
        bicubic_norm = (bicubic / DEPTH_MAX).clamp(0.0, 1.0)
        conf_hi = F.interpolate((lconf >= confidence_threshold).float(), size=hw,
                                mode="bilinear", align_corners=False)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype):
            pred_norm = cnn_model(rgb_t, bicubic_norm, conf_hi)

        bicubic_m = bicubic[0, 0].float().cpu().numpy()
        cnn_m = (pred_norm[0, 0].float() * DEPTH_MAX).clamp(DEPTH_MIN, DEPTH_MAX).cpu().numpy()

        marigold_m = None
        if marigold_pipe is not None:
            try:
                marigold_sample = dataset.get_marigold_sample(idx)
                _, rgb_uint8, sparse_depth, _ = marigold_sample
                marigold_m = run_marigold_on_sample(marigold_pipe, rgb_uint8, sparse_depth)
            except Exception as e:
                print(f"[WARN] Marigold sample {idx} failed: {e}")

        out_path = out_dir / f"sample_{rank:02d}_idx{idx:05d}.png"
        make_sample_grid(idx, batch, cnn_m, bicubic_m, marigold_m, out_path)

    print(f"\nAll {len(indices)} sample grids saved to {out_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description="Generate sample visualization grids.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split-path", type=Path, default=Path("splits/test.txt"))
    p.add_argument("--data-root", type=Path, default=Path("data/arkitscenes"))
    p.add_argument("--out-dir", type=Path, default=Path("results/sample_viz"))
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--target-height", type=int, default=768)
    p.add_argument("--target-width", type=int, default=1024)
    p.add_argument("--confidence-threshold", type=int, default=2)
    p.add_argument("--no-marigold", action="store_true", help="Skip Marigold inference")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_visualization(
        checkpoint_path=args.checkpoint,
        split_path=args.split_path,
        data_root=args.data_root,
        out_dir=args.out_dir,
        n_samples=args.n_samples,
        target_hw=(args.target_height, args.target_width),
        confidence_threshold=args.confidence_threshold,
        run_marigold=not args.no_marigold,
    )
