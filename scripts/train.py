"""Training script for RGBGuidedDepthUpsampler.

Usage:
    python -m scripts.train --config configs/baseline.yaml
    python -m scripts.train --config configs/ablation_filter_1x1.yaml --device cuda:0
    python -m scripts.train --config configs/baseline.yaml --max-steps 10  # smoke test
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import METRIC_KEYS, compute_metrics, summarize_metric_rows
from scripts.arkit_dataset import make_loaders
from scripts.model import RGBGuidedDepthUpsampler, count_parameters

DEPTH_MAX = 10.0


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs()
    return (diff * mask).sum() / mask.sum().clamp(min=1)


def _masked_grad_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]

    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    loss_y = _masked_l1(dy_pred, dy_tgt, mask_y)
    loss_x = _masked_l1(dx_pred, dx_tgt, mask_x)
    return (loss_y + loss_x) * 0.5


def depth_loss(
    pred_norm: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    lambda_grad: float = 0.5,
) -> torch.Tensor:
    gt_norm = gt / DEPTH_MAX
    mask = valid.float()
    l1 = _masked_l1(pred_norm, gt_norm, mask)
    grad = _masked_grad_l1(pred_norm, gt_norm, mask)
    return l1 + lambda_grad * grad


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg_model: dict, device: torch.device) -> RGBGuidedDepthUpsampler:
    model = RGBGuidedDepthUpsampler(
        pretrained_rgb=cfg_model.get("pretrained_rgb", True),
        residual_scale=cfg_model.get("residual_scale", 0.2),
        depth_encoder_levels=cfg_model.get("depth_encoder_levels", 4),
        depth_filter_size=cfg_model.get("depth_filter_size", 3),
        depth_channels=tuple(cfg_model.get("depth_channels", [32, 64, 128, 256])),
        fusion_decoder_channels=tuple(cfg_model.get("fusion_decoder_channels", [256, 128, 64, 32])),
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: RGBGuidedDepthUpsampler,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    lambda_grad: float,
    grad_clip_norm: float,
    max_steps: int | None = None,
    log_every: int = 50,
) -> float:
    model.train()
    model.rgb_enc.eval()  # keep BN frozen if RGB encoder is frozen

    total_loss = 0.0
    n_steps = 0

    for batch_idx, batch in enumerate(loader):
        if max_steps is not None and n_steps >= max_steps:
            break

        rgb = batch["rgb_norm"].to(device)
        bicubic_norm = batch["bicubic_norm"].to(device)
        conf_hi = batch["conf_hi"].to(device)
        gt = batch["gt"].to(device)
        valid = batch["valid"].to(device)

        pred_norm = model(rgb, bicubic_norm, conf_hi)
        loss = depth_loss(pred_norm, gt, valid, lambda_grad=lambda_grad)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_steps += 1

        if (batch_idx + 1) % log_every == 0:
            print(f"  step {batch_idx + 1}: loss={loss.item():.4f}")

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def validate(
    model: RGBGuidedDepthUpsampler,
    loader,
    device: torch.device,
    min_depth: float,
    max_depth: float,
    max_steps: int | None = None,
) -> Dict[str, float]:
    model.eval()
    rows: List[dict] = []

    for batch_idx, batch in enumerate(loader):
        if max_steps is not None and batch_idx >= max_steps:
            break

        rgb = batch["rgb_norm"].to(device)
        bicubic_norm = batch["bicubic_norm"].to(device)
        conf_hi = batch["conf_hi"].to(device)
        gt = batch["gt"]

        pred_norm = model(rgb, bicubic_norm, conf_hi)
        pred_m = (pred_norm * DEPTH_MAX).clamp(min_depth, max_depth)

        for i in range(pred_m.shape[0]):
            pred_np = pred_m[i, 0].cpu().numpy()
            gt_np = gt[i, 0].numpy()
            m = compute_metrics(pred_np, gt_np, min_depth=min_depth, max_depth=max_depth)
            rows.append(m)

    if not rows:
        return {k: float("nan") for k in METRIC_KEYS}
    return summarize_metric_rows(rows, METRIC_KEYS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Limit steps per epoch (train and val). For smoke tests.",
    )
    p.add_argument("--run-name", type=str, default=None, help="Override auto-generated run name.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- data ----
    cfg_ds = cfg["dataset"]
    cfg_loader = cfg["loader"]
    train_loader, val_loader = make_loaders(
        root=Path(cfg_ds["root"]),
        train_split_path=Path(cfg_ds["train_split_path"]),
        val_split_path=Path(cfg_ds["val_split_path"]),
        target_hw=tuple(cfg_ds["target_hw"]),
        train_crop_hw=tuple(cfg_ds["train_crop_hw"]),
        confidence_threshold=cfg_ds["confidence_threshold"],
        min_depth=cfg_ds["min_depth"],
        max_depth=cfg_ds["max_depth"],
        batch_size=cfg_loader["batch_size"],
        num_workers=cfg_loader["num_workers"],
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---- model ----
    model = build_model(cfg["model"], device)
    model.freeze_rgb()  # always start frozen; optionally unfreeze later
    total_p, trainable_p = count_parameters(model)
    print(f"Parameters: total={total_p:,}, trainable={trainable_p:,}")

    # ---- optimizer + scheduler ----
    cfg_opt = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg_opt["learning_rate"],
        weight_decay=cfg_opt["weight_decay"],
    )
    cfg_train = cfg["training"]
    epochs = cfg_train["epochs"]
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    lambda_grad = cfg_train.get("lambda_grad", 0.5)
    grad_clip_norm = cfg_train.get("grad_clip_norm", 1.0)
    freeze_rgb_epochs = cfg_train.get("freeze_rgb_epochs", 1)
    min_depth = cfg_ds["min_depth"]
    max_depth = cfg_ds["max_depth"]

    # ---- output dirs ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.config.stem}_{timestamp}"
    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_csv = ckpt_dir / "train_log.csv"
    log_fields = ["epoch", "train_loss"] + [f"val_{k}" for k in METRIC_KEYS]
    csv_file = open(log_csv, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=log_fields)
    csv_writer.writeheader()

    best_absrel = float("inf")
    print(f"\nStarting training — run '{run_name}'")
    print(f"Checkpoints: {ckpt_dir}\n")

    for epoch in range(1, epochs + 1):
        # freeze/unfreeze RGB encoder per schedule
        if epoch <= freeze_rgb_epochs:
            model.freeze_rgb()
        else:
            model.unfreeze_rgb()
            # rebuild optimizer to include RGB params on first unfreeze
            if epoch == freeze_rgb_epochs + 1:
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=cfg_opt["learning_rate"],
                    weight_decay=cfg_opt["weight_decay"],
                )
                remaining = (epochs - epoch + 1) * len(train_loader)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(remaining, 1))

        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            lambda_grad=lambda_grad,
            grad_clip_norm=grad_clip_norm,
            max_steps=args.max_steps,
        )
        val_metrics = validate(
            model, val_loader, device,
            min_depth=min_depth,
            max_depth=max_depth,
            max_steps=args.max_steps,
        )
        elapsed = time.perf_counter() - t0

        row = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"val_{k}": val_metrics[k] for k in METRIC_KEYS})
        csv_writer.writerow(row)
        csv_file.flush()

        absrel = val_metrics.get("absrel", float("inf"))
        print(
            f"Epoch {epoch}/{epochs} | loss={train_loss:.4f} | "
            f"absrel={absrel:.4f} rmse={val_metrics.get('rmse', float('nan')):.4f} "
            f"d1={val_metrics.get('d1', float('nan')):.4f} | {elapsed:.1f}s"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, ckpt_dir / "last.pth")

        if absrel < best_absrel:
            best_absrel = absrel
            torch.save(ckpt, ckpt_dir / "best.pth")
            print(f"  -> new best absrel={best_absrel:.4f}, saved best.pth")

    csv_file.close()
    print(f"\nTraining complete. Best val absrel: {best_absrel:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
