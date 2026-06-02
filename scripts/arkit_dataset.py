from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

NATIVE_LOW = (192, 256)
MM_TO_M = 1000.0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEPTH_MIN = 0.1
DEPTH_MAX = 10.0

FOLD_MAP = {
    "train": "Training",
    "training": "Training",
    "val": "Validation",
    "validation": "Validation",
}


@dataclass(frozen=True)
class SampleRecord:
    identifier: str
    split_folder: str
    video_id: str
    frame_name: str
    sky_direction: str


def _normalize_fold_name(value: str) -> str:
    key = str(value).strip().lower()
    if key not in FOLD_MAP:
        raise ValueError(f"unknown fold name {value!r}")
    return FOLD_MAP[key]


def _rotate_np(img: np.ndarray, direction: str) -> np.ndarray:
    if direction == "Up":
        return img
    if direction == "Left":
        return np.rot90(img, k=-1).copy()
    if direction == "Right":
        return np.rot90(img, k=1).copy()
    if direction == "Down":
        return np.rot90(img, k=2).copy()
    raise ValueError(f"unknown sky_direction {direction!r}")


def _resize_image(arr: np.ndarray, size: Tuple[int, int], mode: str) -> np.ndarray:
    height, width = size
    if arr.shape[:2] == (height, width):
        return arr

    pil = Image.fromarray(arr)
    if arr.ndim == 2:
        resample = Image.BILINEAR if mode == "bilinear" else Image.NEAREST
    else:
        resample = Image.BILINEAR
    return np.array(pil.resize((width, height), resample=resample))


def _load_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def _load_metadata(data_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    metadata_path = data_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing ARKitScenes metadata file: {metadata_path}")

    meta = pd.read_csv(metadata_path)
    required = {"video_id", "fold", "sky_direction"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {sorted(missing)}")

    video_to_fold = {}
    video_to_sky = {}
    for row in meta.itertuples(index=False):
        video_id = str(row.video_id)
        video_to_fold[video_id] = _normalize_fold_name(row.fold)
        video_to_sky[video_id] = str(row.sky_direction)
    return video_to_fold, video_to_sky


def parse_sample_ids(lines: Iterable[str]) -> list[str]:
    sample_ids = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sample_ids.append(line)
    return sample_ids


def load_sample_ids(split_path: Path) -> list[str]:
    return parse_sample_ids(split_path.read_text().splitlines())


class ARKitUpsampleDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split_path: str | Path,
        target_hw: Tuple[int, int] = (768, 1024),
        crop_hw: Tuple[int, int] | None = None,
        augment: bool = False,
        confidence_threshold: int = 2,
        min_depth: float = DEPTH_MIN,
        max_depth: float = DEPTH_MAX,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser()
        self.split_path = Path(split_path)
        self.target_hw = target_hw
        self.crop_hw = crop_hw
        self.augment = augment
        self.confidence_threshold = confidence_threshold
        self.min_depth = min_depth
        self.max_depth = max_depth

        height, width = target_hw
        if height % 4 != 0 or width % 4 != 0:
            raise ValueError("target resolution must be divisible by 4")

        self.upsample_factor = width // NATIVE_LOW[1]
        if height // NATIVE_LOW[0] != self.upsample_factor:
            raise ValueError("target resolution must preserve the native 4:3 low-res aspect ratio")
        self.lowres_hw = (height // self.upsample_factor, width // self.upsample_factor)

        video_to_fold, video_to_sky = _load_metadata(self.root)
        self.records = self._build_records(
            sample_ids=load_sample_ids(self.split_path),
            video_to_fold=video_to_fold,
            video_to_sky=video_to_sky,
        )

        self._rgb_norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self._color_jitter = (
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            if augment
            else None
        )

    def _build_records(
        self,
        sample_ids: list[str],
        video_to_fold: dict[str, str],
        video_to_sky: dict[str, str],
    ) -> list[SampleRecord]:
        records = []
        for sample_id in sample_ids:
            parts = Path(sample_id).parts
            if len(parts) != 2:
                raise ValueError(f"sample ID must be 'video_id/frame.png', got {sample_id!r}")
            video_id, frame_name = parts
            if video_id not in video_to_fold:
                raise KeyError(f"video ID {video_id!r} missing from metadata.csv")
            record = SampleRecord(
                identifier=sample_id,
                split_folder=video_to_fold[video_id],
                video_id=video_id,
                frame_name=frame_name,
                sky_direction=video_to_sky[video_id],
            )
            self._assert_modalities_exist(record)
            records.append(record)
        return records

    def _assert_modalities_exist(self, record: SampleRecord) -> None:
        base = self.root / record.split_folder / record.video_id
        required = (
            base / "wide" / record.frame_name,
            base / "highres_depth" / record.frame_name,
            base / "lowres_depth" / record.frame_name,
            base / "confidence" / record.frame_name,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing ARKitScenes modalities for {record.identifier}: {missing}")

    def __len__(self) -> int:
        return len(self.records)

    def _load_one(self, record: SampleRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        base = self.root / record.split_folder / record.video_id
        rgb = _load_png(base / "wide" / record.frame_name)
        gt_mm = _load_png(base / "highres_depth" / record.frame_name)
        lo_mm = _load_png(base / "lowres_depth" / record.frame_name)
        conf = _load_png(base / "confidence" / record.frame_name)

        rgb = _rotate_np(rgb, record.sky_direction)
        gt_mm = _rotate_np(gt_mm, record.sky_direction)
        lo_mm = _rotate_np(lo_mm, record.sky_direction)
        conf = _rotate_np(conf, record.sky_direction)
        return rgb, gt_mm, lo_mm, conf

    def _prepare_modalities(
        self,
        record: SampleRecord,
        apply_augmentation: bool,
        include_raw_arrays: bool = False,
    ) -> dict[str, torch.Tensor | str | np.ndarray]:
        rgb, gt_mm, lo_mm, conf = self._load_one(record)

        height, width = self.target_hw
        low_h, low_w = self.lowres_hw

        rgb = _resize_image(rgb, (height, width), mode="bilinear")
        gt_mm = _resize_image(gt_mm, (height, width), mode="nearest")
        if lo_mm.shape != (low_h, low_w):
            lo_mm = _resize_image(lo_mm, (low_h, low_w), mode="nearest")
            conf = _resize_image(conf, (low_h, low_w), mode="nearest")

        gt = gt_mm.astype(np.float32) / MM_TO_M
        lo = lo_mm.astype(np.float32) / MM_TO_M
        lo_filtered = np.where(conf >= self.confidence_threshold, lo, 0.0).astype(np.float32)

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        gt_t = torch.from_numpy(gt).unsqueeze(0)
        lo_t = torch.from_numpy(lo_filtered).unsqueeze(0)
        conf_t = torch.from_numpy(conf.astype(np.uint8)).unsqueeze(0)

        valid = (
            (gt_t > self.min_depth)
            & (gt_t < self.max_depth)
            & torch.isfinite(gt_t)
        )

        if apply_augmentation:
            if self._color_jitter is not None:
                rgb_t = self._color_jitter(rgb_t)

            if random.random() < 0.5:
                rgb_t = torch.flip(rgb_t, dims=[2])
                gt_t = torch.flip(gt_t, dims=[2])
                valid = torch.flip(valid, dims=[2])
                lo_t = torch.flip(lo_t, dims=[2])
                conf_t = torch.flip(conf_t, dims=[2])

            if self.crop_hw is not None:
                crop_h, crop_w = self.crop_hw
                low_crop_h = crop_h // self.upsample_factor
                low_crop_w = crop_w // self.upsample_factor
                if crop_h % self.upsample_factor != 0 or crop_w % self.upsample_factor != 0:
                    raise ValueError("crop size must be divisible by the upsample factor")
                if low_crop_h > low_h or low_crop_w > low_w:
                    raise ValueError("crop size exceeds low-res input size")
                low_y = random.randint(0, low_h - low_crop_h)
                low_x = random.randint(0, low_w - low_crop_w)
                high_y = low_y * self.upsample_factor
                high_x = low_x * self.upsample_factor

                lo_t = lo_t[:, low_y:low_y + low_crop_h, low_x:low_x + low_crop_w]
                conf_t = conf_t[:, low_y:low_y + low_crop_h, low_x:low_x + low_crop_w]
                rgb_t = rgb_t[:, high_y:high_y + crop_h, high_x:high_x + crop_w]
                gt_t = gt_t[:, high_y:high_y + crop_h, high_x:high_x + crop_w]
                valid = valid[:, high_y:high_y + crop_h, high_x:high_x + crop_w]
                height, width = crop_h, crop_w

        bicubic = F.interpolate(
            lo_t.unsqueeze(0).float(),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        bicubic = bicubic.clamp(min=0.0, max=self.max_depth)

        conf_hi = F.interpolate(
            (conf_t >= self.confidence_threshold).float().unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        bicubic_norm = (bicubic / self.max_depth).clamp(0.0, 1.0)
        rgb_norm = self._rgb_norm(rgb_t)

        sample = {
            "rgb": rgb_norm,
            "lowres": lo_t,
            "lowres_conf": conf_t,
            "bicubic": bicubic,
            "bicubic_norm": bicubic_norm,
            "conf_hi": conf_hi,
            "gt": gt_t,
            "valid": valid,
            "identifier": record.identifier,
        }
        if include_raw_arrays:
            sample["rgb_uint8"] = rgb
            sample["gt_np"] = gt
            sample["lowres_np"] = lo_filtered
        return sample

    def __getitem__(self, idx: int):
        return self._prepare_modalities(
            self.records[idx],
            apply_augmentation=self.augment,
            include_raw_arrays=False,
        )

    def get_marigold_sample(self, idx: int) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        sample = self._prepare_modalities(
            self.records[idx],
            apply_augmentation=False,
            include_raw_arrays=True,
        )
        rgb = np.asarray(sample["rgb_uint8"], dtype=np.uint8)
        gt = np.asarray(sample["gt_np"], dtype=np.float32)
        lowres = np.asarray(sample["lowres_np"], dtype=np.float32)

        height, width = gt.shape
        low_h, low_w = lowres.shape
        ys = np.arange(low_h) * self.upsample_factor + self.upsample_factor // 2
        xs = np.arange(low_w) * self.upsample_factor + self.upsample_factor // 2
        ys = np.clip(ys, 0, height - 1)
        xs = np.clip(xs, 0, width - 1)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")

        sparse = np.zeros((height, width), dtype=np.float32)
        valid = np.isfinite(lowres) & (lowres > 0)
        sparse[grid_y[valid], grid_x[valid]] = lowres[valid]
        return str(sample["identifier"]), rgb, sparse, gt


def make_loaders(
    root: str | Path,
    target_hw: Tuple[int, int],
    train_crop_hw: Tuple[int, int] | None,
    batch_size: int,
    num_workers: int,
    train_split_path: str | Path = "splits/train.txt",
    val_split_path: str | Path = "splits/val.txt",
    confidence_threshold: int = 2,
    min_depth: float = DEPTH_MIN,
    max_depth: float = DEPTH_MAX,
):
    from torch.utils.data import DataLoader

    train_set = ARKitUpsampleDataset(
        root=root,
        split_path=train_split_path,
        target_hw=target_hw,
        crop_hw=train_crop_hw,
        augment=True,
        confidence_threshold=confidence_threshold,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    val_set = ARKitUpsampleDataset(
        root=root,
        split_path=val_split_path,
        target_hw=target_hw,
        crop_hw=None,
        augment=False,
        confidence_threshold=confidence_threshold,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_set, val_set
