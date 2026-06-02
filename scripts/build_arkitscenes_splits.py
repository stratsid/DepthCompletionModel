from __future__ import annotations

import argparse
import random
from pathlib import Path

from scripts.arkit_dataset import _load_metadata


def _collect_valid_frames(data_root: Path, split_folder: str, video_id: str) -> list[str]:
    video_dir = data_root / split_folder / video_id
    wide_dir = video_dir / "wide"
    if not wide_dir.is_dir():
        return []

    frames = []
    for rgb_path in sorted(wide_dir.glob("*.png")):
        frame_name = rgb_path.name
        required = (
            video_dir / "highres_depth" / frame_name,
            video_dir / "lowres_depth" / frame_name,
            video_dir / "confidence" / frame_name,
        )
        if all(path.is_file() for path in required):
            frames.append(f"{video_id}/{frame_name}")
    return frames


def build_splits(data_root: Path, output_dir: Path, train_ratio: float, seed: int) -> None:
    video_to_fold, _ = _load_metadata(data_root)

    training_videos = sorted(video_id for video_id, fold in video_to_fold.items() if fold == "Training")
    validation_videos = sorted(video_id for video_id, fold in video_to_fold.items() if fold == "Validation")

    rng = random.Random(seed)
    rng.shuffle(training_videos)

    num_train_videos = int(round(len(training_videos) * train_ratio))
    num_train_videos = min(max(num_train_videos, 1), max(len(training_videos) - 1, 1))
    train_video_ids = set(training_videos[:num_train_videos])
    val_video_ids = set(training_videos[num_train_videos:])

    train_ids = []
    for video_id in sorted(train_video_ids):
        train_ids.extend(_collect_valid_frames(data_root, "Training", video_id))

    val_ids = []
    for video_id in sorted(val_video_ids):
        val_ids.extend(_collect_valid_frames(data_root, "Training", video_id))

    test_ids = []
    for video_id in validation_videos:
        test_ids.extend(_collect_valid_frames(data_root, "Validation", video_id))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.txt").write_text("\n".join(train_ids) + ("\n" if train_ids else ""))
    (output_dir / "val.txt").write_text("\n".join(val_ids) + ("\n" if val_ids else ""))
    (output_dir / "test.txt").write_text("\n".join(test_ids) + ("\n" if test_ids else ""))

    print(f"train samples: {len(train_ids)}")
    print(f"val samples: {len(val_ids)}")
    print(f"test samples: {len(test_ids)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build project train/val/test split files for ARKitScenes upsampling.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_splits(
        data_root=args.data_root.expanduser(),
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
