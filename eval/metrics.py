import numpy as np


METRIC_KEYS = ["rmse", "mae", "absrel", "d1", "d2", "d3"]


def compute_metrics(pred, gt, min_depth=0.1, max_depth=10.0, eps=1e-6):
    """
    Compute depth completion metrics between predicted and ground truth depth maps.

    Args:
        pred: predicted dense depth map (H, W), in meters
        gt: ground truth depth map (H, W), in meters
        min_depth: ignore pixels <= this depth, default 0.1m
        max_depth: ignore pixels >= this depth, default 10.0m
        eps: numerical stability constant for divisions

    Returns:
        dict with:
            rmse: root mean squared error, lower is better
            mae: mean absolute error, lower is better
            absrel: absolute relative error, lower is better
            d1: threshold accuracy under 1.25, higher is better
            d2: threshold accuracy under 1.25^2, higher is better
            d3: threshold accuracy under 1.25^3, higher is better
            valid_pixels: number of pixels included in the metric computation
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have the same shape, got {pred.shape} and {gt.shape}")

    # Mask out invalid GT pixels. Prediction is expected to be dense, so validity is
    # defined by the ground truth depth map.
    mask = (
        (gt > min_depth)
        & (gt < max_depth)
        & np.isfinite(gt)
        & np.isfinite(pred)
    )

    valid_pixels = int(mask.sum())
    if valid_pixels == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "absrel": np.nan,
            "d1": np.nan,
            "d2": np.nan,
            "d3": np.nan,
            "valid_pixels": 0,
        }

    pred_valid = np.maximum(pred[mask], eps)
    gt_valid = np.maximum(gt[mask], eps)

    error = pred_valid - gt_valid
    abs_error = np.abs(error)

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(abs_error)
    absrel = np.mean(abs_error / gt_valid)

    # Threshold accuracy: percent of valid pixels where prediction and GT agree
    # within a multiplicative threshold.
    thresh = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    d1 = np.mean(thresh < 1.25)
    d2 = np.mean(thresh < 1.25 ** 2)
    d3 = np.mean(thresh < 1.25 ** 3)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "absrel": float(absrel),
        "d1": float(d1),
        "d2": float(d2),
        "d3": float(d3),
        "valid_pixels": valid_pixels,
    }


def summarize_metric_rows(rows, metric_keys=METRIC_KEYS):
    """
    Average per-sample metric dictionaries into one summary dictionary.

    Args:
        rows: list of dictionaries returned by compute_metrics, possibly with
              additional fields such as runtime.
        metric_keys: metric names to average.

    Returns:
        dict with mean metrics and total valid pixel count.
    """
    if len(rows) == 0:
        return {key: np.nan for key in metric_keys} | {"valid_pixels": 0}

    summary = {}
    for key in metric_keys:
        values = np.array([row[key] for row in rows if key in row], dtype=np.float64)
        values = values[np.isfinite(values)]
        summary[key] = float(values.mean()) if len(values) > 0 else np.nan

    summary["valid_pixels"] = int(sum(int(row.get("valid_pixels", 0)) for row in rows))
    return summary
