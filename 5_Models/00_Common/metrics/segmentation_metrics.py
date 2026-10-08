"""
Standardized Segmentation Metric Suite for BUSI Phase-1 Model Comparison.
Provides exact mathematical implementations for Dice, IoU, Precision, Recall, Specificity, HD95, and NSD @ 1,2,3 px.
Handles edge cases (empty masks, single empty set) explicitly and deterministically.
"""

from typing import Dict, List, Union, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import torch


def _to_numpy_binary(mask: Union[np.ndarray, torch.Tensor], threshold: float = 0.5) -> np.ndarray:
    """Helper to convert input array/tensor into 2D boolean numpy array."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask after squeezing, got shape {mask.shape}")
    return (mask > threshold).astype(bool)


def compute_pixel_confusion_matrix(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Computes confusion matrix elements (TP, FP, FN, TN) between binary pred and gt.
    
    Returns:
        Tuple[int, int, int, int]: (TP, FP, FN, TN)
    """
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return tp, fp, fn, tn


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Computes Dice Similarity Coefficient (DSC).
    Formula: 2 * TP / (2 * TP + FP + FN)
    """
    tp, fp, fn, _ = compute_pixel_confusion_matrix(pred, gt)
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 1.0  # Both prediction and GT are empty
    return (2.0 * tp) / float(denominator)


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU / Jaccard Index).
    Formula: TP / (TP + FP + FN)
    """
    tp, fp, fn, _ = compute_pixel_confusion_matrix(pred, gt)
    denominator = tp + fp + fn
    if denominator == 0:
        return 1.0  # Both prediction and GT are empty
    return float(tp) / float(denominator)


def compute_precision(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Computes Pixel Precision (Positive Predictive Value).
    Formula: TP / (TP + FP)
    """
    tp, fp, _, _ = compute_pixel_confusion_matrix(pred, gt)
    if tp + fp == 0:
        # No positive predictions made
        return 1.0 if not gt.any() else 0.0
    return float(tp) / float(tp + fp)


def compute_recall(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Computes Pixel Recall / Sensitivity (True Positive Rate).
    Formula: TP / (TP + FN)
    """
    tp, _, fn, _ = compute_pixel_confusion_matrix(pred, gt)
    if tp + fn == 0:
        # No ground-truth lesion present
        return 1.0 if not pred.any() else 0.0
    return float(tp) / float(tp + fn)


def compute_specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Computes Pixel Specificity (True Negative Rate).
    Formula: TN / (TN + FP)
    """
    _, fp, _, tn = compute_pixel_confusion_matrix(pred, gt)
    if tn + fp == 0:
        return 1.0
    return float(tn) / float(tn + fp)


def extract_surface_boundaries(mask: np.ndarray) -> np.ndarray:
    """
    Extracts 1-pixel thick boundary contours of a 2D binary mask.
    Uses binary erosion: boundary = mask AND NOT erosion(mask).
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
    boundary = np.logical_and(mask, np.logical_not(eroded))
    return boundary


def compute_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Computes 95th Percentile Hausdorff Distance (HD95) in pixels.
    
    Edge Case Handling:
    - If both Pred and GT are empty: HD95 = 0.0 (perfect match of empty sets).
    - If one is empty and the other is non-empty: Returns np.nan (invalid surface comparison).
      The caller handles/logs NaNs during aggregation.
    """
    pred_has = pred.any()
    gt_has = gt.any()

    if not pred_has and not gt_has:
        return 0.0
    if not pred_has or not gt_has:
        return np.nan

    # Extract boundary point surface contours
    surf_pred = extract_surface_boundaries(pred)
    surf_gt = extract_surface_boundaries(gt)

    if not surf_pred.any() or not surf_gt.any():
        # Handle 1-pixel tiny regions where boundary extraction yields 0 pixels
        surf_pred = pred
        surf_gt = gt

    # Exact Euclidean distance transform of background (~surf)
    # dist_map_gt[p] gives distance from p to nearest point in surf_gt
    dist_map_gt = ndimage.distance_transform_edt(~surf_gt)
    dist_map_pred = ndimage.distance_transform_edt(~surf_pred)

    distances_pred_to_gt = dist_map_gt[surf_pred]
    distances_gt_to_pred = dist_map_pred[surf_gt]

    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])
    if len(all_distances) == 0:
        return 0.0

    return float(np.percentile(all_distances, 95))


def compute_nsd(pred: np.ndarray, gt: np.ndarray, tolerance_px: float = 1.0) -> float:
    """
    Computes Normalized Surface Distance (NSD) at a specified pixel tolerance tau.
    Formula: (|{p in S_p: d(p, S_g) <= tau}| + |{q in S_g: d(q, S_p) <= tau}|) / (|S_p| + |S_g|)
    
    Edge Case Handling:
    - Both empty: NSD = 1.0
    - One empty: NSD = 0.0
    """
    pred_has = pred.any()
    gt_has = gt.any()

    if not pred_has and not gt_has:
        return 1.0
    if not pred_has or not gt_has:
        return 0.0

    surf_pred = extract_surface_boundaries(pred)
    surf_gt = extract_surface_boundaries(gt)

    if not surf_pred.any() or not surf_gt.any():
        surf_pred = pred
        surf_gt = gt

    n_pred_points = surf_pred.sum()
    n_gt_points = surf_gt.sum()

    if n_pred_points + n_gt_points == 0:
        return 1.0

    dist_map_gt = ndimage.distance_transform_edt(~surf_gt)
    dist_map_pred = ndimage.distance_transform_edt(~surf_pred)

    pred_to_gt_within_tau = (dist_map_gt[surf_pred] <= tolerance_px).sum()
    gt_to_pred_within_tau = (dist_map_pred[surf_gt] <= tolerance_px).sum()

    nsd_val = float(pred_to_gt_within_tau + gt_to_pred_within_tau) / float(n_pred_points + n_gt_points)
    return nsd_val


def calculate_sample_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    gt: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    sample_id: Optional[str] = None
) -> Dict[str, Union[float, str]]:
    """
    Calculates all required metrics for a single image-mask sample.
    
    Returns dict containing:
        - dice, iou, precision, recall, specificity
        - hd95
        - nsd_1px, nsd_2px, nsd_3px
        - sample_id (if provided)
    """
    pred_b = _to_numpy_binary(pred, threshold=threshold)
    gt_b = _to_numpy_binary(gt, threshold=threshold)

    res = {
        "dice": compute_dice(pred_b, gt_b),
        "iou": compute_iou(pred_b, gt_b),
        "precision": compute_precision(pred_b, gt_b),
        "recall": compute_recall(pred_b, gt_b),
        "specificity": compute_specificity(pred_b, gt_b),
        "hd95": compute_hd95(pred_b, gt_b),
        "nsd_1px": compute_nsd(pred_b, gt_b, tolerance_px=1.0),
        "nsd_2px": compute_nsd(pred_b, gt_b, tolerance_px=2.0),
        "nsd_3px": compute_nsd(pred_b, gt_b, tolerance_px=3.0),
    }

    if sample_id is not None:
        res["sample_id"] = sample_id

    return res


def calculate_batch_metrics(
    preds: Union[np.ndarray, torch.Tensor],
    gts: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    sample_ids: Optional[List[str]] = None
) -> List[Dict[str, Union[float, str]]]:
    """
    Calculates per-sample metrics across a batch of predictions and ground-truth masks.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(gts, torch.Tensor):
        gts = gts.detach().cpu().numpy()

    batch_size = preds.shape[0]
    metrics_list = []

    for i in range(batch_size):
        sid = sample_ids[i] if sample_ids and i < len(sample_ids) else f"sample_{i}"
        sample_res = calculate_sample_metrics(
            pred=preds[i],
            gt=gts[i],
            threshold=threshold,
            sample_id=sid
        )
        metrics_list.append(sample_res)

    return metrics_list


def aggregate_metrics(metrics_list: List[Dict[str, Union[float, str]]]) -> Dict[str, float]:
    """
    Aggregates per-sample metrics into cohort summary stats (mean and std).
    Preserves all valid HD95 entries using nanmean/nanstd and records valid_hd95_count.
    
    Returns:
        Dict[str, float]: Aggregated dictionary with mean, std, and valid count keys.
    """
    if not metrics_list:
        return {}

    df = pd.DataFrame(metrics_list)
    metric_keys = [
        "dice", "iou", "precision", "recall", "specificity",
        "hd95", "nsd_1px", "nsd_2px", "nsd_3px"
    ]

    aggregated = {}
    for key in metric_keys:
        if key in df.columns:
            vals = pd.to_numeric(df[key], errors="coerce").values
            aggregated[f"{key}_mean"] = float(np.nanmean(vals))
            aggregated[f"{key}_std"] = float(np.nanstd(vals))
            if key == "hd95":
                aggregated["hd95_valid_count"] = int(np.sum(~np.isnan(vals)))

    return aggregated
