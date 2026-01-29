"""Metric helpers for segmentation evaluation."""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from .timing_utils import time_end, time_start, DEBUG_TIMING, DEBUG_TIMING_VERBOSE

logger = logging.getLogger(__name__)

def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Compute precision/recall/IoU/F1 and confusion counts for binary masks.

    Args:
        pred_mask (np.ndarray): Predicted binary mask.
        gt_mask (np.ndarray): Ground-truth binary mask.

    Returns:
        dict: Metrics and confusion counts.

    Examples:
        >>> import numpy as np
        >>> pred = np.array([[1, 0], [1, 0]])
        >>> gt = np.array([[1, 0], [0, 0]])
        >>> metrics = compute_metrics(pred, gt)
        >>> (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"])
        (1, 1, 0, 2)
    """
    t0 = time.perf_counter() if DEBUG_TIMING and DEBUG_TIMING_VERBOSE else None
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    metrics = {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "f1": float(f1),
    }
    if DEBUG_TIMING and DEBUG_TIMING_VERBOSE:
        time_end("compute_metrics", t0)
    return metrics


def compute_metrics_batch_gpu(score_map: np.ndarray,
                              thresholds: list[float],
                              sh_mask: np.ndarray | None,
                              gt_mask: np.ndarray,
                              device: torch.device,
                              batch_size: int = 8) -> list[dict]:
    """Evaluate many thresholds in parallel on GPU; returns list of metric dicts.

    Args:
        score_map (np.ndarray): Score map.
        thresholds (list[float]): Thresholds to evaluate.
        sh_mask (np.ndarray | None): Optional SH buffer mask.
        gt_mask (np.ndarray): Ground-truth mask.
        device (torch.device): Torch device.
        batch_size (int): Threshold batch size.

    Returns:
        list[dict]: Metric dictionaries per threshold.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> score = np.zeros((2, 2), dtype=np.float32)
        >>> gt = np.zeros((2, 2), dtype=np.uint8)
        >>> _ = compute_metrics_batch_gpu(score, [0.5], None, gt, device=torch.device("cpu"))  # doctest: +SKIP
    """
    t0 = time_start()
    score_t = torch.from_numpy(score_map.astype(np.float32)).to(device).flatten()
    gt_t = torch.from_numpy(gt_mask.astype(np.bool_)).to(device).flatten()
    sh_t = torch.from_numpy(sh_mask.astype(np.bool_)).to(device).flatten() if sh_mask is not None else None
    metrics = []
    eps = 1e-8
    for start in range(0, len(thresholds), batch_size):
        thr_chunk = thresholds[start:start + batch_size]
        thr_t = torch.tensor(thr_chunk, device=device, dtype=torch.float32).view(-1, 1)
        mask_thr = score_t.unsqueeze(0) >= thr_t
        if sh_t is not None:
            mask_thr = mask_thr & sh_t
        tp = (mask_thr & gt_t).sum(dim=1).float()
        fp = (mask_thr & (~gt_t)).sum(dim=1).float()
        fn = ((~mask_thr) & gt_t).sum(dim=1).float()
        tn = ((~mask_thr) & (~gt_t)).sum(dim=1).float()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        for i, thr in enumerate(thr_chunk):
            metrics.append({
                "threshold": float(thr),
                "tp": int(tp[i].item()),
                "fp": int(fp[i].item()),
                "fn": int(fn[i].item()),
                "tn": int(tn[i].item()),
                "precision": float(precision[i].item()),
                "recall": float(recall[i].item()),
                "iou": float(iou[i].item()),
                "f1": float(f1[i].item()),
            })
    time_end("compute_metrics_batch_gpu", t0)
    return metrics


def compute_metrics_batch_cpu(score_map: np.ndarray,
                              thresholds: list[float],
                              sh_mask: np.ndarray | None,
                              gt_mask: np.ndarray,
                              batch_size: int = 16) -> list[dict]:
    """CPU fallback for batched threshold evaluation.

    Args:
        score_map (np.ndarray): Score map.
        thresholds (list[float]): Thresholds to evaluate.
        sh_mask (np.ndarray | None): Optional SH buffer mask.
        gt_mask (np.ndarray): Ground-truth mask.
        batch_size (int): Threshold batch size.

    Returns:
        list[dict]: Metric dictionaries per threshold.

    Examples:
        >>> import numpy as np
        >>> score = np.array([[0.2, 0.8], [0.6, 0.1]])
        >>> gt = np.array([[0, 1], [1, 0]])
        >>> metrics = compute_metrics_batch_cpu(score, [0.5], None, gt, batch_size=1)
        >>> round(metrics[0]["iou"], 3)
        1.0
    """
    t0 = time_start()
    flat_scores = score_map.astype(np.float32).reshape(1, -1)
    flat_gt = gt_mask.astype(bool).reshape(1, -1)
    flat_sh = sh_mask.astype(bool).reshape(1, -1) if sh_mask is not None else None
    metrics = []
    eps = 1e-8
    for start in range(0, len(thresholds), batch_size):
        thr_chunk = np.array(thresholds[start:start + batch_size], dtype=np.float32).reshape(-1, 1)
        mask = flat_scores >= thr_chunk
        if flat_sh is not None:
            mask = np.logical_and(mask, flat_sh)
        tp = np.logical_and(mask, flat_gt).sum(axis=1).astype(np.float64)
        fp = np.logical_and(mask, ~flat_gt).sum(axis=1).astype(np.float64)
        fn = np.logical_and(~mask, flat_gt).sum(axis=1).astype(np.float64)
        tn = np.logical_and(~mask, ~flat_gt).sum(axis=1).astype(np.float64)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        for i, thr in enumerate(thr_chunk[:, 0]):
            metrics.append({
                "threshold": float(thr),
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "tn": int(tn[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "iou": float(iou[i]),
                "f1": float(f1[i]),
            })
    time_end("compute_metrics_batch_cpu", t0)
    return metrics


def compute_oracle_upper_bound(gt_mask: np.ndarray,
                               sh_mask: np.ndarray) -> dict:
    """Compute oracle IoU if predictions are clipped to SH buffer.

    Args:
        gt_mask (np.ndarray): Ground-truth mask.
        sh_mask (np.ndarray): SH buffer mask.

    Returns:
        dict: Metrics for the oracle mask.

    Examples:
        >>> import numpy as np
        >>> gt = np.array([[1, 0], [1, 0]])
        >>> sh = np.array([[1, 1], [0, 0]])
        >>> metrics = compute_oracle_upper_bound(gt, sh)
        >>> (metrics["tp"], metrics["fn"])
        (1, 1)
    """
    t0 = time_start()
    oracle_mask = np.logical_and(gt_mask.astype(bool), sh_mask.astype(bool))
    metrics = compute_metrics(oracle_mask, gt_mask)
    logger.info(
        "oracle SH buffer upper bound -> IoU=%.3f, F1=%.3f, P=%.3f, R=%.3f",
        metrics["iou"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
    )
    time_end("oracle_upper_bound_SH_buffer", t0)
    return metrics
