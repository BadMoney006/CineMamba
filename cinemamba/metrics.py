"""CMRxRecon 2023 official evaluation metrics — shared by train.py validate()
and the inference scripts, so training-time and inference-time numbers use
the SAME protocol and are directly comparable to PromptMR / the leaderboard.

Copied verbatim (semantics) from the official PromptMR repo:
  hellopipu/PromptMR : promptmr_examples/cmrxrecon/utils.py

Key protocol details that make PromptMR's PSNR high (~45 dB) and why our
earlier per-frame full-image numbers (~35 dB) looked much lower:
  1. Metrics are computed on the ENTIRE 4D volume (t, z/slice, x, y) at once,
     not per-frame. maxval = global max over the whole volume.
  2. A center crop to (H/3, W/2) is applied before metrics (cardiac ROI).
  3. PSNR uses data_range = gt.max() over the cropped volume.

Use compute_cmrx_metrics(gt_vol, pred_vol) where both are real magnitude
arrays shaped (T, H, W) or (T, Z, H, W) for a single (subject, view, slice,
acc) unit. The caller is responsible for grouping frames into volumes.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# -----------------------------------------------------------------------------
# Metrics — exact PromptMR semantics
# -----------------------------------------------------------------------------


def nmse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Normalized Mean Squared Error."""
    return float(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> float:
    """PSNR with data_range = gt.max() (global over the whole array)."""
    if maxval is None:
        maxval = gt.max()
    return float(peak_signal_noise_ratio(gt, pred, data_range=maxval))


def ssim_3d(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> float:
    """Mean per-slice SSIM over a 3D array (slice, x, y)."""
    assert gt.ndim == 3 and gt.shape == pred.shape
    maxval = gt.max() if maxval is None else maxval
    acc = 0.0
    for i in range(gt.shape[0]):
        acc += structural_similarity(gt[i], pred[i], data_range=maxval)
    return acc / gt.shape[0]


def ssim_4d(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> float:
    """Mean SSIM over a 4D array (t, slice, x, y) — PromptMR's ssim_4d."""
    assert gt.ndim == 4 and gt.shape == pred.shape
    maxval = gt.max() if maxval is None else maxval
    acc = 0.0
    for t in range(gt.shape[0]):
        acc += ssim_3d(gt[t], pred[t], maxval=maxval)
    return acc / gt.shape[0]


# -----------------------------------------------------------------------------
# Center crop — exact PromptMR crop_submission semantics for the cine task
# -----------------------------------------------------------------------------


def matlab_round(n: float) -> int:
    return int(n + 0.5) if n > 0 else int(n - 0.5)


def _crop_last2(a: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    """Center-crop the last two dims (H, W) of an array using MATLAB-style
    floor(dim/2) + ceil(±crop/2) bounds, matching PromptMR's _crop."""
    h, w = a.shape[-2], a.shape[-1]
    h0 = math.floor(h / 2) + math.ceil(-crop_h / 2)
    h1 = math.floor(h / 2) + math.ceil(crop_h / 2)
    w0 = math.floor(w / 2) + math.ceil(-crop_w / 2)
    w1 = math.floor(w / 2) + math.ceil(crop_w / 2)
    return a[..., h0:h1, w0:w1]


def crop_for_cmrx_cine(a: np.ndarray) -> np.ndarray:
    """Center crop to (H/3, W/2) — the CMRxRecon cine evaluation ROI.
    Works on arrays whose last two dims are (H, W)."""
    crop_h = matlab_round(a.shape[-2] / 3)
    crop_w = matlab_round(a.shape[-1] / 2)
    return _crop_last2(a, crop_h, crop_w)


# -----------------------------------------------------------------------------
# Volume-level metric bundle
# -----------------------------------------------------------------------------


def compute_cmrx_metrics(gt_vol: np.ndarray, pred_vol: np.ndarray, do_crop: bool = True) -> dict:
    """Compute PromptMR-style metrics on a single volume.

    Args:
        gt_vol, pred_vol: real magnitude arrays, shape (T, H, W) or (T, Z, H, W).
        do_crop: apply the CMRxRecon center crop (H/3 x W/2) before metrics.

    Returns dict with psnr, ssim, nmse. maxval is the global max of the
    (cropped) gt volume — the whole-volume convention that gives PromptMR's
    high PSNR numbers.
    """
    gt = gt_vol
    pred = pred_vol
    if do_crop:
        gt = crop_for_cmrx_cine(gt)
        pred = crop_for_cmrx_cine(pred)

    # Ensure 4D (t, slice, x, y) for ssim_4d. If (T, H, W), add a slice axis.
    if gt.ndim == 3:
        gt4 = gt[:, None, :, :]
        pred4 = pred[:, None, :, :]
    elif gt.ndim == 4:
        gt4 = gt
        pred4 = pred
    else:
        raise ValueError(f"expected 3D or 4D volume, got {gt.ndim}D")

    maxval = float(gt4.max())
    if maxval <= 0:
        maxval = 1.0

    return {
        "psnr": psnr(gt4, pred4, maxval=maxval),
        "ssim": ssim_4d(gt4, pred4, maxval=maxval),
        "nmse": nmse(gt4, pred4),
    }
