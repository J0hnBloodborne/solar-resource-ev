"""Forecast error metrics for GHI prediction.

All metrics accept an optional boolean ``mask`` so callers can restrict scoring to
daytime hours (night zeros otherwise inflate R^2 toward 0.99 and hide real skill).
The headline metric is the forecast :func:`skill_score` vs. smart persistence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _prep(
    y_true: ArrayLike, y_pred: ArrayLike, mask: ArrayLike | None
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to float arrays, drop non-finite pairs, and apply an optional mask."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    keep = np.isfinite(yt) & np.isfinite(yp)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    if not keep.any():
        raise ValueError("no valid (finite, unmasked) samples to score")
    return yt[keep], yp[keep]


def mae(y_true: ArrayLike, y_pred: ArrayLike, mask: ArrayLike | None = None) -> float:
    """Mean absolute error."""
    yt, yp = _prep(y_true, y_pred, mask)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike, mask: ArrayLike | None = None) -> float:
    """Root mean squared error."""
    yt, yp = _prep(y_true, y_pred, mask)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mbe(y_true: ArrayLike, y_pred: ArrayLike, mask: ArrayLike | None = None) -> float:
    """Mean bias error (positive = over-forecasting)."""
    yt, yp = _prep(y_true, y_pred, mask)
    return float(np.mean(yp - yt))


def r2(y_true: ArrayLike, y_pred: ArrayLike, mask: ArrayLike | None = None) -> float:
    """Coefficient of determination."""
    yt, yp = _prep(y_true, y_pred, mask)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def nrmse(y_true: ArrayLike, y_pred: ArrayLike, mask: ArrayLike | None = None) -> float:
    """RMSE normalized by the mean of (masked) observed values, as a fraction."""
    yt, yp = _prep(y_true, y_pred, mask)
    denom = float(np.mean(yt))
    if denom == 0.0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)) / denom)


def skill_score(rmse_model: float, rmse_reference: float) -> float:
    """Forecast skill score ``S = 1 - RMSE_model / RMSE_reference``.

    Reference should be *smart/clear-sky persistence* (not plain persistence).
    ``S > 0`` means the model beats the reference.
    """
    if rmse_reference == 0.0:
        return float("nan")
    return 1.0 - rmse_model / rmse_reference
