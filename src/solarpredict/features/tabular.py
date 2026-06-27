"""Supervised tabular features for the classical-ML tier (hour-ahead GHI).

Builds lag / rolling / cyclical-time / clear-sky features that are all available at
t-1 (or deterministic at t), so predicting y(t) from them is a leakage-free
one-hour-ahead forecast. Concurrent covariates are NOT used (only their lags) — the
radiation components in particular would leak the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarpredict.config.sites import SAFE_COVARIATES

GHI_LAGS: tuple[int, ...] = (1, 2, 3, 24)
COV_LAGS: tuple[int, ...] = (1, 24)
ROLL_WINDOWS: tuple[int, ...] = (3, 24)


def build_features(
    prepared: pd.DataFrame,
    *,
    target: str = "y",
    series_col: str = "unique_id",
    covariates: tuple[str, ...] = SAFE_COVARIATES,
    horizon: int = 1,
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(frame_with_features, feature_columns)``.

    Lag/rolling features are shifted so each row only sees information available at
    the forecast origin, i.e. ``horizon`` steps before t (horizon=1 is hour-ahead,
    24 is day-ahead); cyclical-time and clear-sky GHI are deterministic at t and
    stay usable at any horizon. Warm-up rows with NaN features are dropped.
    """
    df = prepared.sort_values("ds").reset_index(drop=True).copy()
    features: list[str] = []
    extra = horizon - 1  # push every autoregressive lag past the forecast gap

    for lag in GHI_LAGS:
        col = f"ghi_lag_{lag + extra}"
        df[col] = df.groupby(series_col)[target].shift(lag + extra)
        features.append(col)

    for cov in covariates:
        if cov not in df.columns:
            continue
        for lag in COV_LAGS:
            col = f"{cov}_lag_{lag + extra}"
            df[col] = df.groupby(series_col)[cov].shift(lag + extra)
            features.append(col)

    for window in ROLL_WINDOWS:
        mean_col, std_col = f"ghi_roll_mean_{window}", f"ghi_roll_std_{window}"
        grouped = df.groupby(series_col)[target]
        df[mean_col] = grouped.transform(
            lambda s, w=window: s.rolling(w).mean().shift(horizon)
        )
        df[std_col] = grouped.transform(
            lambda s, w=window: s.rolling(w).std().shift(horizon)
        )
        features += [mean_col, std_col]

    if "clear_sky_index" in df.columns:
        col = f"kt_lag_{horizon}"
        df[col] = df.groupby(series_col)["clear_sky_index"].shift(horizon)
        features.append(col)
    if "clearsky_ghi" in df.columns:  # deterministic at t — safe to use directly
        features.append("clearsky_ghi")

    ds = pd.DatetimeIndex(df["ds"])
    df["hour_sin"] = np.sin(2 * np.pi * ds.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ds.hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * ds.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * ds.dayofyear / 365.25)
    features += ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

    df = df.dropna(subset=features).reset_index(drop=True)
    return df, features
