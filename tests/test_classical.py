"""Tests for tabular features + the classical adapter (sklearn-only, CI-safe).

xgboost/lightgbm/catboost are only *instantiated* in the real benchmark run, not
here, so these tests pass without the ``classical`` extra installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarpredict.evaluation import evaluate
from solarpredict.features import build_features


def _synthetic_prepared(n_days: int = 30) -> pd.DataFrame:
    times = pd.date_range("2023-01-01", periods=n_days * 24, freq="h", tz="UTC")
    hour = times.hour.to_numpy()
    clearsky = np.clip(900 * np.sin((hour - 6) * np.pi / 12), 0, None)
    kt = 0.85 + 0.1 * np.sin(np.arange(len(times)) / 11.0)
    y = clearsky * kt
    csi = np.clip(
        np.where(clearsky > 5, y / np.where(clearsky > 5, clearsky, 1), 0), 0, 1.5
    )
    return pd.DataFrame(
        {
            "unique_id": "s",
            "ds": times,
            "y": y,
            "clearsky_ghi": clearsky,
            "clear_sky_index": csi,
            "is_day": clearsky > 5,
            "temperature_2m": 20 + 5 * np.sin(hour * np.pi / 12),
        }
    )


def test_build_features_adds_lags_and_drops_warmup() -> None:
    df = _synthetic_prepared()
    feat, cols = build_features(df)
    for expected in ("ghi_lag_1", "ghi_lag_24", "hour_sin", "kt_lag_1", "clearsky_ghi"):
        assert expected in cols
    assert "temperature_2m_lag_1" in cols  # covariate lag
    assert feat[cols].notna().to_numpy().all()  # no NaN after warmup drop
    assert len(feat) < len(df)  # warmup rows dropped


def test_classical_models_run_and_score() -> None:
    feat, cols = build_features(_synthetic_prepared())
    res = evaluate(
        feat,
        ["smart_persistence", "ridge", "random_forest"],
        covariate_cols=tuple(cols),
    )
    assert set(res["model"]) == {"smart_persistence", "ridge", "random_forest"}
    assert (res["rmse"] > 0).all()
    assert res.loc[res["model"] == "ridge", "tier"].iloc[0] == "tier-1"
