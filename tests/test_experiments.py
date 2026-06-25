"""Test the data-efficiency experiment (baseline only -> CI-safe)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarpredict.evaluation import data_efficiency


def _prepared(n_days: int = 400) -> pd.DataFrame:
    times = pd.date_range("2022-01-01", periods=n_days * 24, freq="h", tz="UTC")
    hour = times.hour.to_numpy()
    clearsky = np.clip(900 * np.sin((hour - 6) * np.pi / 12), 0, None)
    y = clearsky * (0.85 + 0.1 * np.sin(np.arange(len(times)) / 13))
    return pd.DataFrame(
        {
            "unique_id": "s",
            "ds": times,
            "y": y,
            "clearsky_ghi": clearsky,
            "is_day": clearsky > 5,
        }
    )


def test_data_efficiency_varies_train_size_with_fixed_test() -> None:
    out = data_efficiency(
        _prepared(), ["smart_persistence"], [0.1, 0.3], test_rows=24 * 30
    )
    assert set(out["train_years"].unique()) == {0.1, 0.3}
    assert (out["train_rows"] > 0).all()
    # bigger training window -> more rows
    rows_by_size = out.groupby("train_years")["train_rows"].first()
    assert rows_by_size[0.3] > rows_by_size[0.1]
    assert {"model", "skill", "rmse"}.issubset(out.columns)
