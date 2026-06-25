"""Tests for the deep-learning tier.

Registration is checked always (CI-safe — no torch/neuralforecast needed to import
the module). The actual training path is exercised only when neuralforecast is
installed, so CI (which installs `.[dev]` only) skips it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solarpredict.models import list_forecasters


def test_dl_models_registered() -> None:
    assert {"nhits", "lstm"}.issubset(set(list_forecasters()))


def test_nhits_adapter_runs() -> None:
    pytest.importorskip("neuralforecast")
    from solarpredict.models import ForecastData
    from solarpredict.models.dl import NeuralForecastAdapter

    n = 24 * 40
    times = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    hour = times.hour.to_numpy()
    clearsky = np.clip(900 * np.sin((hour - 6) * np.pi / 12), 0, None)
    df = pd.DataFrame(
        {
            "unique_id": "s",
            "ds": times,
            "y": clearsky * 0.85,
            "clearsky_ghi": clearsky,
            "temperature_2m": 25.0,
        }
    )
    # disjoint + contiguous, so the concatenated series has no gaps/duplicates
    train = ForecastData(frame=df.iloc[: 24 * 25].copy())
    test = ForecastData(frame=df.iloc[24 * 25 :].copy())

    model = NeuralForecastAdapter("t", "NHITS", input_size=24, max_steps=5)
    preds = model.fit(train).predict(test)

    assert len(preds) == len(test.frame)
    assert np.isfinite(preds).all()
    assert (preds >= 0).all()
