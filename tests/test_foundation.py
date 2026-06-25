"""Tests for the foundation-model tier (Chronos).

Registration is always checked (CI-safe). The zero-shot inference path runs only
when chronos is installed (Bolt-tiny is tiny and cached), so CI skips it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solarpredict.models import list_forecasters


def test_chronos_models_registered() -> None:
    expected = {"chronos2", "chronos_bolt_small", "chronos_bolt_tiny"}
    assert expected.issubset(set(list_forecasters()))


def test_chronos_bolt_tiny_runs() -> None:
    pytest.importorskip("chronos")
    from solarpredict.models import ForecastData, get_forecaster

    times = pd.date_range("2023-01-01", periods=200, freq="h", tz="UTC")
    hour = times.hour.to_numpy()
    clearsky = np.clip(900 * np.sin((hour - 6) * np.pi / 12), 0, None)
    df = pd.DataFrame({"unique_id": "s", "ds": times, "y": clearsky * 0.85})

    model = get_forecaster("chronos_bolt_tiny")
    preds = model.fit(ForecastData(frame=df)).predict(ForecastData(frame=df))

    assert len(preds) == len(df)
    assert np.isfinite(preds).all()
    assert (preds >= 0).all()
