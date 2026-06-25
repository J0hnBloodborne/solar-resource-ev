"""Tests for solar geometry, clear-sky index, and site preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarpredict.config.sites import Location
from solarpredict.features.clearsky import (
    clear_sky_index,
    daytime_mask,
    solar_geometry,
)
from solarpredict.features.prepare import prepare_site


def test_daytime_mask() -> None:
    mask = daytime_mask(np.array([10.0, 95.0, 89.9]))
    assert mask.tolist() == [True, False, True]


def test_clear_sky_index_bounds() -> None:
    ghi = np.array([0.0, 500.0, 1000.0, 2.0])
    cs = np.array([0.0, 500.0, 800.0, 100.0])
    kt = clear_sky_index(ghi, cs)
    assert kt[0] == 0.0  # clear-sky below floor -> 0
    assert abs(kt[1] - 1.0) < 1e-9
    assert kt[2] == 1.25  # 1000/800, under clip_max
    assert abs(kt[3] - 0.02) < 1e-9


def test_clear_sky_index_clips_high() -> None:
    assert clear_sky_index(np.array([2000.0]), np.array([800.0]))[0] == 1.5


def test_solar_geometry_day_night() -> None:
    times = pd.date_range("2023-06-15", periods=24, freq="h", tz="UTC")
    geo = solar_geometry(24.86, 67.01, times)  # Karachi
    assert geo["clearsky_ghi"].max() > 600  # midday clear-sky is high
    assert geo["clearsky_ghi"].min() >= 0
    assert (geo["clearsky_ghi"] < 1.0).any()  # some night hours ~0
    assert geo["zenith"].between(0, 180).all()


def test_prepare_site_adds_expected_columns() -> None:
    times = pd.date_range("2023-06-15", periods=48, freq="h", tz="UTC")
    ghi = np.clip(700 * np.sin((np.arange(48) % 24 - 6) * np.pi / 12), 0, None)
    raw = pd.DataFrame(
        {"timestamp": times, "shortwave_radiation": ghi, "temperature_2m": 30.0}
    )
    out = prepare_site(raw, Location("Karachi", 24.86, 67.01))
    for col in ("ds", "y", "zenith", "clearsky_ghi", "clear_sky_index", "is_day"):
        assert col in out.columns
    assert out["clear_sky_index"].between(0, 1.5).all()
    assert out["unique_id"].iloc[0] == "karachi"
    assert out["is_day"].dtype == bool
