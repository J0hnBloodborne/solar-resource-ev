"""Tests for GHI->power and within-city suitability (CI-safe)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from solarpredict.siting import site_suitability
from solarpredict.solar import (
    annual_energy_kwh_per_m2,
    cell_temperature,
    ghi_to_power_kw,
    ghi_to_power_kw_temp,
    peak_sun_hours,
)


def test_ghi_to_power_clips_and_scales() -> None:
    power = ghi_to_power_kw(
        np.array([-50.0, 1000.0]),
        panel_area_m2=10,
        efficiency=0.2,
        performance_ratio=0.8,
    )
    assert power[0] == 0.0  # negative GHI clipped
    assert abs(power[1] - 1000 * 10 * 0.2 * 0.8 / 1000) < 1e-9  # 1.6 kW


def test_cell_temperature_rises_with_ghi() -> None:
    # NOCT model: T_cell = T_air + (45-20)/800 * GHI; at 800 W/m^2 that is +25 degC.
    tcell = cell_temperature([0.0, 800.0], [25.0, 25.0])
    assert abs(tcell[0] - 25.0) < 1e-9
    assert abs(tcell[1] - 50.0) < 1e-9


def test_temp_correction_penalises_hot_sites() -> None:
    # Same GHI, hotter air -> hotter cell -> less power (negative temp coefficient).
    cool = ghi_to_power_kw_temp(800.0, 20.0)
    hot = ghi_to_power_kw_temp(800.0, 35.0)
    assert hot < cool
    # At 25 degC cell temperature the correction is a no-op vs the base model.
    base = ghi_to_power_kw(800.0)
    neutral = ghi_to_power_kw_temp(800.0, 25.0 - (45.0 - 20.0) / 800.0 * 800.0)
    assert abs(float(neutral) - float(base)) < 1e-9


def test_peak_sun_hours() -> None:
    assert abs(peak_sun_hours([1000.0] * 5) - 5.0) < 1e-9  # 5h at 1 sun


def test_annual_energy() -> None:
    assert (
        abs(annual_energy_kwh_per_m2(200.0, performance_ratio=0.8) - 200 * 8.766 * 0.8)
        < 1e-6
    )


class _FakeRepo:
    """GHI rises with latitude (north sunnier); temperature constant."""

    def fetch(self, location, start, end, *, force_refresh=False):
        ghi = 400.0 + (location.latitude - 24.8) * 100.0
        times = pd.date_range("2020-01-01", periods=90 * 24, freq="h", tz="UTC")
        return pd.DataFrame(
            {"timestamp": times, "shortwave_radiation": ghi, "temperature_2m": 26.0}
        )


def test_site_suitability_ranks_northern_site_first() -> None:
    ranked = site_suitability(
        _FakeRepo(), "Karachi", date(2020, 1, 1), date(2020, 4, 1)
    )
    assert ranked.iloc[0]["site"] == "Gadap"  # highest latitude -> highest yield
    assert {"suitability", "annual_kwh_m2"}.issubset(ranked.columns)
    assert ranked["suitability"].iloc[0] >= ranked["suitability"].iloc[-1]
