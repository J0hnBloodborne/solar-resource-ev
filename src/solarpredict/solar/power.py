"""GHI -> PV power conversion and peak-sun-hours.

A deliberately simple, transparent PV model (linear in GHI with a performance
ratio), reused so the GHI gradient between sites reads as an annual-energy
difference for a real charging station.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# Default fixed PV system per site for the report's power-domain numbers.
DEFAULT_EFFICIENCY = 0.20  # module efficiency
DEFAULT_PERFORMANCE_RATIO = 0.80  # inverter/thermal/soiling losses
STC_IRRADIANCE = 1000.0  # W/m^2
NOCT_C = 45.0  # nominal operating cell temperature (typical c-Si module)
TEMP_COEFF_PER_C = -0.004  # power temperature coefficient, ~-0.4%/degC (c-Si)


def ghi_to_power_kw(
    ghi_wm2: ArrayLike,
    *,
    panel_area_m2: float = 1.0,
    efficiency: float = DEFAULT_EFFICIENCY,
    performance_ratio: float = DEFAULT_PERFORMANCE_RATIO,
) -> np.ndarray:
    """Instantaneous DC power (kW) from GHI for a flat panel of ``panel_area_m2``."""
    ghi = np.clip(np.asarray(ghi_wm2, dtype=float), 0.0, None)
    return ghi * panel_area_m2 * efficiency * performance_ratio / 1000.0


def cell_temperature(
    ghi_wm2: ArrayLike, air_temp_c: ArrayLike, *, noct_c: float = NOCT_C
) -> np.ndarray:
    """Panel cell temperature (degC) from GHI + air temperature (NOCT model)."""
    ghi = np.clip(np.asarray(ghi_wm2, dtype=float), 0.0, None)
    return np.asarray(air_temp_c, dtype=float) + (noct_c - 20.0) / 800.0 * ghi


def ghi_to_power_kw_temp(
    ghi_wm2: ArrayLike,
    air_temp_c: ArrayLike,
    *,
    panel_area_m2: float = 1.0,
    efficiency: float = DEFAULT_EFFICIENCY,
    performance_ratio: float = DEFAULT_PERFORMANCE_RATIO,
    temp_coeff: float = TEMP_COEFF_PER_C,
    noct_c: float = NOCT_C,
) -> np.ndarray:
    """Temperature-corrected DC power (kW): like ``ghi_to_power_kw`` but the module
    efficiency is derated for cell temperature above the 25 degC reference, so hot
    inland sites lose more than cool coastal ones for the same GHI."""
    base = ghi_to_power_kw(
        ghi_wm2,
        panel_area_m2=panel_area_m2,
        efficiency=efficiency,
        performance_ratio=performance_ratio,
    )
    tcell = cell_temperature(ghi_wm2, air_temp_c, noct_c=noct_c)
    derate = np.clip(1.0 + temp_coeff * (tcell - 25.0), 0.0, None)
    return base * derate


def annual_energy_kwh_per_m2(
    mean_ghi_wm2: float, *, performance_ratio: float = DEFAULT_PERFORMANCE_RATIO
) -> float:
    """Annual delivered energy (kWh/m^2/yr) for a 1 m^2 panel from mean GHI."""
    return float(mean_ghi_wm2) * 8.766 * performance_ratio


def peak_sun_hours(daily_ghi_wm2: ArrayLike) -> float:
    """Peak sun hours: daily GHI energy / 1000 (equivalent full-sun hours)."""
    total = float(np.sum(np.clip(np.asarray(daily_ghi_wm2, dtype=float), 0.0, None)))
    return total / STC_IRRADIANCE
