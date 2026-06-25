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


def annual_energy_kwh_per_m2(
    mean_ghi_wm2: float, *, performance_ratio: float = DEFAULT_PERFORMANCE_RATIO
) -> float:
    """Annual delivered energy (kWh/m^2/yr) for a 1 m^2 panel from mean GHI."""
    return float(mean_ghi_wm2) * 8.766 * performance_ratio


def peak_sun_hours(daily_ghi_wm2: ArrayLike) -> float:
    """Peak sun hours: daily GHI energy / 1000 (equivalent full-sun hours)."""
    total = float(np.sum(np.clip(np.asarray(daily_ghi_wm2, dtype=float), 0.0, None)))
    return total / STC_IRRADIANCE
