"""Solar geometry + clear-sky GHI (pvlib) for night masking and the clear-sky index.

The clear-sky GHI uses the Haurwitz model (zenith-only, no turbidity data file
needed). It underpins three things the benchmark needs: a daytime mask (night
zeros otherwise inflate R^2), the clear-sky index ``kt = GHI / GHI_cs`` (the
stationary quantity the statistical models forecast), and the smart-persistence
skill-score reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def solar_geometry(
    latitude: float,
    longitude: float,
    times: pd.DatetimeIndex,
    *,
    altitude: float = 0.0,
) -> pd.DataFrame:
    """Return apparent zenith (deg) and Haurwitz clear-sky GHI (W/m^2) for ``times``.

    ``times`` must be a tz-aware (UTC) DatetimeIndex.
    """
    import pvlib  # lazy: keeps the pure-numpy helpers import-light

    loc = pvlib.location.Location(latitude, longitude, altitude=altitude)
    solpos = loc.get_solarposition(times)
    clearsky = loc.get_clearsky(times, model="haurwitz")
    return pd.DataFrame(
        {
            "zenith": solpos["apparent_zenith"].to_numpy(),
            "clearsky_ghi": clearsky["ghi"].to_numpy(),
        },
        index=times,
    )


def daytime_mask(zenith: np.ndarray, *, max_zenith: float = 90.0) -> np.ndarray:
    """Boolean mask: True where the sun is above ``max_zenith`` (daytime)."""
    return np.asarray(zenith, dtype=float) < max_zenith


def clear_sky_index(
    ghi: np.ndarray,
    clearsky_ghi: np.ndarray,
    *,
    floor: float = 5.0,
    clip_max: float = 1.5,
) -> np.ndarray:
    """Clear-sky index ``kt = GHI / GHI_cs``, 0 at night, clipped to [0, clip_max].

    ``floor`` (W/m^2) guards the near-sunrise/sunset divide-by-(near-)zero.
    """
    ghi_arr = np.asarray(ghi, dtype=float)
    cs_arr = np.asarray(clearsky_ghi, dtype=float)
    safe_cs = np.where(cs_arr > floor, cs_arr, 1.0)
    kt = np.where(cs_arr > floor, ghi_arr / safe_cs, 0.0)
    return np.clip(kt, 0.0, clip_max)
