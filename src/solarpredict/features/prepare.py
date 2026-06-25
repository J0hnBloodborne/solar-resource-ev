"""Turn a site's raw hourly frame into the canonical modeling frame.

Adds solar geometry (zenith, clear-sky GHI), the clear-sky index, and a daytime
mask, then renames to the Nixtla long-format columns (``unique_id``/``ds``/``y``)
that every Forecaster and the harness expect.
"""

from __future__ import annotations

import pandas as pd

from solarpredict.config.sites import Location

from .clearsky import clear_sky_index, daytime_mask, solar_geometry


def prepare_site(
    raw: pd.DataFrame,
    location: Location,
    *,
    target: str = "shortwave_radiation",
    timestamp_col: str = "timestamp",
    altitude: float = 0.0,
) -> pd.DataFrame:
    """Return a modeling frame: ds/y + zenith, clearsky_ghi, clear_sky_index, is_day."""
    df = raw.sort_values(timestamp_col).reset_index(drop=True).copy()
    times = pd.DatetimeIndex(pd.to_datetime(df[timestamp_col], utc=True))

    geo = solar_geometry(
        location.latitude, location.longitude, times, altitude=altitude
    )
    df["zenith"] = geo["zenith"].to_numpy()
    df["clearsky_ghi"] = geo["clearsky_ghi"].to_numpy()
    df["clear_sky_index"] = clear_sky_index(
        df[target].to_numpy(dtype=float), df["clearsky_ghi"].to_numpy()
    )
    df["is_day"] = daytime_mask(df["zenith"].to_numpy())

    df = df.rename(columns={timestamp_col: "ds", target: "y"})
    df["unique_id"] = location.slug
    return df
