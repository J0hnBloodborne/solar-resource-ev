"""Candidate cities, intra-city named sites, and the Open-Meteo variable set.

For Part B we rank sites *within one large city*. ERA5(-Land) is ~9-25 km, so the
named districts below are deliberately spread across each metro to fall in
different grid cells (the "Karachi/Lahore are huge" bet). City auto-selection (M1)
measures which of the two actually yields usable intra-city GHI spread.

Coordinates are approximate district centroids — fine for which ERA5 cell a site
lands in; refine later if a sharper map is wanted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    """A named geographic point. ``city`` is set for intra-city sites."""

    name: str
    latitude: float
    longitude: float
    city: str | None = None

    @property
    def slug(self) -> str:
        """Lowercase, filesystem-safe identifier (unique across city+district)."""
        base = f"{self.city} {self.name}" if self.city else self.name
        return re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")

    @property
    def display(self) -> str:
        """Human label, e.g. ``Karachi · Clifton``."""
        return f"{self.city} · {self.name}" if self.city else self.name


# Candidate cities (centroids) — inter-city context + benchmark.
CANDIDATE_CITIES: tuple[Location, ...] = (
    Location("Islamabad", 33.6844, 73.0479),
    Location("Lahore", 31.5204, 74.3587),
    Location("Karachi", 24.8607, 67.0011),
    Location("Peshawar", 34.0151, 71.5249),
    Location("Multan", 30.1575, 71.5249),
    Location("Quetta", 30.1798, 66.9750),
    Location("Gilgit", 35.9208, 74.3083),
)

# Part B focuses on the two large metros (per user steer); the data check picks one.
FOCUS_CITIES: tuple[str, ...] = ("Karachi", "Lahore")

# Named real districts spread across each metro (different ERA5 cells where possible).
INTRACITY_SITES: dict[str, tuple[Location, ...]] = {
    "Karachi": (
        Location("Clifton", 24.810, 67.030, city="Karachi"),
        Location("DHA", 24.790, 67.070, city="Karachi"),
        Location("Korangi", 24.830, 67.135, city="Karachi"),
        Location("Malir", 24.893, 67.205, city="Karachi"),
        Location("Gulshan-e-Iqbal", 24.920, 67.100, city="Karachi"),
        Location("North Nazimabad", 24.950, 67.040, city="Karachi"),
        Location("SITE", 24.910, 66.990, city="Karachi"),
        Location("Gadap", 25.050, 67.110, city="Karachi"),
    ),
    "Lahore": (
        Location("Walled City", 31.585, 74.310, city="Lahore"),
        Location("Gulberg", 31.515, 74.350, city="Lahore"),
        Location("DHA", 31.470, 74.405, city="Lahore"),
        Location("Johar Town", 31.470, 74.270, city="Lahore"),
        Location("Shahdara", 31.625, 74.300, city="Lahore"),
        Location("Wagah", 31.605, 74.565, city="Lahore"),
        Location("Bahria Town", 31.365, 74.185, city="Lahore"),
        Location("Raiwind", 31.250, 74.210, city="Lahore"),
    ),
}


def intracity_sites(*cities: str) -> list[Location]:
    """Flat list of named sites for the given cities (default: all focus cities)."""
    keys = cities or FOCUS_CITIES
    sites: list[Location] = []
    for key in keys:
        sites.extend(INTRACITY_SITES.get(key, ()))
    return sites


# Hourly variables requested from Open-Meteo. shortwave_radiation is GHI (W/m^2);
# the radiation components (direct/dni/diffuse) are NOT used as t+1 features
# (deterministic parts of the target — see leakage discipline in the plan).
OPEN_METEO_HOURLY: tuple[str, ...] = (
    "shortwave_radiation",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "direct_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
)

# Covariates safe to use as predictors for hour-ahead GHI (lagged in features/).
SAFE_COVARIATES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
)
