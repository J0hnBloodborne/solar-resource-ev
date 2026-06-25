"""Candidate cities and the Open-Meteo variable set.

The city catalog seeds Part B's data-driven city selection (M1): we pull a few
candidates and pick the one with the most intra-/inter-site GHI spread. The four
S2Cool cities are kept; a few climatically distinct ones (Multan, Quetta, Gilgit)
are added so the selection has real variation to find.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    """A named geographic point."""

    name: str
    latitude: float
    longitude: float

    @property
    def slug(self) -> str:
        """Lowercase, filesystem-safe identifier."""
        return self.name.lower().replace(" ", "_")


# Candidate cities for the benchmark + data-driven city selection.
CANDIDATE_CITIES: tuple[Location, ...] = (
    Location("Islamabad", 33.6844, 73.0479),
    Location("Lahore", 31.5204, 74.3587),
    Location("Karachi", 24.8607, 67.0011),
    Location("Peshawar", 34.0151, 71.5249),
    Location("Multan", 30.1575, 71.5249),
    Location("Quetta", 30.1798, 66.9750),
    Location("Gilgit", 35.9208, 74.3083),
)

# Hourly variables requested from Open-Meteo. shortwave_radiation is GHI (W/m^2);
# the radiation components (direct/dni/diffuse) are NOT used as t+1 features
# (they are deterministic parts of the target — see leakage discipline in plan).
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
