"""Configuration: runtime settings and the candidate-site catalog."""

from __future__ import annotations

from .settings import Settings, get_settings
from .sites import (
    CANDIDATE_CITIES,
    CITY_BBOX,
    FOCUS_CITIES,
    INTRACITY_SITES,
    OPEN_METEO_HOURLY,
    SAFE_COVARIATES,
    BBox,
    Location,
    city_grid,
    intracity_sites,
)

__all__ = [
    "CANDIDATE_CITIES",
    "CITY_BBOX",
    "FOCUS_CITIES",
    "INTRACITY_SITES",
    "OPEN_METEO_HOURLY",
    "SAFE_COVARIATES",
    "BBox",
    "Location",
    "Settings",
    "city_grid",
    "get_settings",
    "intracity_sites",
]
