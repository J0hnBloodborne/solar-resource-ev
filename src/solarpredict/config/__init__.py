"""Configuration: runtime settings and the candidate-site catalog."""

from __future__ import annotations

from .settings import Settings, get_settings
from .sites import (
    CANDIDATE_CITIES,
    FOCUS_CITIES,
    INTRACITY_SITES,
    OPEN_METEO_HOURLY,
    SAFE_COVARIATES,
    Location,
    intracity_sites,
)

__all__ = [
    "CANDIDATE_CITIES",
    "FOCUS_CITIES",
    "INTRACITY_SITES",
    "OPEN_METEO_HOURLY",
    "SAFE_COVARIATES",
    "Location",
    "Settings",
    "get_settings",
    "intracity_sites",
]
