"""Configuration: runtime settings and the candidate-site catalog."""

from __future__ import annotations

from .settings import Settings, get_settings
from .sites import CANDIDATE_CITIES, Location

__all__ = ["CANDIDATE_CITIES", "Location", "Settings", "get_settings"]
