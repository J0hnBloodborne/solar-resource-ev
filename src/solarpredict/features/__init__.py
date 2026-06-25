"""Feature engineering: solar geometry, clear-sky index, and site preparation.

Lag/rolling/cyclical features for the classical-ML tier land in M2; this module
provides the clear-sky foundation the skill-score spine needs.
"""

from __future__ import annotations

from .clearsky import clear_sky_index, daytime_mask, solar_geometry
from .prepare import prepare_site

__all__ = ["clear_sky_index", "daytime_mask", "prepare_site", "solar_geometry"]
