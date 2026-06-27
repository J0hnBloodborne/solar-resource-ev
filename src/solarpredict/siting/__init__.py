"""Part B: city selection, inter-city solar comparison, and site suitability."""

from __future__ import annotations

from .cities import REFERENCE_YIELDS, city_summary, monthly_ghi
from .seasonal import SEASON_ORDER, seasonal_site_suitability
from .selection import (
    rank_cities_by_intracity_spread,
    select_city,
    site_mean_ghi,
)
from .suitability import site_suitability

__all__ = [
    "REFERENCE_YIELDS",
    "SEASON_ORDER",
    "city_summary",
    "monthly_ghi",
    "rank_cities_by_intracity_spread",
    "seasonal_site_suitability",
    "select_city",
    "site_mean_ghi",
    "site_suitability",
]
