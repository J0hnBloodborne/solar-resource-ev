"""Part B: city selection, inter-city solar comparison, and site suitability."""

from __future__ import annotations

from .cities import REFERENCE_YIELDS, city_summary, monthly_ghi
from .selection import (
    rank_cities_by_intracity_spread,
    select_city,
    site_mean_ghi,
)
from .suitability import site_suitability

__all__ = [
    "REFERENCE_YIELDS",
    "city_summary",
    "monthly_ghi",
    "rank_cities_by_intracity_spread",
    "select_city",
    "site_mean_ghi",
    "site_suitability",
]
