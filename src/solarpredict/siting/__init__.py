"""Part B: data-driven city selection, plus site sampling/suitability (M6).

Site sampling, the suitability index, and ranking land in M6; the city-selection
utility (which formalizes the Karachi pick) is here now.
"""

from __future__ import annotations

from .selection import (
    rank_cities_by_intracity_spread,
    select_city,
    site_mean_ghi,
)

__all__ = ["rank_cities_by_intracity_spread", "select_city", "site_mean_ghi"]
