"""Data-driven Part B city selection.

Ranks the focus cities by their *intra-city* daytime-GHI spread across the named
districts and picks the most variable one (the site with the strongest siting
signal). Empirically this is Karachi; this module makes that choice reproducible.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from solarpredict.config.sites import FOCUS_CITIES, INTRACITY_SITES, Location
from solarpredict.data.repository import DataRepository


def site_mean_ghi(
    repo: DataRepository,
    sites: list[Location],
    start: date,
    end: date,
    *,
    target: str = "shortwave_radiation",
) -> pd.DataFrame:
    """Per-site mean GHI (all hours and daytime-only) for the given sites."""
    rows = []
    for site in sites:
        ghi = repo.fetch(site, start, end)[target].astype(float)
        daytime = ghi[ghi > 0]
        rows.append(
            {
                "city": site.city,
                "site": site.name,
                "mean_ghi": float(ghi.mean()),
                "mean_ghi_day": float(daytime.mean()),
            }
        )
    return pd.DataFrame(rows)


def rank_cities_by_intracity_spread(
    repo: DataRepository,
    start: date,
    end: date,
    *,
    cities: tuple[str, ...] = FOCUS_CITIES,
) -> pd.DataFrame:
    """Rank ``cities`` by daytime-GHI spread across their named districts."""
    sites = [site for city in cities for site in INTRACITY_SITES.get(city, ())]
    per_site = site_mean_ghi(repo, sites, start, end)

    summaries = []
    for city, group in per_site.groupby("city"):
        means = group["mean_ghi_day"]
        spread = float(means.max() - means.min())
        summaries.append(
            {
                "city": city,
                "n_sites": len(group),
                "mean_ghi_day": round(float(means.mean()), 1),
                "spread": round(spread, 1),
                "spread_pct": round(100 * spread / float(means.mean()), 2),
                "std": round(float(means.std()), 2),
            }
        )
    return (
        pd.DataFrame(summaries)
        .sort_values("spread_pct", ascending=False)
        .reset_index(drop=True)
    )


def select_city(
    repo: DataRepository,
    start: date,
    end: date,
    *,
    cities: tuple[str, ...] = FOCUS_CITIES,
) -> tuple[str, pd.DataFrame]:
    """Return ``(chosen_city, ranking)`` — the highest intra-city-spread city."""
    ranking = rank_cities_by_intracity_spread(repo, start, end, cities=cities)
    return str(ranking.iloc[0]["city"]), ranking
