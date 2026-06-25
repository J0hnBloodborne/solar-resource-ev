"""Ingestion orchestration: pull configured points into the parquet cache.

Thin glue over a ``DataRepository`` — picks the target points and date range, then
loops fetches. Logic stays here (testable); the CLI just calls it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from datetime import date, timedelta

import pandas as pd

from solarpredict.config.settings import Settings, get_settings
from solarpredict.config.sites import CANDIDATE_CITIES, Location, intracity_sites

from .repository import DataRepository

logger = logging.getLogger(__name__)

# ERA5 lags real time by ~5 days; pad so end_date is always available.
ARCHIVE_LATENCY_DAYS = 7


def default_date_range(
    years: int | None = None, *, settings: Settings | None = None
) -> tuple[date, date]:
    """Return ``(start, end)`` covering the last ``years`` of available archive data."""
    settings = settings or get_settings()
    span = years if years is not None else settings.history_years
    end = date.today() - timedelta(days=ARCHIVE_LATENCY_DAYS)
    start = end - timedelta(days=round(365.25 * span))
    return start, end


def default_ingest_targets() -> list[Location]:
    """Candidate cities (inter-city) + Karachi/Lahore named sites (intra-city)."""
    return [*CANDIDATE_CITIES, *intracity_sites()]


def ingest_points(
    repo: DataRepository,
    locations: Iterable[Location],
    start: date,
    end: date,
    *,
    sleep: float = 1.0,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch + cache every location, returning ``{slug: hourly_frame}``."""
    results: dict[str, pd.DataFrame] = {}
    locations = list(locations)
    for index, loc in enumerate(locations):
        frame = repo.fetch(loc, start, end, force_refresh=force_refresh)
        results[loc.slug] = frame
        logger.info("[ingest] %s rows=%d", loc.slug, len(frame))
        if sleep and index < len(locations) - 1:
            time.sleep(sleep)
    return results
