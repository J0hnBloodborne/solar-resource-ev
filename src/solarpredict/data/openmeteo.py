"""Open-Meteo Historical Archive (ERA5) repository with a parquet cache.

Open-Meteo weights a request by range x variables, so a multi-year x 8-variable
pull trips the per-minute limit in a single call. We therefore fetch in
year-sized chunks, back off 60s when the minute limit hits, and concatenate. Each
location's full series is cached to one parquet file (keyed by site, range, and
variable set). The heavy optional deps (``openmeteo-requests`` etc.) are imported
lazily so the core package stays importable without the ``ingest`` extra.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from solarpredict.config.settings import get_settings
from solarpredict.config.sites import MODELING_HOURLY, Location

from .repository import DataRepository

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CHUNK_DAYS = 365  # keep each request's weighted cost under the per-minute limit
RATE_LIMIT_SLEEP_SECONDS = 62.0
MAX_RATE_LIMIT_RETRIES = 6


def _is_rate_limit_error(exc: Exception) -> bool:
    return "limit" in str(exc).lower()


def _build_frame(
    times: pd.DatetimeIndex, columns: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Assemble a tidy hourly frame from a time index and per-variable arrays."""
    data: dict[str, Any] = {"timestamp": times}
    data.update(columns)
    return pd.DataFrame(data)


class OpenMeteoArchiveRepository(DataRepository):
    """Fetch hourly ERA5 archive data, caching each location to parquet."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        hourly: tuple[str, ...] = MODELING_HOURLY,
        chunk_days: int = CHUNK_DAYS,
        inter_request_sleep: float = 1.0,
        client: Any = None,
    ) -> None:
        self._hourly = tuple(hourly)
        self._cache_dir = cache_dir or get_settings().cache_dir
        self._chunk_days = chunk_days
        self._sleep = inter_request_sleep
        self._client: Any = client  # built lazily on first remote fetch

    def _cache_path(self, location: Location, start: date, end: date) -> Path:
        vhash = hashlib.md5(",".join(self._hourly).encode()).hexdigest()[:6]
        return (
            self._cache_dir
            / f"{location.slug}_{start.isoformat()}_{end.isoformat()}_{vhash}.parquet"
        )

    def fetch(
        self, location: Location, start: date, end: date, *, force_refresh: bool = False
    ) -> pd.DataFrame:
        cache = self._cache_path(location, start, end)
        if cache.exists() and not force_refresh:
            logger.info("[cache] hit %s", cache.name)
            return pd.read_parquet(cache)
        df = self._fetch_remote(location, start, end)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
        logger.info("[cache] wrote %s (%d rows)", cache.name, len(df))
        return df

    def _get_client(self) -> Any:
        if self._client is None:
            import openmeteo_requests
            import requests_cache
            from retry_requests import retry

            self._cache_dir.mkdir(parents=True, exist_ok=True)
            session = retry(
                requests_cache.CachedSession(
                    str(self._cache_dir / "http_cache"), expire_after=-1
                ),
                retries=3,
                backoff_factor=0.4,
            )
            self._client = openmeteo_requests.Client(session=session)
        return self._client

    def _request(self, params: dict[str, Any]) -> Any:
        """Call the archive API, backing off on the per-minute rate limit."""
        client = self._get_client()
        for attempt in range(MAX_RATE_LIMIT_RETRIES):
            try:
                return client.weather_api(ARCHIVE_URL, params=params)[0]
            except Exception as exc:
                last = attempt == MAX_RATE_LIMIT_RETRIES - 1
                if _is_rate_limit_error(exc) and not last:
                    logger.warning(
                        "[ratelimit] minute limit hit; sleeping %.0fs (try %d/%d)",
                        RATE_LIMIT_SLEEP_SECONDS,
                        attempt + 1,
                        MAX_RATE_LIMIT_RETRIES,
                    )
                    time.sleep(RATE_LIMIT_SLEEP_SECONDS)
                    continue
                raise
        raise RuntimeError("unreachable")  # pragma: no cover

    def _response_to_frame(self, response: Any) -> pd.DataFrame:
        hourly = response.Hourly()
        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
        columns = {
            name: hourly.Variables(i).ValuesAsNumpy()
            for i, name in enumerate(self._hourly)
        }
        return _build_frame(times, columns)

    def _fetch_remote(self, location: Location, start: date, end: date) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        meta: dict[str, float] = {}
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + timedelta(days=self._chunk_days - 1), end)
            params = {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
                "hourly": list(self._hourly),
                "timezone": "UTC",
            }
            logger.info("[fetch] %s %s -> %s", location.slug, chunk_start, chunk_end)
            response = self._request(params)
            frames.append(self._response_to_frame(response))
            if not meta:
                meta = {
                    "cell_latitude": float(response.Latitude()),
                    "cell_longitude": float(response.Longitude()),
                    "elevation": float(response.Elevation()),
                }
            chunk_start = chunk_end + timedelta(days=1)
            if chunk_start <= end and self._sleep:
                time.sleep(self._sleep)

        df = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        # ERA5 grid cell the point resolved to (in-memory only; parquet drops attrs).
        df.attrs.update({"site": location.slug, **meta})
        return df
