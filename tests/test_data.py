"""Tests for the site catalog and the Open-Meteo repository (no network)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from solarpredict.config.sites import (
    CANDIDATE_CITIES,
    INTRACITY_SITES,
    Location,
    intracity_sites,
)
from solarpredict.data import OpenMeteoArchiveRepository
from solarpredict.data.ingest import default_date_range, default_ingest_targets


def test_all_coordinates_valid() -> None:
    everything = [*CANDIDATE_CITIES, *intracity_sites()]
    for loc in everything:
        assert -90 <= loc.latitude <= 90
        assert -180 <= loc.longitude <= 180


def test_intracity_sites_have_unique_slugs() -> None:
    sites = intracity_sites()
    slugs = [s.slug for s in sites]
    assert len(slugs) == len(set(slugs)), "site slugs must be unique"
    # "DHA" exists in both metros but slugs are namespaced by city.
    assert "karachi_dha" in slugs
    assert "lahore_dha" in slugs
    for s in sites:
        assert s.city in INTRACITY_SITES


def test_default_date_range_spans_years() -> None:
    start, end = default_date_range(3)
    assert end > start
    assert 3 * 360 <= (end - start).days <= 3 * 370


def test_default_targets_include_cities_and_sites() -> None:
    targets = default_ingest_targets()
    slugs = {t.slug for t in targets}
    assert "karachi" in slugs  # candidate city centroid
    assert "karachi_clifton" in slugs  # intra-city named site


def test_repository_caches_to_parquet(tmp_path, monkeypatch) -> None:
    repo = OpenMeteoArchiveRepository(cache_dir=tmp_path)
    calls = {"n": 0}

    def fake_remote(location, start, end):
        calls["n"] += 1
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC"),
                "shortwave_radiation": [0.0, 1.0, 2.0],
            }
        )

    monkeypatch.setattr(repo, "_fetch_remote", fake_remote)
    loc = Location("Test", 24.8, 67.0)
    first = repo.fetch(loc, date(2020, 1, 1), date(2020, 1, 2))
    second = repo.fetch(loc, date(2020, 1, 1), date(2020, 1, 2))

    assert calls["n"] == 1  # second call served from the parquet cache
    assert len(first) == len(second) == 3
    assert list(tmp_path.glob("test_*.parquet"))  # cached to parquet
