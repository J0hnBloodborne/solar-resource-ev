"""Boundary geometry for the map figures.

Province (ADM1) and district (ADM2) outlines from geoBoundaries (gbOpen, PAK) are
committed under ``assets/geo/`` so every map renders offline and reproducibly.
``geopandas`` is imported lazily so the rest of ``viz`` stays importable without
the heavy geo stack.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd


def _geo_dir() -> Path:
    here = Path(__file__).resolve().parents[3] / "assets" / "geo"
    return here if here.exists() else Path.cwd() / "assets" / "geo"


@cache
def provinces() -> gpd.GeoDataFrame:
    """Pakistan provinces / territories (ADM1)."""
    import geopandas as gpd

    return gpd.read_file(_geo_dir() / "pakistan_adm1.geojson")


@cache
def country() -> gpd.GeoDataFrame:
    """National outline (provinces dissolved into one polygon)."""
    return provinces().dissolve()


@cache
def city_outline(name: str = "Karachi") -> gpd.GeoDataFrame:
    """District (ADM2) polygon whose name contains ``name`` (e.g. Karachi)."""
    import geopandas as gpd

    adm2 = gpd.read_file(_geo_dir() / "pakistan_adm2.geojson")
    sel = adm2[adm2["shapeName"].str.contains(name, case=False, na=False)]
    if sel.empty:
        raise ValueError(f"no ADM2 boundary matching {name!r}")
    return sel
