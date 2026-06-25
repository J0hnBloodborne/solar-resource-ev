"""Data layer: the DataRepository abstraction (Repository pattern).

The interface decouples every caller from *where* data comes from. The default
Open-Meteo archive + parquet implementation is here; the abstraction lets us swap
to the S2Cool Neon DB later without touching callers.
"""

from __future__ import annotations

from .openmeteo import OpenMeteoArchiveRepository
from .repository import DataRepository

__all__ = ["DataRepository", "OpenMeteoArchiveRepository"]
