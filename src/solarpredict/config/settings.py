"""Project settings (env-overridable via ``SOLARPREDICT_*`` variables).

Config-driven by design (no hardcoded paths/params scattered across modules) so
experiments are reproducible and contributors don't edit shared code to retarget
a run. Override any field with an env var, e.g. ``SOLARPREDICT_HISTORY_YEARS=5``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global, env-overridable settings."""

    model_config = SettingsConfigDict(env_prefix="SOLARPREDICT_", env_file=".env")

    # Paths (relative to repo root by default; created on demand by callers).
    data_dir: Path = Field(default=Path("data"))
    cache_dir: Path = Field(default=Path("data/cache"))
    artifacts_dir: Path = Field(default=Path("artifacts"))

    # Data window. 1-2 years is the floor; a longer range is the same number of
    # Open-Meteo requests (one per site), so we default higher — see the plan's
    # "Data ingestion strategy" section.
    history_years: int = Field(default=5, ge=1, le=40)

    # Forecasting setup.
    target: str = Field(default="shortwave_radiation")  # GHI (W/m^2)
    freq: str = Field(default="h")
    horizon: int = Field(default=1)  # hour-ahead

    # Reproducibility.
    random_seed: int = Field(default=42)


@lru_cache
def get_settings() -> Settings:
    """Return the cached singleton settings instance."""
    return Settings()
