"""Models: the Forecaster Strategy interface + a self-populating Registry.

Importing this package imports the built-in model modules so their ``@register``
decorators run and the registry is populated. Adding a model = drop a module here
and import it below (or rely on the explicit imports) — no edits to shared code.
"""

from __future__ import annotations

# Import built-in model modules for their registration side effects.
from . import baselines  # noqa: F401  (registers tier-0 baselines)
from .base import ForecastData, Forecaster
from .registry import get_forecaster, list_forecasters, register

__all__ = [
    "ForecastData",
    "Forecaster",
    "get_forecaster",
    "list_forecasters",
    "register",
]
