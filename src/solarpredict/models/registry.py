"""Model registry (Registry pattern).

Models self-register with ``@register("name")``; the benchmark harness discovers
them through ``list_forecasters`` / ``get_forecaster``. This is the mechanism that
keeps the tiers decoupled — a contributor adds a model by writing one module and
decorating it, never by editing the harness.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from .base import Forecaster

# Maps a lowercase name to a zero-arg-or-kwargs factory producing a Forecaster.
_REGISTRY: dict[str, Callable[..., Forecaster]] = {}

F = TypeVar("F", bound=Callable[..., Forecaster])


def register(name: str) -> Callable[[F], F]:
    """Decorator registering a Forecaster class (or factory) under ``name``."""

    def decorator(factory: F) -> F:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"Forecaster {name!r} is already registered")
        _REGISTRY[key] = factory
        return factory

    return decorator


def get_forecaster(name: str, **kwargs: object) -> Forecaster:
    """Instantiate a registered forecaster by name."""
    key = name.lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown forecaster {name!r}. Registered: {known}")
    return _REGISTRY[key](**kwargs)


def list_forecasters() -> list[str]:
    """Return the sorted list of registered forecaster names."""
    return sorted(_REGISTRY)
