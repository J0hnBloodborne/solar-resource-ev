"""DataRepository interface (Repository pattern).

Abstracts fetch + cache of hourly weather/solar series for a location and date
range. Concrete implementations (``OpenMeteoArchiveRepository``, a future
``NeonRepository``) live alongside this module and are selected by callers via
this interface only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from solarpredict.config.sites import Location


class DataRepository(ABC):
    """Fetch hourly weather/solar data for a location and date range."""

    @abstractmethod
    def fetch(
        self, location: Location, start: date, end: date, *, force_refresh: bool = False
    ) -> pd.DataFrame:
        """Return an hourly dataframe for ``location`` between ``start`` and ``end``.

        The frame carries a UTC ``timestamp`` column plus one column per requested
        Open-Meteo hourly variable (see
        :data:`solarpredict.config.sites.OPEN_METEO_HOURLY`). Implementations with a
        cache should bypass it when ``force_refresh`` is true.
        """
        raise NotImplementedError
