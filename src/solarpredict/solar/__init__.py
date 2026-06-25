"""Solar physics: GHI -> power conversion (pvlib PVWatts) and PSH.

Populated in M5/M6. Reuses the linear STC scaling from S2Cool's math_model plus a
fixed PV system per site so power-domain error is reproducible.
"""

from __future__ import annotations
