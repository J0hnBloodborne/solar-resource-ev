# Data Summary — Solar Resource Prediction

A short summary of the chosen data and what we expect from it. (Numbers below are
from 1–2 year scoping pulls; the final analysis uses ~5 years — magnitudes/ordering
hold, exact values will shift slightly.)

## Source & method
- **Open-Meteo Historical Archive (ERA5 reanalysis)** — free, hourly, no API key.
- **8 hourly variables**: GHI (`shortwave_radiation`), temperature, humidity, wind,
  cloud cover, and the direct/DNI/diffuse components.
- **Resolution ~9–25 km** (ERA5-Land/ERA5). This matters for *within-city* siting
  (below). **~5 years** of history per point; cached locally; reproducible.

## Sites chosen
- **7 candidate cities** (inter-city benchmark): Islamabad, Lahore, Karachi,
  Peshawar, Multan, Quetta, Gilgit.
- **Part B city — Karachi** (chosen from the data, see below). Sampled with a
  **25-point ~7 km grid** that resolves into **~16 distinct ERA5 cells** (for the
  GHI heatmap), plus **8 named districts** (Clifton, DHA, Korangi, Malir,
  Gulshan-e-Iqbal, North Nazimabad, SITE, Gadap) for human-readable recommendations.
- **Lahore** kept as the comparison metro.

## Variance — the key numbers

**Inter-city (large, real):** mean GHI varies **28%** across the 7 cities.

| City | Daytime GHI (W/m²) | Annual energy (kWh/m²/yr) |
| --- | --- | --- |
| Quetta | 466 | 2187 |
| Gilgit | 421 | 1959 |
| **Karachi** | 419 | 1951 |
| Multan | 383 | 1796 |
| Islamabad | 366 | 1696 |
| Lahore | 368 | 1694 |
| Peshawar | 357 | 1661 |

→ Range **1661–2187 kWh/m²/yr** (Quetta, high desert, is ~32% sunnier than the
fog/smog-prone Peshawar–Lahore–Islamabad belt).

**Intra-city (granular, but resolvable):**

| City | Distinct ERA5 cells | GHI spread | Pattern |
| --- | --- | --- | --- |
| **Karachi** | ~16 | **3.3%** (~7 W/m²) | clear coastal → inland gradient |
| Lahore | ~7 | 1.4% | nearly flat (uniform plain) |

→ Karachi has ~2× Lahore's spread and a physical story (coastal humidity/marine
layer lowers GHI; inland/north is sunnier). Best-vs-worst Karachi site ≈ **+50–65
kWh/m²/yr** — small in % but meaningful as annual energy. **Karachi is the chosen
Part B city.**

## Expected results (to be confirmed by the full runs)

**Part A — model benchmark (hour-ahead GHI):**
- Hour-ahead GHI is highly predictable (strong diurnal cycle), so all models score
  high raw R² — which is why we report a **skill score vs. smart (clear-sky)
  persistence** instead. Expected modest positive skill (~**0.05–0.20**) at this
  horizon; the genuine forecasting value lives in cloudy hours.
- Expected ordering: **XGBoost / LightGBM** lead or tie; the modern **NHITS** and
  the **Chronos-2 foundation model** are competitive; a locally-trained tree likely
  edges the zero-shot foundation model on this near-deterministic target — an honest,
  publishable finding. The covariate ablation shows weather inputs help most in
  cloudy conditions.

**Part B — site suitability / charging-station siting:**
- Within Karachi, **northern/inland sites** (Gadap, SITE) rank highest, **coastal
  sites** (Clifton, DHA) lowest; the suitability index blends GHI + PSH + seasonal
  consistency (+ temperature), reported as annual energy so the ~3% gradient reads
  as a concrete kWh/m²/yr difference.
- Deliverable: a ranked recommended-sites table + a Karachi GHI heatmap.

## Caveats
- ERA5 is **reanalysis, ~9–25 km** — it captures the city-scale resource and a real
  coastal→inland gradient, but not sub-cell microclimate; intra-city differences are
  genuinely small. A higher-res satellite source (or the proposed physical sensor)
  is future work.
- Inter-city differences are the strong signal; intra-city is the granular,
  honestly-reported siting contribution.
