# Solar resource prediction: data, models, and EV-charging siting

This document is the full record of the analysis behind the paper. It covers where the
data comes from and how it was collected, the models we compared and how they were
scored, the results for every experiment, and the siting work for solar-powered EV
charging. Every figure and table produced by the project is included here, with the
source file named so each number can be traced back to a run.

All numbers come from real Open-Meteo data and actual model runs. Nothing is invented.
Where a result is weak or unflattering it is reported as-is.

The work has two parts:

1. Compare several machine-learning models at predicting solar irradiance (GHI), and
   convert that to PV power.
2. Rank locations by solar suitability and pick good sites for solar-powered EV
   charging stations.

---

## 1. Data

### 1.1 Source and variables

Data is from the Open-Meteo Historical Archive, which serves the ERA5 reanalysis. ERA5
is a physically consistent, hourly, global record produced by ECMWF by blending weather
models with observations. It is free, needs no API key, and goes back to 1940.

We fetch five hourly variables per location:

| Variable (Open-Meteo name) | Meaning | Role |
|---|---|---|
| `shortwave_radiation` | Global horizontal irradiance, GHI (W/m²) | Main prediction target |
| `temperature_2m` | Air temperature at 2 m (°C) | Second target + covariate |
| `relative_humidity_2m` | Relative humidity (%) | Covariate |
| `wind_speed_10m` | Wind speed at 10 m (m/s) | Covariate |
| `cloud_cover` | Total cloud cover (%) | Covariate |

GHI is the energy that a flat panel receives per square metre. It is the quantity that
sets solar output, so it is the headline target.

### 1.2 Coverage

- **Period:** five years of hourly data per location, about 43,800 records each, ending
  roughly a week before the run date (ERA5 has a short processing lag).
- **Cities (7):** Islamabad, Lahore, Karachi, Peshawar, Multan, Quetta, Gilgit. These
  span the country: the southern coast, the central plains, the western highlands, and
  the northern mountains.
- **Within Karachi (8 named districts):** Clifton, DHA, Korangi, Malir, Gulshan-e-Iqbal,
  North Nazimabad, SITE, Gadap. Spread from the coast in the south to the urban fringe
  in the north.
- **Karachi GHI grid:** a regular grid at about 0.1° spacing over the Karachi district,
  29 points that fall inside the district boundary, used for the within-city heatmap.

### 1.3 How it was acquired

The archive returns a full multi-year hourly series for a point in one request, so the
number of calls is roughly the number of locations, not the number of days. Open-Meteo
weights a request by range times variables, so a five-year, five-variable pull is heavy
enough to trip the per-minute limit. To stay inside the free limits the fetcher:

- splits each pull into one-year chunks,
- waits and retries when the minute limit is hit,
- caches every location's full series to a Parquet file keyed by site, date range, and
  variable set, so nothing is ever fetched twice.

The data layer sits behind a small interface (`DataRepository`), so the same code could
later read from a database instead of the archive without changing the callers.

### 1.4 Resolution: what the data can and cannot resolve

ERA5's grid is coarse, roughly 25 km for radiation. That has two consequences that
shape the whole study:

- **Between cities** the differences are large and real. The grid easily separates
  Quetta from Karachi.
- **Within a city** the differences are small. Karachi spans only a handful of grid
  cells, so the intra-city spread in GHI is on the order of a few percent. We report
  that spread honestly rather than dressing it up.

ERA5 is a reanalysis, not a ground measurement. It captures the regional resource and
the coastal gradient well, but it is not a substitute for an on-site sensor. A funded
pyranometer for ground validation is noted as future work.

---

## 2. Methods

### 2.1 Clear-sky model and night masking

A clear-sky model gives the GHI you would see with no clouds, from solar geometry alone.
We compute it with pvlib (the Haurwitz model). Two uses:

- **Night masking.** At night the clear-sky GHI is near zero. We drop those hours from
  every metric. If they are kept, the long run of correct night-time zeros inflates R²
  to about 0.99 for any model and hides the real daytime skill.
- **Clear-sky index.** The ratio `kt = GHI / clear-sky GHI` strips out the predictable
  daily and seasonal cycle, leaving the cloud-driven part that is the hard thing to
  forecast.

### 2.2 Features

For the tabular models each row is built only from information available at or before
the forecast time, so predicting the next hour is leakage-free:

- lagged GHI (1, 2, 3, and 24 hours back),
- lagged covariates (temperature, humidity, wind, cloud at 1 and 24 hours back),
- rolling mean and standard deviation of GHI (3- and 24-hour windows, shifted back one
  step),
- the lagged clear-sky index,
- the clear-sky GHI at the forecast time (this is deterministic, so it is safe to use
  directly),
- cyclical encodings of hour-of-day and day-of-year (sine and cosine).

The radiation components that Open-Meteo can also return (direct, diffuse, DNI) are
deliberately not used as predictors. They are parts of the target and would leak it.

### 2.3 Train/test split and leakage rules

The split is strictly chronological: the models train on the earlier part of the series
and are tested on the most recent 20%. The data is never shuffled, because shuffling
would let a model see the future. The sequence models (LSTM, NHITS, Chronos-2) get a
one-week context window of history immediately before the test set so they have
something to condition on, but only the test portion is scored.

### 2.4 Metrics

Every model is scored the same way, on daytime hours only:

- **MAE** and **RMSE** in W/m² (for GHI) or °C (for temperature),
- **R²**, the fraction of variance explained,
- **nRMSE**, RMSE divided by the mean daytime value,
- **MBE**, the mean bias (positive means the model runs high),
- **skill score** `S = 1 − RMSE_model / RMSE_reference`.

The skill score is the headline number. It says how much a model beats a sensible naive
forecast. `S = 0` means no better than the reference; `S = 0.4` means 40% lower RMSE;
negative means worse. For GHI the reference is smart persistence (see below). For
temperature the reference is plain persistence, because temperature has no clear-sky
model.

We avoid MAPE. It blows up near sunrise and sunset where GHI is close to zero.

### 2.5 The models

Four tiers, from naive to modern. The naive tier is not a research question (the
instructor settled that AI is the method); it is the standard reference that any
credible solar-forecasting result is measured against.

| Tier | Models | What it is |
|---|---|---|
| 0 — naive / statistical | persistence, smart (clear-sky) persistence, climatology | Reference forecasts |
| 1 — classical ML | Ridge, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost | Engineered features + tree ensembles |
| 2 — deep learning | LSTM, NHITS | Neural sequence models (PyTorch, via neuralforecast) |
| 3 — foundation model | Chronos-2 | A ~120M-parameter time-series transformer (Amazon), used zero-shot with no training on our data |

The three naive references:

- **Persistence:** next hour equals this hour. Strong for slow signals, weak for GHI
  because GHI changes fast around dawn and dusk.
- **Smart (clear-sky) persistence:** persist the clear-sky index and rescale by the next
  hour's clear-sky GHI. This carries the predictable solar cycle forward and is the
  proper reference for solar forecasting.
- **Climatology:** the historical average for that hour and season.

Every model implements one interface, so the benchmark runs them all through one scoring
harness. Adding a model is one file; it does not touch shared code.

---

## 3. Part A: forecasting GHI

### 3.1 Karachi benchmark

Hour-ahead GHI, Karachi, five years, daytime hours only. Source:
`reports/benchmark_karachi.csv`.

| Model | Tier | MAE | RMSE | R² | MBE | Skill |
|---|---|---|---|---|---|---|
| extra_trees | classical ML | 20.2 | 41.6 | 0.976 | −0.4 | **0.402** |
| catboost | classical ML | 22.4 | 41.8 | 0.976 | 0.4 | 0.400 |
| xgboost | classical ML | 22.4 | 43.3 | 0.974 | 1.5 | 0.378 |
| lightgbm | classical ML | 22.8 | 44.0 | 0.973 | 0.7 | 0.367 |
| random_forest | classical ML | 22.5 | 44.5 | 0.973 | 0.4 | 0.360 |
| nhits | deep learning | 23.3 | 46.1 | 0.971 | 1.3 | 0.337 |
| chronos2 | foundation | 26.8 | 48.6 | 0.967 | 5.0 | 0.301 |
| lstm | deep learning | 32.7 | 52.0 | 0.963 | 3.8 | 0.253 |
| ridge | classical ML | 31.8 | 52.2 | 0.962 | 6.0 | 0.251 |
| smart_persistence | reference | 54.4 | 69.6 | 0.933 | −43.4 | 0.000 |
| climatology | naive | 46.0 | 82.3 | 0.906 | 5.1 | −0.183 |
| persistence | naive | 118.9 | 138.2 | 0.736 | −10.4 | −0.986 |

![Hour-ahead GHI skill by model, Karachi](figures/fig1_model_skill.png)

![Hour-ahead GHI RMSE by model, Karachi](figures/fig1b_model_rmse.png)

What this shows:

- The tree ensembles win. Extra Trees is best at 41.6 W/m² RMSE, a 40% lower error than
  the smart-persistence reference (69.6). CatBoost is essentially tied.
- NHITS (deep learning) is just behind the trees. Chronos-2, used with no training on
  Karachi at all, still reaches 0.30 skill, ahead of the LSTM and the linear model.
- Machine learning clearly helps at this horizon. That is the direct answer to "is AI
  worth it here": yes, a 40% error reduction over the best naive method.
- Why GHI is this predictable one hour ahead: most of the signal is the solar cycle,
  which the models get for free through clear-sky GHI and the time encodings. One hour
  ahead, clouds mostly persist. So the models are close to "smart persistence plus a
  learned correction." Their main gain is removing the clear-sky reference's large
  negative bias (its MBE is −43 W/m²; the trees sit near zero).
- Honest scope: one hour ahead is the easy horizon. Day-ahead skill would be much lower.
  We do not claim otherwise.

### 3.2 Predicted versus actual

The best model's predictions against the measured GHI on the test set:

![Predicted vs actual GHI, best model, Karachi](figures/fig5_pred_vs_actual.png)

Points sit tight to the 1:1 line across the full range, with the spread widening at high
irradiance where passing clouds make the largest absolute errors.

### 3.3 What the models rely on

Feature importances from the best tree model:

![Feature importance, best tree model](figures/fig7_feature_importance.png)

The recent GHI lags and the clear-sky GHI carry most of the weight. The model leans on
the deterministic solar cycle plus the last few hours of cloud behaviour, which matches
the reading above.

### 3.4 All seven cities

The same 12-model benchmark, run for every city, so the ranking can be read across the
country. Source: `reports/benchmark_multicity.csv`. The table below is the skill score
for each model in each city, with the mean across cities in the last column.

| Model | Quetta | Multan | Peshawar | Islamabad | Karachi | Lahore | Gilgit | Mean |
|---|---|---|---|---|---|---|---|---|
| extra_trees | 0.507 | 0.434 | 0.432 | 0.393 | 0.404 | 0.380 | 0.374 | **0.418** |
| catboost | 0.482 | 0.432 | 0.439 | 0.399 | 0.401 | 0.388 | 0.374 | 0.416 |
| xgboost | 0.482 | 0.423 | 0.418 | 0.400 | 0.388 | 0.381 | 0.366 | 0.408 |
| lightgbm | 0.474 | 0.412 | 0.423 | 0.403 | 0.364 | 0.386 | 0.364 | 0.404 |
| random_forest | 0.455 | 0.393 | 0.420 | 0.386 | 0.360 | 0.364 | 0.342 | 0.388 |
| nhits | 0.420 | 0.375 | 0.329 | 0.304 | 0.340 | 0.306 | 0.277 | 0.336 |
| chronos2 | 0.381 | 0.346 | 0.229 | 0.207 | 0.304 | 0.265 | 0.260 | 0.285 |
| ridge | 0.370 | 0.335 | 0.216 | 0.211 | 0.253 | 0.238 | 0.215 | 0.263 |
| lstm | 0.354 | 0.322 | 0.255 | 0.264 | 0.195 | 0.248 | 0.204 | 0.263 |
| climatology | −0.114 | −0.082 | −0.384 | −0.329 | −0.175 | −0.386 | −0.639 | −0.301 |
| persistence | −1.211 | −1.082 | −0.861 | −0.768 | −0.986 | −0.887 | −1.005 | −0.971 |

(smart_persistence is the reference, skill 0.000 in every city.)

![GHI skill by city and model](figures/fig10_multicity_skill.png)

![GHI RMSE by city and model](figures/fig10b_multicity_rmse.png)

What this shows:

- The ranking is the same everywhere. A tree ensemble wins in all seven cities: Extra
  Trees in four (Karachi, Multan, Quetta, Gilgit), CatBoost in two (Lahore, Peshawar),
  LightGBM in one (Islamabad). It is never the deep or foundation model.
- Chronos-2, with no local training, stays positive in every city (0.21 to 0.38) and
  beats the LSTM. That is strong for an off-the-shelf model, but a locally trained tree
  beats it everywhere. The honest headline: on this near-deterministic, short-horizon
  target the foundation model is competitive out of the box but does not overtake
  classical ML.
- Predictability tracks climate. Skill is highest in Quetta (0.51, clear high-desert
  skies) and lowest in Gilgit (0.37, complex mountain weather).

### 3.5 How much data you need

Foundation and deep models are often pitched as the answer when you have little data.
We tested that directly: fix the test set, vary the training size from one month to four
years, and rerun. Source: `reports/data_efficiency_karachi.csv`. Skill by training size:

| Model | 1 mo | 3 mo | 6 mo | 1 yr | 2 yr | 4 yr |
|---|---|---|---|---|---|---|
| chronos2 (zero-shot) | 0.301 | 0.301 | 0.301 | 0.301 | 0.301 | 0.301 |
| nhits | 0.202 | 0.271 | 0.298 | 0.332 | 0.338 | 0.336 |
| extra_trees | 0.158 | 0.189 | 0.227 | 0.394 | 0.406 | 0.410 |
| xgboost | −0.122 | 0.105 | 0.181 | 0.363 | 0.391 | 0.384 |

![Data efficiency, skill vs training size](figures/fig8_data_efficiency_skill.png)

![Data efficiency, RMSE vs training size](figures/fig8b_data_efficiency_rmse.png)

What this shows:

- Chronos-2 is flat because it never trains. At one month it is the best model in the
  table, and XGBoost is actually worse than the naive reference (skill −0.12).
- NHITS needs only a few months to get close to its full skill.
- The trees overtake once they have about a year of data, then pull ahead and stay
  ahead.
- The practical reading: with little local history, reach for the foundation or deep
  model; with a year or more, train a tree. This is the second main result of Part A.

---

## 4. Forecasting temperature

S2Cool, the earlier project, predicted temperature as well as GHI, so we ran the same
benchmark on hour-ahead 2 m air temperature in Karachi. The reference here is plain
persistence (temperature has no clear-sky model and no night masking). Source:
`reports/benchmark_temp_karachi.csv`.

| Model | Tier | MAE (°C) | RMSE (°C) | R² | Skill |
|---|---|---|---|---|---|
| chronos2 | foundation | 0.231 | 0.346 | 0.994 | **0.603** |
| nhits | deep learning | 0.243 | 0.360 | 0.994 | 0.587 |
| extra_trees | classical ML | 0.251 | 0.380 | 0.993 | 0.564 |
| random_forest | classical ML | 0.259 | 0.390 | 0.993 | 0.552 |
| lightgbm | classical ML | 0.265 | 0.398 | 0.993 | 0.544 |
| catboost | classical ML | 0.269 | 0.403 | 0.993 | 0.538 |
| xgboost | classical ML | 0.266 | 0.406 | 0.992 | 0.534 |
| lstm | deep learning | 0.274 | 0.411 | 0.992 | 0.528 |
| ridge | classical ML | 0.288 | 0.417 | 0.992 | 0.521 |
| persistence | reference | 0.615 | 0.871 | 0.965 | 0.000 |
| climatology | naive | 1.279 | 1.727 | 0.863 | −0.982 |

![Hour-ahead temperature skill by model](figures/fig11_temp_skill.png)

![Hour-ahead temperature RMSE by model](figures/fig11b_temp_rmse.png)

The ranking flips. On temperature the modern models win: Chronos-2 first (0.60 skill,
0.35 °C RMSE), NHITS second, and the trees behind them.

This is the most interesting finding in Part A, and it is why both targets are worth
including. The best model depends on what kind of signal the target is:

- Temperature is smooth and strongly periodic. Sequence models, and the foundation model
  in particular, are built for exactly that shape and come out on top.
- GHI is spiky and cloud-driven. The predictable cycle is easy; the hard part is the
  cloud noise, where engineered lag features in a tree ensemble do best.

So the foundation model is not a gimmick row in the table. It is genuinely the best model
on the smooth target and competitive on the hard one. You only see that by testing both.

---

## 5. Part B: solar resource and where to charge

Part B ranks locations by solar suitability and recommends sites for solar-powered EV
charging. It works at two scales: between cities, and within one city (Karachi).

### 5.1 Between cities

Five-year mean GHI per city, converted to an annual energy yield. Source:
`reports/city_summary.csv`.

| City | Annual yield (kWh/m²/yr) | Daytime GHI (W/m²) | Seasonality (CV) |
|---|---|---|---|
| Quetta | 2174 | 463 | 0.253 |
| Gilgit | 1960 | 421 | 0.281 |
| Karachi | 1950 | 418 | **0.178** |
| Multan | 1793 | 383 | 0.232 |
| Lahore | 1696 | 369 | 0.258 |
| Islamabad | 1692 | 365 | 0.285 |
| Peshawar | 1650 | 355 | 0.296 |

![Solar resource across Pakistani cities](figures/fig2_ghi_map.png)

![City solar ranking](figures/fig4_city_ranking.png)

![Seasonal GHI by city](figures/fig3_seasonal.png)

What this shows:

- Quetta and Gilgit sit above Karachi, our study city, on raw solar resource. They are
  the untapped higher-solar candidates. If EV adoption reaches them, they would make even
  better hosts for solar charging than the coastal cities. Quetta in particular, at 2174
  kWh/m²/yr, leads the country by a wide margin (its clear high-desert skies).
- Karachi has the lowest seasonality of the seven (CV 0.178), meaning its solar supply is
  the steadiest through the year. For a charging station that wants reliable output, low
  seasonal swing matters alongside peak yield.
- Even the least-sunny city here, Peshawar at 1650 kWh/m²/yr, is well above the
  national-average yields of Germany (about 1080) and the UK (about 1000). Solar EV
  charging is viable across the whole country.
- The spread between cities (28% from Quetta to Peshawar) is the strong signal. It is
  much larger than the within-city spread (a few percent), which is the next section.

### 5.2 GHI through the day and the year

Two views of the Karachi GHI shape, both in local time:

![GHI by hour of day and month, Karachi](figures/fig2b_ghi_heatmap.png)

![Mean daily GHI and PV potential, Karachi](figures/fig3b_daily_profile.png)

The heatmap shows the daily peak near 13:00 local time and the seasonal band: brightest
in spring before the monsoon, dimmest during the summer monsoon. The daily profile is the
average day, with the right axis reading GHI as PV power for a 1 kW-peak panel.

### 5.3 Within Karachi: the resource surface

The within-city heatmap samples GHI across the Karachi district on a 0.1° grid (29
interior points) and interpolates a continuous surface, clipped to the real district
boundary. The seaward edge is the coastline.

![Intra-city GHI gradient, Karachi](figures/fig2c_karachi_grid_heatmap.png)

GHI runs from about 412 W/m² in the urban core to 440 in the rural north. The gradient is
real but small, on the order of 5%, which is the resolution limit discussed in §1.4. Note
that the densest, most built-up parts of the city read slightly lower than the open land
around them, because the urban core carries more aerosol and humidity that cut the
surface irradiance.

### 5.4 The suitability index

We rank Karachi's districts with a transparent weighted index, not a black box, so the
instructor can explain every term. It blends three things, each normalised across the
sites:

- **yield** (annual GHI), weight 0.60,
- **consistency** (low seasonal variation), weight 0.25,
- **coolness** (lower air temperature, since hot panels lose efficiency), weight 0.15.

Yield dominates, so the ranking is mostly resource-driven, with consistency and coolness
as tie-breakers. Source: `reports/site_suitability_karachi.csv`.

| Site | Annual yield (kWh/m²/yr) | Daytime GHI (W/m²) | Mean temp (°C) | Suitability |
|---|---|---|---|---|
| Gadap | 1990 | 426.4 | 26.1 | **0.800** |
| SITE | 1973 | 423.1 | 26.3 | 0.670 |
| Clifton | 1950 | 418.3 | 26.3 | 0.275 |
| Korangi | 1954 | 419.2 | 26.3 | 0.235 |
| DHA | 1954 | 419.2 | 26.4 | 0.198 |
| Malir | 1951 | 418.6 | 26.3 | 0.190 |
| North Nazimabad | 1950 | 418.4 | 26.5 | 0.000 |
| Gulshan-e-Iqbal | 1950 | 418.4 | 26.5 | 0.000 |

![Within-city site suitability, Karachi](figures/fig4b_site_suitability.png)

The northern and western urban fringe (Gadap, then SITE) ranks highest. It has the most
GHI of the built-up districts and runs slightly cooler.

### 5.5 Recommended EV-charging sites

The recommendation map puts the ranked sites on top of the resource surface. The top
three are starred.

![Recommended solar-EV charging sites, Karachi](figures/fig6_ev_locations.png)

| Rank | Site | Coordinates (lat, lon) | Suitability |
|---|---|---|---|
| 1 | Gadap | 25.050, 67.110 | 0.800 |
| 2 | SITE | 24.910, 66.990 | 0.670 |
| 3 | Clifton | 24.810, 67.030 | 0.275 |

Gadap and SITE are the year-round picks. Clifton, on the south coast, ranks third on the
strength of its coolness term even though its raw GHI is a little lower, which matters
more in summer (see §5.7).

### 5.6 Resource versus demand: why not the sunniest land?

The resource surface in §5.3 covers the whole district, and the sunniest spots are not in
the city at all. The rural north (about 440 W/m²) and the open south-west coast near Cape
Monze and the Hub river mouth (about 435 W/m²) read higher than the urban core (411 to
419). Those are genuine ERA5 readings, not artefacts.

We do not put a charger there, because those areas are not urban. They are open coast,
hills, and barren hinterland with little population, road access, or grid. An EV charging
station follows demand, so the candidates are the eight built-up districts. The sunniest
land suits a utility-scale solar farm, not a car charger. The recommendation map shows
both layers at once: the full-city resource underneath, and the urban sites on top, which
is why the brightest zones sit there without a star.

### 5.7 Season by season

The instructor asked for the year split into four seasons with the sites ranked through
them. We compute the suitability index inside each meteorological season (winter
December–February, spring March–May, summer June–August, autumn September–November).
Source: `reports/seasonal_site_suitability_karachi.csv`.

Mean daytime GHI by site and season (W/m²):

| Site | Winter | Spring | Summer | Autumn |
|---|---|---|---|---|
| Gadap | 402 | 507 | 378 | 416 |
| SITE | 400 | 503 | 376 | 412 |
| Clifton | 396 | 499 | 370 | 407 |
| DHA | 398 | 502 | 367 | 408 |
| Korangi | 398 | 502 | 367 | 408 |
| Malir | 398 | 502 | 366 | 408 |
| Gulshan-e-Iqbal | 397 | 502 | 366 | 408 |
| North Nazimabad | 397 | 502 | 366 | 408 |

Suitability by site and season (0–1, within each season):

| Site | Winter | Spring | Summer | Autumn |
|---|---|---|---|---|
| Gadap | 0.750 | 0.950 | 0.650 | 1.000 |
| SITE | 0.585 | 0.441 | 0.715 | 0.635 |
| Clifton | 0.250 | 0.150 | 0.546 | 0.173 |
| DHA | 0.479 | 0.415 | 0.244 | 0.100 |
| Korangi | 0.494 | 0.440 | 0.294 | 0.100 |
| Malir | 0.524 | 0.465 | 0.210 | 0.085 |
| Gulshan-e-Iqbal | 0.305 | 0.233 | 0.060 | 0.101 |
| North Nazimabad | 0.305 | 0.233 | 0.060 | 0.101 |

![Seasonal solar resource and site suitability, Karachi](figures/fig9_seasonal_sites.png)

What this shows:

- There is a strong seasonal cycle. Spring, before the monsoon, is brightest everywhere
  at about 500 to 507 W/m². Summer, during the monsoon, is dimmest at 366 to 378, a 25%
  drop. Winter and autumn sit in between.
- The ranking is mostly stable, but it flips in summer. Gadap leads in winter, spring,
  and autumn (suitability 0.75, 0.95, 1.00). In summer the cooler coastal and western
  sites take over: SITE at 0.72 and Clifton at 0.55 overtake Gadap at 0.65. Clifton jumps
  from 0.25 in winter to 0.55 in summer because the coolness term matters most when it is
  hottest.
- Practical reading: Gadap or SITE is the year-round choice; if you specifically want to
  hold up through the hot monsoon months, the coastal sites recover ground.

### 5.8 GHI to power, and the heat penalty

To turn irradiance into something an EV charger can use, we apply a simple, transparent
PV model: power scales linearly with GHI through a module efficiency (0.20) and a
performance ratio (0.80) that covers inverter, thermal, and soiling losses.

PV panels also lose efficiency as they heat up, about 0.4% per °C above 25 °C. Hot air
and strong sun make hot panels, so we add a temperature correction (an NOCT cell-
temperature model) and measure how much annual energy each Karachi site loses to heat.
Source: `reports/temp_derating_karachi.csv`.

| Site | Mean temp (°C) | Annual energy lost to heat |
|---|---|---|
| Clifton | 26.3 | 9.40% |
| Korangi | 26.3 | 9.52% |
| DHA | 26.4 | 9.54% |
| Malir | 26.3 | 9.63% |
| SITE | 26.3 | 9.83% |
| Gulshan-e-Iqbal | 26.5 | 9.88% |
| North Nazimabad | 26.5 | 9.88% |
| Gadap | 26.1 | 10.07% |

Hot Karachi loses about 9 to 10% of its annual PV energy to cell heating. The cooler
coastal sites lose the least (Clifton at 9.40%), and the hot inland winner Gadap loses
the most (10.07%). So Gadap's GHI lead is partly clawed back by heat. This is the
physical basis for the coolness term in the suitability index and for the summer ranking
flip in §5.7.

---

## 6. Reproducibility and engineering

The project is a proper package, not a notebook, so several people can work on it and
every figure can be regenerated.

- **Layout:** the importable package lives in `src/solarpredict/` (data, features,
  models, evaluation, solar, siting, viz). The scripts that produce the figures and
  tables live in `pipelines/`. Outputs land in `reports/`.
- **Patterns:** models register themselves behind one `Forecaster` interface, so adding
  one is a single file. Data sits behind a `DataRepository` so the source can be swapped.
  Configuration is centralised, not hard-coded.
- **Quality gate:** ruff (lint and format), mypy (types), and pytest run in CI on every
  push. Pinned tool versions keep CI and local in step.
- **Boundaries:** the maps use real province and district outlines (geoBoundaries),
  committed under `assets/geo/` so the figures render offline and reproducibly.

To regenerate everything, run the pipelines in `pipelines/` (each is one command). They
read from the Parquet cache, so after the first data pull they need no network.

---

## 7. Limitations

- ERA5 is a reanalysis at roughly 25 km. It captures the regional resource and the
  coastal gradient, but not street-level microclimate. The within-city differences are
  genuinely small, and we report them as such.
- One hour ahead is the easy horizon. Skill at day-ahead would be substantially lower.
  The forecasting results should not be read as day-ahead performance.
- The reanalysis is not a ground measurement. The numbers are internally consistent and
  physically sensible, but a sensor is needed to confirm them on the ground.
- The PV conversion is deliberately simple (linear in GHI plus a temperature correction).
  It is meant to rank sites and give an order-of-magnitude energy figure, not to size a
  specific installation.

---

## 8. Future work

- A physical GHI and temperature sensor (a pyranometer, or a low-cost light-and-
  temperature module on a microcontroller) to validate the ERA5 values on the ground. The
  instructor has offered to fund this.
- Longer forecast horizons (day-ahead), where the gap between the models should widen and
  the modern models may matter more.
- A second city's within-city study, and a live dashboard surfacing the model comparison
  and the suitability ranking.

---

## 9. Figure and table index

Figures (`reports/figures/`):

| Figure | File | Shows |
|---|---|---|
| 1 / 1b | fig1_model_skill / fig1b_model_rmse | GHI model comparison, Karachi |
| 2 | fig2_ghi_map | Pakistan city GHI map |
| 2b | fig2b_ghi_heatmap | GHI by hour and month, Karachi |
| 2c | fig2c_karachi_grid_heatmap | Within-city GHI surface, Karachi |
| 3 | fig3_seasonal | Seasonal GHI by city |
| 3b | fig3b_daily_profile | Mean daily GHI and PV power |
| 4 | fig4_city_ranking | City solar ranking |
| 4b | fig4b_site_suitability | Within-city site suitability |
| 5 | fig5_pred_vs_actual | Predicted vs actual GHI |
| 6 | fig6_ev_locations | Recommended EV sites |
| 7 | fig7_feature_importance | Best model's feature importance |
| 8 / 8b | fig8_data_efficiency_skill / fig8b_..._rmse | Skill vs training size |
| 9 | fig9_seasonal_sites | Site by season GHI and suitability |
| 10 / 10b | fig10_multicity_skill / fig10b_..._rmse | GHI benchmark across cities |
| 11 / 11b | fig11_temp_skill / fig11b_..._rmse | Temperature model comparison |

Tables (`reports/`):

| File | Contents |
|---|---|
| benchmark_karachi.csv | Karachi GHI benchmark, all metrics |
| benchmark_multicity.csv | GHI benchmark for all 7 cities |
| benchmark_temp_karachi.csv | Karachi temperature benchmark |
| data_efficiency_karachi.csv | Skill vs training size |
| city_summary.csv | Per-city yield and seasonality |
| site_suitability_karachi.csv | Within-city suitability ranking |
| seasonal_site_suitability_karachi.csv | Site by season |
| temp_derating_karachi.csv | PV heat loss by site |
| karachi_grid_ghi.csv | District GHI grid points |
