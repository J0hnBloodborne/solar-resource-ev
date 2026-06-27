# Solar resource prediction and solar-EV charging siting in Pakistan

This document records the data, methods, and results behind the paper in full. Every
figure and table the project produces is included, and the source file for each number is
named so it can be traced to a run. It is written for a reader fluent in time-series
forecasting and PV, and it states what was done and what came out.

The work has two parts. The first compares machine-learning models at predicting global
horizontal irradiance (GHI) one hour ahead, and converts the forecast to PV power. The
second ranks locations by solar suitability and identifies good sites for solar-powered EV
charging, between cities and within one city (Karachi).

---

## 1. Data

### 1.1 Source and variables

The data is ERA5, ECMWF's fifth-generation reanalysis, served hourly through the
Open-Meteo Historical Archive (free, no key, record back to 1940). ERA5 assimilates
observations into a frozen IFS model cycle by 4D-Var and distributes surface fields on a
0.25° grid. Surface shortwave is the radiation scheme's output, so it already carries the
model's assimilated cloud and aerosol state rather than a raw satellite retrieval. That
makes it a physically consistent, gap-free series, which is what the sequence models need,
at the cost of the spatial smoothing discussed in §1.4.

Five hourly variables per location:

| Variable (Open-Meteo name) | Quantity | Role |
|---|---|---|
| `shortwave_radiation` | GHI, W/m² | Primary target |
| `temperature_2m` | 2 m air temperature, °C | Second target and covariate |
| `relative_humidity_2m` | Relative humidity, % | Covariate |
| `wind_speed_10m` | 10 m wind speed, m/s | Covariate |
| `cloud_cover` | Total cloud cover, % | Covariate |

GHI is the broadband irradiance on a horizontal plane and is the quantity that sets a
fixed flat panel's output, so it is the headline target. Temperature enters twice, as a
covariate for the GHI models and as a forecast target in its own right (§4), and it drives
the cell-temperature derate in §5.8.

### 1.2 Coverage

- **Period.** Five years of hourly data per location, roughly 43,800 records each, ending
  about a week before the run date (ERA5's processing latency).
- **Cities (7).** Islamabad, Lahore, Karachi, Peshawar, Multan, Quetta, Gilgit. These span
  the climatic range of the country: the Arabian Sea coast, the Punjab plains, the western
  high desert, and the northern mountains.
- **Karachi districts (8).** Clifton, DHA, Korangi, Malir, Gulshan-e-Iqbal, North
  Nazimabad, SITE, Gadap, spread from the southern coast to the northern urban fringe.
- **Karachi GHI grid.** A regular 0.1° grid over the Karachi district, the 29 points that
  fall inside the district polygon, used for the intra-city resource surface.

### 1.3 Acquisition pipeline

The archive returns a full multi-year hourly series for a point in a single request, so
the call count scales with the number of locations, not the number of days. Open-Meteo
prices a request by a cost roughly proportional to (variables × days), so a five-year,
five-variable pull for one point is worth several hundred units against the per-minute
budget. The fetcher accommodates that by:

- splitting each location into 365-day chunks to keep a single request under the
  per-minute ceiling,
- backing off 62 s and retrying when a limit error is returned,
- writing each location's full series to a Parquet file keyed by site, date range, and a
  hash of the variable set, so nothing is fetched twice.

The fetcher sits behind a `DataRepository` interface, so the archive can later be swapped
for a database without touching the callers.

### 1.4 What the resolution resolves

At a ~25 km effective scale a metro the size of Karachi occupies only a 3×4 block of grid
cells, so any intra-urban contrast is a blend of a handful of cell values rather than a
street-level field. Two consequences run through the whole study. Between cities, separated
by hundreds of kilometres, the contrasts are fully resolved and the inter-city ranking is
solid. Within a city the dynamic range is about 5%, so the intra-city work is presented as
an interpolated surface and a ranking of coarse cells, not as point claims about
neighbourhoods. ERA5 is also a reanalysis, not a ground measurement: internally consistent
and physically sensible, but unvalidated against a local pyranometer.

---

## 2. Methods

### 2.1 Clear-sky model and night masking

A clear-sky model gives the GHI expected under cloud-free skies from solar geometry alone;
we use pvlib's Haurwitz model. It serves two purposes.

Night masking. Night hours are long runs of true zeros that every model predicts
correctly. Because R² = 1 − SS_res/SS_tot, those zeros inflate the total sum of squares
(the variance about the 24 h mean) far more than the residual sum of squares, which pushes
R² toward 0.99 for any model and hides the daytime skill. Dropping hours with solar zenith
beyond about 90° (clear-sky GHI near zero) removes that and leaves only the daytime
variance the forecast actually has to explain.

Clear-sky index. The ratio kt = GHI / GHI_cs detrends the deterministic astronomical and
seasonal component, isolating the stochastic cloud transmittance, which is the only
genuinely hard part of the signal to predict.

### 2.2 Features

Each row for the tabular models is built only from information available at or before the
forecast time, so the one-step-ahead prediction is leakage-free:

- lagged GHI at 1, 2, 3, and 24 hours,
- lagged covariates (temperature, humidity, wind, cloud) at 1 and 24 hours,
- rolling mean and standard deviation of GHI over 3 h and 24 h windows, shifted back one
  step,
- the lagged clear-sky index,
- the clear-sky GHI at the forecast hour, which is deterministic and therefore safe to use
  concurrently,
- sine and cosine encodings of hour-of-day and day-of-year.

The radiation components the archive can also serve (direct, diffuse, DNI) are excluded as
predictors. They are an algebraic decomposition of the target (GHI = DNI·cos θ + DHI), so
feeding them concurrently is target leakage.

### 2.3 Splitting and leakage control

The split is a strictly chronological holdout: the models train on the earlier part of the
series and are scored on the most recent 20%. No k-fold or shuffling, because the strong
temporal autocorrelation would place near-duplicate neighbours on both sides of a fold and
leak the answer. The sequence models receive a 168 h causal context window immediately
before the test set so they have history to condition on; only the held-out tail is scored.
Every feature at time t uses values realised at t−1 or earlier, plus terms deterministic at
t (clear-sky GHI, harmonic time).

### 2.4 Metrics

All models are scored identically, on daytime hours only: MAE and RMSE (W/m² for GHI, °C
for temperature), R², nRMSE (RMSE over the mean daytime value, for cross-city comparison),
MBE (mean bias), and the forecast skill score

S = 1 − RMSE_model / RMSE_reference.

The skill score is the headline. For GHI the reference is smart persistence (persist kt,
rescale by the next hour's clear-sky GHI), the standard reference in the solar-forecasting
literature. For temperature the reference is plain persistence, since temperature has no
clear-sky model and no night to mask. MBE is reported because the clear-sky reference
carries a large negative bias that the learned models neutralise (§3.1). MAPE is excluded;
it is singular near sunrise and sunset where GHI approaches zero.

### 2.5 The model lineup

Four tiers, from naive to modern. The naive tier is the reference any solar-forecasting
result is measured against, not an open question about whether ML is worthwhile.

| Tier | Models | Inductive bias |
|---|---|---|
| 0 — naive / statistical | persistence, smart (clear-sky) persistence, climatology | Reference forecasts |
| 1 — classical ML | Ridge, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost | Axis-aligned partitioning over engineered lag / rolling / clear-sky features; strong on heteroscedastic tabular targets, no extrapolation |
| 2 — deep learning | LSTM, NHITS | Learned sequence representations; NHITS uses multi-rate pooling and hierarchical interpolation for multi-scale seasonality |
| 3 — foundation model | Chronos-2 | ~120M-parameter encoder-decoder, tokenises the series and forecasts in-context, zero-shot with no gradient step on our data |

The three references: persistence carries this hour forward (strong for slow signals, weak
for GHI which swings fast at the edges of the day); smart persistence carries the clear-sky
index forward and rescales it by the next hour's clear-sky GHI; climatology returns the
historical mean for that hour and season. The deep models run on PyTorch through
neuralforecast (with `num_workers=0`, gradient clipping, and input standardisation to keep
NHITS stable on Windows). Every model implements one `Forecaster` interface and is scored
through one harness, so adding a model is a single file.

---

## 3. Forecasting GHI (Part A)

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

**What this shows.** The five tree ensembles cluster within 3 W/m² of each other (41.6 to
44.5 RMSE); the ordering among them is at the noise level, and Extra Trees' extra split
randomisation gives it a marginal edge on this noisy target. The gap from the trees to
NHITS (46.1) to Chronos-2 (48.6) is real but small, and the LSTM and the linear model trail
at about 52. The best model cuts RMSE 40% below the smart-persistence reference (41.6
versus 69.6).

The MBE column carries part of the story. Smart persistence sits at −43 W/m², the Haurwitz
clear-sky underestimate propagated through the rescaling; every learned model collapses the
magnitude of that bias below about 6 W/m². So a substantial share of their gain is bias
removal layered on top of variance reduction, not variance reduction alone.

The reason GHI is this forecastable one step ahead falls out of the decomposition
GHI = GHI_cs·kt. GHI_cs is deterministic and supplied to the models directly; kt is
strongly persistent at an hourly lag because cloud fields advect slowly relative to one
hour. The conditional-mean predictor is therefore close to kt(t−1)·GHI_cs(t) plus a learned
correction, and the headroom over smart persistence is exactly the learnable part of kt's
hour-to-hour evolution. One hour ahead that headroom is worth 40%; at a day-ahead horizon
it would be far smaller, because kt persistence decays.

### 3.2 Predicted against actual

The best model's predictions against measured GHI on the test set:

![Predicted vs actual GHI, best model, Karachi](figures/fig5_pred_vs_actual.png)

Points sit tight to the 1:1 line across the full range. The scatter widens toward high
irradiance, where broken cloud produces the largest absolute deviations; this is the
heteroscedastic, cloud-driven residual that no model fully removes and that sets the skill
ceiling.

### 3.3 What the models rely on

Gini importances from the best tree model:

![Feature importance, best tree model](figures/fig7_feature_importance.png)

The recent GHI lags and the concurrent clear-sky GHI carry almost all the weight. The model
leans on the deterministic solar geometry plus the last few hours of cloud state, which is
the same reading the decomposition in §3.1 gives. The covariate lags (cloud, humidity) add
a thin margin; the cyclical-time terms matter little once clear-sky GHI is present, because
clear-sky GHI already encodes the phase.

### 3.4 The same benchmark across seven cities

The 12-model benchmark, rerun for every city, so the ranking can be read across the
country's climates. Source: `reports/benchmark_multicity.csv`. Skill score per model per
city, mean across cities in the last column:

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

**What this shows.** The ranking is invariant across all seven climates: a tree ensemble
wins in every city (Extra Trees in four, CatBoost in two, LightGBM in one), and it is never
the deep or foundation model. That invariance is the substantive result, because it shows
the tree-ensemble advantage on hour-ahead GHI is not a Karachi artefact but holds from the
coast to the high desert to the mountains.

Skill tracks the variance structure of kt rather than the absolute resource. Quetta's clear,
low-variability high-desert skies give the most learnable signal (0.51); Gilgit's orographic
convection gives the least (0.37). Absolute RMSE is actually lowest where mean GHI is lower
(Multan 34.8, Quetta 33.3 W/m²), which is why nRMSE and skill, not raw RMSE, are the fair
cross-city comparison. Chronos-2 stays positive everywhere zero-shot (0.21 to 0.38) and
beats the trained LSTM in every city, which is a strong result for a model that never sees
local data; a locally trained tree still beats it everywhere.

### 3.5 How much training data the models need

Foundation and deep models are usually pitched as the answer when local history is short.
We tested that directly: fix the test set, vary the training window from one month to four
years, rerun. Source: `reports/data_efficiency_karachi.csv`. Skill by training size:

| Model | 1 mo | 3 mo | 6 mo | 1 yr | 2 yr | 4 yr |
|---|---|---|---|---|---|---|
| chronos2 (zero-shot) | 0.301 | 0.301 | 0.301 | 0.301 | 0.301 | 0.301 |
| nhits | 0.202 | 0.271 | 0.298 | 0.332 | 0.338 | 0.336 |
| extra_trees | 0.158 | 0.189 | 0.227 | 0.394 | 0.406 | 0.410 |
| xgboost | −0.122 | 0.105 | 0.181 | 0.363 | 0.391 | 0.384 |

![Data efficiency, skill vs training size](figures/fig8_data_efficiency_skill.png)

![Data efficiency, RMSE vs training size](figures/fig8b_data_efficiency_rmse.png)

**What this shows.** Chronos-2's skill is flat at 0.30 because it never trains; that is its
zero-shot operating point, invariant to local sample size, and it is the best model in the
table at three months or less. XGBoost is below the naive reference at one month (skill
−0.12), overfitting the short warm-up before it has seen enough of the kt distribution. The
crossover is at roughly six to twelve months: the trees overtake only once they have about a
year of history, then plateau near 0.41 by one to two years. NHITS reaches about 0.30 within
six months, generalising from less data than the trees because its multi-rate structure
captures the seasonal shape from fewer examples.

The operational reading for a newly instrumented site is concrete: deploy Chronos-2 or
NHITS from day one and switch to a trained tree after roughly a year of local data has
accumulated. For a solar-EV operator rolling out chargers into cities with no local
forecasting history, that means a usable forecast on day one without waiting to collect a
training set.

---

## 4. Forecasting temperature, and the model-target match

### 4.1 Temperature benchmark

The earlier S2Cool project predicted temperature alongside GHI, so we ran the same tiers on
hour-ahead 2 m air temperature in Karachi. The reference is plain persistence. Source:
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

### 4.2 Why the ranking inverts

On temperature the ranking inverts. Chronos-2 leads (0.60 skill, 0.35 °C RMSE), NHITS is
second (0.59), and the trees sit behind them (0.56 and down). R² is about 0.99 for every
model, including ridge, which says the target is nearly deterministic.

The reason is the signal's spectral character. Temperature's variance is concentrated in the
diurnal and seasonal harmonics, with high lag-1 autocorrelation and little broadband noise.
A learned continuous sequence representation matches that smooth, periodic phase structure
closely; the trees' piecewise-constant fits over hand-built lags are a coarser approximation
to a smooth field and lose a few hundredths of skill to it. GHI is the opposite case: the
deterministic part is easy, and the hard part is the high-frequency, regime-switching cloud
residual, where a gradient-boosted tree over engineered cloud and lag features does best.

The general statement for the paper is that model choice should follow the target's signal
character rather than its recency. Smooth, strongly periodic targets such as temperature (or
electrical load) favour sequence and foundation models; spiky, regime-switching targets with
informative tabular covariates, such as irradiance under cloud, favour gradient-boosted
trees. Running both targets through one harness is what exposes the crossover, and it is why
the foundation model earns its place in the lineup rather than being a token modern row.

---

## 5. Solar resource and where to put a charger (Part B)

Part B ranks locations by solar suitability and turns that into siting for solar-powered EV
charging, first between cities, then within Karachi, and finally into delivered charging
energy.

### 5.1 Resource and EV potential across cities

Five-year mean GHI per city, converted to an annual energy yield on the horizontal plane.
Source: `reports/city_summary.csv`.

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

**What this shows.** Every city in the set delivers more than the national-average yield of
Germany (about 1080 kWh/m²/yr) or the UK (about 1000), both of which run large EV-charging
fleets on grids with a substantial solar share. The solar resource is therefore not the
binding constraint on solar-EV charging anywhere in Pakistan; demand density, grid
connection, and capital are. That reframes the siting question from "is there enough sun"
to "where does the sun coincide with charging demand."

Two cities sit above the coastal study city on raw resource. Quetta leads the country by a
wide margin (2174 kWh/m²/yr, the clear high-desert skies), and Gilgit (1960) edges Karachi.
As EV adoption moves inland and north, these are the highest-yield hosts for solar charging,
and a solar canopy there delivers proportionally more energy per square metre than the same
canopy on the coast. They are the untapped potential the paper should flag.

Karachi earns the within-city study for a different reason: it has the lowest seasonality of
the seven (CV 0.178), so its month-to-month solar supply is the steadiest in the country.
For a charging station that quantity matters as much as peak yield, because a flat seasonal
profile is a more uniform solar duty cycle and needs the smallest storage or grid buffer to
hold a target uptime through the year. A coastal Karachi charger trades a little peak yield
for the most predictable year-round supply.

### 5.2 The diurnal and seasonal cycle

Two views of the Karachi GHI field, both in local time:

![GHI by hour of day and month, Karachi](figures/fig2b_ghi_heatmap.png)

![Mean daily GHI and PV potential, Karachi](figures/fig3b_daily_profile.png)

The heatmap places the diurnal peak near 13:00 local and the seasonal band clearly: a
pre-monsoon spring maximum and a summer monsoon minimum. The daily profile is the mean day,
with the right axis reading GHI directly as PV power for a 1 kW-peak panel. For charging,
the shape sets the natural solar window, roughly 09:00 to 16:00, within which a solar-only
station meets demand directly and outside which it draws on storage or the grid.

### 5.3 The intra-city resource surface

The within-city surface samples GHI across the Karachi district on the 0.1° grid (29
interior points) and interpolates a continuous field clipped to the real district boundary;
the seaward edge is the coastline.

![Intra-city GHI gradient, Karachi](figures/fig2c_karachi_grid_heatmap.png)

GHI runs from about 412 W/m² in the urban core to 440 in the rural north, a ~5% range that
is real but at the resolution floor of §1.4. The densest built-up cells read slightly lower
than the open land around them, consistent with the higher aerosol and humidity loading over
the urban core depressing surface irradiance. The gradient is monotonic from the coast
inland, which is what makes the northern and western fringe the resource-favoured part of
the city.

### 5.4 The suitability index

The districts are ranked by a transparent weighted index, not a black box, so every term is
explainable. It blends three normalised components: annual GHI yield (weight 0.60), seasonal
consistency (0.25), and panel coolness (0.15, since a hotter cell loses efficiency). Yield
dominates, so the ranking is resource-led with consistency and coolness as tie-breakers.
Source: `reports/site_suitability_karachi.csv`.

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

**What this shows.** The northern and western fringe (Gadap, then SITE) tops the ranking: it
holds the most GHI of the built-up districts and runs marginally cooler. The spread in raw
yield across the districts is only about 2% (1950 to 1990 kWh/m²/yr), so the index is really
separating sites on small, consistent differences; the suitability scores spread wider than
the underlying yields because each component is min-max normalised across only eight sites.
The practical content is the ordering, not the gaps: Gadap and SITE are resource-favoured,
the southern coastal cluster is close behind, and within that cluster the coolness term
decides.

### 5.5 Recommended charging sites

The recommendation map places the ranked sites on the resource surface, top three starred.

![Recommended solar-EV charging sites, Karachi](figures/fig6_ev_locations.png)

| Rank | Site | Coordinates (lat, lon) | Suitability | Charging-relevant character |
|---|---|---|---|---|
| 1 | Gadap | 25.050, 67.110 | 0.800 | Northern urban fringe, growing, lower density, most resource |
| 2 | SITE | 24.910, 66.990 | 0.670 | Industrial estate, heavy daytime vehicle flow, good resource |
| 3 | Clifton | 24.810, 67.030 | 0.275 | Dense commercial coast, high demand, coolest site |

Gadap and SITE are the resource-led year-round picks. Clifton ranks third on the strength of
its coolness term despite slightly lower raw GHI, and it has the strongest charging demand of
the three as a dense commercial district, which is the demand-side point developed next.

### 5.6 Why the sunniest ground is not a charging site

The resource surface in §5.3 covers the whole district, and its brightest cells are not in
the city. The rural north (about 440 W/m²) and the open south-west coast near Cape Monze and
the Hub river mouth (about 435) read higher than the urban core (411 to 419), and those are
genuine ERA5 readings. A charger does not go there, because those cells are open coast,
hills, and barren hinterland with little population, no arterial roads, and no grid. An EV
charging station is sited by demand first: it belongs on commercial corridors and dense
districts where vehicles actually dwell. The sunniest ground suits a utility-scale solar
farm feeding the grid, not a forecourt charger.

This is the core siting tension, and the recommendation map shows both layers at once. The
resource surface underneath identifies where the energy is; the urban site markers identify
where the demand is; and the right charger sits where the two overlap best, which is the
sunnier northern and western fringe of the built-up area rather than the resource peak. The
~5% intra-city resource gradient is small enough that demand and grid access dominate the
final choice, with the suitability ranking acting as the resource tie-breaker among
otherwise comparable urban candidates.

### 5.7 Suitability through the four seasons

The year is split into the four meteorological seasons (winter DJF, spring MAM, summer JJA,
autumn SON) and the suitability index is recomputed inside each, so the seasonal movement is
visible. Source: `reports/seasonal_site_suitability_karachi.csv`.

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

Suitability by site and season (0–1, normalised within each season):

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

**What this shows.** The seasonal cycle is strong and matters directly for charger sizing.
Spring, before the monsoon, is brightest everywhere at about 500 to 507 W/m²; the summer
monsoon is dimmest at 366 to 378, a 25% drop; winter and autumn sit between. A solar-only
charger therefore sees its supply swing by about a quarter between spring and summer, which
is the gap that storage or a grid tie has to cover if the station is to hold throughput
year-round. Karachi's low inter-annual seasonality (§5.1) keeps that swing smaller than it
would be at an inland site.

The ranking is mostly stable but it inverts in summer. Gadap leads in winter, spring, and
autumn (0.75, 0.95, 1.00); in summer the cooler coastal and western sites take over, with
SITE at 0.72 and Clifton at 0.55 overtaking Gadap at 0.65. Clifton climbs from 0.25 in
winter to 0.55 in summer because the coolness term carries most weight when ambient
temperature is highest and cell derating bites hardest. For siting this means the
resource-optimal location is itself mildly season-dependent: the northern fringe for the
dry, bright three-quarters of the year, the coast for the hot monsoon quarter when a panel's
thermal losses are largest.

### 5.8 From irradiance to delivered charge

To make the resource concrete for a charging operator we apply a transparent PV model,
power = GHI × area × η × PR with module efficiency η = 0.20 and performance ratio PR = 0.80,
and a NOCT cell-temperature correction that derates output by about 0.4% per °C above 25 °C.

Per square metre of array at the top-ranked site (Gadap, 1990 kWh/m²/yr incident), the model
delivers 1990 × 0.20 × 0.80 ≈ 318 kWh/m²/yr of DC electricity, falling to about 286
kWh/m²/yr (0.78 kWh/m²/day) after the ~10% heat derate below. A realistic 200 m² solar
canopy over the charging bays therefore yields roughly 57 MWh/yr, about 157 kWh on the mean
day. Against a 50 kWh EV battery that is about three full charges per day from solar alone,
or on the order of 1,000 vehicle-kilometres per day at 0.16 kWh/km, before any grid support.

The seasonal swing in §5.7 carries through to delivered energy, though longer summer days
offset part of the lower midday irradiance, so the daily total moves by roughly a fifth
across the year rather than the full quarter seen in midday GHI. The same canopy gives around
3.5 charges on a bright spring day and closer to 3 in the monsoon, so a solar-only station
sized to the spring peak is undersupplied in summer. Sizing the array to the summer floor, or
pairing it with a battery buffer or a grid trickle, is what holds a constant charging
throughput across the year.

The cell-heating derate is the temperature target paying off in the power domain. Source:
`reports/temp_derating_karachi.csv`.

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

Hot Karachi loses about 9 to 10% of its annual PV energy to cell heating, which at 200 m² is
roughly a third of a charge per day given away to temperature. The cooler coastal sites keep
the most of it (Clifton at 9.40%) and the hot inland resource winner Gadap the least (10.07%),
which is the physical basis for the coolness term in the suitability index and for the summer
ranking flip in §5.7. A reflective or ventilated mounting that lowers cell temperature
recovers part of this loss, and a canopy form factor over the bays has the side benefit of
shading vehicles in Karachi's heat.

### 5.9 What this means for a Karachi solar-EV station

Pulling Part B together: the binding levers for a Karachi solar-EV charger are array area
and demand-led siting, not the ~5% intra-city resource gradient. The resource is ample
everywhere in the city and far above European baselines, so the design choices that move the
numbers are how large a canopy the site allows and how well it sits in the charging demand.
Among the built-up districts the sunnier, cooler northern and western fringe (Gadap, SITE)
is the resource-favoured pick for a year-round station, while the dense coastal commercial
districts (Clifton, with Korangi and DHA nearby) trade a little resource for stronger demand
and the best summer performance. A practical specification reads: choose a high-demand
built-up district on the northern or coastal fringe, size the canopy to the summer-monsoon
floor with a battery or grid buffer for the spring surplus, and expect on the order of 0.8
kWh per m² of array per day delivered after thermal losses.

---

## 6. Reproducibility and engineering

The project is a package rather than a notebook, so several contributors can work in
parallel and every figure regenerates from cached data.

- **Layout.** The importable package is `src/solarpredict/` (data, features, models,
  evaluation, solar, siting, viz). The scripts that produce the figures and tables are in
  `pipelines/`; outputs land in `reports/`.
- **Patterns.** Models register behind one `Forecaster` interface, so adding one is a single
  file with no edits to shared code. Data sits behind a `DataRepository` so the source is
  swappable. Configuration is centralised.
- **Quality gate.** ruff (lint and format), mypy (types), and pytest run in CI on every
  push, with pinned tool versions so CI and local stay in step.
- **Maps.** The figures draw real province and district outlines (geoBoundaries), committed
  under `assets/geo/` so they render offline; the intra-city surface is interpolated with
  scipy and masked to the district polygon.

Each pipeline is one command and reads from the Parquet cache, so after the first data pull
the whole set regenerates with no network.

---

## 7. Limitations

- ERA5 is a reanalysis at a ~25 km effective scale. It resolves the regional resource and
  the coastal gradient but not sub-cell microclimate, so the intra-city differences are
  small (~5%) and the within-city work is a ranking of coarse cells, not of neighbourhoods.
- One hour ahead is the easy horizon. Day-ahead skill would be materially lower because the
  clear-sky-index persistence the models exploit decays with lead time, so these numbers
  should not be read as day-ahead performance.
- The reanalysis is internally consistent and physically sensible but is not validated
  against a local pyranometer.
- The PV conversion is first-order (linear in GHI plus a NOCT temperature derate). It is
  built to rank sites and give an order-of-magnitude energy and charging figure, not to size
  a specific installation, which would need tilt, orientation, soiling, and a measured
  module model.

---

## 8. Figure and table index

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
