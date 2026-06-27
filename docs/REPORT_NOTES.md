# Report source notes — Solar Resource Prediction

Working notes for the instructor report (M7). Raw material and decisions, not the
final prose. Numbers are from the committed runs (`reports/*.csv`, `reports/figures/`).
All results come from real Open-Meteo data and actual model runs — no fabricated
numbers.

## What the paper is (from the meeting transcript)

Conference paper, "Solar Resource Prediction." The student team supplies
results/figures; the instructor writes the prose. Two contributions:

1. Compare the predictive performance of several ML/AI models for estimating solar
   power generation potential (predict GHI, convert to power).
2. Rank locations by solar suitability and identify good sites for solar-powered EV
   charging stations.

The instructor decided AI *is* the method (the "is AI needed?" debate is closed).
Baselines stay only as the standard skill-score reference. Single city, multiple
sites, with a city chosen where variation exists. Data from Open-Meteo. No
plagiarism / no fabricated results.

## Data

- Source: Open-Meteo Historical Archive (ERA5 reanalysis), hourly, free, no key.
- Variables fetched: GHI (shortwave_radiation) + temperature, humidity, wind, cloud
  cover (5 "modeling" variables). 5 years per location (≈43,824 hourly rows).
- Resolution ≈ 9–25 km. This matters for within-city work (below).
- 7 candidate cities: Islamabad, Lahore, Karachi, Peshawar, Multan, Quetta, Gilgit.

## Part A — model benchmark (hour-ahead GHI, Karachi, 5 years)

Test = last 20% (chronological). Metrics computed daytime-only (night zeros masked).
Headline metric = forecast skill score vs smart (clear-sky) persistence.
Full table: `reports/benchmark_karachi.csv`. Figures 1, 5, 7.

| model | tier | RMSE | R² | skill |
|---|---|---|---|---|
| extra_trees | classical ML | 41.6 | 0.976 | **+0.40** |
| catboost | classical ML | 41.8 | 0.976 | +0.40 |
| xgboost | classical ML | 43.3 | 0.974 | +0.38 |
| lightgbm | classical ML | 44.0 | 0.973 | +0.37 |
| random_forest | classical ML | 44.5 | 0.973 | +0.36 |
| nhits | deep learning | 46.1 | 0.971 | +0.34 |
| chronos2 | foundation model | 48.6 | 0.967 | +0.30 |
| lstm | deep learning | 52.0 | 0.963 | +0.25 |
| ridge | classical ML | 52.2 | 0.962 | +0.25 |
| smart_persistence | baseline (ref) | 69.6 | 0.933 | 0.00 |
| climatology | baseline | 82.3 | 0.906 | −0.18 |
| persistence | baseline | 138.2 | 0.736 | −0.99 |

Findings / talking points:
- Tree ensembles lead; the modern NHITS is close behind; the zero-shot Chronos-2
  foundation model is competitive without any training on this site.
- ML cuts hour-ahead error ~40% vs the best naive method — concrete answer to "does
  AI help?": yes, clearly, at this horizon.
- Why so predictable: most of the signal is the deterministic solar geometry (fed in
  as clear-sky GHI + cyclical time); 1 hour ahead clouds persist, so the models are
  essentially "smart persistence plus learned corrections," and they remove the
  clear-sky model's ~−43 W/m² bias (see MBE column).
- Honest scope: this is the easy horizon. Day-ahead skill would be much lower
  (probably ~0.1–0.2). State this so it isn't oversold.

### Data-efficiency experiment (the scarce-data story) — figure 8

Fixed test, training size varied 1 month → 4 years. `reports/data_efficiency_karachi.csv`.

Skill by training size (Karachi):

| model | 1mo | 3mo | 6mo | 1y | 4y |
|---|---|---|---|---|---|
| chronos2 (zero-shot) | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 |
| nhits | 0.20 | 0.27 | 0.30 | 0.33 | 0.34 |
| extra_trees | 0.16 | 0.19 | 0.23 | 0.39 | 0.41 |
| xgboost | −0.12 | 0.11 | 0.18 | 0.36 | 0.38 |

- Chronos-2 is flat (it never trains) and best at 1–3 months; xgboost is *worse than
  the naive baseline* at 1 month.
- Cross-over at ~6–12 months: trained models overtake only once they have a year+.
- Conclusion: foundation/deep models are the better choice when data is scarce; trees
  win with plenty. This is the second main contribution.

## Part B — siting

### City choice
Karachi is the data-chosen city for the within-city study because it has the most
intra-city GHI spread (it splits into ~16 ERA5 cells with a coastal→inland gradient;
Lahore is nearly flat). See `select-city` / earlier checks. Intra-city differences
are real but small (~3%) — honest framing required.

### Inter-city comparison — figures 2, 3, 4. `reports/city_summary.csv`
Annual GHI yield (kWh/m²/yr), 5-year mean:

| city | yield | daytime GHI (W/m²) | seasonality CV |
|---|---|---|---|
| Quetta | 2174 | 463 | 0.25 |
| Gilgit | 1960 | 421 | 0.28 |
| Karachi | 1950 | 418 | **0.18** (most consistent) |
| Multan | 1793 | 383 | 0.23 |
| Lahore | 1696 | 369 | 0.26 |
| Islamabad | 1692 | 365 | 0.29 |
| Peshawar | 1650 | 355 | 0.30 |

Talking points:
- Quetta and Gilgit sit *above* our study city (Karachi) on solar resource — they are
  the untapped higher-solar candidates. If EV adoption reaches them, they would be
  even better hosts for solar charging than the coastal cities.
- Karachi has the lowest seasonality (steadiest year-round solar) — useful for a
  charging station that needs consistent supply, even though its peak yield is lower
  than Quetta's.
- Viability floor: even Pakistan's least-sunny city (Peshawar, 1650) exceeds the
  national-average yield of Germany/UK (~1000–1100). Solar-EV is viable nationwide.
- Inter-city spread is large (28% from Quetta to Peshawar) — the real, strong signal,
  vs the small (~3%) within-city spread.

### Within-Karachi siting — (to build) figures 6 + intra-city heatmap
- Sample the ~7 km grid over Karachi (16 cells) for a GHI heatmap; map the named
  districts onto it for the recommended-sites table.
- Suitability index: blend mean/seasonal GHI, PSH, low-irradiance (cloud) penalty,
  and consistency. Northern/inland sites (Gadap, SITE) rank highest, coastal (Clifton,
  DHA) lowest.
- Convert GHI → power with a fixed PV model so the gradient reads as annual energy.

## Methodology (for the methods section)
- Forecast skill score S = 1 − RMSE_model / RMSE_reference, reference = smart
  (clear-sky) persistence (persist the clear-sky index, rescale by next-hour clear-sky
  GHI). Standard in solar forecasting; avoids the misleading high R² that night zeros
  produce.
- Night mask: drop hours with solar zenith ≳ 90° (clear-sky GHI ≈ 0) from all metrics.
- Clear-sky GHI via pvlib (Haurwitz).
- Strictly chronological train/val/test (no shuffling — that leaks). Context window
  prepended to the test set so sequence models have history.
- Leakage discipline: features at t use only lagged GHI/covariates + deterministic
  terms (clear-sky GHI, cyclical time); no concurrent radiation components.
- Models: tier-0 persistence/smart-persistence/climatology; tier-1 Ridge, RF,
  ExtraTrees, XGBoost, LightGBM, CatBoost; tier-2 LSTM + NHITS (neuralforecast,
  covariate-aware); tier-3 Chronos-2 (Amazon, ~120M-param transformer, zero-shot).

## Multi-city benchmark (Part A breadth — fig10 / fig10b)
Full 12-model benchmark run for all 7 cities (not just Karachi), 5-year hourly,
hour-ahead, daytime-masked, skill vs smart persistence. Source:
reports/benchmark_multicity.csv; figures fig10_multicity_skill.png (city x model
skill heatmap) + fig10b_multicity_rmse.png.
- The ranking is the SAME in every city: tree ensembles win. Mean skill across
  cities — extra_trees 0.418, catboost 0.416, xgboost 0.408, lightgbm 0.404,
  random_forest 0.388; then nhits 0.336, chronos2 0.285, lstm 0.263 (~ ridge 0.263).
  Naive: climatology -0.30, plain persistence -0.97 (both worse than smart persistence).
- Best per city: extra_trees wins 4/7 (Karachi, Multan, Quetta, Gilgit), catboost 2
  (Lahore, Peshawar), lightgbm 1 (Islamabad) — i.e. it is always a gradient-boosted /
  bagged tree, never the deep or foundation model.
- Chronos-2 (zero-shot, NO local training) still posts positive skill everywhere
  (0.21-0.38) and beats the LSTM — strong for an off-the-shelf model — but a locally
  trained tree beats it in all 7 cities. This is the honest headline: on a
  near-deterministic, short-horizon target the modern foundation model is competitive
  out-of-the-box but does not dethrone classical ML.
- Predictability tracks climate: highest skill in Quetta (0.51, clear high-desert
  skies), lowest in Gilgit (0.37, complex mountain weather).

## Second target: temperature + the model-vs-target insight (fig11 / fig11b)
S2Cool also predicted temperature, so we benchmark hour-ahead 2 m air temperature
on the same tiers (skill vs plain persistence; no clear-sky reference / night mask
for temperature). Source: reports/benchmark_temp_karachi.csv.
- THE RANKING FLIPS vs GHI. On temperature the modern models win: Chronos-2 best
  (skill 0.603, RMSE 0.35 degC), NHITS second (0.587), THEN the trees
  (extra_trees 0.564, random_forest 0.552, ...). R^2 ~ 0.99 for everything (temperature
  is smooth and strongly periodic).
- Interpretation (the paper's nuance): the best model depends on the target's
  character. Temperature is smooth + periodic -> sequence/foundation models
  (Chronos-2, NHITS) shine. GHI is spiky + cloud-driven -> engineered-lag tree
  ensembles win. So the foundation model is not a gimmick: it is genuinely best on
  the smooth target and competitive on the hard one. Including both targets is what
  makes that visible.
- PV heat-derating (reports/temp_derating_karachi.csv, NOCT model, -0.4%/degC): hot
  Karachi loses ~9.4-10.1% of annual PV energy to cell heating. Cooler coastal sites
  lose least (Clifton 9.40%) and the hot inland winner Gadap most (10.07%) — so
  Gadap's GHI lead is partly clawed back by heat, giving the coastal-coolness term in
  the suitability index a concrete physical basis (and explaining the summer flip).

## Figure inventory (reports/figures/)
- fig1_model_skill / fig1b_model_rmse — model comparison (Part A).
- fig2_ghi_map — Pakistan city GHI map.
- fig3_seasonal — seasonal GHI by city.
- fig4_city_ranking — city solar ranking (Quetta/Gilgit highlighted).
- fig5_pred_vs_actual — predicted vs actual, best model.
- fig7_feature_importance — best tree's features.
- fig8_data_efficiency_skill / fig8b_..._rmse — scarce-data story.
- fig2b_ghi_heatmap (hour x month) + fig3b_daily_profile (GHI/PV, local time).
- fig2c_karachi_grid_heatmap — continuous GHI surface clipped to the real Karachi
  district outline (interpolated from a ~0.1 deg interior grid).
- fig4b_site_suitability (within-city ranking) + fig6_ev_locations (recommended sites
  over the GHI surface).
- fig9_seasonal_sites — site x season GHI + suitability heatmaps.
- fig10_multicity_skill / fig10b_multicity_rmse — (city x model) benchmark heatmaps.
- fig11_temp_skill / fig11b_temp_rmse — hour-ahead temperature model comparison.

## Maps (real geography, geoBoundaries gbOpen PAK ADM1/ADM2)
- Inter-city (fig2) draws Pakistan's province outlines + national border; intra-city
  figures draw the true Karachi district polygon (coastline = the seaward edge).
- Boundaries committed under assets/geo/ so figures render offline + reproducibly
  (geopandas + shapely; surface interpolation via scipy.griddata, masked to the polygon).

## Within-city siting & the resource-vs-demand point (addresses "why not SW Karachi?")
- The GHI surface covers the WHOLE district. The sunniest zones are genuinely the
  rural north (~440 W/m^2) and the open south-west coast (~435 W/m^2, e.g. Cape Monze /
  Hub-river mouth) — both real ERA5 readings, ~5% above the 411-419 W/m^2 urban core
  (the dense city has more aerosol/humidity loading that depresses surface GHI).
- We do NOT site a charger in those high-GHI zones because they are non-urban: open
  coast, hills, and barren hinterland with little population, road access, or grid.
  An EV charging station follows demand, so candidates are the 8 built-up districts
  (Clifton, DHA, Korangi, Malir, Gulshan, North Nazimabad, SITE, Gadap). Among those,
  the ranking still favours the northern/western urban fringe (Gadap, SITE), which is
  both higher-resource and cooler.
- Framing for the paper: solar *resource* peaks outside the urban core (those areas
  suit utility-scale solar farms); EV *siting* trades a little resource for demand and
  accessibility. fig6 shows both at once — full-city resource surface + urban sites.

## Seasonal site suitability — 4 seasons (fig9, the instructor's explicit ask)
Year split into meteorological seasons (Winter DJF / Spring MAM / Summer JJA /
Autumn SON); suitability (yield + day-to-day consistency + coolness) computed within
each season and normalised across the 8 Karachi districts. Source:
reports/seasonal_site_suitability_karachi.csv; figure fig9_seasonal_sites.png.
- Strong seasonal cycle: Spring (pre-monsoon) is brightest everywhere (~500-507
  W/m^2 daytime), Summer (monsoon) is the dimmest (~366-378, a ~25% drop from spring),
  Winter/Autumn sit in between (~396-416).
- The ranking is mostly stable but it FLIPS in summer: Gadap (northern fringe) is the
  best site in Winter/Spring/Autumn (suit 0.75 / 0.95 / 1.00), but in Summer the cooler
  coastal/western sites win — SITE 0.71 and Clifton 0.55 overtake Gadap 0.65. Clifton's
  summer jump (0.55 vs 0.25 in winter) is the coastal-coolness term mattering most when
  it is hottest. Good, defensible "creativity" point for the paper.
- Practical read: a Gadap/SITE charger is the year-round pick; if optimising for the
  hot monsoon months specifically, the coastal sites claw back some ground.

## Caveats / honesty
- ERA5 is reanalysis at 9–25 km: captures city resource + coastal gradient, not
  sub-cell microclimate. Within-city differences are genuinely small.
- Hour-ahead is the easy horizon; do not overstate skill.
- Reanalysis is not ground truth; a physical sensor (funded, future work) would
  validate it.
