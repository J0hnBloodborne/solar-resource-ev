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

## Figure inventory (reports/figures/)
- fig1_model_skill / fig1b_model_rmse — model comparison (Part A).
- fig2_ghi_map — Pakistan city GHI map.
- fig3_seasonal — seasonal GHI by city.
- fig4_city_ranking — city solar ranking (Quetta/Gilgit highlighted).
- fig5_pred_vs_actual — predicted vs actual, best model.
- fig7_feature_importance — best tree's features.
- fig8_data_efficiency_skill / fig8b_..._rmse — scarce-data story.
- fig6 (EV-locations map) + the intra-Karachi heatmap/ranking — to build.

## Caveats / honesty
- ERA5 is reanalysis at 9–25 km: captures city resource + coastal gradient, not
  sub-cell microclimate. Within-city differences are genuinely small.
- Hour-ahead is the easy horizon; do not overstate skill.
- Reanalysis is not ground truth; a physical sensor (funded, future work) would
  validate it.
