# solar-resource-ev — Project Plan & Roadmap

> Repo (suggested GitHub name): **`solar-resource-ev`**. Python package: `solarpredict`.
> This is the team-facing roadmap. **M0 is done** (see the git history); M1–M8 below are
> the remaining work, dependency-ordered so several people can build in parallel.

## Context

We produce **results and figures for a conference paper** (not a journal paper) titled
around **"Solar Resource Prediction."** The student team supplies the results/figures; the
instructor writes the prose. The brief, decoded from a meeting transcript, has **two
concrete contributions**:

1. **Compare the predictive performance of several ML models** for estimating solar power
   generation potential — predict **GHI** (Global Horizontal Irradiance) from weather data,
   convert it to solar power, and report which model predicts best.
2. **Rank multiple sites *within one city* by solar suitability** across the four seasons,
   and **identify optimal locations for solar-powered EV charging stations.**

Plus four named figures: a **model-performance comparison bar chart**, a **GHI heatmap**, a
**GHI / solar-potential time series**, and a **suitability-score ranking chart**.

Key steers from the instructor:
- **Conference** paper, not journal (creativity over rigor).
- **The "is AI even needed?" debate is CLOSED** — AI is the method; the paper compares
  models to find the best predictor. Naive/statistical baselines remain only as the
  **standard skill-score reference** (you can't report a credible RMSE ranking without one).
- **Single city, multiple sites**; pick a city where GHI variation actually exists
  (Karachi / Peshawar / Lahore were floated). Differences are *granular*.
- Data source: the **Open-Meteo API**.
- A physical GHI/temperature sensor was offered (funded) — **software-only for now**, sensor
  noted as future work.
- Hard rule: **real results, no plagiarism.** Presentation may favour the modern models, but
  **no number is ever invented or altered.**

### Scope decisions

- **Deliverable:** a reproducible analysis producing **paper-ready figures + tables**, built
  as a **properly engineered, multi-contributor repo**. Lean where it helps, but no
  tech-debt corners.
- **Stack:** **PyTorch only — no TensorFlow/Keras** (runs natively on Windows). The S2Cool
  Keras LSTM is **re-implemented in torch** via `neuralforecast` so every deep model shares
  one framework/API.
- **Environment:** Python 3.11 + **PyTorch ≥ 2.x** (install your own CUDA/CPU build; the dev
  box runs torch 2.11+cu130 on an RTX 3070). Don't let the project reinstall torch. Deps are
  pip-managed (this supersedes the earlier "uv" idea). A local GPU means the DL and
  foundation-model tiers run on-device — no Colab needed.
- **Study area:** **let the data choose the city** — pull candidate Pakistani cities, pick
  the one with the most intra-/inter-site GHI spread, then sample sites across it and rank.
- **Pace:** ASAP, **no fixed calendar**; dependency-ordered milestones, parallelizable.
- **Models:** keep traditional ML as the comparison floor, **add modern AI** (a
  covariate-aware DL model + a time-series foundation model) — *not* DQN/RL/traffic-sim.

## What we reuse — the `S2Cool-Magic-Box` repo (≈80% reusable)

Prior work already implements, for the four instructor-named cities (Islamabad, Lahore,
Karachi, Peshawar):

| Capability | Where | Reuse |
| --- | --- | --- |
| **Open-Meteo ingestion** of GHI (`shortwave_radiation`), temp, humidity, wind, cloud, DNI, DHI | `ingest/src/ingest/api.py`, `db.py`, `main.py` | **High** — copy fetch logic; drop the Neon/Postgres layer |
| Historical backfill (extendable to ~10 y) | `ingest/src/ingest/api.py` | High |
| **GHI → solar power formula** (`estimate_solar_generation_kw`, linear STC scaling) | `backend/services/math_model.py:37` | **High** — reuse |
| **PSH** with monthly seasonal factors | `backend/services/math_model.py:97` | High — feeds the suitability score |
| Feature engineering (lags, rolling, cyclical time) | `Static Model/scripts/features.py`, `pipeline/preprocessing.py` | High |
| **XGBoost** trainer + metrics for next-hour GHI | `Static Model/scripts/`, `pipeline/train_models.py` | High — classical anchor |
| LSTM **approach** (Keras code **not** reused — rebuilt in torch) | `pipeline/train_models.py` | Low — reference only |
| **GitHub Actions**: CI, scheduled ingest, historic backfill, HF sync | `.github/workflows/*.yml` | **High** — CI + auto-ingestion templates |
| Dashboard charts (model-comparison, city-comparison, seasonal, GHI heatmap, feature-importance, backtest) | `frontend/src/components/*.jsx` | **High** — port target (M8) |

Reference metric already on disk: XGBoost GHI MAE 56.4 / RMSE 86.3 / R² 0.83 (Islamabad).

## Explicitly OUT of scope (and why)

- **SUMO / TraCI / traffic microsimulation, Gym, DQN/RL, charging-station-placement
  optimization.** The instructor set the old traffic project aside; "charging station" here
  is a **site-selection label on top of solar suitability**, not a traffic/queue/wait-time
  problem.
- **The `ccn-proj` repo is a red herring** — a Computer-Networks data-offloading simulation
  (OSMnx + IDM, greedy/predictive heuristics). No SUMO, no RL, no solar/GHI, no charging.
  Do not build on it.

---

## Deliverables

### Part A — ML model benchmark for GHI / solar-power prediction

A **tiered benchmark** on hour-ahead GHI, reported as a **forecast skill score vs. smart
(clear-sky) persistence** — the rigorous, standard way to rank solar forecasters and to show
the modern models' margin. Convert GHI → solar power with a fixed PV model
(reuse `estimate_solar_generation_kw`) and report power-domain error too.

**Recommended lineup — torch-only, Windows-native (core in bold; extended = parallelizable breadth):**

| Tier | Models | Library | Role |
| --- | --- | --- | --- |
| 0 — Naive/Statistical | **plain persistence**; **smart/clear-sky persistence** (skill denominator); climatology; SARIMAX (exog) | numpy/pandas, **pvlib**, statsforecast | Baselines / skill-score reference |
| 1 — Classical ML | **XGBoost** (reuse) + **LightGBM**; Linear/Ridge (floor); RandomForest/ExtraTrees; CatBoost | scikit-learn, xgboost, lightgbm, catboost | The bar modern AI must clear |
| 2 — Deep learning (all torch) | **LSTM** (re-implemented) + **NHITS** (covariate-aware); TFT / TimeXer (stretch) | **`neuralforecast` (torch/Lightning)** | Classic + modern DL, one framework |
| 3 — Foundation model (latest gen) | **Chronos-2** (Amazon, 2025, native covariates); **TimesFM-2.5** / **TabPFN-TS** (stretch); Chronos-Bolt (fallback) | `chronos-forecasting>=2.0`, `timesfm`, `tabpfn-time-series` | The "latest-and-greatest AI" headline |

Models are decoupled behind one `Forecaster` interface, so each tier can be built by a
different contributor in parallel.

**The headline four** (one per tier): **smart persistence → XGBoost → NHITS → Chronos-2** —
a clean "naive → classical → deep → foundation" story. The rest are supporting/ablation,
including the key **with- vs without-covariates ablation** for NHITS and Chronos-2.

**Novelty hook:** *the first reproducible benchmark of a 2025-generation, covariate-aware
time-series foundation model (Chronos-2) against classical ML and deep learning for
hour-ahead GHI prediction in Pakistani cities, reported as a clear-sky skill score.* Expected
honest finding: a locally-trained XGBoost typically matches or beats a zero-shot foundation
model on a near-deterministic target, while the FM still posts a positive skill score over
smart persistence.

**Evaluation protocol (non-negotiables):**
- Metrics: MAE, RMSE, R², nRMSE (normalize by mean *daytime* GHI), MBE, and the **skill
  score `S = 1 − RMSE_model/RMSE_reference`** vs smart persistence. Avoid MAPE (explodes near
  zero GHI).
- **Night mask:** drop hours with solar zenith ≳ 85–90° (pvlib clear-sky ≈ 0) from *all*
  metrics — night zeros otherwise inflate R² to ~0.99 and hide real skill.
- **Strictly chronological** train/val/test (e.g. train→2021 / val 2022 / test 2023–24);
  never shuffle. Model the **clear-sky index `kt = GHI/GHI_cs`** for statistical models; clip
  predictions ≥ 0.
- **Leakage discipline:** for t+1 use only lagged GHI/covariates; do **not** feed concurrent
  `direct_radiation`/`DNI`/`diffuse_radiation` (deterministic components of the target). For
  NHITS use `hist_exog_list` (realized-to-t), not `futr_exog_list`.
- Report metrics split by **clear vs. cloudy/high-variability hours** — where AI beats
  persistence and where the contribution lives.
- Figures: **model-performance comparison bar chart** + **predicted-vs-actual time series**.

**Traps the research flagged (avoid — they silently waste days):**
- **PatchTST / iTransformer in neuralforecast silently drop exogenous covariates** → don't
  use them as "covariate-aware"; the right covariate-aware patch-transformer is **TimeXer**.
- **Chronos version trap:** classic `chronos`/`chronos-bolt` are univariate; you must
  `pip install "chronos-forecasting>=2.0"` and use `amazon/chronos-2` for covariates.
- **Windows (torch-only):** set `num_workers=0` for neuralforecast/Lightning to avoid the
  DataLoader hang — then NHITS / LSTM / Chronos-2 all run natively. No TF/Keras, no Colab.
- **Why some models are out (not "old vs new"):** **TimeGPT** = closed/hosted/paid, breaks
  reproducibility (optional labeled row only). **Lag-Llama** = first-gen univariate,
  superseded. **Prophet/pmdarima** = Windows compiler pain → use Nixtla `statsforecast`.
- **Fallback:** if Chronos-2 fights the env, degrade to **Chronos-Bolt** zero-shot; the paper
  still stands on Tiers 0–2 with NHITS covering the "modern AI" claim.

### Part B — Site suitability ranking & charging-station siting
- For the **data-chosen city**, sample **N candidate sites** (coordinates across the city),
  pull multi-year hourly GHI for each.
- Compute a **solar-suitability score** per site per season — a transparent weighted index
  over mean/seasonal GHI, PSH, low-irradiance (cloud) penalty, and inter-day consistency.
  (Explainable, no black box.)
- **Rank sites** and recommend the top solar-EV-charging-station locations.
- Output: **GHI heatmap** (site × month or hour × month), **GHI/solar-potential time
  series**, **suitability-score ranking chart**, plus a recommended-sites table with coords.

### Packaging for the instructor
A short **results note** (markdown/PDF): best model + why, recommended city + top sites with
coordinates, the four figures, and a metrics table — plus the reproducible code.

---

## Architecture & design patterns

Design goals: **parallel contributors, low tech debt, trivial to add a model, reproducible.**

- **Strategy + Registry for models** — one `Forecaster` interface
  (`fit`/`predict` + metadata: `name, tier, supports_covariates, is_probabilistic`). Each
  model self-registers via `@register("nhits")`; the harness scores all registry entries
  identically. *Adding a model = one file + a decorator, no shared-code edits.*
  (Implemented in M0: `models/base.py`, `models/registry.py`.)
- **Adapter** wrappers bring scikit-learn / xgboost / `neuralforecast` / `chronos` under the
  common interface.
- **Repository pattern for data** — `DataRepository` abstracts fetch + cache;
  `OpenMeteoArchiveRepository` (archive API → parquet) is the default, swappable for the
  S2Cool Neon DB without touching callers. (Interface in M0: `data/repository.py`.)
- **Config-driven** via `pydantic-settings` + YAML experiment configs. (M0: `config/`.)
- **Thin orchestration** — `typer` CLI; logic stays in the importable package.

**Repo structure (src layout):**
```
solar-resource-ev/
├── .github/workflows/        # ci.yml (✅) + ingest-current/ingest-historic/benchmark (M1/M7)
├── src/solarpredict/
│   ├── config/    (✅)        # pydantic-settings + city/site catalog
│   ├── data/      (✅ iface)  # DataRepository + OpenMeteoArchiveRepository (M1) + parquet cache
│   ├── features/  (M1/M2)     # lags, rolling, cyclical time, clear-sky index, solar zenith
│   ├── models/    (✅ core)   # Forecaster base + registry + one module per model (+ adapters)
│   ├── evaluation/(✅ metrics)# metrics + skill-score harness, night mask, splits, baselines
│   ├── solar/     (M5/M6)     # GHI->power (pvlib PVWatts), PSH
│   ├── siting/    (M6)        # site sampling, suitability index, ranking
│   └── viz/       (M5/M6)     # figure generators (matplotlib/plotly)
├── pipelines/                 # typer CLIs: run_ingest / run_benchmark / run_siting / make_figures
├── configs/                   # experiment YAMLs
├── dashboard/                 # React port of S2Cool charts (M8)
├── tests/         (✅)        # pytest: registry + metrics smoke; grows per milestone
├── data/  artifacts/          # parquet cache / metrics+figures (gitignored)
└── pyproject.toml  (✅)       # deps + ruff + mypy + pytest config
```

**Tooling:** ruff (lint+format), mypy, pytest, pre-commit; deps pinned via the project venv.

## DevOps: CI + automated ingestion (mirroring S2Cool)

- **`ci.yml`** (✅) — PR/push: ruff + mypy + pytest (+ a benchmark smoke test once M1 lands).
- **`ingest-current.yml`** (M1) — scheduled cron: pull the latest Open-Meteo hours and append.
- **`ingest-historic.yml`** (M1) — manual: chunked multi-year backfill to seed the dataset.
- **`benchmark.yml`** (M7) — manual: run the full benchmark, publish metrics + figures.
- *(optional)* **`deploy-dashboard.yml`** once M8 lands.

Default storage is credential-free (parquet in a data branch / artifacts); the Repository
abstraction lets us swap to Neon DB + HF sync later via repo secrets.

## Data ingestion strategy (fast, within free limits)

Historical ingestion is fast and **not rate-limited at our scale** (Open-Meteo, verified):
- **No API key**; free for non-commercial/educational. Limits: <600/min, <5,000/hr, <10,000/day.
- The Historical Archive (ERA5) API serves multi-year hourly data, but Open-Meteo **weights
  each request by range × variables** — a multi-year × 8-variable pull trips the **per-minute
  limit in a single call** (confirmed in practice). So the repository **chunks by ~1 year**
  and **backs off 60 s** on the minute limit (mirrors S2Cool's chunking). Still cheap overall
  (well under 10k/day); a full multi-year, multi-site seed is a one-time, cached job.
- Official **`openmeteo-requests`** client + **`requests-cache`** + **`retry-requests`**;
  persist each location to **parquet** (keyed by site / range / variable-set).
- ERA5 has ~5-day latency (irrelevant for a historical benchmark).
- **How much data:** 1–2 years is the floor; default **~5 years**
  (`SOLARPREDICT_HISTORY_YEARS=5`).
- **Resolution caveat (Part B):** ERA5(-Land) is ~9–25 km. Live check confirmed Karachi's
  named sites span **6 distinct cells** (2.6% GHI spread, coastal→inland) vs Lahore's 1.4% →
  **Karachi is the data-chosen city**; intra-city differences are real but small, so the
  suitability index blends GHI with PSH/seasonal/temperature and is framed in annual-energy.

## Milestones (dependency-ordered; ∥ = parallelizable)

> Hard ordering: **M0 → M1 → {M2,M3,M4,M6} → M5 → M7 → M8.** The ∥ milestones run in parallel
> once M1 lands.

- **M0 — Repo skeleton & CI** ✅ **DONE**: src layout, `pyproject`, ruff/mypy/pytest/pre-commit,
  `Forecaster` interface + registry, `DataRepository` interface, metrics + skill score, typer
  CLI, `ci.yml`. (See git history.)
- **M1 — Data + skill-score spine** *(critical dependency)*: `OpenMeteoArchiveRepository` +
  parquet cache; pvlib clear-sky GHI, night mask, clear-sky-index transform; chronological
  split; **persistence / smart-persistence / climatology baselines + the shared skill-score
  harness**; `ingest-current.yml` + `ingest-historic.yml`. **City auto-selection.**
- **M2 ∥ — Classical ML tier**: feature engineering; **XGBoost (reuse) + LightGBM** core, then
  Linear/Ridge, RandomForest/ExtraTrees, CatBoost, SARIMAX. *(needs M1)*
- **M3 ∥ — Deep-learning tier (torch)**: **LSTM + NHITS** via `neuralforecast` with the
  covariate ablation; TFT / TimeXer stretch. *(needs M1)*
- **M4 ∥ — Foundation-model tier**: **Chronos-2** zero-shot, covariate ablation, predictions
  cached; TabPFN-TS stretch; Chronos-Bolt fallback. *(needs M1)*
- **M5 — Benchmark consolidation**: run the registry through the shared harness → metrics
  table (skill scores; clear-vs-cloudy split; nRMSE cross-city) + the two Part-A figures.
  *(needs M2–M4)*
- **M6 ∥ — Part B / siting**: sample sites in the chosen city, multi-year GHI per site, the
  **suitability index**, ranking, GHI→power; the three Part-B figures + recommended-sites
  table. *(needs M1; ∥ with M2–M4)*
- **M7 — Results package**: results note + four figures + tables + reproducible run
  instructions; `benchmark.yml` publishing artifacts.
- **M8 — Dashboard port** *(immediately after core)*: surface the new model-comparison +
  suitability-ranking charts in the S2Cool React dashboard, wired to M5/M6 outputs.

## Verification

- **Reproducibility:** the `typer` CLI / pipelines regenerate every figure/table from cached
  data with fixed seeds (2–3-seed spread for DL).
- **CI gate:** `ci.yml` runs ruff + mypy + pytest + a benchmark smoke test on every PR.
- **Sanity checks:** smart-persistence skill score reported; GHI zero at night in plots;
  metrics in sane W/m² ranges vs the XGBoost reference (MAE ~56, RMSE ~86, R² ~0.83).
- **Honesty gate:** every number from real Open-Meteo data + actual runs. Framing **may**
  favour the modern models (lead with novelty, headline where they win), but **never invent
  or alter a number**, and report genuine losses.

## Future work (noted in the paper, not built now)

- **Physical IoT sensor** (pyranometer or BH1750 + DHT22 on a microcontroller) for
  real-vs-API GHI validation — funding offered by the instructor.
- **Heavier transformer baselines** (Informer / Autoformer / Mamba) and a second foundation
  model — low payoff at hour-ahead; add only if a reviewer asks.
