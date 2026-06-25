# solarpredict

A tiered ML/AI benchmark for **hour-ahead GHI (solar irradiance) forecasting** and
**solar-suitability site ranking** for solar-powered EV charging stations, using
Open-Meteo weather data for Pakistani cities. Results feed a conference paper
("Solar Resource Prediction").

Two contributions:

- **Part A — model benchmark:** naive/statistical baselines → classical ML
  (XGBoost, LightGBM, …) → deep learning (LSTM, NHITS) → a **time-series
  foundation model (Chronos-2)**, ranked by a **forecast skill score vs. smart
  (clear-sky) persistence**.
- **Part B — siting:** rank sites within a data-chosen city by a transparent
  solar-suitability index and recommend charging-station locations.

## Requirements

- **Python 3.11**
- **PyTorch ≥ 2.x**, installed separately for your platform — a CUDA build is
  recommended (the deep-learning and foundation-model tiers use the GPU; CPU also
  works). The core install does **not** pull in torch, so it won't clobber an
  existing CUDA build.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# install PyTorch for your platform first: https://pytorch.org/get-started/locally/
pip install -e ".[dev]"
```

Pull in each tier's dependencies as you reach its milestone:
`".[classical]"`, `".[dl]"`, `".[fm]"`, `".[ingest]"`, `".[viz]"`.

## Usage

```powershell
solarpredict version        # package version
solarpredict models         # list the registered forecasters (benchmark roster)
```

## Layout

```
src/solarpredict/
  config/      settings (pydantic) + candidate-city catalog
  data/        DataRepository interface (Open-Meteo impl in M1)
  features/    lags, rolling, cyclical time, clear-sky index
  models/      Forecaster interface + registry + one module per model
  evaluation/  metrics + skill-score harness
  solar/       GHI -> power (pvlib), PSH
  siting/      suitability index + ranking (Part B)
  viz/         paper figures
pipelines/     typer CLI orchestration
tests/         pytest (registry + metrics smoke tests)
```

## Design

- **Strategy + Registry** for models: add a model = one module + `@register(...)`,
  no edits to the shared harness. See `models/registry.py`.
- **Repository pattern** for data: callers depend on `DataRepository`, not the
  concrete Open-Meteo/DB source.
- **Config-driven** via `pydantic-settings` (`SOLARPREDICT_*` env overrides).

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the multi-contributor workflow,
[`docs/PLAN.md`](docs/PLAN.md) for the full plan + milestone breakdown (M0–M8), and
[`docs/DATA_SUMMARY.md`](docs/DATA_SUMMARY.md) for the chosen-data summary (sites,
variance, expected results — instructor-facing).

> Repo name: **`solar-resource-ev`** · Python package: `solarpredict`.
