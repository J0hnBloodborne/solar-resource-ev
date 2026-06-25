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

## Environment

This project uses the existing venv at **`F:\Documents\vit\vitvenv`**
(Python 3.11, **torch 2.11+cu130**, CUDA 13.0, RTX 3070). Do **not** reinstall
torch. Install the package editable into that venv:

```powershell
& F:\Documents\vit\vitvenv\Scripts\python.exe -m pip install -e ".[dev]"
```

Add tier dependencies as you reach each milestone:
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

See `CONTRIBUTING.md` for the multi-contributor workflow and the project plan for
the full milestone breakdown (M0–M8).
