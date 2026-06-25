# Contributing

Multi-contributor project. The architecture is built so the model tiers, the data
layer, siting, and figures can be worked on **in parallel** without touching shared
code.

## Setup

Use the project venv (`F:\Documents\vit\vitvenv`) — do not create a new one or
reinstall torch.

```powershell
& F:\Documents\vit\vitvenv\Scripts\python.exe -m pip install -e ".[dev]"
pre-commit install   # optional but recommended
```

## Workflow

- **Branch per feature**, open a PR. CI (ruff + mypy + pytest) must be green to merge.
- Run locally before pushing: `ruff check . && ruff format . && mypy && pytest`.

## Adding a model (the common case)

1. Create `src/solarpredict/models/<your_model>.py`.
2. Subclass `Forecaster`, set `name` / `tier` / `supports_covariates`, implement
   `fit` and `predict`.
3. Decorate the class with `@register("<your_model>")`.
4. Import the module in `models/__init__.py` so registration runs.
5. Add a smoke test under `tests/`.

That's it — the benchmark harness picks it up from the registry automatically.

## Suggested ownership (decoupled by interfaces)

| Area | Modules |
| --- | --- |
| Data + DevOps | `data/`, `.github/workflows/` |
| Classical ML | `models/` (tier-1), `features/` |
| Deep learning + foundation models | `models/` (tier-2/3) |
| Siting / suitability | `siting/`, `solar/` |
| Figures / dashboard | `viz/`, `dashboard/` |

## Conventions

- Line length 88; ruff owns lint + format (replaces black/isort/flake8).
- Keep orchestration thin (in `pipelines/` / `cli.py`); put logic in the package.
- Config via `SOLARPREDICT_*` env vars or `.env`, never hardcoded paths.
