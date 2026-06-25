"""``solarpredict`` command-line entrypoint (thin orchestration).

All logic lives in the package; the CLI only wires it together. Subcommands for
ingest / benchmark / siting / figures are added as the milestones land.
"""

from __future__ import annotations

import typer

from solarpredict import __version__

app = typer.Typer(help="Solar resource prediction benchmark.", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def models() -> None:
    """List every registered forecaster (the benchmark roster)."""
    from solarpredict.models import list_forecasters

    for name in list_forecasters():
        typer.echo(name)


@app.command()
def ingest(
    years: int | None = typer.Option(
        None, help="History length in years (default: settings)."
    ),
    sites_only: bool = typer.Option(
        False, "--sites-only", help="Only the Karachi/Lahore named sites."
    ),
    force: bool = typer.Option(False, "--force", help="Refetch even if cached."),
) -> None:
    """Pull Open-Meteo archive data into the parquet cache (cities + named sites)."""
    import logging

    from solarpredict.config.sites import intracity_sites
    from solarpredict.data import OpenMeteoArchiveRepository
    from solarpredict.data.ingest import (
        default_date_range,
        default_ingest_targets,
        ingest_points,
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start, end = default_date_range(years)
    targets = intracity_sites() if sites_only else default_ingest_targets()
    typer.echo(f"Ingesting {len(targets)} points: {start} -> {end}")
    repo = OpenMeteoArchiveRepository()
    results = ingest_points(repo, targets, start, end, force_refresh=force)
    total = sum(len(df) for df in results.values())
    typer.echo(f"Done: {len(results)} points, {total:,} rows cached.")


if __name__ == "__main__":  # pragma: no cover
    app()
