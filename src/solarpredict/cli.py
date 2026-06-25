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


if __name__ == "__main__":  # pragma: no cover
    app()
