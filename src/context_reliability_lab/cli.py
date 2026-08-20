"""Command-line entry point."""

import typer

from context_reliability_lab import __version__

app = typer.Typer(no_args_is_help=True, help="Evaluate stateful memory behavior.")


@app.callback()
def main() -> None:
    """Context Reliability Lab command group."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
