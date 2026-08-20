from typer.testing import CliRunner

from context_reliability_lab import __version__
from context_reliability_lab.cli import app


def test_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__
