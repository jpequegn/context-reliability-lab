"""Command-line workflows for the Context Reliability Lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from context_reliability_lab import __version__
from context_reliability_lab.corpus import generate_corpus
from context_reliability_lab.graders import grade_run
from context_reliability_lab.policies import MemoryPolicy, default_policies
from context_reliability_lab.replay import compact_case, counterfactual_replay
from context_reliability_lab.reports import EvaluationReport, write_demo
from context_reliability_lab.runner import run_case
from context_reliability_lab.storage import RunStore, export_otlp
from context_reliability_lab.tournament import run_tournament

app = typer.Typer(no_args_is_help=True, help="Evaluate stateful memory behavior.")


@app.callback()
def main() -> None:
    """Context Reliability Lab command group."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("corpus")
def corpus_command(
    output: Annotated[Path, typer.Option(help="Destination JSON file.")] = Path("corpus.json"),
) -> None:
    """Generate the deterministic 50-case corpus."""
    data = generate_corpus()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(data.model_dump_json(indent=2) + "\n")
    typer.echo(f"wrote {len(data.cases)} cases to {output}")


@app.command("run")
def run_command(
    case_id: Annotated[str, typer.Option(help="Synthetic case ID.")],
    policy: Annotated[str, typer.Option(help="Memory policy name.")] = "critic-gated",
    database: Annotated[Path, typer.Option(help="DuckDB evidence ledger.")] = Path("runs.duckdb"),
) -> None:
    """Run one case and persist its evidence."""
    data = generate_corpus()
    case = next((item for item in data.cases if item.case_id == case_id), None)
    if case is None:
        raise typer.BadParameter(f"unknown case: {case_id}")
    selected = _policy(policy)
    result = run_case(case, selected)
    grades = grade_run(result, data.hidden_truth[case.case_id])
    with RunStore(database) as store:
        store.save_run(result, grades, tenant_id="cli")
        exported = store.export_json(result.lineage.run_id)
    typer.echo(json.dumps(exported, indent=2, sort_keys=True))


@app.command()
def tournament(
    output: Annotated[Path, typer.Option(help="Destination report JSON.")] = Path(
        "tournament.json"
    ),
) -> None:
    """Run the 900-trial repeated-seed tournament."""
    report = run_tournament(generate_corpus())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n")
    typer.echo(f"wrote {len(report.trials)} trials to {output}")


@app.command()
def replay(case_id: Annotated[str, typer.Option(help="Synthetic case ID.")]) -> None:
    """Show compaction and counterfactual replay evidence for a case."""
    data = generate_corpus()
    case = next((item for item in data.cases if item.case_id == case_id), None)
    if case is None:
        raise typer.BadParameter(f"unknown case: {case_id}")
    truth = data.hidden_truth[case.case_id]
    payload = {
        "compaction": [
            compact_case(case, truth, budget).model_dump(mode="json") for budget in (40, 80, 180)
        ],
        "counterfactual": counterfactual_replay(case, truth).model_dump(mode="json"),
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def inspect(
    database: Annotated[Path, typer.Option(help="DuckDB evidence ledger.")],
    run_id: Annotated[str, typer.Option(help="Run identifier.")],
) -> None:
    """Inspect a sanitized stored run."""
    with RunStore(database) as store:
        typer.echo(json.dumps(store.export_json(run_id), indent=2, sort_keys=True))


@app.command()
def export(
    database: Annotated[Path, typer.Option(help="DuckDB evidence ledger.")],
    run_id: Annotated[str, typer.Option(help="Run identifier.")],
    format: Annotated[str, typer.Option(help="json or otlp.")] = "json",
) -> None:
    """Export a sanitized stored run or OTLP trace envelope."""
    with RunStore(database) as store:
        if format == "json":
            payload = store.export_json(run_id)
        elif format == "otlp":
            payload = export_otlp(store.load_run(run_id), store.load_grades(run_id))
        else:
            raise typer.BadParameter("format must be json or otlp")
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def status(report: Annotated[Path, typer.Option(help="Evaluation report JSON.")]) -> None:
    """Return release-gate status; exit one when attention is required."""
    evaluation = EvaluationReport.model_validate_json(report.read_text())
    state = "passed" if evaluation.quality_gate.passed else "attention"
    typer.echo(f"quality gate: {state}")
    typer.echo(f"cases: {evaluation.cases}")
    typer.echo(f"trials: {len(evaluation.tournament.trials)}")
    typer.echo(f"regressions: {evaluation.regressions_replayed}")
    if not evaluation.quality_gate.passed:
        raise typer.Exit(1)


@app.command()
def demo(
    output_dir: Annotated[Path, typer.Option(help="Demo artifact directory.")] = Path(
        "artifacts/demo"
    ),
) -> None:
    """Run the complete deterministic evaluation workflow."""
    report = write_demo(output_dir)
    typer.echo(f"demo complete: {output_dir}")
    typer.echo(f"quality gate: {'passed' if report.quality_gate.passed else 'attention'}")
    typer.echo(f"cases: {report.cases}")
    typer.echo(f"trials: {len(report.tournament.trials)}")
    typer.echo(f"regressions: {report.regressions_replayed}")
    if not report.quality_gate.passed:
        raise typer.Exit(1)


def _policy(name: str) -> MemoryPolicy:
    selected = next((policy for policy in default_policies() if policy.name == name), None)
    if selected is None:
        raise typer.BadParameter(f"unknown policy: {name}")
    return selected


if __name__ == "__main__":
    app()
