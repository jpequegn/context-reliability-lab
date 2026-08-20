import json
from xml.etree import ElementTree

from typer.testing import CliRunner

from context_reliability_lab.cli import app
from context_reliability_lab.reports import (
    EvaluationReport,
    QualityGate,
    build_report,
    render_junit,
    render_markdown,
    write_demo,
)


def test_release_report_passes_all_deterministic_gates() -> None:
    report, regressions = build_report()
    assert report.quality_gate.passed
    assert report.cases == 50
    assert report.families == 5
    assert len(report.tournament.trials) == 900
    assert report.compaction_checks == 150
    assert report.compaction_passed == 150
    assert report.regressions_replayed == len(regressions) == 10
    assert report.calibration.agreement_rate == 0.9
    markdown = render_markdown(report)
    assert "retrieval" in markdown
    assert "recovery" in markdown
    ElementTree.fromstring(render_junit(report))


def test_demo_writes_complete_redacted_artifact_set(tmp_path) -> None:
    report = write_demo(tmp_path)
    expected = {
        "corpus.json",
        "evaluation-report.json",
        "evaluation-report.md",
        "evaluation-report.xml",
        "runs.duckdb",
        "otlp-trace.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    assert len(list((tmp_path / "regressions").glob("*.json"))) == 10
    assert "SECRET_CANARY_" not in (tmp_path / "evaluation-report.json").read_text()
    assert "SECRET_CANARY_" not in (tmp_path / "otlp-trace.json").read_text()
    assert report.quality_gate.passed


def test_cli_demo_status_and_attention_exit(tmp_path) -> None:
    runner = CliRunner()
    output = tmp_path / "demo"
    demo = runner.invoke(app, ["demo", "--output-dir", str(output)])
    assert demo.exit_code == 0, demo.output
    assert "trials: 900" in demo.output
    report_path = output / "evaluation-report.json"
    status = runner.invoke(app, ["status", "--report", str(report_path)])
    assert status.exit_code == 0
    assert "quality gate: passed" in status.output
    report = EvaluationReport.model_validate_json(report_path.read_text())
    failing = report.model_copy(
        update={"quality_gate": QualityGate(passed=False, failures=("forced",))}
    )
    failing_path = tmp_path / "failing.json"
    failing_path.write_text(failing.model_dump_json())
    attention = runner.invoke(app, ["status", "--report", str(failing_path)])
    assert attention.exit_code == 1
    assert "quality gate: attention" in attention.output


def test_cli_run_inspect_and_otlp_export(tmp_path) -> None:
    runner = CliRunner()
    database = tmp_path / "runs.duckdb"
    executed = runner.invoke(
        app,
        [
            "run",
            "--case-id",
            "retrieval-00",
            "--policy",
            "critic-gated",
            "--database",
            str(database),
        ],
    )
    assert executed.exit_code == 0, executed.output
    run_id = "run-retrieval-00-critic-gated-10000"
    inspected = runner.invoke(app, ["inspect", "--database", str(database), "--run-id", run_id])
    assert inspected.exit_code == 0
    assert json.loads(inspected.output)["lineage"]["run_id"] == run_id
    exported = runner.invoke(
        app,
        [
            "export",
            "--database",
            str(database),
            "--run-id",
            run_id,
            "--format",
            "otlp",
        ],
    )
    assert exported.exit_code == 0
    assert "resourceSpans" in json.loads(exported.output)
    assert "SECRET_CANARY_" not in exported.output
