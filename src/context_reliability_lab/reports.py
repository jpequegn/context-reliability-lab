"""Evaluation reports, release gates, and the deterministic demo."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

from context_reliability_lab.contracts import Corpus, StrictModel
from context_reliability_lab.corpus import generate_corpus
from context_reliability_lab.graders import grade_run
from context_reliability_lab.policies import CriticGatedPolicy
from context_reliability_lab.replay import (
    RegressionCase,
    compact_case,
    export_minimized_regressions,
    replay_regression,
)
from context_reliability_lab.runner import run_case
from context_reliability_lab.storage import RunStore, export_otlp
from context_reliability_lab.tournament import TournamentReport, run_tournament


class HumanCalibration(StrictModel):
    reviewed_cases: int
    agreements: int
    agreement_rate: float
    note: str


class QualityGate(StrictModel):
    passed: bool
    failures: tuple[str, ...]


class EvaluationReport(StrictModel):
    schema_version: str = "context-reliability-evaluation-v1"
    corpus_digest: str
    cases: int
    families: int
    tournament: TournamentReport
    compaction_checks: int
    compaction_passed: int
    regressions_exported: int
    regressions_replayed: int
    calibration: HumanCalibration
    quality_gate: QualityGate


def build_report(
    corpus: Corpus | None = None,
) -> tuple[EvaluationReport, tuple[RegressionCase, ...]]:
    data = corpus or generate_corpus()
    tournament = run_tournament(data)
    regressions = export_minimized_regressions(data)
    replayed = sum(replay_regression(item) for item in regressions)
    compactions = [
        compact_case(case, data.hidden_truth[case.case_id], budget)
        for case in data.cases
        for budget in (40, 80, 180)
    ]
    compaction_passed = sum(item.critical_evidence_survived for item in compactions)
    calibration = HumanCalibration(
        reviewed_cases=20,
        agreements=18,
        agreement_rate=0.9,
        note=(
            "Deterministic double-review fixture; replace with domain reviewers "
            "before production use."
        ),
    )
    failures: list[str] = []
    if len(data.cases) < 50:
        failures.append("fewer_than_50_cases")
    if len(tournament.policies) < 4 or len(tournament.seeds) < 3:
        failures.append("insufficient_policy_or_seed_coverage")
    if tournament.budget_violations:
        failures.append("budget_violations")
    if any(item.isolation_leaks for item in tournament.overall):
        failures.append("isolation_leaks")
    if replayed < 10:
        failures.append("insufficient_replayable_regressions")
    if compaction_passed != len(compactions):
        failures.append("critical_evidence_lost_during_compaction")
    if calibration.agreement_rate < 0.8:
        failures.append("grader_human_agreement_below_threshold")
    report = EvaluationReport(
        corpus_digest=data.digest,
        cases=len(data.cases),
        families=len({case.family for case in data.cases}),
        tournament=tournament,
        compaction_checks=len(compactions),
        compaction_passed=compaction_passed,
        regressions_exported=len(regressions),
        regressions_replayed=replayed,
        calibration=calibration,
        quality_gate=QualityGate(passed=not failures, failures=tuple(failures)),
    )
    return report, regressions


def render_markdown(report: EvaluationReport) -> str:
    lines = [
        "# Context Reliability Evaluation",
        "",
        f"- Quality gate: **{'PASSED' if report.quality_gate.passed else 'FAILED'}**",
        f"- Corpus: {report.cases} cases / {report.families} families",
        f"- Tournament: {len(report.tournament.trials)} trials / "
        f"{len(report.tournament.policies)} policies / {len(report.tournament.seeds)} seeds",
        f"- Compaction: {report.compaction_passed}/{report.compaction_checks} passed",
        f"- Regressions: {report.regressions_replayed}/{report.regressions_exported} replayed",
        f"- Human calibration fixture: {report.calibration.agreement_rate:.3f}",
        f"- Graph decision: {report.tournament.graph_decision.reason}",
        "",
        "## Policy Results",
        "",
        "| Policy | Success | Retrieval recall | Adherence | Avg tokens | Isolation leaks |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report.tournament.overall:
        lines.append(
            f"| {item.policy} | {item.task_success:.3f} | {item.retrieval_recall:.3f} | "
            f"{item.adherence:.3f} | {item.average_tokens:.1f} | {item.isolation_leaks:.0f} |"
        )
    lines.extend(["", "## Task-Family Slices", ""])
    for item in report.tournament.by_family:
        lines.append(
            f"- {item.policy} / {item.family.value}: success {item.task_success:.3f}, "
            f"recall {item.retrieval_recall:.3f}, tokens {item.average_tokens:.1f}"
        )
    if report.quality_gate.failures:
        lines.extend(["", "## Gate Failures", ""])
        lines.extend(f"- {failure}" for failure in report.quality_gate.failures)
    return "\n".join(lines) + "\n"


def render_junit(report: EvaluationReport) -> str:
    suite = ElementTree.Element(
        "testsuite",
        name="context-reliability-release-gate",
        tests="5",
        failures=str(len(report.quality_gate.failures)),
    )
    checks = {
        "case-coverage": report.cases >= 50,
        "policy-tournament": len(report.tournament.trials) >= 600,
        "compaction": report.compaction_passed == report.compaction_checks,
        "regression-replay": report.regressions_replayed >= 10,
        "isolation": not any(item.isolation_leaks for item in report.tournament.overall),
    }
    for name, passed in checks.items():
        case = ElementTree.SubElement(suite, "testcase", name=name)
        if not passed:
            ElementTree.SubElement(case, "failure", message=f"{name} failed")
    return ElementTree.tostring(suite, encoding="unicode") + "\n"


def write_demo(output_dir: Path) -> EvaluationReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = generate_corpus()
    report, regressions = build_report(corpus)
    (output_dir / "corpus.json").write_text(corpus.model_dump_json(indent=2) + "\n")
    (output_dir / "evaluation-report.json").write_text(report.model_dump_json(indent=2) + "\n")
    (output_dir / "evaluation-report.md").write_text(render_markdown(report))
    (output_dir / "evaluation-report.xml").write_text(render_junit(report))
    regression_dir = output_dir / "regressions"
    regression_dir.mkdir(exist_ok=True)
    for regression in regressions:
        (regression_dir / f"{regression.regression_id}.json").write_text(
            regression.model_dump_json(indent=2) + "\n"
        )
    database = output_dir / "runs.duckdb"
    with RunStore(database) as store:
        for case in corpus.cases:
            truth = corpus.hidden_truth[case.case_id]
            result = run_case(case, CriticGatedPolicy())
            grades = grade_run(result, truth)
            store.save_run(result, grades, tenant_id="synthetic-demo")
        first = run_case(corpus.cases[0], CriticGatedPolicy())
        first_grades = grade_run(first, corpus.hidden_truth[first.case.case_id])
    (output_dir / "otlp-trace.json").write_text(
        json.dumps(export_otlp(first, first_grades), indent=2, sort_keys=True) + "\n"
    )
    return report
