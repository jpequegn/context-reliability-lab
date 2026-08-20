from context_reliability_lab.corpus import generate_corpus
from context_reliability_lab.tournament import bootstrap_interval, run_tournament


def test_tournament_is_repeated_balanced_and_deterministic() -> None:
    corpus = generate_corpus()
    first = run_tournament(corpus)
    second = run_tournament(corpus)
    assert first == second
    assert first.cases == 50
    assert len(first.policies) == 6
    assert len(first.seeds) == 3
    assert len(first.trials) == 50 * 6 * 3
    assert first.budget_violations == 0
    assert len(first.by_family) == 6 * 5


def test_all_policy_slices_report_uncertainty_and_no_isolation_leaks() -> None:
    report = run_tournament(generate_corpus())
    for item in report.overall:
        assert 0 <= item.success_ci_low <= item.success_ci_high <= 1
        assert item.isolation_leaks == 0


def test_graph_promotion_is_auditable() -> None:
    decision = run_tournament(generate_corpus()).graph_decision
    assert decision.baseline != "no-memory"
    assert decision.reason in {
        "held_out_quality_improved",
        "non_inferior_quality_with_efficiency_gain",
        "no_measured_quality_or_efficiency_gain",
    }
    if decision.promoted:
        assert decision.quality_delta > 0 or decision.token_reduction >= 0.05


def test_bootstrap_interval_is_stable() -> None:
    assert bootstrap_interval([0.0, 1.0, 1.0]) == bootstrap_interval([0.0, 1.0, 1.0])
