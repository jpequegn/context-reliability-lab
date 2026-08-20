from context_reliability_lab.contracts import TaskFamily
from context_reliability_lab.corpus import generate_case, generate_corpus
from context_reliability_lab.policies import CuratedPolicy, SeededFaultPolicy
from context_reliability_lab.replay import (
    compact_case,
    counterfactual_replay,
    export_minimized_regressions,
    replay_regression,
    state_diff,
)
from context_reliability_lab.runner import run_case


def test_compaction_preserves_critical_evidence_at_multiple_budgets() -> None:
    corpus = generate_corpus()
    for case in corpus.cases:
        truth = corpus.hidden_truth[case.case_id]
        for budget in (40, 80, 180):
            result = compact_case(case, truth, budget)
            assert result.tokens_used <= budget
            assert result.critical_evidence_survived


def test_counterfactual_preserves_original_lineage_and_changes_outcome() -> None:
    case, truth = generate_case(TaskFamily.RETRIEVAL, 0)
    original_ids = {item.memory_id for item in case.initial_memory}
    result = counterfactual_replay(case, truth)
    assert result.outcome_changed
    assert result.original_outcome == "route red"
    assert result.counterfactual_outcome == "route green"
    assert result.removed_memory_id in original_ids
    assert result.original_evidence_id
    assert result.original_case_digest != result.counterfactual_case_digest


def test_state_diff_localizes_first_divergent_transition() -> None:
    case, _ = generate_case(TaskFamily.RECOVERY, 0)
    baseline = run_case(case, SeededFaultPolicy())
    candidate = run_case(case, CuratedPolicy())
    diff = state_diff(baseline, candidate)
    assert diff.first_divergent_index == 0
    assert diff.baseline_transition_id
    assert diff.candidate_transition_id
    assert diff.baseline_memory_ids != diff.candidate_memory_ids


def test_ten_minimized_regressions_replay_independently() -> None:
    regressions = export_minimized_regressions(generate_corpus())
    assert len(regressions) == 10
    assert {item.case.family for item in regressions} == set(TaskFamily)
    assert all(len(item.case.initial_memory) == 2 for item in regressions)
    assert all(replay_regression(item) for item in regressions)
