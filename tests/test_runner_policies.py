from context_reliability_lab.contracts import EventKind, TaskFamily
from context_reliability_lab.corpus import generate_case, generate_corpus
from context_reliability_lab.policies import CuratedPolicy, default_policies
from context_reliability_lab.runner import run_case


def test_all_policies_run_full_corpus_reproducibly() -> None:
    corpus = generate_corpus()
    policies = default_policies()
    assert len(policies) == 6
    for policy in policies:
        first = [run_case(case, policy) for case in corpus.cases]
        second = [run_case(case, policy) for case in corpus.cases]
        assert first == second
        assert all(result.lineage.policy_version == f"{policy.name}:1" for result in first)


def test_authorization_is_enforced_before_retrieval() -> None:
    case, truth = generate_case(TaskFamily.RETRIEVAL, 0)
    for policy in default_policies():
        result = run_case(case, policy)
        evidence = " ".join(result.answer.evidence_ids)
        assert truth.canary not in result.answer.content
        assert "restricted" not in evidence


def test_equal_budget_is_respected() -> None:
    case, _ = generate_case(TaskFamily.RETRIEVAL, 0)
    for policy in default_policies():
        result = run_case(case, policy)
        assert all(transition.token_cost <= case.token_budget for transition in result.transitions)
        assert all(transition.tool_cost <= case.tool_budget for transition in result.transitions)


def test_correction_preserves_original_memory_history() -> None:
    case, _ = generate_case(TaskFamily.ADHERENCE, 0)
    result = run_case(case, CuratedPolicy())
    original_ids = {memory.memory_id for memory in case.initial_memory}
    assert original_ids <= {memory.memory_id for memory in result.final_memory}
    correction = next(event for event in case.events if event.kind is EventKind.CORRECT)
    assert any(item.event_id == correction.event_id for item in result.transitions)


def test_lineage_and_transition_digests_are_complete() -> None:
    case, _ = generate_case(TaskFamily.RECOVERY, 0)
    result = run_case(case, CuratedPolicy())
    assert result.lineage.input_digest
    assert len(result.transitions) == len(case.events)
    assert all(item.state_before and item.state_after for item in result.transitions)
