from context_reliability_lab.contracts import FailureClass, TaskFamily
from context_reliability_lab.corpus import generate_case, generate_corpus
from context_reliability_lab.graders import first_failure, grade_run
from context_reliability_lab.policies import CuratedPolicy, SeededFaultPolicy
from context_reliability_lab.runner import run_case


def test_every_seeded_failure_is_caught_at_state_or_behavior_level() -> None:
    corpus = generate_corpus()
    caught: set[FailureClass] = set()
    for case in corpus.cases:
        result = run_case(case, SeededFaultPolicy())
        grades = grade_run(result, corpus.hidden_truth[case.case_id])
        state_or_behavior = [item for item in grades if item.level in {"memory_state", "behavior"}]
        assert any(not item.passed for item in state_or_behavior)
        caught.add(case.seeded_failure)
    assert caught == set(FailureClass)


def test_curated_policy_has_exact_grader_evidence() -> None:
    case, truth = generate_case(TaskFamily.RETRIEVAL, 0)
    result = run_case(case, CuratedPolicy())
    grades = grade_run(result, truth)
    assert all(item.case_id == case.case_id for item in grades)
    assert grades[1].metrics["recall"] == 1.0
    assert grades[3].metrics["task_success"] == 1.0
    assert first_failure(grades) is None


def test_fault_is_attributed_to_first_component() -> None:
    case, truth = generate_case(TaskFamily.GENERALIZATION, 1)
    grades = grade_run(run_case(case, SeededFaultPolicy()), truth)
    assert first_failure(grades) == ("storage", "scope_leak")
    failed = [item for item in grades if not item.passed]
    assert all(item.evidence_transition_ids for item in failed)


def test_no_memory_localizes_retrieval_before_outcome() -> None:
    from context_reliability_lab.policies import NoMemoryPolicy

    case, truth = generate_case(TaskFamily.RETRIEVAL, 0)
    grades = grade_run(run_case(case, NoMemoryPolicy()), truth)
    assert first_failure(grades) == ("retrieval", "missed_relevant")
