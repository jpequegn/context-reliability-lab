"""Compaction, counterfactual replay, divergence, and regression export."""

from __future__ import annotations

from datetime import datetime

from context_reliability_lab.contracts import (
    Corpus,
    EvalCase,
    Event,
    HiddenTruth,
    MemoryRecord,
    RunResult,
    StrictModel,
    TaskFamily,
    digest,
)
from context_reliability_lab.graders import first_failure, grade_run
from context_reliability_lab.policies import MemoryPolicy, SeededFaultPolicy, active, eligible
from context_reliability_lab.runner import run_case


class CompactionResult(StrictModel):
    case_id: str
    token_budget: int
    selected_memory_ids: tuple[str, ...]
    tokens_used: int
    critical_evidence_survived: bool


class StateDiff(StrictModel):
    case_id: str
    baseline_policy: str
    candidate_policy: str
    first_divergent_index: int | None
    baseline_transition_id: str | None
    candidate_transition_id: str | None
    baseline_action: str | None
    candidate_action: str | None
    baseline_memory_ids: tuple[str, ...]
    candidate_memory_ids: tuple[str, ...]
    baseline_state_digest: str | None
    candidate_state_digest: str | None


class CounterfactualResult(StrictModel):
    case_id: str
    removed_memory_id: str
    original_evidence_id: str
    original_case_digest: str
    counterfactual_case_digest: str
    original_outcome: str
    counterfactual_outcome: str
    outcome_changed: bool


class RegressionCase(StrictModel):
    schema_version: str = "context-reliability-regression-v1"
    regression_id: str
    case: EvalCase
    truth: HiddenTruth
    expected_component: str
    expected_reason: str
    source_case_digest: str


class FailureFirstFallbackPolicy(MemoryPolicy):
    """Select the injected failure, otherwise fall back to valid scoped memory."""

    name = "failure-first-fallback"

    def select(
        self,
        case: EvalCase,
        event: Event,
        memory: tuple[MemoryRecord, ...],
        now: datetime,
    ) -> tuple[MemoryRecord, ...]:
        suffix = "stale" if case.seed % 2 == 0 else "restricted"
        injected = tuple(record for record in memory if record.memory_id.endswith(suffix))
        if injected:
            return injected
        return active(eligible(memory, case), now)[:1]


def compact_case(case: EvalCase, truth: HiddenTruth, token_budget: int) -> CompactionResult:
    correct = set(truth.correct_memory_ids)
    ranked = sorted(
        eligible(case.initial_memory, case),
        key=lambda item: (item.memory_id not in correct, item.poisoned, -item.confidence),
    )
    selected: list[MemoryRecord] = []
    used = 0
    for record in ranked:
        cost = len(record.content.split())
        if used + cost <= token_budget:
            selected.append(record)
            used += cost
    selected_ids = tuple(item.memory_id for item in selected)
    return CompactionResult(
        case_id=case.case_id,
        token_budget=token_budget,
        selected_memory_ids=selected_ids,
        tokens_used=used,
        critical_evidence_survived=correct <= set(selected_ids),
    )


def state_diff(baseline: RunResult, candidate: RunResult) -> StateDiff:
    first: int | None = None
    for index, pair in enumerate(zip(baseline.transitions, candidate.transitions, strict=False)):
        left, right = pair
        if (
            left.action != right.action
            or left.retrieved_memory_ids != right.retrieved_memory_ids
            or left.state_after != right.state_after
        ):
            first = index
            break
    if first is None and len(baseline.transitions) != len(candidate.transitions):
        first = min(len(baseline.transitions), len(candidate.transitions))
    left = (
        baseline.transitions[first]
        if first is not None and first < len(baseline.transitions)
        else None
    )
    right = (
        candidate.transitions[first]
        if first is not None and first < len(candidate.transitions)
        else None
    )
    return StateDiff(
        case_id=baseline.case.case_id,
        baseline_policy=baseline.policy,
        candidate_policy=candidate.policy,
        first_divergent_index=first,
        baseline_transition_id=left.transition_id if left else None,
        candidate_transition_id=right.transition_id if right else None,
        baseline_action=left.action if left else None,
        candidate_action=right.action if right else None,
        baseline_memory_ids=left.retrieved_memory_ids if left else (),
        candidate_memory_ids=right.retrieved_memory_ids if right else (),
        baseline_state_digest=left.state_after if left else None,
        candidate_state_digest=right.state_after if right else None,
    )


def counterfactual_replay(case: EvalCase, truth: HiddenTruth) -> CounterfactualResult:
    failure = next(
        item for item in case.initial_memory if item.memory_id == truth.failure_memory_id
    )
    policy = FailureFirstFallbackPolicy()
    original = run_case(case, policy)
    modified = case.model_copy(
        update={
            "initial_memory": tuple(
                item for item in case.initial_memory if item.memory_id != truth.failure_memory_id
            )
        }
    )
    counterfactual = run_case(modified, policy)
    return CounterfactualResult(
        case_id=case.case_id,
        removed_memory_id=failure.memory_id,
        original_evidence_id=failure.source.evidence_id,
        original_case_digest=digest(case.model_dump(mode="json")),
        counterfactual_case_digest=digest(modified.model_dump(mode="json")),
        original_outcome=original.answer.content,
        counterfactual_outcome=counterfactual.answer.content,
        outcome_changed=original.answer.content != counterfactual.answer.content,
    )


def export_minimized_regressions(corpus: Corpus, count: int = 10) -> tuple[RegressionCase, ...]:
    selected: list[EvalCase] = []
    for family in TaskFamily:
        selected.extend([case for case in corpus.cases if case.family is family][:2])
    regressions: list[RegressionCase] = []
    for case in selected[:count]:
        truth = corpus.hidden_truth[case.case_id]
        keep = set(truth.correct_memory_ids) | {truth.failure_memory_id}
        minimized = case.model_copy(
            update={
                "initial_memory": tuple(
                    memory for memory in case.initial_memory if memory.memory_id in keep
                )
            }
        )
        grades = grade_run(run_case(minimized, SeededFaultPolicy()), truth)
        attribution = first_failure(grades)
        if attribution is None:
            raise ValueError(f"regression {case.case_id} no longer fails")
        regressions.append(
            RegressionCase(
                regression_id=f"regression-{case.case_id}",
                case=minimized,
                truth=truth,
                expected_component=attribution[0],
                expected_reason=attribution[1],
                source_case_digest=digest(case.model_dump(mode="json")),
            )
        )
    return tuple(regressions)


def replay_regression(regression: RegressionCase) -> bool:
    result = run_case(regression.case, SeededFaultPolicy())
    attribution = first_failure(grade_run(result, regression.truth))
    return attribution == (regression.expected_component, regression.expected_reason)
