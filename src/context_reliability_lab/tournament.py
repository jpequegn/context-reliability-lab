"""Repeated-seed memory policy tournament and promotion gates."""

from __future__ import annotations

import random
from collections import defaultdict

from pydantic import Field

from context_reliability_lab.contracts import Corpus, StrictModel, TaskFamily
from context_reliability_lab.graders import grade_run
from context_reliability_lab.policies import MemoryPolicy, default_policies
from context_reliability_lab.runner import run_case


class Trial(StrictModel):
    case_id: str
    family: TaskFamily
    policy: str
    trial_seed: int
    task_success: float
    retrieval_recall: float
    adherence: float
    token_cost: int
    tool_cost: int
    isolation_leaks: float


class PolicySlice(StrictModel):
    policy: str
    family: TaskFamily | None = None
    trials: int
    task_success: float
    retrieval_recall: float
    adherence: float
    average_tokens: float
    isolation_leaks: float
    success_ci_low: float
    success_ci_high: float


class PromotionDecision(StrictModel):
    candidate: str
    baseline: str
    promoted: bool
    quality_delta: float
    token_reduction: float
    paired_ci_low: float
    paired_ci_high: float
    reason: str


class TournamentReport(StrictModel):
    schema_version: str = "context-reliability-tournament-v1"
    seeds: tuple[int, ...]
    cases: int
    policies: tuple[str, ...]
    trials: tuple[Trial, ...]
    overall: tuple[PolicySlice, ...]
    by_family: tuple[PolicySlice, ...]
    graph_decision: PromotionDecision
    budget_violations: int = Field(ge=0)


def run_tournament(
    corpus: Corpus,
    policies: tuple[MemoryPolicy, ...] | None = None,
    seeds: tuple[int, ...] = (1, 2, 3),
) -> TournamentReport:
    selected_policies = policies or default_policies()
    trials: list[Trial] = []
    budget_violations = 0
    for trial_seed in seeds:
        for case in corpus.cases:
            trial_case = case.model_copy(update={"seed": case.seed + trial_seed})
            truth = corpus.hidden_truth[case.case_id]
            for policy in selected_policies:
                result = run_case(trial_case, policy)
                grades = grade_run(result, truth)
                retrieval = next(item for item in grades if item.level == "retrieval")
                behavior = next(item for item in grades if item.level == "behavior")
                outcome = next(item for item in grades if item.level == "outcome")
                state = next(item for item in grades if item.level == "memory_state")
                tokens = sum(item.token_cost for item in result.transitions)
                tools = sum(item.tool_cost for item in result.transitions)
                if any(item.token_cost > case.token_budget for item in result.transitions):
                    budget_violations += 1
                trials.append(
                    Trial(
                        case_id=case.case_id,
                        family=case.family,
                        policy=policy.name,
                        trial_seed=trial_seed,
                        task_success=outcome.metrics["task_success"],
                        retrieval_recall=retrieval.metrics["recall"],
                        adherence=behavior.metrics["adherence"],
                        token_cost=tokens,
                        tool_cost=tools,
                        isolation_leaks=state.metrics["scope_leaks"],
                    )
                )
    overall = tuple(
        _slice(policy.name, None, [trial for trial in trials if trial.policy == policy.name])
        for policy in selected_policies
    )
    by_family = tuple(
        _slice(
            policy.name,
            family,
            [trial for trial in trials if trial.policy == policy.name and trial.family is family],
        )
        for policy in selected_policies
        for family in TaskFamily
    )
    decision = _graph_promotion(trials, overall)
    return TournamentReport(
        seeds=seeds,
        cases=len(corpus.cases),
        policies=tuple(policy.name for policy in selected_policies),
        trials=tuple(trials),
        overall=overall,
        by_family=by_family,
        graph_decision=decision,
        budget_violations=budget_violations,
    )


def _slice(policy: str, family: TaskFamily | None, trials: list[Trial]) -> PolicySlice:
    successes = [trial.task_success for trial in trials]
    low, high = bootstrap_interval(successes)
    count = len(trials)
    return PolicySlice(
        policy=policy,
        family=family,
        trials=count,
        task_success=sum(successes) / count,
        retrieval_recall=sum(item.retrieval_recall for item in trials) / count,
        adherence=sum(item.adherence for item in trials) / count,
        average_tokens=sum(item.token_cost for item in trials) / count,
        isolation_leaks=sum(item.isolation_leaks for item in trials),
        success_ci_low=low,
        success_ci_high=high,
    )


def bootstrap_interval(values: list[float], *, repetitions: int = 500) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(238)
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(repetitions)
    )
    return means[int(0.025 * repetitions)], means[int(0.975 * repetitions) - 1]


def _graph_promotion(trials: list[Trial], overall: tuple[PolicySlice, ...]) -> PromotionDecision:
    graph = next(item for item in overall if item.policy == "graph-linked")
    baselines = [item for item in overall if item.policy not in {"graph-linked", "no-memory"}]
    baseline = max(baselines, key=lambda item: (item.task_success, -item.average_tokens))
    quality_delta = graph.task_success - baseline.task_success
    token_reduction = (
        (baseline.average_tokens - graph.average_tokens) / baseline.average_tokens
        if baseline.average_tokens
        else 0.0
    )
    grouped: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for trial in trials:
        if trial.policy in {graph.policy, baseline.policy}:
            grouped[(trial.case_id, trial.trial_seed)][trial.policy] = trial.task_success
    paired = [
        values[graph.policy] - values[baseline.policy]
        for values in grouped.values()
        if graph.policy in values and baseline.policy in values
    ]
    low, high = bootstrap_interval(paired)
    promoted = quality_delta > 0 or (quality_delta >= 0 and token_reduction >= 0.05)
    if quality_delta > 0:
        reason = "held_out_quality_improved"
    elif quality_delta >= 0 and token_reduction >= 0.05:
        reason = "non_inferior_quality_with_efficiency_gain"
    else:
        reason = "no_measured_quality_or_efficiency_gain"
    return PromotionDecision(
        candidate=graph.policy,
        baseline=baseline.policy,
        promoted=promoted,
        quality_delta=quality_delta,
        token_reduction=token_reduction,
        paired_ci_low=low,
        paired_ci_high=high,
        reason=reason,
    )
