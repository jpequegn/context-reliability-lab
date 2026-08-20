"""Deterministic stateful case runner."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from context_reliability_lab.contracts import (
    Answer,
    EvalCase,
    Event,
    EventKind,
    MemoryRecord,
    RunLineage,
    RunResult,
    Transition,
    digest,
)
from context_reliability_lab.policies import MemoryPolicy


class AgentAdapter(Protocol):
    name: str

    def answer(self, event: Event, selected: tuple[MemoryRecord, ...]) -> tuple[str, bool]: ...


class DeterministicAgent:
    name = "deterministic-agent-v1"

    def answer(self, event: Event, selected: tuple[MemoryRecord, ...]) -> tuple[str, bool]:
        if event.kind is not EventKind.QUERY:
            return event.kind.value, False
        for record in selected:
            lowered = record.content.lower()
            for route in ("green", "red", "blue", "black"):
                if f"route {route}" in lowered:
                    return f"route {route}", False
        return "abstain", True


def _state_digest(memory: tuple[MemoryRecord, ...]) -> str:
    return digest([record.model_dump(mode="json") for record in memory])


def run_case(
    case: EvalCase,
    policy: MemoryPolicy,
    *,
    agent: AgentAdapter | None = None,
    run_id: str | None = None,
) -> RunResult:
    adapter = agent or DeterministicAgent()
    memory = case.initial_memory
    initial_digest = _state_digest(memory)
    transitions: list[Transition] = []
    final_content, abstained = "abstain", True
    final_evidence: tuple[str, ...] = ()
    for index, event in enumerate(case.events):
        before = _state_digest(memory)
        selected = policy.select(case, event, memory, event.at)
        selected = _fit_budget(selected, case.token_budget)
        content, event_abstained = adapter.answer(event, selected)
        if event.kind is EventKind.QUERY:
            final_content, abstained = content, event_abstained
            final_evidence = tuple(record.source.evidence_id for record in selected)
        transitions.append(
            Transition(
                transition_id=f"{case.case_id}-{policy.name}-{index:02d}",
                case_id=case.case_id,
                policy=policy.name,
                event_id=event.event_id,
                state_before=before,
                state_after=_state_digest(memory),
                retrieved_memory_ids=tuple(record.memory_id for record in selected),
                action=content,
                evidence_ids=tuple(record.source.evidence_id for record in selected),
                token_cost=sum(len(record.content.split()) for record in selected),
                tool_cost=0,
            )
        )
    identifier = run_id or f"run-{case.case_id}-{policy.name}-{case.seed}"
    lineage = RunLineage(
        run_id=identifier,
        corpus_digest=digest(case.model_dump(mode="json")),
        policy_version=f"{policy.name}:{policy.version}",
        seed=case.seed,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        input_digest=initial_digest,
    )
    return RunResult(
        lineage=lineage,
        case=case,
        policy=policy.name,
        transitions=tuple(transitions),
        answer=Answer(
            case_id=case.case_id,
            policy=policy.name,
            content=final_content,
            evidence_ids=final_evidence,
            abstained=abstained,
        ),
        final_memory=memory,
    )


def _fit_budget(records: tuple[MemoryRecord, ...], token_budget: int) -> tuple[MemoryRecord, ...]:
    selected: list[MemoryRecord] = []
    used = 0
    for record in records:
        cost = len(record.content.split())
        if used + cost <= token_budget:
            selected.append(record)
            used += cost
    return tuple(selected)
