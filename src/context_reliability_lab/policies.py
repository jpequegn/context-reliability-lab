"""Deterministic provider-neutral memory policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from context_reliability_lab.contracts import EvalCase, Event, MemoryRecord


class MemoryPolicy(ABC):
    name: str
    version = "1"

    @abstractmethod
    def select(
        self,
        case: EvalCase,
        event: Event,
        memory: tuple[MemoryRecord, ...],
        now: datetime,
    ) -> tuple[MemoryRecord, ...]:
        """Select policy-visible memory for an event."""


def eligible(memory: tuple[MemoryRecord, ...], case: EvalCase) -> tuple[MemoryRecord, ...]:
    """Enforce authorization before any policy ranks records."""
    return tuple(
        record
        for record in memory
        if record.user_id == case.user_id and record.task_id == case.task_id
    )


def active(records: tuple[MemoryRecord, ...], now: datetime) -> tuple[MemoryRecord, ...]:
    return tuple(
        record
        for record in records
        if (record.valid_to is None or record.valid_to >= now)
        and (record.expires_at is None or record.expires_at >= now)
    )


class NoMemoryPolicy(MemoryPolicy):
    name = "no-memory"

    def select(
        self,
        case: EvalCase,
        event: Event,
        memory: tuple[MemoryRecord, ...],
        now: datetime,
    ) -> tuple[MemoryRecord, ...]:
        return ()


class AppendOnlyPolicy(MemoryPolicy):
    name = "append-only"

    def select(
        self,
        case: EvalCase,
        event: Event,
        memory: tuple[MemoryRecord, ...],
        now: datetime,
    ) -> tuple[MemoryRecord, ...]:
        return eligible(memory, case)


class SummaryPolicy(MemoryPolicy):
    name = "summary"

    def select(
        self,
        case: EvalCase,
        event: Event,
        memory: tuple[MemoryRecord, ...],
        now: datetime,
    ) -> tuple[MemoryRecord, ...]:
        scoped = eligible(memory, case)
        return tuple(sorted(scoped, key=lambda item: item.observed_at, reverse=True)[:1])


class CuratedPolicy(MemoryPolicy):
    name = "curated"

    def select(
        self,
        case: EvalCase,
        event: Event,
        memory: tuple[MemoryRecord, ...],
        now: datetime,
    ) -> tuple[MemoryRecord, ...]:
        return active(eligible(memory, case), now)


class CriticGatedPolicy(MemoryPolicy):
    name = "critic-gated"

    def select(
        self,
        case: EvalCase,
        event: Event,
        memory: tuple[MemoryRecord, ...],
        now: datetime,
    ) -> tuple[MemoryRecord, ...]:
        return tuple(
            record
            for record in active(eligible(memory, case), now)
            if not record.poisoned and record.confidence >= 0.8
        )


class GraphLinkedPolicy(MemoryPolicy):
    name = "graph-linked"

    def select(
        self,
        case: EvalCase,
        event: Event,
        memory: tuple[MemoryRecord, ...],
        now: datetime,
    ) -> tuple[MemoryRecord, ...]:
        allowed_evidence = set(event.evidence_ids)
        return tuple(
            record
            for record in active(eligible(memory, case), now)
            if record.source.evidence_id in allowed_evidence
        )


class SeededFaultPolicy(MemoryPolicy):
    """Deliberately unsafe policy used to validate grader sensitivity."""

    name = "seeded-fault"

    def select(
        self,
        case: EvalCase,
        event: Event,
        memory: tuple[MemoryRecord, ...],
        now: datetime,
    ) -> tuple[MemoryRecord, ...]:
        suffix = "stale" if case.seed % 2 == 0 else "restricted"
        return tuple(record for record in memory if record.memory_id.endswith(suffix))


def default_policies() -> tuple[MemoryPolicy, ...]:
    return (
        NoMemoryPolicy(),
        AppendOnlyPolicy(),
        SummaryPolicy(),
        CuratedPolicy(),
        CriticGatedPolicy(),
        GraphLinkedPolicy(),
    )
