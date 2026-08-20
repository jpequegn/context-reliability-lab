"""Versioned contracts for stateful memory evaluations."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "context-reliability-v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskFamily(StrEnum):
    RETRIEVAL = "retrieval"
    ADHERENCE = "adherence"
    GENERALIZATION = "generalization"
    HYGIENE = "hygiene"
    RECOVERY = "recovery"


class EventKind(StrEnum):
    OBSERVE = "observe"
    QUERY = "query"
    CORRECT = "correct"
    COMPACT = "compact"
    FAIL = "fail"
    RESUME = "resume"


class FailureClass(StrEnum):
    MISSED_RELEVANT = "missed_relevant"
    STALE_ADOPTION = "stale_adoption"
    INVALID_ADHERENCE = "invalid_adherence"
    OVER_GENERALIZATION = "over_generalization"
    DUPLICATE_PROMOTION = "duplicate_promotion"
    POISON_ADOPTION = "poison_adoption"
    FAILED_RECOVERY = "failed_recovery"
    SCOPE_LEAK = "scope_leak"


class EvidenceRef(StrictModel):
    evidence_id: str
    uri: str
    digest: str


class MemoryRecord(StrictModel):
    memory_id: str
    user_id: str
    task_id: str
    content: str
    source: EvidenceRef
    observed_at: datetime
    valid_from: datetime
    valid_to: datetime | None = None
    expires_at: datetime | None = None
    confidence: float = Field(ge=0, le=1)
    sensitivity: str = "internal"
    supersedes: tuple[str, ...] = ()
    poisoned: bool = False

    @model_validator(mode="after")
    def validate_times(self) -> MemoryRecord:
        if self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("valid_to precedes valid_from")
        if self.expires_at and self.expires_at < self.observed_at:
            raise ValueError("expires_at precedes observed_at")
        return self


class Event(StrictModel):
    event_id: str
    kind: EventKind
    at: datetime
    content: str
    evidence_ids: tuple[str, ...] = ()


class ExpectedTransition(StrictModel):
    event_id: str
    required_memory_ids: tuple[str, ...] = ()
    forbidden_memory_ids: tuple[str, ...] = ()
    expected_action: str


class GraderRubric(StrictModel):
    required_evidence_ids: tuple[str, ...]
    acceptable_actions: tuple[str, ...]
    release_blocking_failures: tuple[FailureClass, ...]


class EvalCase(StrictModel):
    schema_version: str = SCHEMA_VERSION
    case_id: str
    family: TaskFamily
    seed: int
    user_id: str
    task_id: str
    token_budget: int = Field(gt=0)
    tool_budget: int = Field(ge=0)
    initial_memory: tuple[MemoryRecord, ...]
    events: tuple[Event, ...]
    expected_transitions: tuple[ExpectedTransition, ...]
    rubric: GraderRubric
    seeded_failure: FailureClass


class HiddenTruth(StrictModel):
    case_id: str
    correct_memory_ids: tuple[str, ...]
    expected_answer: str
    failure_memory_id: str
    canary: str


class Transition(StrictModel):
    transition_id: str
    case_id: str
    policy: str
    event_id: str
    state_before: str
    state_after: str
    retrieved_memory_ids: tuple[str, ...] = ()
    action: str
    evidence_ids: tuple[str, ...] = ()
    token_cost: int = Field(ge=0)
    tool_cost: int = Field(ge=0)


class Answer(StrictModel):
    case_id: str
    policy: str
    content: str
    evidence_ids: tuple[str, ...]
    abstained: bool = False


class GraderResult(StrictModel):
    case_id: str
    policy: str
    level: str
    passed: bool
    reason_code: str
    evidence_transition_ids: tuple[str, ...]
    metrics: dict[str, float] = Field(default_factory=dict)


class RunLineage(StrictModel):
    run_id: str
    corpus_digest: str
    policy_version: str
    seed: int
    started_at: datetime
    input_digest: str


class Corpus(StrictModel):
    schema_version: str = SCHEMA_VERSION
    cases: tuple[EvalCase, ...]
    hidden_truth: dict[str, HiddenTruth]

    @model_validator(mode="after")
    def validate_truth(self) -> Corpus:
        case_ids = {case.case_id for case in self.cases}
        if case_ids != set(self.hidden_truth):
            raise ValueError("hidden truth must map one-to-one to cases")
        return self

    @property
    def digest(self) -> str:
        return digest(self.model_dump(mode="json"))


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()
