"""DuckDB evidence ledger, sanitized exports, and deletion receipts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from context_reliability_lab.contracts import (
    GraderResult,
    RunResult,
    StrictModel,
    digest,
)


class DeletionReceipt(StrictModel):
    schema_version: str = "context-reliability-deletion-v1"
    tenant_id: str
    run_ids: tuple[str, ...]
    deleted_rows: int
    issued_at: datetime
    receipt_digest: str


class RunStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.connection = duckdb.connect(str(self.path))
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _migrate(self) -> None:
        statements = (
            "CREATE TABLE IF NOT EXISTS runs (run_id VARCHAR PRIMARY KEY, "
            "tenant_id VARCHAR, created_at TIMESTAMPTZ, payload VARCHAR, "
            "payload_digest VARCHAR)",
            "CREATE TABLE IF NOT EXISTS cases (case_id VARCHAR PRIMARY KEY, payload VARCHAR)",
            "CREATE TABLE IF NOT EXISTS lineage (run_id VARCHAR PRIMARY KEY, payload VARCHAR)",
            "CREATE TABLE IF NOT EXISTS transitions (run_id VARCHAR, "
            "transition_id VARCHAR, payload VARCHAR, PRIMARY KEY(run_id, transition_id))",
            "CREATE TABLE IF NOT EXISTS retrievals (run_id VARCHAR, "
            "transition_id VARCHAR, memory_id VARCHAR, "
            "PRIMARY KEY(run_id, transition_id, memory_id))",
            "CREATE TABLE IF NOT EXISTS grader_results (run_id VARCHAR, "
            "level VARCHAR, payload VARCHAR, PRIMARY KEY(run_id, level))",
            "CREATE TABLE IF NOT EXISTS outcomes (run_id VARCHAR PRIMARY KEY, payload VARCHAR)",
            "CREATE TABLE IF NOT EXISTS decisions (run_id VARCHAR PRIMARY KEY, payload VARCHAR)",
        )
        for statement in statements:
            self.connection.execute(statement)

    def save_run(
        self,
        result: RunResult,
        grades: tuple[GraderResult, ...],
        *,
        tenant_id: str,
        decision: dict[str, Any] | None = None,
    ) -> bool:
        run_id = result.lineage.run_id
        payload = result.model_dump_json()
        payload_digest = digest(result.model_dump(mode="json"))
        existing = self.connection.execute(
            "SELECT payload_digest FROM runs WHERE run_id = ?", [run_id]
        ).fetchone()
        if existing:
            if existing[0] != payload_digest:
                raise ValueError(f"run id collision for {run_id}")
            return False
        self.connection.execute("BEGIN TRANSACTION")
        try:
            self.connection.execute(
                "INSERT OR IGNORE INTO cases VALUES (?, ?)",
                [result.case.case_id, result.case.model_dump_json()],
            )
            self.connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
                [run_id, tenant_id, result.lineage.started_at, payload, payload_digest],
            )
            self.connection.execute(
                "INSERT INTO lineage VALUES (?, ?)",
                [run_id, result.lineage.model_dump_json()],
            )
            for transition in result.transitions:
                self.connection.execute(
                    "INSERT INTO transitions VALUES (?, ?, ?)",
                    [run_id, transition.transition_id, transition.model_dump_json()],
                )
                for memory_id in transition.retrieved_memory_ids:
                    self.connection.execute(
                        "INSERT INTO retrievals VALUES (?, ?, ?)",
                        [run_id, transition.transition_id, memory_id],
                    )
            for grade in grades:
                self.connection.execute(
                    "INSERT INTO grader_results VALUES (?, ?, ?)",
                    [run_id, grade.level, grade.model_dump_json()],
                )
            self.connection.execute(
                "INSERT INTO outcomes VALUES (?, ?)", [run_id, result.answer.model_dump_json()]
            )
            self.connection.execute(
                "INSERT INTO decisions VALUES (?, ?)",
                [run_id, json.dumps(decision or {"state": "unreviewed"}, sort_keys=True)],
            )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return True

    def load_run(self, run_id: str) -> RunResult:
        row = self.connection.execute(
            "SELECT payload FROM runs WHERE run_id = ?", [run_id]
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return RunResult.model_validate_json(row[0])

    def verify_run(self, run_id: str) -> bool:
        row = self.connection.execute(
            "SELECT payload, payload_digest FROM runs WHERE run_id = ?", [run_id]
        ).fetchone()
        if row is None:
            return False
        result = RunResult.model_validate_json(row[0])
        return digest(result.model_dump(mode="json")) == row[1]

    def export_json(self, run_id: str) -> dict[str, Any]:
        return redact(self.load_run(run_id).model_dump(mode="json"))

    def delete(self, tenant_id: str, *, run_id: str | None = None) -> DeletionReceipt:
        query = "SELECT run_id FROM runs WHERE tenant_id = ?"
        parameters: list[Any] = [tenant_id]
        if run_id is not None:
            query += " AND run_id = ?"
            parameters.append(run_id)
        run_ids = tuple(row[0] for row in self.connection.execute(query, parameters).fetchall())
        deleted = 0
        self.connection.execute("BEGIN TRANSACTION")
        try:
            for identifier in run_ids:
                for table in (
                    "retrievals",
                    "transitions",
                    "grader_results",
                    "outcomes",
                    "decisions",
                    "lineage",
                    "runs",
                ):
                    count = self.connection.execute(
                        f"SELECT count(*) FROM {table} WHERE run_id = ?", [identifier]
                    ).fetchone()[0]
                    self.connection.execute(f"DELETE FROM {table} WHERE run_id = ?", [identifier])
                    deleted += count
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        issued_at = datetime.now(UTC)
        unsigned = {
            "schema_version": "context-reliability-deletion-v1",
            "tenant_id": tenant_id,
            "run_ids": run_ids,
            "deleted_rows": deleted,
            "issued_at": issued_at.isoformat(),
        }
        return DeletionReceipt(
            tenant_id=tenant_id,
            run_ids=run_ids,
            deleted_rows=deleted,
            issued_at=issued_at,
            receipt_digest=digest(unsigned),
        )


def verify_receipt(receipt: DeletionReceipt) -> bool:
    unsigned = {
        "schema_version": receipt.schema_version,
        "tenant_id": receipt.tenant_id,
        "run_ids": receipt.run_ids,
        "deleted_rows": receipt.deleted_rows,
        "issued_at": receipt.issued_at.isoformat(),
    }
    return digest(unsigned) == receipt.receipt_digest


def redact(value: Any) -> Any:
    if isinstance(value, str):
        return "[REDACTED_CANARY]" if "SECRET_CANARY_" in value else value
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    return value


def export_otlp(result: RunResult, grades: tuple[GraderResult, ...]) -> dict[str, Any]:
    trace_id = hashlib.sha256(result.lineage.run_id.encode()).hexdigest()[:32]
    spans: list[dict[str, Any]] = []
    grade_failures = [grade.reason_code for grade in grades if not grade.passed]
    for index, transition in enumerate(result.transitions):
        span_id = hashlib.sha256(transition.transition_id.encode()).hexdigest()[:16]
        attributes = {
            "context.case_id": result.case.case_id,
            "context.policy": result.policy,
            "context.event_id": transition.event_id,
            "context.action": transition.action,
            "context.token_cost": transition.token_cost,
            "context.failure_reasons": ",".join(grade_failures),
        }
        spans.append(
            {
                "traceId": trace_id,
                "spanId": span_id,
                "name": "context.memory.transition",
                "kind": 1,
                "startTimeUnixNano": str(1_800_000_000_000_000_000 + index * 1_000_000),
                "endTimeUnixNano": str(1_800_000_000_000_500_000 + index * 1_000_000),
                "attributes": [
                    {"key": key, "value": _otlp_value(value)} for key, value in attributes.items()
                ],
                "status": {"code": 1 if not grade_failures else 2},
            }
        )
    envelope = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": "context-reliability-lab"},
                        }
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "context-reliability-lab", "version": "0.1.0"},
                        "spans": spans,
                    }
                ],
            }
        ]
    }
    return redact(envelope)


def _otlp_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    return {"stringValue": str(value)}
