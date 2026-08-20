import json

from context_reliability_lab.contracts import TaskFamily
from context_reliability_lab.corpus import generate_case
from context_reliability_lab.graders import grade_run
from context_reliability_lab.policies import CuratedPolicy
from context_reliability_lab.runner import run_case
from context_reliability_lab.storage import RunStore, export_otlp, verify_receipt


def _result(index: int = 0):
    case, truth = generate_case(TaskFamily.RETRIEVAL, index)
    result = run_case(case, CuratedPolicy(), run_id=f"stored-run-{index}")
    return result, grade_run(result, truth), truth


def test_store_is_idempotent_and_rebuilds_identically(tmp_path) -> None:
    result, grades, _ = _result()
    with RunStore(tmp_path / "runs.duckdb") as store:
        assert store.save_run(result, grades, tenant_id="tenant-a")
        assert not store.save_run(result, grades, tenant_id="tenant-a")
        assert store.load_run(result.lineage.run_id) == result
        assert store.verify_run(result.lineage.run_id)


def test_sanitized_json_and_otlp_never_export_canaries(tmp_path) -> None:
    result, grades, truth = _result()
    with RunStore(tmp_path / "runs.duckdb") as store:
        store.save_run(result, grades, tenant_id="tenant-a")
        exported = json.dumps(store.export_json(result.lineage.run_id), sort_keys=True)
    otlp = json.dumps(export_otlp(result, grades), sort_keys=True)
    assert truth.canary not in exported
    assert truth.canary not in otlp
    assert "resourceSpans" in otlp
    assert "context.memory.transition" in otlp


def test_deletion_is_tenant_scoped_and_receipted(tmp_path) -> None:
    first, first_grades, _ = _result(0)
    second, second_grades, _ = _result(1)
    with RunStore(tmp_path / "runs.duckdb") as store:
        store.save_run(first, first_grades, tenant_id="tenant-a")
        store.save_run(second, second_grades, tenant_id="tenant-b")
        receipt = store.delete("tenant-a", run_id=first.lineage.run_id)
        assert verify_receipt(receipt)
        assert receipt.deleted_rows > 0
        assert not store.verify_run(first.lineage.run_id)
        assert store.verify_run(second.lineage.run_id)
