# Usage and Extensions

## Typical Usage

### Develop a memory policy

Implement the `MemoryPolicy` interface, add it to a tournament policy set, and
compare it under the same cases, seeds, token budget, and tool budget. Inspect
family slices instead of relying only on an aggregate score.

### Diagnose a failure

Use `uv run --no-editable context-lab replay` to inspect compaction survival and the outcome after
removing the earliest bad memory. The state diff identifies the first transition
where retrieval, action, or state diverged.

### Gate a release

Run the demo in CI and consume `evaluation-report.xml` as JUnit evidence. Preserve
`evaluation-report.json`, the DuckDB ledger, and minimized regressions as build
artifacts. `uv run --no-editable context-lab status` exits nonzero when a gate requires attention.

### Feed operational reliability

Use `uv run --no-editable context-lab export --format otlp` to send sanitized transition spans to the
#227 Agent Trace Reliability Control Plane. Exported failure regressions can be
adapted to #167 and #185 without exposing hidden truth at runtime.

## Follow-On Projects

- **#239 MemoryFS Guard:** run its raw-search and compiled-pack policies against
  this corpus; semantic Git diffs can select the affected regressions.
- **#240 Sleep-Time Memory Curator:** require every proposed background memory
  change to pass hygiene, recovery, and counterfactual gates before promotion.
- **#241 Adaptive Context Mounting Router:** use these cases and equal-budget
  receipts to measure marginal value of each mounted context class.
- **#242 Operational Reasoning Graph:** retain graph steering only where its
  held-out quality or efficiency beats flat and compact baselines.

## Innovative Extensions

- **Memory mutation testing:** delete, reorder, expire, duplicate, or contradict
  one record and predict the exact affected cases before replay.
- **Causal failure bisect:** binary-search state history to find the earliest
  memory whose removal changes a verified outcome.
- **Reviewer-time optimization:** optimize policy promotion for corrected failures
  per reviewer minute, with quality and privacy as hard constraints.
- **Live trace-to-memory loop:** consume sanitized #227 anomalies, minimize the
  implicated memory trajectory, and propose a regression for reviewed promotion.
- **Provider tournament:** add external adapters behind the protocol and compare
  them under identical visible context, while retaining deterministic graders.

## Production Adoption Checklist

1. Replace synthetic cases with explicitly sanitized, reviewed domain cases.
2. Double-review representative labels and publish agreement by task family.
3. Define domain-specific release thresholds and costly-error classes.
4. Add real tokenizer and provider cost adapters where billing precision matters.
5. Send traces through a configured OpenTelemetry Collector with retention rules.
6. Keep a deterministic safe policy and rollback path for every promoted policy.
