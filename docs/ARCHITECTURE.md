# Architecture

## Boundary

Context Reliability Lab evaluates memory behavior. The completed
`agent-memory-integrity-lab` (#200) owns durable memory correction and recovery
primitives; this project supplies reusable stateful eval cases, graders, policy
comparisons, and release evidence.

## Components

```text
versioned contracts + hidden truth
              |
      synthetic 50-case corpus
              |
 provider-neutral runner ---- six deterministic policies
              |
 state / retrieval / behavior / outcome graders
              |
 repeated-seed tournament + bootstrap intervals
              |
 compaction / counterfactual / first divergence
              |
 DuckDB evidence ledger ---- OTLP and CI regressions
              |
 JSON / JUnit / Markdown release gate
```

`contracts.py` contains immutable Pydantic envelopes. `corpus.py` creates ten
cases for each task family and stores hidden truth separately. `runner.py` knows
only an agent protocol and a policy protocol. Authorization filtering occurs in
`policies.py` before ranking.

`graders.py` never inspects hidden reasoning. It grades observable state,
retrievals, actions, evidence, and outcomes. `tournament.py` holds task, policy,
tool, and token budgets constant. `replay.py` minimizes failures and preserves
the original source digest. `storage.py` stores complete run lineage, while its
exports redact synthetic canaries.

## Determinism

Inputs, policies, seeds, budgets, source evidence, state digests, and policy
versions are recorded. Bootstrap sampling uses a fixed seed. DuckDB writes are
idempotent by run ID and payload digest. A run-ID collision with different
content is rejected.

## Trust Boundary

- Hidden truth is unavailable to policies and the agent adapter.
- User and task scope are filtered before retrieval ranking.
- Corrections do not delete original observations.
- Exported reports and traces contain IDs, metrics, and reason codes, not private
  memory payloads or hidden chain-of-thought.
- Graph links are retrieval signals, not truth or causal evidence.

## V1 Limitations

- The corpus and double-review calibration labels are synthetic.
- Policy behavior is deterministic; provider adapters remain an extension point.
- Token cost is a stable word-count proxy, not provider billing.
- Confidence intervals quantify repeated fixture outcomes, not external validity.
- The OTLP exporter emits JSON trace envelopes; a live collector receiver is not
  included.
- The graph policy uses bounded evidence links rather than a production graph.

