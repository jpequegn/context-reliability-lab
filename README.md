# Context Reliability Lab

Provider-neutral evaluation gym for stateful memory retrieval, adherence,
generalization, hygiene, and recovery. It evaluates observable memory state and
agent behavior across transitions; it is not another memory store.

## Quick Start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/jpequegn/context-reliability-lab.git
cd context-reliability-lab
uv sync --locked --all-groups --no-editable --reinstall-package context-reliability-lab
uv run --no-editable context-lab demo --output-dir artifacts/demo
uv run --no-editable context-lab status --report artifacts/demo/evaluation-report.json
```

The demo requires no model, provider credentials, or network access. It runs 50
versioned cases across six policies and three seeds, producing 900 trial records.

## What It Evaluates

- **Retrieval:** relevant-memory recall, distractors, temporal correctness, and cost.
- **Adherence:** valid instructions, supersession, exceptions, and abstention.
- **Generalization:** transfer without broadening user or task scope.
- **Hygiene:** duplication, transient observations, poisoning, and expiry.
- **Recovery:** resuming from failed or misleading trajectories without repetition.

Graders separately inspect memory state, retrieval, behavior, and outcome, then
localize the first failing component. The release gate also checks scope isolation,
critical-evidence survival under compaction, portable regression replay, and a
deterministic human-calibration fixture.

## CLI

```bash
# Generate the corpus and run the policy tournament
uv run --no-editable context-lab corpus --output artifacts/corpus.json
uv run --no-editable context-lab tournament --output artifacts/tournament.json

# Run and persist one case
uv run --no-editable context-lab run --case-id recovery-00 --policy critic-gated \
  --database artifacts/runs.duckdb

# Inspect counterfactual and compaction evidence
uv run --no-editable context-lab replay --case-id recovery-00

# Inspect or export a stored run
uv run --no-editable context-lab inspect --database artifacts/runs.duckdb \
  --run-id run-recovery-00-critic-gated-10000
uv run --no-editable context-lab export --database artifacts/runs.duckdb \
  --run-id run-recovery-00-critic-gated-10000 --format otlp
```

The demo writes JSON, JUnit XML, Markdown, DuckDB, ten minimized regressions, and
an OpenTelemetry-compatible trace envelope. Reports and traces redact secret
canaries; the authoritative synthetic corpus retains them to test leak detection.

## Development

```bash
make check
make demo
```

`make check` performs a locked non-editable install, Ruff checks, pytest/Hypothesis
tests, and wheel/sdist builds. CI repeats the checks and uploads demo evidence.

See [Architecture](docs/ARCHITECTURE.md) and
[Usage and Extensions](docs/USAGE_AND_EXTENSIONS.md).

## Guardrails

- Use synthetic or explicitly sanitized data only.
- Never capture hidden chain-of-thought.
- Treat model graders as proposals until calibrated against domain reviewers.
- Filter authorization before ranking or model exposure.
- Do not generalize benchmark results beyond the tested corpus and harness.

Source project idea: https://github.com/jpequegn/project-ideas/issues/238
