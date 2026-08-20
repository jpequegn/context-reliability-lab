# Context Reliability Lab

Provider-neutral evaluation gym for stateful memory retrieval, adherence,
generalization, hygiene, and recovery.

## Development

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --all-groups
make check
uv run context-lab version
```

## Guardrails

- Use synthetic or explicitly sanitized data only.
- Never capture hidden chain-of-thought.
- Treat model graders as uncalibrated proposals until human-reviewed.
- Keep provider access behind optional adapters; core tests require no credentials.
- Do not generalize benchmark results beyond the tested corpus and harness.

Source project idea: https://github.com/jpequegn/project-ideas/issues/238

