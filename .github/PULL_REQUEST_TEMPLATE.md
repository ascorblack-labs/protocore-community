## What changed

<!-- What the change does, in a sentence or two. -->

## Why

<!-- The reason it needed changing. If it fixes something subtle, what was the
     symptom? -->

## Checklist

- [ ] `uv run pytest .` passes (tests plus the 90% coverage floor)
- [ ] `uv run ruff check .` is clean
- [ ] `uv run mypy --strict` is clean — no path argument, it replaces the
      configured file list and silently skips the tests
- [ ] `uv run bandit -r protocore -q -c pyproject.toml` is clean
- [ ] Behaviour changes have a test that fails without the change
- [ ] New tunables are `RuntimeConstants` fields with a description, and
      default to the previous behaviour
- [ ] Docs updated if anything user-facing moved
