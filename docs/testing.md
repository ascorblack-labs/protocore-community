# Testing the Core

How to test, lint, and type-check the pure core (`protocore`). All commands
assume [`uv`](https://docs.astral.sh/uv/) and are run from the repository root.

## Setup

Install the package together with its development dependencies (pytest,
pytest-asyncio, pytest-cov, ruff, mypy):

```bash
uv sync --extra dev
```

## The three checks

The core is verified by three independent gates. Run all three before proposing
a change; CI runs the same commands.

```bash
uv run pytest .              # tests (+ coverage)
uv run ruff check .          # lint + import sorting
uv run mypy protocore        # static type-check (strict, package only)
```

Notes:

- **Tests** — `pytest` discovers from `tests/` (configured by
  `testpaths = ["tests"]` in `pyproject.toml`). `asyncio_mode = "auto"`, so an
  `async def test_*` runs without a per-test `@pytest.mark.asyncio` marker.
- **Lint** — `ruff` is configured with `line-length = 120`,
  `target-version = "py312"`, and the `E,F,W,I,B,UP,RUF` rule sets.
- **Type-check** targets the `protocore` *package*, not the test tree.

## Test layout

Tests live at the **repository root under `tests/`**, *not* inside the
`protocore` package. The package ships no `test_*.py` files of its own — keeping
runtime code and test code physically separate.

```
tests/
  conftest.py                      # shared fixtures (e.g. tenant_id)
  test_core_import_boundary.py     # import-boundary guard (see below)
  test_constructible.py
  test_runtime_constants_sse.py
  unit/
    contracts/                     # contract types + protocol shape tests
    runtime/                       # loop, dispatch, context, resilience, ...
    tools/                         # @tool decorator, lean verbs, ...
    prompts/
    test_*.py                      # token counting, chain parser, JSON utils, ...
```

The suite is fast and uses minimal mocking: the core has **no external service
dependencies** (no database, no HTTP, no Kubernetes — those live behind the
contracts that the host implements), so nothing needs to be stood up to run
the tests.

## Import-boundary guard test

`tests/test_core_import_boundary.py` enforces the central architectural
invariant: **the core never imports upward.** It AST-parses every `.py` file in
the `protocore` package tree and fails if any top-level import names a package
that sits above the core.

The rule is the namespace, not a list. Everything the core may import from
itself lives under the single `protocore` package; a host puts its adapters,
service layer, and frontends in sibling distributions whose import names all
begin `protocore_`. Naming the shape instead of enumerating today's siblings
means a package added tomorrow is caught on the day it appears.

In other words, importing any `protocore.*` module must pull in **zero**
symbols from the layers above. Outside-facing capabilities are
`Protocol`s the core defines and the host implements (see
[`contracts.md`](contracts.md)); you add behaviour through contracts, adapters,
or `RuntimeConstants`, never by importing upward (see
[`extending.md`](extending.md)). A violation fails CI immediately and reports
each offending `file: imports 'package'`.

## Coverage stance

Coverage is collected by `pytest-cov` against `source = ["protocore"]` with
`branch = true` (see `[tool.coverage.run]` / `[tool.coverage.report]` in
`pyproject.toml`; `__init__.py` files are omitted and Protocol-stub/`overload`
bodies are excluded from the denominator).

The core holds a **strict minimum coverage threshold** — most new capabilities
ship default-off, so their *enabled* paths are exercised by unit tests rather
than by a live run, and the threshold guards against silently shipping
unexercised branches. CI enforces it by running the suite with
`--cov-fail-under=90` (see `.gitlab-ci.yml`); a change that drops coverage below
90% fails the pipeline.

## What to run before a change

Run all three gates locally before proposing a change — they are exactly what CI
runs:

```bash
uv sync --extra dev          # once, to install the dev toolchain
uv run pytest .              # tests + coverage (must meet the threshold)
uv run ruff check .          # lint + import sorting
uv run mypy protocore        # strict static type-check
```

The import-boundary guard (above) runs as part of `pytest`, so a reverse import
into a sibling package fails the test run immediately.
