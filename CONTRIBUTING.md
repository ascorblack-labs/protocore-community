# Contributing to Protocore

Thanks for considering it. This document is short because the project has few
rules — but the ones it has are load-bearing.

## The one rule that is not negotiable

**The core never imports upward.** `protocore/` may import from `protocore` and
from its four runtime dependencies. It may not import any package whose name
begins `protocore_`, and it may not grow a database driver, an HTTP client, a
cloud SDK, or a model-provider SDK.

If the core needs something from the outside, that need is expressed as a
`Protocol` in `protocore/contracts/` and injected. `tests/test_core_import_boundary.py`
enforces this and will fail your pull request rather than argue about it.

## Setup

```bash
uv sync --extra dev
```

Python ≥ 3.12. No services, no containers, no fixtures to provision — the whole
suite runs offline against in-memory doubles and finishes in well under a minute.

## The gates

Run all four before you open a pull request. CI runs the same ones on Python
3.12, 3.13, and 3.14.

```bash
uv run pytest .                                     # tests + 90% coverage floor
uv run ruff check .                                 # lint
uv run mypy --strict                                # typing, strict, no path argument
uv run bandit -r protocore -q -c pyproject.toml     # security
```

Two things people trip over:

- **Do not pass a path to `mypy`.** A path on the command line *replaces* the
  configured `files` list, which silently excludes the test tree. Bare
  `mypy --strict` checks the surface `pyproject.toml` declares.
- **Do not pipe a gate into `tail` or `head`.** A shell pipeline reports the
  *last* command's status, so a failing gate reads as a pass. Redirect to a file
  and read the exit code separately. This is not hypothetical; the security gate
  sat red for a while because of exactly this.

## Conventions

- **No magic numbers in the executable path.** A tunable value belongs on
  `RuntimeConstants` as a field with a description, so an operator can change it
  without a deploy. `protocore/constants.py` is only for memory-safety caps that
  are not per-tenant.
- **New behaviour defaults off**, or to a value that reproduces the previous
  behaviour exactly. A tenant opts in deliberately; an upgrade never changes
  what a running system does.
- **No module-level mutable state.** No module-level dicts, no locks held as
  module state. Correctness-affecting state lives per-run on the `QueryEngine`
  so the runtime stays safe to scale horizontally.
- **Log at WARNING or above** on paths that run in production. `logger.info` in
  the hot loop is noise someone else has to pay for.
- **Docstrings explain why**, not what. The code says what it does. The comment
  is for the reason it does it that way — usually a failure mode that is not
  obvious from the surrounding lines.

## Tests

A change to behaviour needs a test that fails without it. The suite is large
(2973 tests) precisely so that the loop's edge cases stay pinned; adding to it is
the normal cost of a change, not an extra.

Coverage is enforced at 90% and the current margin is thin. If your change adds
a branch, cover it.

## Commits and pull requests

Write the commit message for someone reading `git log` in a year with no other
context. Say what changed and why it needed changing; if the change fixes
something subtle, say what the symptom was.

Keep a pull request to one concern. A refactor bundled with a behaviour change
is two reviews wearing one hat, and the behaviour change is the one that gets
skimmed.

## Reporting bugs

Open an issue with a reproduction. A failing test is the ideal form of one.

Security problems do **not** go in the issue tracker — see
[`SECURITY.md`](SECURITY.md).

## License of contributions

By contributing you agree that your contribution is licensed under the
[Mozilla Public License 2.0](LICENSE), the same terms as the rest of the
project. There is no CLA.
