# RuntimeConstants

`RuntimeConstants` is the single configuration surface that flows across the
core ↔ the host boundary. It is the mechanism behind the project's hard rule:
**no inline magic numbers.** Every runtime-tunable threshold — token-budget
fractions, operational caps, timeouts, feature kill-switches — is a typed field
on one frozen Pydantic snapshot that is default-safe and dashboard-configurable.
Runtime code reads from the snapshot; it never embeds a literal.

This page is the conceptual reference for that model. The operational day-to-day
(the dashboard flow, override precedence, and anti-patterns) lives with 
the host service and its administration dashboard, which read and persist the
per-tenant overrides this model describes. The deep architecture treatment is in
the [RuntimeConstants system](architecture.md#runtimeconstants-system) section of
[`architecture.md`](architecture.md).

## The model: frozen Pydantic, `extra="forbid"`

`RuntimeConstants` lives in `protocore/contracts/runtime_constants.py`. It is a
`pydantic.BaseModel` whose configuration is:

```python
class RuntimeConstants(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    ...
```

Two properties matter, and both are deliberate.

- **`frozen=True`** — a snapshot is immutable once constructed. The runtime
  binds one snapshot at engine construction — it is stored on
  `QueryEngineConfig.rc`, and `query(engine)` reads it from `engine.config.rc` —
  so a run's behaviour cannot drift mid-flight even if an operator edits a value
  while the run is in progress. There is no global state and no module-level
  cache; the snapshot is the authority for the lifetime of the engine. New
  values apply on the next run, never retroactively.

- **`extra="forbid"`** — an unknown field is a hard validation error, not a
  silently-ignored extra. The practical consequence: **core and the host must
  deploy paired.** The host builds each snapshot from per-tenant data and
  hands it to the core; if the host sends a field the deployed core model does
  not declare (or vice-versa), construction rejects rather than swallowing the
  mismatch. A core/the host version skew therefore fails loudly at the
  boundary instead of corrupting behaviour. Treat the core and its host as a single
  release unit when adding or renaming fields.

All fields are **canonical inputs only**. Formula-derived values (for example
token budgets and the compaction trigger in tokens) are *computed* from these
inputs elsewhere in the runtime — they are not stored as fields. Adding a
derived value as its own field would invite two sources of truth.

## The provider: `RuntimeConstantsProvider`

Core does not know how snapshots are built or where tenant overrides live — that
is the host's job, expressed through a Protocol that core defines and
the host implements:

```python
@runtime_checkable
class RuntimeConstantsProvider(Protocol):
    async def get(self, tenant_id: str) -> RuntimeConstants:
        """Return the latest snapshot for ``tenant_id``."""
        ...
```

The production implementation (in the host) reads per-tenant overrides from
Postgres, caches them in Redis, watches Redis pub/sub for invalidation, and
rebuilds a fresh frozen snapshot whenever an override changes. The core only
ever sees the result of `get(tenant_id)`: a ready-to-use, immutable snapshot.
This keeps the persistence and caching machinery entirely on the host side
of the boundary.

## Defaults and the in-memory provider

Every field carries a **default-safe, default-off** value defined inline on the
field. The default is the single source of truth for that value — that is
precisely *why* it lives on the field and not as a scattered literal. New
behavioural surfaces ship behind a boolean that defaults to `False` (or a cap
that defaults to a conservative value), so deploying the code does not change
behaviour until an operator opts in per tenant. Feature kill-switches default
`True` only when the feature is the established steady-state path and the switch
exists for incident rollback.

Read the live `Field(...)` default — do not infer it from older docs. Two
exceptions worth naming because they have been mis-stated:

- `workspace_enabled` defaults to **`True`** (the workspace subsystem is
  available; the host does not expose the legacy Workspace* LLM tools by
  default).
- Intent, ledger, tree, lanes, typed hooks, telemetry, manual compact, and
  steer/follow-up are default-**off**: `intent_settlement_enabled`,
  `usage_ledger_enabled`, `session_tree_enabled`, `lanes_enabled`,
  `typed_hooks_enabled`, `telemetry_spans_enabled`,
  `compaction_manual_enabled`, `steer_follow_up_enabled`.

Personal API-key policy is part of this tunable surface. By default, one user
may hold at most 10 active keys in an account
(`personal_api_key_active_limit`), and successful key use updates its durable
last-used timestamp at most once every 300 seconds
(`personal_api_key_last_used_write_interval_seconds`). Setting the write
interval to `0` records every authenticated use; the active-key limit must
remain positive.

For tests and the in-memory smoke runtime — anywhere there is no Postgres-backed
provider — core ships two helpers in `protocore/runtime/runtime_constants.py`:

- `default_runtime_constants(**overrides)` — returns a `RuntimeConstants`
  built entirely from field defaults, with optional keyword overrides for the
  fields a test needs to vary.
- `StaticRuntimeConstantsProvider` — a `RuntimeConstantsProvider` that returns
  one fixed snapshot for every `tenant_id`. Useful as the provider in a
  single-tenant smoke run or unit test.

Both are re-exported from the package top level (`from protocore import
default_runtime_constants, StaticRuntimeConstantsProvider`). Production pods do
**not** use these; they wire in the Postgres + Redis provider described above.

## Static caps in `constants.py`

`RuntimeConstants` is for values that should be tunable per tenant through the
dashboard. A small, separate set of values must **never** vary per scope: memory
safety ceilings and protocol identifiers. Those live as module-level constants
in `protocore/constants.py` — for example `MAX_TOOL_CALL_ARGUMENT_BYTES`,
`MAX_ARTIFACTS`, `MAX_STRUCTURED_JSON_CHARS`, `PROTOCOL_VERSION`, and
`DEFAULT_MODEL`. These are non-negotiable backstops enforced regardless of what
a snapshot says; they are not exposed for dashboard editing because making them
tunable would let a misconfiguration defeat a safety bound.

The decision rule:

- **Should an operator be able to tune it per tenant?** → `RuntimeConstants`
  field.
- **Is it a hard safety ceiling or a protocol/identity constant that must hold
  everywhere?** → `constants.py`.

A value belongs to exactly one of these. It is never both, and it is never an
inline literal in runtime logic.

## Adding a tunable: the 3-edit rule

Because the configuration surface spans the core and its host, adding one new tunable is a
**three-edit** operation. Skipping any edit yields a field that either rejects
at the boundary (`extra="forbid"`) or never reaches the dashboard.

1. **Core Pydantic field.** Add the field to `RuntimeConstants` in
   `protocore/contracts/runtime_constants.py`, with a `Field(...)` declaration
   that carries a default-safe/off value, validation bounds where applicable,
   and a `description`. The default on the field is the canonical value.

2. **The host `_FIELD_MAP` identity entry.** Register the field in 
   the host `_FIELD_MAP` so the provider knows to read and write it. Without
   this entry the dashboard cannot persist an override for the field.

3. **Migration catalog seed.** Add the field to the host migration catalog
   seed so existing tenants get a row for it. Without the seed the field exists
   in the model but has no catalog presence for the dashboard to enumerate.

With all three edits in place, the dashboard Constants page auto-discovers the
field and renders an editor for it — no further registration is needed. The
exact the host file locations and the dashboard persistence flow live with 
the host service that owns the `_FIELD_MAP` and the migration catalog.

## How to read a field default

Each field declares its own default and (where numeric) its validation bounds.
To learn the current behaviour of a tunable, read its `Field(...)` definition in
`protocore/contracts/runtime_constants.py` — the `default=` is authoritative and
the `description=` explains the value's intent and history. That field
definition is the *only* place the value is written down; runtime code reads it
from the live snapshot:

```python
# Correct — read the tunable from the injected snapshot.
if iteration >= rc.max_iterations:
    stop()

# Wrong — an inline literal is a magic number. It bypasses the snapshot,
# cannot be tuned per tenant, and violates the no-magic-numbers rule.
if iteration >= 50:
    stop()
```

The literal in the "wrong" example is exactly what the model exists to
eliminate: the bound lives on the `max_iterations` field, not in the branch.

## See also

- [`architecture.md`](architecture.md) — the RuntimeConstants system in the
  full core architecture, including where the snapshot is bound to the engine.
- `contracts.md` — the wider contract surface (the interface Protocols
  the host implements), of which `RuntimeConstantsProvider` is one.
