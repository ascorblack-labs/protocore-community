# Tool surface

> Audience: an engineer adding or reasoning about agent-facing tools in the
> **pure core** (`protocore/`).
> Scope: the current core library (`protocore/`). The core owns
> tool *contracts, names, dispatch, gating, retrieval, and ordering*. It ships a
> few concrete tools whose entire surface **is** the protocol contract — ask-user
> (`tools/ask_user.py`) and memory (`tools/memory.py`) — but it registers **no**
> concrete *lean default*
> (sandbox / exec / read / write) tool itself; those production, backend-bound
> tools live in the sibling the host repo. (The `IWorkspace` **contract** lives
> in core, but its concrete tools are host-only — see the workspace
> subsystem in [`architecture.md`](architecture.md).)
> For the wider picture see [`architecture.md`](architecture.md).

The core deliberately keeps the agent-visible tool list **small and stable**: a
small universal pool keeps the prompt cheap for smaller local models, and a
deterministic ordering keeps the KV-prefix cache reusable across turns. For the
*lean default* surface (sandbox / exec / read / write) the core registers no
concrete tool — it defines the *shape* and the machinery that runs one tool call
safely, and the host binds the backend-backed implementations. (The core does
ship its own protocol-surface tools — ask-user and memory — see the
[scope note](#tool-surface) above.)

---

## The lean 7-verb surface

`contracts/lean_tool_surface.py` defines a canonical pool of **exactly seven
verbs**. A capable model hand-composes operations against the runtime instead of
filling a dozen bespoke typed schemas. The names are stable, domain-neutral
identifiers a backend SHOULD adopt verbatim so the agent-facing surface is
uniform across backends.

The constants `LEAN_TOOL_EXEC`, `LEAN_TOOL_READ`, `LEAN_TOOL_READ_SILENT`,
`LEAN_TOOL_WRITE`, `LEAN_TOOL_FIND`, `LEAN_TOOL_SEARCH`, `LEAN_TOOL_ANSWER`
collect into the ordered tuple `LEAN_TOOL_NAMES`:

| Verb | Constant | What it does |
|---|---|---|
| `exec` | `LEAN_TOOL_EXEC` | Run a registered runtime binary by path with an argv list and optional stdin (`{path, args, stdin}`). This is **not** a `/bin/sh` — there is no shell metacharacter handling. The single power tool that lets one capable model invoke any runtime binary without a bespoke typed verb per operation. |
| `read` | `LEAN_TOOL_READ` | Read a file/record **and record its path as observed evidence** (grounding-tracked). Use for any source you may cite. |
| `read_silent` | `LEAN_TOOL_READ_SILENT` | Read a file/record **without** recording it as evidence. Identical content to `read`; only the grounding side effect differs. Lets the model browse or compare candidates without polluting its citation set. |
| `write` | `LEAN_TOOL_WRITE` | Create or modify a file/record (the mutation verb). Optimistic-write backends may accept a content hash for compare-and-swap — a backend concern, not part of the universal name. |
| `find` | `LEAN_TOOL_FIND` | Locate files/records by name or path pattern (filename-level discovery). |
| `search` | `LEAN_TOOL_SEARCH` | Search file/record *contents* by query (content-level discovery). |
| `answer` | `LEAN_TOOL_ANSWER` | Produce the final, terminal answer for the task with supporting citations (`{message, outcome, refs}`). Every lean surface MUST guarantee a terminal answer is always producible. |

The `answer` verb's `outcome` field is a **free-form string** whose accepted
values are defined by the backend's own answer contract — core owns only the
field's *shape*, never a fixed value set.

### Grounding-tracked reads

`GROUNDING_TRACKED_TOOLS` is the frozenset `{read}` — only successful `read`
calls record observed evidence:

```python
GROUNDING_TRACKED_TOOLS: frozenset[str] = frozenset({LEAN_TOOL_READ})
```

A downstream grounding/citation layer treats every recorded `read` path as
observed evidence, so the model's own minimal reads become its citation set.
`read_silent` returns identical content but is explicitly **not** recorded.
Exposing this split as a core contract means any backend implements the same
read-vs-browse discipline. A host adapter can assert its (non-)recording
behaviour against this frozenset.

### Profiles and backend binding

Core owns only the *contracts and names*; it registers no concrete `Tool` for
the lean pool. Helper functions build the canonical specs and validate the
active profile:

- `lean_tool_surface()` → the full pool as `ToolDefinition` specs, in
  `LEAN_TOOL_NAMES` order (stable for cache reuse and deterministic tests).
- `lean_tool_definition(name)` / `is_lean_tool(name)` — per-verb spec / membership.
- `select_tool_surface_profile(profile)` — normalise the active profile to one
  of `TOOL_SURFACE_PROFILES` (`"legacy"` | `"lean"`), falling back to the
  behaviour-preserving `"legacy"` default for an unrecognised value.

Whether a tenant sees the lean pool or a backend's pre-existing typed surface is
governed by `RuntimeConstants.tool_surface_profile` plus per-tool
`tool_surface_*_enabled` flags. The host binds each canonical name to a
concrete backend. See [`runtime-constants.md`](runtime-constants.md) for how
these flags are read.

---

## The `@tool` decorator

`tools/decorator.py` provides a lightweight in-core helper to turn an async
function into a `Tool` subclass. A Pydantic `TypeAdapter` derives the parameter
JSON Schema from the function's type hints; the special `context: ToolContext`
parameter is skipped. The decorated callable is **replaced** with the generated
`Tool` subclass.

`tool`, `ToolContext`, and `ToolResult` are re-exported at the package top
level:

```python
from protocore import tool, ToolContext, ToolResult


@tool(name="echo", description="Echo back the input.")
async def echo(context: ToolContext, text: str) -> ToolResult:
    return ToolResult(tool_call_id="...", content=text)
```

The wrapped function must be `async` — `@tool` raises `TypeError` otherwise.
`echo` is now a `Tool` subclass whose `.definition` carries `name`,
`description`, and the schema built from the `text: str` hint. The
backend-backed *lean default* tools use this decorator but live in 
the host package; the core itself registers no lean-default tool (though it
does ship the ask-user / memory protocol-surface tools). For richer
or stateful tools, implement the `Tool` ABC (`contracts/tools.py`) directly.

---

## Dispatch pipeline

`runtime/tool_dispatch.py` — `ToolDispatcher.dispatch(...)` is the **single core
entry point** for executing one `ToolCall` after the LLM has finished emitting
it. The dispatcher is agnostic to the tool implementation: it depends only on
`Tool.invoke`. It **never raises** on a tool error — every failure mode is
translated into a `tool_result(success=false)` block so the model can recover on
the next turn and the run never hard-crashes.

`dispatch()` is an async generator that yields `TurnEvent` envelopes and, as its
final item, a `DispatchOutcome`. The lifecycle:

1. **Registry lookup** — `DispatchErrorKind.unknown_tool` if the name is not
   registered.
2. **Schema validation** — the byte-cap / JSON-serialisability invariant on the
   input dict (tool-specific Pydantic validation lives in the host adapter).
3. **Permission gate** — fans out to `ToolPermissionGate.check(...)` (below). A
   `require_approval` verdict surfaces a `tool_call_pending` event and emits no
   tool result; the caller transitions the loop to `AWAITING`.
4. **Preconditions** — the tool's `ToolDefinition.preconditions` DAG is checked
   (below); an unsatisfied tool short-circuits to a failure when
   `RuntimeConstants.tool_preconditions_enabled` is set.
5. **Execute** — `tool.invoke(ctx)` wrapped in `asyncio.wait_for` honouring
   `rc.tool_timeout_seconds`.
6. **`PostToolUse` hook** — fire-and-await; may rewrite the output.

`DispatchOutcome` (frozen) carries `success`, `content`, `is_error`,
`error_kind`, `approval_required` / `approval_token`, `ask_user_required` /
`ask_user_payload`, `duration_ms`, and a `metadata` bag. `DispatchErrorKind` is
the failure taxonomy: `validation | permission | execution | timeout |
rate_limit | unknown_tool | consecutive_error_cap`. The last is a guard — once
the per-run consecutive-identical-error streak exceeds
`tool_dispatch_consecutive_error_cap`, the dispatcher rewrites the failure into
this kind so a stuck loop terminates instead of burning iterations.

A tool may attach a machine-readable `structured_error` dict (e.g.
`{"finalization_recommended": True, "reason": ...}`) to a raised exception; the
dispatch except-branch forwards it on `DispatchOutcome.metadata` and the loop
surfaces a finalize hint to the model.

See the [Tool dispatch + gating](architecture.md#tool-dispatch--gating) section
for the event-emission contract.

---

## Permission gate

`runtime/tool_permission.py` — `ToolPermissionGate.check(...)` runs an async
pipeline and returns the **first non-allow** `ToolPermissionDecision`. The
decision's `outcome` is one of `allow` / `deny` / `require_approval`, with an
optional rewritten `modified_input` and `approval_token`. Each decision also
records the `PermissionStage` at which it was reached (for telemetry).

The `PermissionStage` StrEnum names the gate's four ordered stages, plus a
no-op default:

| Order | Stage (`PermissionStage`) | What it checks |
|---|---|---|
| 1 | `whitelist` | The `ToolVisibilityPolicy` (and any subagent narrowing whitelist) must permit the tool name — `blocked` always denies, `visible` (when non-empty) is a strict allow-list. |
| 2 | `safety_policy` | Per-side-effect-class checks via the `IToolSafetyPolicy` chain. |
| 3 | `rate_limit` | Host-only; the baseline is a no-op `allow`. Compose a Redis-backed bucket policy via `register_policy` to deny here. |
| 4 | `hook` | The `PreToolUse` hook — the final, highest-leverage stage; it can flip `allow` → `deny` / `require_approval` / modify the args. Skipped when no hook manager is wired. |
| — | `default` | The decision's StrEnum default value, used for the implicit `allow` when no stage objected. |

A safety policy implements `IToolSafetyPolicy` — `applies_to(side_effect_class)`
plus `evaluate(tool, arguments, ctx)`. The default policy stack contains exactly
one policy:

- `ShellSafetyPolicyAdapter` — wraps `DefaultShellSafetyPolicy`, inspecting
  `arguments['command']` for the sandbox side-effect class.

`HttpDnsAllowlistPolicy` and `WorkspacePathPolicy` are provided but **not** in
the default stack; the host stacks them on at runtime via `register_policy`,
which keeps the core API frozen. The gate is always on. The `PreToolUse` hook is
the highest-leverage seam for an LLM-as-policy gate.

See the [Permission gate](architecture.md#permission-gate) section for the
gate's place in the dispatch flow and the side-effect class map.

---

## The 3-layer effective surface

`runtime/tool_registry.py` — `ToolRegistry` implements the `IToolRegistry`
contract (`contracts/tool_registry.py`). Its `compute_effective_surface(...)` is
the per-turn filter that keeps the LLM's tool list small and relevant while
preserving a byte-deterministic ordering. It is called by the loop each turn and
applies three layers:

1. **Policy** — apply the `ToolVisibilityPolicy` (`visible` allow-list /
   `blocked` deny-list / `pinned` always-include), yielding the tenant's
   visible set.
2. **Clipping** — if `top_k is None` or the visible set is already `<= top_k`,
   return it sorted by name (no retrieval at all).
3. **Progressive discovery** — otherwise BM25-rank the visible set by the recent
   user `query` (`runtime/tool_retrieval.py`), always include the `pinned`
   tools, and keep the top-K by score.

```python
def compute_effective_surface(
    self,
    tenant_id: str,
    policy: ToolVisibilityPolicy,
    *,
    query: str = "",
    top_k: int | None = None,
) -> Sequence[ToolDefinition]: ...
```

Whichever layer runs, the **final ordering is always name-ascending** — the
retrieval order drives *selection*, but the emitted list is sorted by name so
the LLM context stays byte-stable and the KV-prefix cache survives across turns.
The clip threshold is the RC `tool_retrieval_top_k` passed by the loop.

> `runtime/tool_pool.py` (`assemble_tool_pool`) is a parallel assembler that is
> **not** wired into the loop; the loop uses `compute_effective_surface`. It has
> callers only in tests and a re-export.

See the
[Tool retrieval / pool / registry / 3-layer surface](architecture.md#tool-retrieval--pool--registry--3-layer-surface)
section for the retrieval internals.

---

## Tool preconditions

Preconditions enforce tool **ordering**: a tool that requires a prior
observation is masked until its precondition is satisfied, so the model cannot,
for example, mutate a record before reading the governing policy.

There are three systems; they never interact:

- **Runtime DAG** — `runtime/tool_preconditions.py` (`check_preconditions`,
  `resolve_precondition`, `record_satisfaction`, `compute_masked_tools`,
  `load_satisfied_set` / `store_satisfied_set`). A tool's `preconditions` are
  read from its `ToolDefinition`; the satisfied set is round-tripped through the
  engine helper-bag so it survives snapshot/resume. This layer is consumed in
  the dispatch path (step 4 above) and is gated by
  `RuntimeConstants.tool_preconditions_enabled` (default `False`). It
  **blocks** a tool the model chose.
- **Declarative spec** — `contracts/tool_action_preconditions.py` holds the
  typed, string-payload rule spec: `ToolActionPreconditionSpec` /
  `ToolActionPreconditionRule` / `ToolActionPreconditionPredicate` /
  `ToolActionPreconditionResult`, with the predicate kinds
  `PREDICATE_KIND_ARGS_MATCH`, `PREDICATE_KIND_REF_OBSERVED`, and
  `PREDICATE_KIND_DOC_OBSERVED` (collected in `ALL_PREDICATE_KINDS`). This
  targets *argument-pattern + observed-state* preconditions (e.g. a mutating
  binary may only run after a specific document was observed in this run). Rule
  `kind` strings are forward-compatible: an unknown kind is a runtime no-op
  rather than a snapshot-rejecting validation error. Wired only as RC types
  plus a host evaluator — no core loop code reads the spec directly.
  Mode: `tool_action_preconditions_mode` (`off | shadow | block`, default
  `off`).
- **Run-level forcer** — `runtime/run_tool_preconditions.py` plus
  `QueryEngineConfig.tool_preconditions`. An ordered tuple of tools this run
  **must** call before the agent is free to answer; while an entry is
  outstanding the loop sets `LLMRequest.extra['forced_tool_choice']`. Empty
  (the default) is a no-op. This **forces** a tool the model did not choose.

See the [Tool preconditions](architecture.md#tool-preconditions-dag--action-spec--run-level-forcer)
section for the precondition mechanism.

---

## Extending the tool surface

- **Add a tool:** implement the `Tool` ABC (`contracts/tools.py`) or use `@tool`;
  register the resulting `ToolDefinition` with the `IToolRegistry`. For a lean
  agent-facing surface, adopt the canonical names from `LEAN_TOOL_NAMES` for
  cross-backend uniformity.
- **Control visibility:** set `visible` / `blocked` / `pinned` on a
  `ToolVisibilityPolicy`.
- **Add a safety check:** implement `IToolSafetyPolicy` and register it with
  `ToolPermissionGate.register_policy` — it evaluates after the defaults; the
  core gate stays frozen.
- **Gate by observed state:** ship a `ToolActionPreconditionSpec` via the
  precondition RCs (a host evaluator runs it).

See [`extending.md`](extending.md) for the broader "pick your seam" guide and
the import-boundary rule.
