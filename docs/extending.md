# Extending the Core

> Audience: an engineer adding capability to the **pure core** (`protocore/`) or
> wiring a new backend behind it. Scope: the core library (`protocore/`).

The core is a set of `Protocol`s and an immutable ReAct runtime. You extend it
**from the outside** — by implementing a contract, registering a hook, flipping a
`RuntimeConstants` toggle, or injecting a system-prompt section — **never by
editing the loop**. This page is a decision guide for picking the right seam,
followed by the two hard rules that bound every extension.

For the full per-subsystem detail behind each seam, see
[`architecture.md`](./architecture.md) — the
[extension-point table](./architecture.md#extension-points-the-protocols-host-implements),
the [conventions](./architecture.md#conventions), and the per-section
"Extension protocol" notes.

## Pick your seam

There are four seams. They are not interchangeable; each answers a different
question.

| You want to… | Use | Where |
|---|---|---|
| Provide a concrete backend the core only knows as an interface (an LLM, a store, a transport, memory, workspace, skills, …) | **A `Protocol` adapter** | a module in `contracts/`, implemented by the host |
| Observe, deny, or mutate behaviour at a lifecycle point (pre/post tool, prompt submit, session/compaction boundaries) without changing the loop | **A hook** | `contracts/hooks.py` (`IHookManager`) or the pluggy `hooks/` path |
| Turn a capability on/off or tune a numeric/string value per tenant | **An `RuntimeConstants` toggle** | `contracts/runtime_constants.py` |
| Add static guidance/persona/orientation text to the system prompt | **A `system_prompt_section`** | `QueryEngineConfig.system_prompt_sections` |

Decision order, fastest first:

1. **Is it just a value or an on/off switch?** → RC toggle. No new code path.
2. **Is it static text the model should always see?** → `system_prompt_section`.
3. **Is it a reaction at a lifecycle point (deny/modify/observe)?** → hook.
4. **Is it a whole new backend / capability behind an interface?** → `Protocol`
   adapter (in the host).

If none of these fit, you are probably about to modify the loop — **stop** and
re-read [the hard rules](#hard-rules-do-not-cross-these).

---

### Implement a `Protocol` (the host adapter)

**Use when** the core needs a capability it deliberately does not own: talking to
an LLM, persisting runs/sessions/blobs, searching, storing memory or workspace
units, dispatching subagents, transporting tool calls, rendering prompt
templates. The core declares the *shape* as a `Protocol`; the concrete
implementation lives in the host (or any other consumer) and is
**injected** across the boundary.

There is **no single `protocols.py`** — each interface is in its own module under
`contracts/`. The principal interfaces (one row per seam, full list in the
[extension-point table](./architecture.md#extension-points-the-protocols-host-implements)):

| Protocol | Module |
|---|---|
| `ILLMProvider` / `IProviderChain` | `contracts/llm.py` |
| `RuntimeConstantsProvider` | `contracts/runtime_constants.py` |
| `ISessionStore` / `IRunStore` | `contracts/session.py` / `contracts/run.py` |
| `IBlobStore` / `ISearchIndex` / `ITodoStorage` | `contracts/blob.py` / `contracts/search.py` / `contracts/todo.py` |
| `IToolRegistry` / `Tool` (ABC) | `contracts/tool_registry.py` / `contracts/tools.py` |
| `IToolTransport` | `contracts/resilience.py` |
| `IMemory` / `IWorkspace` | `contracts/memory.py` / `contracts/workspace.py` |
| `ISkillStore` | `contracts/skills.py` |
| `IHookManager` | `contracts/hooks.py` |
| `IEventStream` | `contracts/events.py` |
| `IAgentDispatch` | `contracts/agent_dispatch.py` |
| `IPromptTemplateProvider` | `contracts/prompts.py` |
| `IToolSafetyPolicy` | `runtime/tool_permission.py` |

**Tools are a special case.** A concrete tool is the one piece you write *as*
code, but it is still registered through the `IToolRegistry`, not wired into the
loop. Implement the `Tool` ABC (`contracts/tools.py`) directly, or use the
`@tool` decorator (`tools/decorator.py`) for an async function:

```python
from typing import Any

from protocore import tool, ToolContext, ToolResult
from protocore import IToolRegistry  # the registry interface


@tool(name="echo", description="Echo back the input text.")
async def echo(context: ToolContext, text: str) -> ToolResult:
    # `context` (tenant_id / run_id / session_id / metadata) is injected by the
    # loop and is read-only. Non-`context` params become the JSON-Schema args.
    return ToolResult(tool_call_id=context.run_id, content=text)


def install(registry: IToolRegistry) -> None:
    registry.register(echo())  # the decorator returns a Tool subclass
```

The decorator derives the `ToolDefinition` (name, description, JSON-Schema
parameters) from the signature and docstring; the special `context: ToolContext`
parameter is skipped when building the schema. Adopt the canonical agent-visible
verb names from the [lean tool surface](./tools.md) (`exec` / `read` /
`read_silent` / `write` / `find` / `search` / `answer`) for cross-backend
uniformity.

**Extra safety policies** are also a `Protocol` seam: implement
`IToolSafetyPolicy` and register it on the gate via `register_policy(...)`
(`runtime/tool_permission.py`) — the core gate API stays frozen while the policy
stack grows at runtime (this is how `HttpDnsAllowlistPolicy` and
`WorkspacePathPolicy` are added; they are **not** in the default stack).

**Rule of thumb:** if your code has a database driver, an HTTP client, a wire
format, or any I/O, it belongs in an adapter behind a `Protocol` — not in
`protocore/`.

---

### Use a hook

**Use when** you want to react at a lifecycle point — deny a tool call, rewrite
its arguments, observe an event, gate on an LLM-as-judge — **without** changing
the loop. Hooks are the highest-leverage extension point: the loop already calls
them at every boundary.

Three paths exist; they are not the same thing:

- **Production / cross-pod: implement `IHookManager`** (`contracts/hooks.py`).
  This is the contract the loop actually drives — `engine.hooks` is typed
  `IHookManager` and the loop calls a **3-arg** `invoke(event, payload,
  tenant_id) -> HookResult`. The host supplies this adapter. A hook returns a
  `HookResult` whose `action` is one of `HookActionKind.ALLOW` / `DENY` /
  `MODIFY` (`contracts/hooks.py`); for synchronous pre-hooks the first `DENY`
  short-circuits.
- **Typed published hooks: register handlers on `HookRegistry`**
  (`runtime/typed_hooks.py`). The production typed surface is
  `PUBLISHED_HOOKS` — `before_run`, `before_tool`, `after_tool`,
  `transform_context`, `before_compact`, `after_compact`. Gated by
  `typed_hooks_enabled` (default `False`); `correctness_bind.fire_typed_hook`
  dispatches from `_query_raw`. `before_run` / `transform_context` /
  `before_compact` / `after_compact` fire from that flag alone.
  `before_tool` / `after_tool` are nested inside the
  `intent_settlement_enabled` dispatch branch — they do not run if only the
  typed-hooks flag is on. `transform_context` is fired (and may yield
  `hook_fired`) but its `rewrite` outcome is **not** applied to history.
  The host re-exports `PUBLISHED_HOOKS` (the session-correctness route lists
  the published set). This is the seam for in-process, host-registered
  Python handlers — no tenant JS, no exec in the loop pod.
- **In-process pluggy: register `hookimpl`s** against `AgentHookSpecs`
  (`hooks/specs.py`), driven by the core `HookManager` (`hooks/manager.py`).
  `AgentHookSpecs` declares **8 hookspecs**: `pre_tool_use`, `post_tool_use`,
  `user_prompt_submit`, `session_start`, `session_end`, `pre_compact`,
  `post_compact`, `file_changed`. This pluggy manager is exported but is used
  mainly in tests — in production the `IHookManager` adapter drives the loop.

The single most useful hook is `pre_tool_use`: it is the permission gate's
**final stage** (gate order: `whitelist` → `safety_policy` → `rate_limit` →
`hook`, in `runtime/tool_permission.py`), the highest-leverage point where it can
flip an otherwise-allowed call to deny / require-approval or mutate its
arguments. The gate as a whole runs before preconditions and execution
(`runtime/tool_dispatch.py`), making the hook the natural seam for an
LLM-as-policy gate. See the
[Hooks](./architecture.md#hooks-pluggy--injection--scratchpad--context_bootstrap)
section for the full hook wiring and the `IHookManager`-vs-pluggy distinction.

---

### Add a `RuntimeConstants` toggle

**Use when** the change is a tunable value or an on/off switch — not new control
flow. Every tunable in the core flows through `RuntimeConstants`, a **frozen
Pydantic snapshot** with `model_config = ConfigDict(frozen=True, extra="forbid")`
(`contracts/runtime_constants.py`). **No inline magic numbers** — runtime code
reads from the snapshot, never a hard-coded literal.

New capabilities default **off** (or to a value that reproduces prior
behaviour), so a tenant opts in deliberately. Adding a tunable is the **3-edit
rule**:

1. a core Pydantic field on `RuntimeConstants` (default safe/off), and
2. the host `_FIELD_MAP` identity entry, and
3. the migration-catalog seed.

The Constants dashboard page then renders the toggle automatically. Because
`extra="forbid"` rejects unknown fields, **core and the host must deploy
paired** — a field added on one side without the other will reject the snapshot.
See [`runtime-constants.md`](./runtime-constants.md) for the full RC model and
the 3-edit rule.

If your "feature" is really "let a tenant turn X on/off" or "let a tenant set the
threshold for Y", you are done after the 3-edit rule — no other seam is needed.

---

### Inject a `system_prompt_section`

**Use when** you want to add static text to the model's system prompt — a
persona, domain guidance, an orientation block — that should always be present
for a run. These are passed at engine construction via
`QueryEngineConfig.system_prompt_sections` (a `tuple[str, ...]`,
`runtime/query_engine.py`); the loop assembles them into the system message each
turn.

This is the right seam for *prompt content*. It is **not** the right seam for
behaviour that should react to events (use a hook), for a value that should be
tenant-configurable (use an RC toggle), or for dynamic per-turn orientation read
from the environment (that is the `context_bootstrap_*` RC path, see the
[Hooks](./architecture.md#hooks-pluggy--injection--scratchpad--context_bootstrap)
section).

Two adjacent injection points on `QueryEngineConfig` are worth knowing — they are
**injected callables/observers**, not prompt text: `cache_observer`
(`CacheObserverProtocol`, prompt-cache hit-rate sink) and the two terminal-verify
trigger callables (`pre_terminal_self_verify_trigger`,
`pre_dispatch_terminal_verify_trigger`). All default `None` / unset, so the
machinery behind them is inert unless the host injects one *and* flips the
matching RC.

---

## Hard rules (do not cross these)

Two rules bound every extension. Violating either is a process failure, not a
design choice.

### Do not modify the loop structure

The ReAct loop (`runtime/query.py` + `runtime/query_engine.py` +
`runtime/loop_state.py`) is **immutable**. It is the single consumer of every
other subsystem and the basis for cross-pod snapshot/resume. **Do not edit its
structure.** Customise only via the four seams above:

- **hooks** (deny/modify/observe at lifecycle points),
- **`QueryEngineConfig`** injected callables/observers,
- **`RuntimeConstants`** toggles,
- **`system_prompt_sections`**.

If a change seems to require new branches in `query()`, that is a signal you have
picked the wrong seam — re-check the [decision order](#pick-your-seam). Recovery
behaviour, terminal classification, and finalization are all already
RC-gated; you toggle them, you do not rewrite them.

### Never import upward (the import boundary)

The core is the **root** of the dependency graph and must never import upward.
`protocore/` must never import any package whose name begins `protocore_` —
that underscore is what marks a sibling distribution sitting above the core:
adapters, a service layer, frontends, an execution backend, deployment tooling.

Add behaviour through **contracts / adapters / RC**, not by reaching upward. If
the core needs something from a higher layer, express that need as a new
`Protocol` and let the host inject the implementation.

This is enforced by the guard test
[`tests/test_core_import_boundary.py`](../tests/test_core_import_boundary.py).
It walks every `*.py` file under `protocore/`, parses the AST, and **fails CI**
if any top-level `import` / `from … import` references a forbidden package. Run
it (and the rest of the suite) before you push — see [`testing.md`](./testing.md).

---

## Related docs

- [`architecture.md`](./architecture.md) — full per-subsystem detail (the
  [extension-point table](./architecture.md#extension-points-the-protocols-host-implements),
  the [conventions](./architecture.md#conventions), per-section "Extension
  protocol" notes).
- [`contracts.md`](./contracts.md) — the interface surface you implement against.
- [`runtime-constants.md`](./runtime-constants.md) — the RC model and 3-edit rule.
- [`tools.md`](./tools.md) — the lean verb surface and the `@tool` decorator.
- [`testing.md`](./testing.md) — running the suite and the import-boundary guard.
