# Contracts — the core boundary

> Audience: an engineer wiring a host application onto the
> pure core, or anyone who needs to know exactly where the core ends and the
> outside world begins. Scope: the core library `protocore/`.

`protocore` is a set of **contracts** (Python `Protocol`s + typed Pydantic
models) and a protocol-first ReAct runtime. Everything outside-facing is a
`Protocol` that a host implements; the core ships **no** database driver, HTTP
endpoint, or LLM client. This document is the catalogue of that boundary: the
interface protocols the host provides, the core type system that flows across
them, and the conventions that keep the surface stable.

For the deeper "how it fits together" view — the loop, the subsystems, the data
flow of one turn — read [`architecture.md`](architecture.md), the structural
source this page indexes.

---

## The contract package: no monolithic `protocols.py`

The interface surface lives in `protocore/contracts/`, **one module per
domain**. There is deliberately **no single `protocols.py`** — the old
monolithic file was split so each concern (LLM, runs, sessions, memory,
workspace, …) owns a self-contained module with its protocol, its typed models,
and its errors together.

```
protocore/contracts/
  types.py        # the core type system (Message, ContentBlock union, Run, …)
  llm.py          # ILLMProvider + IProviderChain + LLMRequest/LLMResponse + provider deltas
  run.py          # IRunStore
  session.py      # ISessionStore
  blob.py         # IBlobStore
  search.py       # ISearchIndex
  todo.py         # ITodoStorage
  tool_registry.py# IToolRegistry + ToolVisibilityPolicy
  tools.py        # Tool (ABC) + ToolContext + tool errors
  skills.py       # ISkillStore (+ SkillBundle / SkillIndexEntry / SkillFileRef)
  agent_dispatch.py# IAgentDispatch
  events.py       # IEventStream
  hooks.py        # IHookManager + HookResult / HookSpec / HookActionKind
  memory.py       # IMemory (+ IMemoryContentScanner)
  workspace.py    # IWorkspace
  resilience.py   # IToolTransport (+ the resilience taxonomy)
  prompts.py      # IPromptTemplateProvider
  observability.py# CacheObserverProtocol
  runtime_constants.py        # RuntimeConstants + RuntimeConstantsProvider
  lean_tool_surface.py        # the 7 canonical verb names
  references.py               # normalize_ref (grounding comparison)
  terminal_answer_validation.py
  attempt_ledger.py
  tool_action_preconditions.py
  verification.py             # candidate verification lifecycle (imported by QueryEngine)
  tool_chunking.py            # chunkable-content truncation recovery (imported by query())
```

### Contract-first re-export principle

Re-exports are managed through explicit `__all__` lists, and the boundary is
**contract-first**: the public surface leads with the interface protocols, then
the typed models that cross them. Two `__all__` lists matter:

- **`protocore/contracts/__init__.py`** — the contract surface, re-exported as
  the explicit `__all__` list (not every symbol the domain modules define). This
  is the list to import from when implementing an adapter. Some symbols defined
  in their contract modules are deliberately **not** in this `__all__` — e.g.
  `AttemptLedger` (`contracts/attempt_ledger.py`), `ToolActionPreconditionSpec`
  (`contracts/tool_action_preconditions.py`), `TerminalAnswerValidationSpec`
  (`contracts/terminal_answer_validation.py`), `LLMTimeoutError`
  (`contracts/llm.py`), and `SkillNotFoundError` (`contracts/skills.py`) — import
  those from their named module.
- **`protocore/__init__.py`** (the top-level package) — a **curated subset** of
  the same surface for the common case. It re-exports the 11 store/service
  interfaces (`I*`) plus `Tool`, the core type system, `RuntimeConstants`, and a
  handful of runtime utilities.

> **Heads-up:** `IMemory`, `IWorkspace`, `IToolTransport`,
> `IPromptTemplateProvider`, and `CacheObserverProtocol` are exported from
> `protocore.contracts` (and their named modules) but are **not** in the
> top-level `protocore` `__all__`. Import them from `protocore.contracts` (or
> the specific module) rather than the top-level package. `IProviderChain` and
> `SkillFileRef` are not in `protocore.contracts.__all__` either — import them
> from `protocore.contracts.llm` and `protocore.contracts.skills`.

The two loop entry points (`QueryEngine`, `query()`) are **not** re-exported at
either level — import them directly from `protocore.runtime.query_engine` /
`protocore.runtime.query`. See the
[Public API](architecture.md#public-api-protocore__init__py) section.

---

## Interface protocols (what the host provides)

These are the seams. Core declares the `Protocol` (or ABC); the host binds a
concrete adapter. Each is one line: the contract and what a host supplies.

| Protocol | Module | What the host provides |
|---|---|---|
| `ILLMProvider` | `contracts/llm.py` | LLM completions: `stream_with_tools`, `complete_structured` (post-loop JSON schema), `complete_text` (post-loop free-form document), and `count_tokens`; a universal LiteLLM/OpenAI-compatible adapter (OpenRouter / vLLM / OpenAI). |
| `IProviderChain` | `contracts/llm.py` | Ordered remaining providers plus a one-way `advance()` cursor. `QueryEngine` injects it as `provider_chain` for mid-stream failover; `None` leaves existing recovery untouched. Not in `protocore.contracts.__all__` — import from `protocore.contracts.llm`. |
| `RuntimeConstantsProvider` | `contracts/runtime_constants.py` | Per-tenant `RuntimeConstants` (`async get(tenant_id)`), Postgres-backed with a Redis cache. |
| `ISessionStore` | `contracts/session.py` | Session / transcript persistence. |
| `IRunStore` | `contracts/run.py` | Run record create / list / read (durable row + hot record). |
| `IToolRegistry` | `contracts/tool_registry.py` | The concrete `ToolRegistry` ships in core; the host registers concrete `Tool`s and a `ToolVisibilityPolicy`. |
| `Tool` (ABC) / `@tool` | `contracts/tools.py`, `tools/decorator.py` | Concrete tool implementations bound to the canonical verb names. |
| `IToolTransport` | `contracts/resilience.py` | The tool / VM transport the resilience wrapper wraps; optional `rebuild()` hook. |
| `IMemory` | `contracts/memory.py` | A scope-aware FTS/BM25 memory store + an `IMemoryContentScanner` for injection-scanning. |
| `IWorkspace` | `contracts/workspace.py` | A durable byte store + FTS/BM25 manifest, atomic write, per-scope GC. |
| `ISkillStore` | `contracts/skills.py` | Skill-bundle storage and lookup **and** the multi-file API: `list_files` / `load_file` returning `SkillFileRef` rows (at minimum the canonical `SKILL.md` / `SKILL_ENTRY_PATH` entry). The **core loop never calls** `list_files` / `load_file` — it catalogs via `list` / `list_enabled_subset` and loads a triggered body via `load` / `list_subset`. Store reads key on `QueryEngineConfig.account_id`, not `tenant_id`. The catalog is rendered by `render_skills_catalog` (+ `derive_skill_index_budget_tokens`) in `runtime/skill_index.py` as `Skill(skill="{name}")` lines, not file paths. |
| `IHookManager` | `contracts/hooks.py` | The production hook dispatcher (`invoke(event, payload, tenant_id)`) — this drives the loop, not the in-process pluggy `HookManager`. |
| `IEventStream` | `contracts/events.py` | Cross-pod durable event stream for SSE reconnect / replay. |
| `IBlobStore` | `contracts/blob.py` | Content-addressed blob storage used by Tier-1 compaction. |
| `ISearchIndex` | `contracts/search.py` | A generic lexical search index. |
| `ITodoStorage` | `contracts/todo.py` | Per-session todo persistence. |
| `IAgentDispatch` | `contracts/agent_dispatch.py` | Subagent dispatch / lookup. |
| `IPromptTemplateProvider` | `contracts/prompts.py` | System-prompt template rendering. |
| `CacheObserverProtocol` | `contracts/observability.py` | A prompt-cache hit-rate sink, injected via `QueryEngineConfig.cache_observer`. |

The **15 interface protocols** the core's public API leads with remain
`ILLMProvider`, `IRunStore`, `ISessionStore`, `IBlobStore`, `ISearchIndex`,
`ITodoStorage`, `IToolRegistry`, `ISkillStore`, `IAgentDispatch`,
`IEventStream`, `IHookManager`, plus `IMemory`, `IWorkspace`, `IToolTransport`,
and `IPromptTemplateProvider`. Three more protocols complete the seam set but
are not counted in that headline: `RuntimeConstantsProvider` (the per-tenant
config source), `IProviderChain` (mid-stream provider failover; not a
top-level or `contracts.__all__` re-export), and `IToolSafetyPolicy` (extra
permission policies registered at runtime; see the
[Permission gate](architecture.md#permission-gate) section).
`Tool` is an ABC, not a `Protocol` — the host subclasses it (or uses `@tool`).

> `IBlobStore` is declared as an ABC; the rest of the store/service interfaces
> are `Protocol`s. Either way the rule is the same — the core depends only on the
> declared shape and never on a concrete implementation.

---

## The core type system

Every conversation primitive flows as one of these Pydantic models (the
"use `Message` models, never raw dicts" convention). All live in
`contracts/types.py` unless noted. Most are frozen value objects.

### Messages & content

- **`Message`** — the sole conversation primitive (role-scoped). Assistant turns
  carry `content_blocks` (text + tool_use + thinking interleaved).
- **`MessageRole`** (`StrEnum`) — `system` · `user` · `assistant` · `tool`.
- **`ContentBlock`** — a **union type**, not a class:
  `TextBlock | ThinkingBlock | ImageRefBlock | ToolUseBlock | ToolResultBlock`.
- **`ContentBlockKind`** (`StrEnum`) — the discriminant: `text` · `thinking` ·
  `image_ref` · `tool_use` · `tool_result`.
- **`TextBlock`** / **`ThinkingBlock`** — plain text and model reasoning (the
  latter usually stripped before persistence).
- **`ToolUseBlock`** — an assistant-emitted tool invocation (`tool_call_id`,
  `name`, `arguments_json`; arg bytes capped).
- **`ToolResultBlock`** — a tool-call result returned to the model
  (`tool_call_id`, `content`, `is_error`, `metadata`).
- **`ImageRefBlock`** — an image reference whose bytes live in `IBlobStore`.

### Tool calls & results

- **`ToolCall`** — an LLM-emitted invocation surfaced to `Tool.invoke`
  (`id`, `name`, `arguments`). Carries truncation flags
  (`truncated_by_output_cap`, `args_partial_truncated`) the loop uses to detect
  a mid-stream-truncated argument JSON.
- **`ToolResult`** — the result of a single invocation
  (`tool_call_id`, `content`, `is_error`, `metadata`).
- **`ToolContext`** — the per-invocation context handed to a tool
  (tenant scope + metadata).
- **`ToolDefinition`** — the registry entry (name, description, params schema,
  approval flag, category) a `@tool` function or `Tool` subclass produces.
- **`ToolParameterSchema`** — the JSON-Schema shape of a tool's parameters.
- **`ToolError`** (and `ToolInvocationError`, `ToolPolicyDenied`,
  in `contracts/tools.py`) — the tool error hierarchy.

### Runs, sessions, events

Three distinct run shapes — do not conflate them (see
[`architecture.md`](architecture.md)):

- **`RunStatus`** (`StrEnum`) — the **durable** run lifecycle mirrored in the
  persistent `runs.status` column: `queued` · `running` · `completed` ·
  `partial` · `error` · `cancelled` · `incomplete` · `paused`. `partial` is a
  functionally-terminal status for a run that finished its loop but accumulated
  tool-dispatch errors.
- **`Run`** — the durable run record (`id`, `tenant_id`, `session_id`,
  `status`, timestamps, optional detail-blob ref).
- **`RunState`** — the **ephemeral** hot working set (held in a Redis hash by the
  host): `current_turn`, token counters, `last_event_id`. (Distinct again from
  `LoopState`, the in-flight engine FSM in `runtime/loop_state.py`, which is
  *not* a contract type.)
- **`Session`** — the multi-turn conversation root (durable, never deleted).
- **`Event`** — the in-flight event envelope (`run_id`, `name`, `payload`)
  emitted via `IEventStream` / the in-process `EventBus`.
- **`StopReason`** (`StrEnum`) — why a turn terminated: `end_turn` · `tool_use` ·
  `max_tokens` · `max_turns` · `stop_sequence` · `error` · `cancelled`.
- **`ExecutionReport`** — the bounded per-run telemetry rollup (events,
  tool-call records, LLM-call records, warnings, subagent runs, artifacts, and an
  optional `AttemptLedger` snapshot), with structural caps from
  `protocore.constants`.

### LLM request / response

- **`LLMRequest`** — the request the loop assembles for `ILLMProvider`
  (`messages`, `tools`, `max_tokens`, `extra` — including the
  `cache_breakpoints` prompt-cache hints).
- **`LLMResponse`** — a non-streaming response shape returned by
  `complete_structured` and `complete_text`.
- **`LLMObservabilityContext`** — the per-call observability context attached to
  an LLM request.
- **`IProviderChain`** — not a request/response type; the failover cursor
  `QueryEngine` rebinds onto `self.llm` when a mid-stream provider fails.

### Ingress, blobs, compaction

- **`AgentEnvelope`** — the single cross-component ingress contract
  (`kind`, `payload`, `metadata`; payload size capped). Parsed/serialised via
  `parse_envelope` / `serialize_envelope`.
- **`EnvelopeKind`** (`StrEnum`) — `task` · `control` · `result` · `error`.
- **`BlobMetadata`** — a blob index entry (`ref`, `content_type`, `size_bytes`,
  `sha256`).
- **`CompactionSourceRef`** — a pointer to a compacted tool-result blob, persisted
  as a wire-format placeholder during Tier-1 compaction.

### Verification & chunking

- **`VerificationLifecycle`** / **`VerificationDelivery`** /
  **`CandidateBundle`** / **`ReleaseDecision`** (`contracts/verification.py`) —
  the candidate-verification lifecycle `QueryEngine` snapshots as
  `verification` and uses to gate public reader delivery. Re-exported from
  `protocore.contracts`.
- **`is_chunkable_content_mutation`** (`contracts/tool_chunking.py`) — the
  single predicate for Write→AppendFile→FinalizeFile truncation recovery.
  Imported by `query()`. Not in `protocore.contracts.__all__` — import from
  the named module.

### Hooks

- **`HookEvent`** (`StrEnum`) — the **10** lifecycle events: `pre_tool_use` ·
  `post_tool_use` · `user_prompt_submit` · `session_start` · `session_end` ·
  `pre_compact` · `post_compact` · `file_changed` · `subagent_start` ·
  `subagent_stop`.
- **`HookResult`** — a hook's verdict (allow / deny / modify, via
  `HookActionKind`).
- **`HookSpec`** — the declarative spec for a registered hook.

### Skills, subagents, todos

- **`SkillManifest`** / **`SkillIndexEntry`** / **`SkillBundle`** /
  **`SkillFileRef`** — the skill catalogue shapes. `SkillFileRef` is the
  multi-file bundle index row (`path`, `size_bytes`, `mime_type`,
  `content_hash`); fetch bytes via `ISkillStore.load_file`. Every bundle has
  at least `SKILL_ENTRY_PATH` (`SKILL.md`). A legacy single-file skill may
  synthesise that one row from `body_md`. `SkillFileRef` is **not** in
  `protocore.contracts.__all__` — import it from `protocore.contracts.skills`.
  The loop's catalog is `Skill(skill="{name}")` call shapes, not these paths.
- **`SubagentDef`** / **`SubagentTask`** / **`SubagentResult`** — the subagent
  dispatch shapes used by `IAgentDispatch`.
- **`Todo`** / **`TodoStatus`** (`StrEnum`) — per-session todo persistence shape.

---

## Implementing an adapter

To bind the core to a host:

1. Implement the interface protocols you need from `protocore.contracts` (you do
   not need all of them — memory is default-off (`memory_enabled = False`);
   `workspace_enabled` defaults to `True` but the concrete workspace tools
   still live in the host).
2. Accept and return the core type-system models — never raw dicts at the
   boundary.
3. Inject configuration via `RuntimeConstants` (a frozen snapshot) and
   `ToolContext.metadata`; never hard-code tenant policy.
4. Construct a `QueryEngine` with your adapters and drive it with
   `async for evt in query(engine)` (`query()` is a sync function that
   returns that iterator) or `async for evt in engine.run(message)`.

The mechanics of extending the runtime — which seam to choose (protocol vs hook
vs RC toggle vs prompt section) and the hard "do not modify the loop structure"
rule — are covered in `extending.md` and [`architecture.md`](architecture.md).
The import boundary (core never imports a host) is enforced by
`tests/test_core_import_boundary.py`.
