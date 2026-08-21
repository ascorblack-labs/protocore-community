# Protocore Core — Architecture

> Audience: an engineer onboarding to the **pure core** (`protocore/`).
> Scope: this document describes the current core library (`protocore/`) only.
> The host adapters, the FastAPI service, frontends, and
> deployment live in sibling repos and are referenced here only at the boundary.

---

## Overview

`protocore` is the **pure core** of the Protocore agent runtime. It is a
Python 3.12+ library of **contracts (protocols + typed models)** and a
**protocol-first ReAct runtime** that drives one agent turn at a time.

It is, by design, a **universal product core**, not a benchmark harness:

- **Zero upward imports.** Core never imports a package that sits above it —
  anything sharing its name with an underscore after it (`protocore_*`). It has
  no database driver, no HTTP endpoint, no orchestration logic. Everything
  outside-facing is a `Protocol` the host implements. Enforced by
  `tests/test_core_import_boundary.py`.
- **Universal / multi-tenant.** No per-task, per-tenant-id, per-prompt, or
  scorer/rubric-shaped logic in any executable path. Every method is
  tenant-scoped; tenant policy is injected (via `RuntimeConstants` and
  `ToolContext.metadata`), never hard-coded.
- **Everything `RuntimeConstants`-configurable and default-safe.** Tunable
  values flow through `RuntimeConstants` (a frozen Pydantic snapshot) or
  `constants.py` (memory-safety caps). New capabilities default **off** or to a
  value that reproduces prior behaviour, so a tenant opts in deliberately.
- **Horizontal-scale-safe.** No module-level dicts, no `asyncio` locks held as
  module state, no per-process authority. Durable and ephemeral cross-process
  state are both provided across the boundary. Correctness-affecting state
  lives per-run on the `QueryEngine` instance.

### Dependency direction

```
protocore (pure core, no upward imports)
  └─> a host distribution (adapters, service layer, HTTP API)
        ├─> frontends (HTTP/SSE only)
        └─> an execution backend (service API contracts only)
```

`protocore` is the root. It must never import upward. The guard test asserts
that importing any `protocore.*` module pulls in **zero** symbols from the
layers above it.

### Public API (`protocore/__init__.py`)

The public surface is **contract-first**: the re-exports are the 11 store /
service interface `Protocol`s plus the `Tool` ABC — `IAgentDispatch`,
`IBlobStore`, `IEventStream`, `IHookManager`, `ILLMProvider`, `IRunStore`,
`ISearchIndex`, `ISessionStore`, `ISkillStore`, `IToolRegistry`, `ITodoStorage`,
and `Tool`. (`IMemory`, `IWorkspace`, `IToolTransport`, and
`IPromptTemplateProvider` live in their contract modules — `contracts/memory.py`,
`contracts/workspace.py`, `contracts/resilience.py`, `contracts/prompts.py` — but
are **not** top-level re-exports.) The surface also re-exports the core type
system (`Message`, `ToolCall`, `ToolResult`, `Event`, `Run`, `Session`, the
`ContentBlock` union, …), `RuntimeConstants` + `RuntimeConstantsProvider`,
`EventBus`/`EventName`, the pluggy `HookManager`, `DefaultShellSafetyPolicy`, the
`@tool` decorator, the envelope/JSON utilities, and the token-counting helpers
(`LanguageProfile`, `chars_per_token`, `detect_profile`, `estimate_tokens`). It
does **not** re-export `derive_budgets`, `retrieve_tools`, or `bm25_score` — those
are imported directly from their runtime modules
(`runtime/context/budgets.py`, `runtime/tool_retrieval.py`).

The loop machinery (`runtime/query.py` + `runtime/query_engine.py`) is the heart
of the runtime. The loop entry points are imported directly from
`protocore.runtime.query` / `protocore.runtime.query_engine`; they are **not**
re-exported at the top level.

---

## Architecture diagrams

### Layered structure

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ CONTRACTS / PROTOCOLS  (protocore/contracts/)                                   │
│   types.py  (Message, ToolCall, ToolResult, ContentBlock union, Run, Session,   │
│              ExecutionReport, StopReason, AgentEnvelope, …)                      │
│   16 interface Protocols: llm.py ILLMProvider + IProviderChain · run.py          │
│     IRunStore · session.py ISessionStore · blob.py IBlobStore ·                  │
│     search.py ISearchIndex · todo.py ITodoStorage ·                              │
│     tool_registry.py IToolRegistry · skills.py ISkillStore ·                     │
│     agent_dispatch.py IAgentDispatch · events.py IEventStream ·                  │
│     hooks.py IHookManager · memory.py IMemory · workspace.py IWorkspace ·        │
│     resilience.py IToolTransport · prompts.py IPromptTemplateProvider            │
│   runtime_constants.py  RuntimeConstants (frozen, extra="forbid") + Provider     │
│   lean_tool_surface.py · references.py · terminal_answer_validation.py ·         │
│   attempt_ledger.py · tool_action_preconditions.py · observability.py ·          │
│   verification.py · tool_chunking.py                                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                     ▲ implemented by the host / consumed by runtime
┌──────────────────────────────────────────────────────────────────────────────┐
│ RUNTIME — ORCHESTRATION  (protocore/runtime/)                                   │
│                                                                                │
│   QueryEngine (query_engine.py) ── owns mutable per-run state:                  │
│        history · LoopState · CompactionState · TokenUsage ·                     │
│        open_intents · usage_rows · lanes · live_* · steer/follow-up queues ·    │
│        verification · recovery latches ·                                        │
│        snapshot()/resume_from_snapshot()  (any pod can resume)                  │
│   query(engine)  (query.py) ── sync entry: _reset_per_turn_state() then          │
│        returns an async iterator of TurnEvent (no turn-start/end snapshot)      │
│   run() ── appends, snapshots, iterates _query_raw (not query())                │
│   loop_strategies.py ── DirectStrategy | DeepStrategy (run_mode)                │
│   intent.py · usage_ledger.py · session_tree.py · lanes.py ·                    │
│   typed_hooks.py · telemetry.py · correctness_bind.py ·                         │
│   compact_checkpoint.py · live_control.py · run_work_budget.py                  │
│        LoopState (loop_state.py): PENDING→RUNNING→{AWAITING|COMPACTING}→         │
│                                   {COMPLETED|FAILED|CANCELLED}                   │
└──────────────────────────────────────────────────────────────────────────────┘
        │                 │                       │                    │
        ▼                 ▼                       ▼                    ▼
┌───────────────┐ ┌────────────────┐ ┌──────────────────────┐ ┌────────────────┐
│ TOOL SURFACE  │ │ TOOL DISPATCH  │ │ CONTEXT / COMPACTION  │ │ FINALIZATION   │
│ + RETRIEVAL   │ │ + GATING       │ │ context/manager.py    │ │ + GROUNDING    │
│ tool_registry │ │ tool_dispatch  │ │ context/budgets.py    │ │ finalization_  │
│ tool_retrieval│ │ ToolDispatcher │ │ context/compaction.py │ │   gate.py      │
│ tool_pool     │ │ tool_permission│ │ context/session_      │ │ finalization_  │
│ lean surface  │ │   Gate (4 stg) │ │   memory.py           │ │   contract.py  │
│ @tool decorat.│ │ tool_precondi- │ │ compact_checkpoint.py │ │ terminal_      │
│               │ │   tions (DAG)  │ │ token_counting.py     │ │   payload_norm │
│               │ │ run_tool_pre-  │ │ prompt_caching.py     │ │                │
│               │ │   conditions   │ │ json_utils strip-     │ │                │
│               │ │   (run forcer) │ │   thinking            │ │                │
└───────────────┘ └────────────────┘ └──────────────────────┘ └────────────────┘
        │                 │                       │                    │
        ▼                 ▼                       ▼                    ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ MEMORY       │ │ WORKSPACE    │ │ RESILIENCE   │ │ SKILLS       │ │ HOOKS/EVENTS │
│ contracts/   │ │ contracts/   │ │ contracts/   │ │ skill_index  │ │ events.py    │
│   memory.py  │ │  workspace.py│ │  resilience  │ │ contracts/   │ │ runtime/     │
│ tools/       │ │ read_dedup_  │ │ runtime/     │ │  skills.py   │ │  events/*    │
│   memory.py  │ │  cache.py    │ │  resilience  │ │  list_files/ │ │ runtime/llm/ │
│ (IMemory)    │ │ (IWorkspace) │ │ attempt_     │ │  load_file   │ │  delta_bridge│
│              │ │              │ │  ledger ·    │ │  (host API)  │ │ hooks/       │
│              │ │              │ │ adaptive_    │ │              │ │  manager,    │
│              │ │              │ │  safety_band │ │              │ │  specs +     │
│              │ │              │ │ run_work_    │ │              │ │ typed_hooks  │
│              │ │              │ │  budget      │ │              │ │  PUBLISHED_  │
│              │ │              │ │              │ │              │ │  HOOKS       │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
        │                 │                 │                 │                 │
        ▼                 ▼                 ▼                 ▼                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ SAFETY  (protocore/safety/)  shell.py DefaultShellSafetyPolicy + deny patterns │
│         + chain_parser.py (segment/substitution grammar)                       │
├──────────────────────────────────────────────────────────────────────────────┤
│ HOST-ADAPTER BOUNDARY  (lives in the host distribution — NOT core)            │
│   LiteLLM/OpenAI-compat ILLMProvider · PgMemoryStore · IWorkspace store ·      │
│   PostgresStateManager · sandbox-backed exec/file tools · ConnectRPC transport │
│   · IHookManager adapter · RuntimeConstantsProvider (Postgres + Redis cache)   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Data flow of one agent turn

`query(engine)` is a **sync** function: it calls `_reset_per_turn_state()` and
returns an async iterator. Each inner `yield` from `_query_raw` is a stop-check
checkpoint; the executor streams the emitted `TurnEvent`s out over SSE
(Redis pub/sub at the host layer). `query()` does **not** persist
turn-start or turn-end snapshots — `QueryEngine.run()` appends the user
message, snapshots, then iterates `_query_raw` (not `query()`).

```
                       ┌─────────────────────────────────────────────┐
 caller: async for evt │  query(engine)  — sync reset, then iterator  │
   in query(engine):   │  of TurnEvent (one already-prepared turn)    │
                       └─────────────────────────────────────────────┘
                                          │
   (1) STOP CHECK ──────────────────────►│  stop_requested? → synthesize missing
                                          │   tool_results → CANCELLED
        INTENT RECOVERY ─────────────────►│  resume_open_intents +
                                          │   mark_intent_recovery (correctness_bind)
        TYPED before_run ───────────────►│  fire_typed_hook → deny? → stop
        MANUAL /compact ────────────────►│  CompactCheckpoint (RC-gated, default off)
   (2) COMPACTION CHECK ─────────────────►│  needs_compaction()?  ── yes ──┐
                                          │                                 ▼
                                          │                 ┌──────────────────────────┐
                                          │                 │ _run_compaction          │
                                          │                 │  Tier 1: truncate/blob   │
                                          │                 │   big tool_results       │
                                          │                 │  Tier 2: summarise old   │
                                          │                 │   turns → snapshot       │
                                          │                 │  COMPACTING→RUNNING      │
                                          │                 └──────────────────────────┘
   (3) UserPromptSubmit HOOK ────────────►│  _safe_hook_invoke → deny? → FAILED
                                          │
   (4) BUILD CONTEXT ────────────────────►│  tools = registry.compute_effective_surface
                                          │     (policy → clip → BM25 retrieval)
                                          │  skill catalog (alpha Skill() lines) ◄── SKILLS
                                          │  context_manager.build_context(history,…)
                                          │     ◄── MEMORY auto-recall injected (the host)
                                          │     ◄── context_bootstrap env-docs (turn 1)
   (4b) LOOP STRATEGY ───────────────────►│  select_strategy(run_mode)
                                          │     DirectStrategy: no pre-action step
                                          │     DeepStrategy: forced Plan tool +
                                          │       one REASONING_STEP, then shared loop
                                          ▼
   (5-9) STREAM ONE ASSISTANT MESSAGE  _stream_one_assistant_message(engine, context)
   ┌──────────────────────────────────────────────────────────────────────────────────┐
   │  budget max_tokens  ← AdaptiveSafetyBand (drift margin) ◄── RESILIENCE             │
   │  full_messages = system sections + history                                         │
   │  full_messages = _repair_outbound_tool_pairing(...)  (UNCONDITIONAL)               │
   │  cache_breakpoints = apply_system_and_3(full_messages)  ◄── PROMPT CACHE           │
   │  request = LLMRequest(messages, tools, max_tokens, extra={cache_breakpoints})      │
   │  async for delta in _iter_with_idle_watchdog(engine.llm.stream_with_tools(req)):   │
   │      delta → TurnEvent   (ProviderDelta → content_block_* / tool_use_* / usage)    │
   │                          (delta_bridge.py translates provider deltas) ◄── EVENTS   │
   │      usage → record cache_read/creation + cache observer                           │
   └──────────────────────────────────────────────────────────────────────────────────┘
                                          │
              ┌───────────────────────────┴────────────────────────────┐
              ▼                                                          ▼
   pending_tool_calls?  ── yes ──┐                            no tool_calls (end_turn)
                                 ▼                                       │
   ┌───────────────────────────────────────────────┐    stop_requested re-check
   │ for each call: _dispatch_tool(engine, call)    │            │
   │   ┌─────────────────────────────────────────┐ │            ▼
   │   │ ToolDispatcher.dispatch:                 │ │   terminal-tool nudge / backstop?
   │   │ 1. registry lookup → unknown_tool?       │ │   guaranteed-terminal submit?
   │   │ 2. schema / JSON validation              │ │            │
   │   │ 3. ToolPermissionGate.check (4 stages):  │ │            ▼
   │   │    whitelist→policy→rate→hook (pre_tool) │ │   FINALIZATION GATE + GROUNDING:
   │   │ 4. preconditions DAG→masked? (post-gate) │ │    verify deliverables (stat) ·
   │   │ 5. execute tool.invoke(ctx)              │ │    terminal_answer_validation
   │   │ 6. post_tool_use hook                    │ │      (refs ⊆ reads, canonical)
   │   │ → DispatchOutcome (success/err/approval) │ │    terminal_payload_normalize
   │   │     ◄── read records grounding ref       │ │            │
   │   │     ◄── WORKSPACE read-dedup cache       │ │            ▼
   │   └─────────────────────────────────────────┘ │      MESSAGE_STOP → COMPLETED
   │  append tool_result Message to history         │ │
   │  snapshot after every tool_result append       │ │
   │  loop back to (5): next assistant message       │ │
   └────────────────────┬───────────────────────────┘ │
                        │  approval_required? → AWAITING (resume_approved_tool later)
                        ▼  ask_user? → AWAITING (resume on user answer)
                  (recurse) stream next assistant message
```

Where the subsystems hook in:

- **Memory** injects auto-recalled facts before the LLM call (step 4) and is
  read/written by the `read`/`write`/`recall`-style tools during dispatch.
- **Workspace** backs `read`/`write`/`find`/`search` during dispatch; the
  process-local **read-dedup cache** short-circuits a repeated `read` of the
  same path/content.
- **Grounding** records a citation ref whenever a grounding-tracked `read`
  fires; the **finalization gate / terminal-answer validation** consume that
  ref ledger when the terminal `answer` is produced.
- **Resilience** wraps the outbound LLM call budget (AdaptiveSafetyBand) and is
  available as the universal `IToolTransport` wrapper for tool/VM calls
  (the host binds it).
- **Hooks/events** fire at every lifecycle point (UserPromptSubmit, pre/post
  tool, pre/post compact) and every provider delta becomes a `TurnEvent`.
  When `typed_hooks_enabled` is on, `correctness_bind.fire_typed_hook` runs
  `before_run` / `transform_context` / `before_compact` / `after_compact`.
  `before_tool` / `after_tool` also require `intent_settlement_enabled`.
  `transform_context` is fired but its `rewrite` is not applied to history.
- **Intent settlement + usage ledger** (both default-off): when
  `intent_settlement_enabled` is on, **every** dispatched tool commits an
  `IntentRecord` (never-replay vs safe by `intent_never_replay_tools`);
  interrupted never-replay intents are marked on turn start. Usage rows for
  `inference` / `retry` / `compaction` / `abort` / `fail` go through
  `commit_usage` when `usage_ledger_enabled` is on; a **tool**-kind row is
  written only on the intent-settlement dispatch path.
- **Live control** holds steer / follow-up queues and live model/thinking
  overrides; **CompactCheckpoint** is the operator `/compact` path (not a
  third compaction tier).

---

## Technology inventory

One row per core technology. **Wired into loop?** = referenced by the core
runtime loop (`query.py` / `query_engine.py`); subsystems wired only by 
the host adapter are marked accordingly. **RC toggle(s) + default** records the
governing `RuntimeConstants` field(s) and their safe/off default.

| Technology | Core files | RC toggle(s) + default | Wired into loop? | Tested? |
|---|---|---|---|---|
| ReAct loop / orchestrator / query engine | `runtime/query.py`, `runtime/query_engine.py`, `runtime/loop_state.py`, `runtime/loop_strategies.py` | n/a (always on); `run_mode` = `"direct"`; recovery branches RC-gated | Yes | Yes |
| Lean tool surface | `contracts/lean_tool_surface.py`, `tools/decorator.py` | `tool_surface_profile` = `"legacy"` | Yes | Yes |
| Tool dispatch + gating | `runtime/tool_dispatch.py`, `runtime/tool_permission.py` | gate always on; consecutive-error cap RC | Yes | Yes |
| Tool retrieval / pool / registry | `runtime/tool_registry.py`, `runtime/tool_retrieval.py`, `runtime/tool_pool.py` | `tool_retrieval_top_k` (clip threshold) | Yes (registry/retrieval); `tool_pool` **no** | Yes |
| Tool preconditions (three systems) | `runtime/tool_preconditions.py`, `contracts/tool_action_preconditions.py`, `runtime/run_tool_preconditions.py` | `tool_preconditions_enabled` = `False`; `tool_action_preconditions_mode` = `"off"`; run-level `QueryEngineConfig.tool_preconditions` empty | DAG + run-level forcer: Yes; action **spec**: host-only | Yes |
| Universal resilience layer | `contracts/resilience.py`, `runtime/resilience.py` | `resilience_enabled` = `False`; `resilience_transport_max_attempts` = `1` | Ledger/band: Yes; transport wrapper: host-only | Yes |
| Run wind-down (soft stop) | `runtime/soft_stop.py` | `soft_stop_enabled` = `True`, `soft_stop_max_turns` = `3` | Yes | Yes |
| Attempt ledger + adaptive safety band | `contracts/attempt_ledger.py`, `runtime/adaptive_safety_band.py` | band wired via per-call output budget | Yes | Yes |
| Finalization gate + contract | `runtime/finalization_gate.py`, `runtime/finalization_contract.py` | `terminal_tool_nudge_enabled` (`False`), `finalize_prose_gate_enabled` | Yes | Yes |
| Terminal-answer validation + references/grounding | `contracts/terminal_answer_validation.py`, `contracts/references.py`, `runtime/terminal_payload_normalize.py` | `terminal_answer_validation_enabled`, `observed_ref_normalize_enabled`, normalize toggles (all `False`) | Yes | Yes |
| IMemory subsystem | `contracts/memory.py`, `tools/memory.py` | `memory_enabled` = `False`, `memory_auto_recall_enabled` = `False` | Host-wired (tools held by core contract) | Yes |
| IWorkspace + read-dedup cache | `contracts/workspace.py`, `runtime/read_dedup_cache.py` | `workspace_enabled` = `True` | **No** (host-wired) | Yes |
| Context management / two-tier compaction / session memory | `runtime/context/manager.py`, `runtime/context/compaction.py`, `runtime/context/budgets.py`, `runtime/context/session_memory.py`, `runtime/compact_checkpoint.py` | ratios in RC; `compaction_manual_enabled` = `False` | Compaction + `/compact`: Yes; session-memory fold: host-wired | Yes |
| Token counting / language profiles | `runtime/token_counting.py` | `chars_per_token_*` ratios in RC | Yes | Yes |
| Prompt caching | `runtime/prompt_caching.py` | `prompt_cache_wire_enabled` = `True` (kill-switch) | Yes (hints in core; wire translation the host) | Yes |
| Skills routing / surfacing | `runtime/skill_index.py`, `contracts/skills.py` | data-driven (empty store = no block); `skills_hot_reload_enabled` = `False` | Yes (`_ensure_run_skill_catalog`); `list_files`/`load_file` host-only | Yes |
| Hooks (pluggy) + typed hooks + injection / context_bootstrap | `hooks/manager.py`, `hooks/specs.py`, `runtime/typed_hooks.py`, `runtime/correctness_bind.py` | `judge_failure_mode`, `context_bootstrap_enabled` = `False`, `typed_hooks_enabled` = `False` | Core pluggy manager: exported but **the host `IHookManager` drives the loop**; typed `PUBLISHED_HOOKS` default-off; `before_tool`/`after_tool` also need `intent_settlement_enabled` | Yes |
| Events / observability / streaming | `events.py`, `runtime/events/*`, `runtime/llm/delta_bridge.py`, `runtime/telemetry.py` | `telemetry_spans_enabled` = `False` | Yes | Yes |
| Intent / usage ledger / session tree / lanes | `runtime/intent.py`, `runtime/usage_ledger.py`, `runtime/session_tree.py`, `runtime/lanes.py` | `intent_settlement_enabled`, `usage_ledger_enabled`, `session_tree_enabled`, `lanes_enabled` (all `False`) | Intent + ledger: Yes when on; tree/lanes: host-invoked | Yes |
| Live control + run work budget | `runtime/live_control.py`, `runtime/run_work_budget.py` | `steer_follow_up_enabled` = `False`; tree token/run caps | Yes | Yes |
| Safety (shell policy + chain parser) | `safety/shell.py`, `runtime/chain_parser.py` | policy stack via `register_policy` | Yes | Yes |
| RuntimeConstants system | `contracts/runtime_constants.py`, `runtime/runtime_constants.py`, `constants.py` | the system itself | Yes | Yes |

A cross-cutting fact: **most new capabilities are default-off** and have no
exercise on the default tenant, so their *enabled* paths are covered by unit
tests rather than live runs. That is intentional — they are opt-in product
capabilities.

---

## Per-technology sections

The per-subsystem tour below is the deep reference for each inventory row.
The ReAct loop, lean surface, dispatch, retrieval, preconditions, resilience,
attempt ledger, finalization, grounding, memory, workspace, compaction,
pairing repair, and RuntimeConstants sections are unchanged in behaviour
from the code they name; the next sections correct the skill-catalog
wiring and the default-off intent / ledger / tree / lanes / typed-hooks /
telemetry surfaces a new engineer would otherwise miss.

### ReAct loop / orchestrator / query engine / loop state

**What & why.** This is the heart of the runtime: a ReAct (reason→act→observe)
loop that runs **one assistant turn at a time** and yields streaming events. It
is split into state and behaviour so any pod can resume a run after another
crashes. The shared assistant loop is **not** a single immutable path:
`QueryEngineConfig.run_mode` selects `DirectStrategy` or `DeepStrategy` in
`runtime/loop_strategies.py` before that shared loop.

**Key classes/files.**

- `runtime/query_engine.py`
  - `QueryEngine` — one instance per active run. Owns the **mutable
    per-conversation state**: `history` (list of `Message`), the `LoopState`
    machine, `CompactionState`, `TokenUsage`, plus `open_intents`, `usage_rows`,
    `lanes`, live-control queues (`_steer_queue` / `_follow_up_queue`),
    live model/thinking overrides (`_live_model_name` /
    `_live_thinking_enabled` / `_live_reasoning_effort`), optional
    `verification` (`VerificationLifecycle`), and the recovery latches
    (terminal-only, guaranteed-terminal, self-verify, circuit-breaker,
    pending-reads, longfile, tool-precondition index, …). Persistence is
    `snapshot()` ↔ `resume_from_snapshot()`; `run()` snapshots at turn start
    and in `finally`. The snapshot also writes `open_intents`, `usage_rows`,
    `lanes`, the live `live_*` fields, the steer/follow-up queues,
    `verification` (when non-default), and those recovery latches.
  - `QueryEngineConfig` — the **immutable injection surface** bound at engine
    construction: `run_id`/`tenant_id`/`session_id`/`model_name`,
    `account_id` (account-wide skill bank key; empty if unresolved),
    `system_prompt_sections`, `tool_visibility_policy`, the `rc` snapshot,
    `run_mode` (`"direct"` | `"deep"`, default `"direct"`),
    `execution_profile`, `thinking_enabled` / `reasoning_effort`,
    `expected_terminal_tool`, `tool_preconditions` (the run-level forcer;
    empty default), the optional `cache_observer`, optional
    `verification_delivery`, and the two **host-supplied trigger
    callables** (`pre_terminal_self_verify_trigger`,
    `pre_dispatch_terminal_verify_trigger`) — both default `None`, so the
    pre-dispatch veto / self-verify machinery is dead unless the host injects
    a callable *and* flips the matching RC.
  - `QueryEngine.__init__` accepts an optional `provider_chain: IProviderChain`
    for mid-stream provider failover. `None` (every caller that configured no
    priority list) leaves existing recovery untouched.
  - `QueryEngine.run(initial_message)` is the **async-generator** driver: it
    appends the user message (or continues against an existing user-final
    history), increments `turn_count`, resets per-turn state, stamps the run
    clock, persists a turn-start snapshot, binds `_current_turn_task`, then
    iterates `_query_raw` (not `query()`). A turn-end snapshot lands in
    `finally`.
- `runtime/query.py` (11517 lines) — `query(engine)` is a **sync** function.
  It is deliberately not an async generator: it calls
  `_reset_per_turn_state()` at the call site and **returns**
  `_projected_turn_events`, which iterates `_query_raw` and applies the public
  delivery boundary. A turn driven through `query()` has no cross-pod resume
  point and does not bind `_current_turn_task`. `_query_raw` implements the
  turn lifecycle: stop check → resume interrupted intents → typed
  `before_run` → optional `/compact` via `CompactCheckpoint` → compaction
  check → UserPromptSubmit hook → build context → `select_strategy(run_mode).prepare_turn`
  → `_stream_one_assistant_message` (recursive on tool_use) → dispatch →
  finalize. Recovery is broader than the 413 / max-output / thinking-trap /
  empty-nudge / idle-watchdog set: the turn also resumes interrupted intents,
  fires typed `before_run`, handles `/compact` via `CompactCheckpoint`, and
  binds usage/hooks through `runtime/correctness_bind.py`
  (`commit_usage`, `fire_typed_hook`, `mark_intent_recovery`,
  `persist_correctness`). Those older recovery branches remain model-agnostic
  and RC-gated.
- `runtime/loop_strategies.py` — `select_strategy(run_mode)` is the single
  branch point. `DirectStrategy` contributes no pre-action step (the
  auto-tool loop). `DeepStrategy` runs a forced `Plan` tool (native
  `tool_choice` + CoT bounded by `reasoning_effort`), emits exactly one
  `REASONING_STEP` event, then the shared assistant loop drives the real
  action with the full surface.
- `runtime/loop_state.py` — `LoopState` is a pure 7-state machine:
  `PENDING → RUNNING → {AWAITING | COMPACTING} → {COMPLETED | FAILED |
  CANCELLED}`. `assert_transition()` enforces the legal-edge table;
  `TERMINAL_STATES` have no outgoing edges. **Distinct** from
  `RunStatus` (the durable PG-row mirror) and `RunState` (the hot Redis-hash
  record) — `LoopState` is the engine instance's in-flight state.

**How invoked/wired.** The host executor constructs a `QueryEngine` on
run admission, then typically `async for evt in engine.run(message)` per turn
(or `async for evt in query(engine)` after the caller has already seeded
history). Each `TurnEvent` is forwarded to the SSE bridge. The loop is the
single consumer of every other subsystem.

**RC configurability.** `max_turns_per_run`, `agent_max_seconds` (wall-clock
deadline; `<= 0` = inert), the idle/stall watchdog timeouts, and every recovery
toggle are RC fields. `model_name` is required (no baked-in default).
`agent_loop_default_mode` is the tenant default for `run_mode`.

**Extension protocol.** Do **not** edit the loop structure. Customise via (a)
hooks (including typed `PUBLISHED_HOOKS`), (b) `QueryEngineConfig` injected
callables/observers/`run_mode`/`tool_preconditions`/`provider_chain`, (c) RC
toggles, (d) `system_prompt_sections`.

**Terminal-classification notes.** Three terminal-classification behaviours are
worth calling out: (1) `query()` re-checks `stop_requested` after streaming and
routes a cancelled run to CANCELLED (not a clean end-turn); (2)
`_synthesize_missing_tool_results` is called at every teardown checkpoint so a
persisted snapshot is always pairing-valid (see
`_repair_outbound_tool_pairing` / `_synthesize_missing_tool_results` in the
inventory table); (3) the `max_turns` exit
is classified as a resource-**exhaustion** terminal — it keeps
`stop_reason=max_turns` on the wire and is treated as an error/non-success class,
not a clean `COMPLETED`.

### Lean tool surface through pairing repair

The lean 7-verb surface, dispatch + 4-stage permission gate, 3-layer
registry/retrieval, three non-interacting precondition systems, resilience
transport wrapper, attempt ledger + adaptive safety band, finalization gate,
terminal-answer validation / `normalize_ref`, IMemory, IWorkspace + read-dedup,
two-tier compaction + `CompactCheckpoint` + session-memory fold, token
counting, prompt-cache `system_and_3` hints, and `_repair_outbound_tool_pairing`
/ `_synthesize_missing_tool_results` are documented in the inventory table
above and implemented in the files that table names. They are not repeated
here. Continue at [Skills routing / surfacing](#skills-routing--surfacing).

### Skills routing / surfacing

**What & why.** Surfaces a small **catalog** of available skills into the
system prompt each turn, and loads a full skill body on demand when the user
references it — so domain capability can be added as data, never as per-task
prompt hints. The catalog is a compact, alphabetically ordered
`<system-reminder>` block, built once per run (cached on
`engine._skill_catalog_block`) and placed in the static prompt prefix so it
stays byte-stable across turns (preserving the prompt cache). It is **not**
BM25- or top-K-ranked.

**Key classes/files.** `runtime/skill_index.py` — `render_skills_catalog`
emits `SYSTEM_REMINDER_HEADER` ("Skills are tools, not files… call exactly
`Skill(skill="<name>")`") plus one `Skill(skill="{name}") — {description}`
line per enabled skill, alphabetical by name. Over the token budget the
block degrades to call-shapes only (`Skill(skill="{name}")`).
`derive_skill_index_budget_tokens` is `model_context_window ×
skill_index_budget_ratio` (default 1%). `contracts/skills.py` —
`ISkillStore`, `SkillIndexEntry`, `SkillBundle`, `SkillFileRef`,
`SKILL_ENTRY_PATH` (`SKILL.md`). `list_files` / `load_file` are required
protocol methods for multi-file bundles (at minimum the canonical `SKILL.md`
row; a legacy single-file skill may synthesise that row from `body_md`).
The **core loop never calls** `list_files` / `load_file` — it catalogs via
`list` + `list_enabled_subset` and loads a triggered body via `load` /
`list_subset`. Hosts that expose helper files use the file API themselves.

**How wired.** Step 4 calls `_ensure_run_skill_catalog(engine)`. Skill-store
reads key on `QueryEngineConfig.account_id` (the account-wide bank), **not**
`tenant_id`. When `engine.skills is None` or the store is empty → an
empty-string zero-cost block. Failures are isolated with a WARNING; the run
continues. Per-turn, `<command-name>NAME</command-name>` in the latest user
text loads the matching `SkillBundle.body` as a Layer-3 block, capped by
`max_skills_per_run` (default 4). Project pins (`pinned_skill_names`) are
merged through `list_enabled_subset` so a disabled skill stays off the
catalog.

**RC/extension.** Surfacing is data-driven (empty store = no block).
`skills_hot_reload_enabled` (default `False`) skips the per-run cache and
rebuilds the catalog on every `_ensure_run_skill_catalog` call. Implement
`ISkillStore`; there is no ranker to implement.

### Hooks (pluggy) + injection / scratchpad + context_bootstrap

**What & why.** Extensibility seam: deny/modify/observe at every lifecycle
point, without touching the loop. Plus an optional turn-1 **context bootstrap**
that reads the environment's own contract/readme docs and prepends a frozen
`<environment_context>` orientation message.

**Key classes/files.** `hooks/specs.py` — `AgentHookSpecs`: **8 pluggy
hookspecs** (`pre_tool_use`, `post_tool_use`, `user_prompt_submit`,
`session_start`, `session_end`, `pre_compact`, `post_compact`, `file_changed`).
`hooks/manager.py` — `HookManager` (the in-process pluggy registry + aggregator).
`contracts/hooks.py` — the cross-pod `IHookManager` contract, `HookResult`,
`HookActionKind`, `HookSpec`.
`runtime/typed_hooks.py` — `PUBLISHED_HOOKS` (`before_run`, `before_tool`,
`after_tool`, `transform_context`, `before_compact`, `after_compact`) plus
`HookRegistry` / `dispatch_hook`. The host re-exports this published set
(for example the session-correctness route lists `PUBLISHED_HOOKS`).
`runtime/correctness_bind.py` is the glue that fires typed hooks and commits
usage from `_query_raw`.

**How wired (important).** The **core pluggy `HookManager` is exported but does
not drive the loop.** The loop's `engine.hooks` is typed `IHookManager` and
calls a **3-arg** `invoke(event, payload, tenant_id)` — production hooks run via
the host `IHookManager` adapter. The pluggy manager is constructed mainly
in tests. Also note a contract gap: `HookEvent` enumerates 10 events but
`AgentHookSpecs` declares only 8 (no `subagent_start`/`subagent_stop`), so those
two can never fire through the core pluggy manager.

Typed hooks are a **second** production surface: when `typed_hooks_enabled` is
on and `engine.typed_hook_registry` is set, `fire_typed_hook` runs the
matching published handler. Default-off — no registry, no-op allow.
`before_run`, `transform_context`, `before_compact`, and `after_compact`
fire from that flag alone. `before_tool` and `after_tool` are nested inside
the `intent_settlement_enabled` dispatch branch — they do not run if only
the typed-hooks flag is on. `transform_context` is fired (and a `hook_fired`
event may be yielded) but its `rewrite` outcome is **not** applied to history.

**RC/extension.** `judge_failure_mode` (LLM-judge hook fail-open/closed),
`judge_timeout_ms`; `context_bootstrap_enabled` (default `False`),
`context_bootstrap_docs`, `context_bootstrap_tree_depth`;
`typed_hooks_enabled` (default `False`), `typed_hooks_timeout_ms`. Register an
`IHookManager` implementation (the host), pluggy `hookimpl`s, or handlers
on `HookRegistry` for the published typed names.

### Events / observability / streaming

**What & why.** Streaming is mandatory — every provider delta becomes a typed
`TurnEvent` forwarded immediately. Two distinct event surfaces:

- `events.py` — `EventBus` + `EventName` (~70 names): **in-process** typed
  pub/sub for sibling-handler signalling within a pod (used by HookManager,
  ContextManager, …). Distinct from the cross-pod `IEventStream` (Redis
  Streams) used for SSE reconnect/replay.
- `runtime/events/types.py` — `EventType`: the **per-turn streaming** taxonomy
  (Anthropic-aligned: `message_*`, `content_block_*`, `tool_use_*`,
  `tool_result`, `error`, plus Protocore extensions
  `sandbox_*`/`subagent_*`/`hook_fired`/`tool_call_pending`/`state_changed` and
  loop lifecycle `run_started`/`heartbeat`/`compaction_*`). Later additions
  include `reasoning_step` (Deep-mode plan), `intent_committed`,
  `usage_committed`, `session_forked`, `lane_locked`, `recovery_marked`,
  `compact_checkpoint`, steer/follow-up/queue events (`steer_queued`,
  `follow_up_queued`, `queue_update`), live-control
  `model_changed`/`thinking_changed`, and candidate-verification events
  (`candidate_ready`, `verification_started`, `verification_reported`,
  `repair_requested`, `release_decided`, `candidate_released`). Each value is
  the `event:` line surfaced to SSE clients.
- `runtime/events/envelope.py` — `TurnEvent` (the frozen wire envelope).
- `runtime/llm/delta_bridge.py` — translates a provider's stream into
  `ProviderDelta` → `TurnEvent` (`_normalise_finish_reason`, `is_block_end`,
  …).
- `contracts/observability.py` — `CacheObserverProtocol` (the optional
  prompt-cache hit-rate sink injected via `QueryEngineConfig.cache_observer`).

**How wired.** `query()` yields `TurnEvent`s throughout; the usage delta feeds
the cache observer. Tracing/observability sinks are injected across the
boundary.

### Safety (shell policy + chain parser + path isolation + approvals)

**What & why.** Validate model-composed shell commands before execution, with
capability-based deny/approval patterns, and isolate workspace paths.

**Key classes/files.**

- `safety/shell.py` — `DefaultShellSafetyPolicy` + `_DENY_PATTERNS`
  (destructive `rm -rf /`, SUID, base64/dd, ANSI-C `$'...'` and locale `$"..."`
  quoting, `$IFS`/`${...IFS}` word-split injection, …). Returns a
  `ShellPolicyDecision` (allow / deny / require-approval).
- `runtime/chain_parser.py` — `parse_chain(...)`: a small shell grammar that
  splits a command on `;`/`|`/`&&` into `CommandSegment`s and surfaces `$()` /
  backtick **substitution bodies** (collected even inside double quotes, not
  single) so per-segment deny patterns re-arm on substitution bodies.
- Path-isolation + approval policies live in `tool_permission.py`
  (`WorkspacePathPolicy`) and the approval flow is the gate's `require_approval`
  stage (loop → AWAITING → resume).

> Note: `DefaultShellSafetyPolicy` **fails open** on a non-match (no
> fail-closed/ambiguous escalation), and `HttpDnsAllowlistPolicy` /
> `WorkspacePathPolicy` are not in the default stack — the host must register
> them via `register_policy`.

### RuntimeConstants system

**What & why.** The single mechanism for tunable values — **no inline magic
numbers**. Every tunable is a field on a frozen Pydantic snapshot, default-safe,
and dashboard-configurable.

**Key classes/files.**

- `contracts/runtime_constants.py` (7423 lines) — `RuntimeConstants`
  (`model_config = ConfigDict(frozen=True, extra="forbid")`) and the
  `RuntimeConstantsProvider` Protocol (`async get(tenant_id) -> RuntimeConstants`).
  `extra="forbid"` means an unknown key is a validation error (rejected), not
  silently dropped, so **core and the host must deploy paired**. The snapshot
  includes the default-off surfaces `intent_settlement_enabled`,
  `usage_ledger_enabled`, `session_tree_enabled`, `lanes_enabled`,
  `typed_hooks_enabled`, `telemetry_spans_enabled` (and
  `compaction_manual_enabled`, `steer_follow_up_enabled`).
  `workspace_enabled` defaults to `True`.
- `runtime/runtime_constants.py` — `StaticRuntimeConstantsProvider` +
  `default_runtime_constants(**overrides)` (tests + the in-memory smoke runtime;
  production pods supply a Postgres-backed provider with a Redis cache).
- `constants.py` (~70 lines) — module-level memory-safety caps (`MAX_ARTIFACTS`,
  `MAX_TOOL_CALL_ARGUMENT_BYTES`, `PROTOCOL_VERSION`, `DEFAULT_MODEL`, …).

**The 3-edit rule.** Adding a tunable: (1) a core Pydantic field (default
safe/off) + (2) the host `_FIELD_MAP` identity entry + (3) the migration
catalog seed. The Constants dashboard page then renders a toggle for free.

### Intent, usage ledger, session tree, lanes, typed hooks, telemetry, live control, run work budget

Modules that sit beside the shared ReAct loop. Each is default-off unless
the matching RC field says otherwise.

- `runtime/intent.py` — `IntentRecord` / `commit_intent` / `settle_intent` /
  `resume_open_intents` / `replay_policy_for`. When
  `intent_settlement_enabled` is on, **every** dispatched tool call commits
  an `IntentRecord` with reserved result ids before `ToolDispatcher.dispatch`.
  `replay_policy_for` sets `replay="never"` when the tool name is in
  `intent_never_replay_tools` (default `Write,Edit,Bash,Finalize,AppendFile`);
  every other name is `"safe"`. A crash mid-flight: never-replay intents
  become `interrupted` (synthetic error, no replay); safe intents stay
  `open`. `should_skip_never_replay` short-circuits a resumed interrupted
  never-replay call. Snapshot field: `open_intents`.
- `runtime/usage_ledger.py` — append-only `UsageRow` list. When
  `usage_ledger_enabled` is on, `correctness_bind.commit_usage` appends a
  row. `_query_raw` records `inference` / `retry` / `compaction` / `abort`
  / `fail` independently of intent. A **tool**-kind row is appended only
  on the intent-settlement dispatch path (the same
  `if intent_settlement_enabled` block that settles the intent). A failed
  attempt plus its retry is two rows. Snapshot field: `usage_rows`.
- `runtime/session_tree.py` — `fork_session` / `clone_session` copy a path
  of history into a new `SessionBranch` without mutating the source. Gated
  by `session_tree_enabled`; clone requires a settled source;
  `session_tree_max_copy_messages` (default 500) caps the copy. **Host-
  invoked** — the loop does not call these helpers.
- `runtime/lanes.py` — named lanes over shared history. `ensure_main`
  makes `main` exist; extras take exclusive locks (`create_lane` /
  `acquire_lane` / `release_lane`). Gated by `lanes_enabled`;
  `lanes_max_per_session` (default 4) includes main. **Host-invoked**;
  `QueryEngine.lanes` is snapshot-persisted so a resume sees the same
  locks.
- `runtime/typed_hooks.py` — `PUBLISHED_HOOKS` + `HookRegistry` /
  `dispatch_hook`. See the
  [Hooks](#hooks-pluggy--injection--scratchpad--context_bootstrap) section
  for which names fire without a second flag.
- `runtime/telemetry.py` — low-cardinality spans (`run` / `turn` / `step` /
  `tool` / `compact` / `hook`). Gated by `telemetry_spans_enabled`. High-
  cardinality ids stay attributes; `is_prometheus_safe_label` refuses
  `session_id` / `lane_id` / `operation_id` / `run_id` as label keys.
  `mark_recovery` tags a span when an interrupted intent is resumed
  (`correctness_bind.mark_intent_recovery`). Spans live on `engine.spans`
  (in-process); they are **not** in the snapshot.
- `runtime/correctness_bind.py` — glue so intent, ledger, typed hooks, and
  recovery run inside `_query_raw` (`commit_usage`, `fire_typed_hook`,
  `mark_intent_recovery`, `persist_correctness`).
- `runtime/live_control.py` — steer / follow-up queues (`QueuedPrompt`,
  `enqueue`, `place_items`), live model/thinking overrides, and the settled
  helper. Gated by `steer_follow_up_enabled` (default `False`).
- `runtime/run_work_budget.py` — cumulative tree-lifetime budget
  (`max_subagent_runs_per_tree`, `max_total_tokens_per_tree`) for one root
  run and everything it spawns. Sibling of `SubagentTreeBudget` (permits),
  not an extension of it. Exhaustion refuses **delegation**, never the run
  itself.

---

## Extension points (the protocols the host implements)

The core is a set of `Protocol`s; the host provides the concrete adapters.
The full interface surface lives in `contracts/` (each in its own module — there
is **no single `protocols.py`**; the old monolithic file was split per domain).
The principal extension points:

| Protocol | Module | What the host provides |
|---|---|---|
| `ILLMProvider` | `contracts/llm.py` | LLM completions: `stream_with_tools`, `complete_structured`, `complete_text`, and `count_tokens`; a universal LiteLLM/OpenAI-compatible adapter (OpenRouter / vLLM / OpenAI). |
| `IProviderChain` | `contracts/llm.py` | Ordered remaining providers plus a one-way `advance()` cursor. Injected on `QueryEngine(..., provider_chain=...)` for mid-stream failover; `None` leaves existing recovery untouched. |
| `RuntimeConstantsProvider` | `contracts/runtime_constants.py` | Per-tenant `RuntimeConstants` backed by Postgres + Redis cache. |
| `ISessionStore` | `contracts/session.py` | Session/transcript persistence (Postgres). |
| `IRunStore` | `contracts/run.py` | Run record create/list/read (Postgres + Redis hot record). |
| `IToolRegistry` | `contracts/tool_registry.py` | The concrete `ToolRegistry` is in core; the host registers concrete `Tool`s + visibility policy. |
| `Tool` (ABC) / `@tool` | `contracts/tools.py`, `tools/decorator.py` | Concrete tool implementations (sandbox-backed exec/file tools, the lean verbs). |
| `IToolTransport` | `contracts/resilience.py` | The tool/VM transport (e.g. ConnectRPC) the resilience wrapper wraps; optional `rebuild()` hook. |
| `IMemory` | `contracts/memory.py` | `PgMemoryStore` (Postgres FTS/BM25, two-stage idempotent write, drift-guard) + an `IMemoryContentScanner`. |
| `IWorkspace` | `contracts/workspace.py` | Durable byte store + Postgres FTS/BM25 manifest, atomic write, per-scope GC. |
| `ISkillStore` | `contracts/skills.py` | Skill bundle storage + lookup **and** multi-file `list_files` / `load_file` (`SkillFileRef`). The core loop catalogs via `list` / `list_enabled_subset` and loads bodies via `load` / `list_subset`; `list_files` / `load_file` are the host file API. The catalog renderer lives in core, `runtime/skill_index.py`. |
| `IHookManager` | `contracts/hooks.py` | The production 3-arg hook dispatcher (this drives the loop, not the pluggy `HookManager`). Typed `PUBLISHED_HOOKS` are a separate default-off surface in `runtime/typed_hooks.py`. |
| `IEventStream` | `contracts/events.py` | Cross-pod durable event stream (Redis Streams) for SSE reconnect/replay. |
| `IBlobStore` | `contracts/blob.py` | Content-addressed blob storage (S3) used by Tier-1 compaction. |
| `ISearchIndex` | `contracts/search.py` | Generic lexical search index. |
| `ITodoStorage` | `contracts/todo.py` | Per-session todo persistence. |
| `IAgentDispatch` | `contracts/agent_dispatch.py` | Subagent dispatch/lookup. |
| `IPromptTemplateProvider` | `contracts/prompts.py` | System-prompt template rendering. |
| `IToolSafetyPolicy` | `runtime/tool_permission.py` | Extra permission policies (`HttpDnsAllowlistPolicy`, `WorkspacePathPolicy`) registered via `register_policy`. |
| Hook specs (pluggy) | `hooks/specs.py` | In-process `hookimpl`s for the 8 spec events (when using the pluggy path). |
| `CacheObserverProtocol` | `contracts/observability.py` | Prompt-cache hit-rate sink injected via `QueryEngineConfig.cache_observer`. |
| `WorkspaceStatProtocol` | `runtime/finalization_gate.py` | Stat-only workspace facade for the finalization gate. |
| Self-verify trigger callables | `runtime/query_engine.py` | `pre_terminal_self_verify_trigger` / `pre_dispatch_terminal_verify_trigger` on `QueryEngineConfig`. |

---

## Conventions

- **Import boundary.** Core never imports a package whose name begins
  `protocore_` — the sibling distributions that sit above it. Add
  behaviour via contracts / adapters / RC, not by importing upward. Guard:
  `tests/test_core_import_boundary.py`.
- **No inline magic numbers.** Every tunable is a `RuntimeConstants` field
  (frozen, `extra="forbid"`) or a `constants.py` cap. Runtime code reads from
  the RC snapshot, never a hard-coded literal. Adding one is the 3-edit rule
  (core field + the host `_FIELD_MAP` + migration seed).
- **Horizontal-scale-safe.** No module-level dicts, no module-held locks, no
  per-pod in-memory authority. Correctness-affecting state lives per-run on the
  `QueryEngine` (snapshot/resume); ephemeral cross-pod state is Redis, durable
  state is Postgres — both injected across the boundary. The token-bucket and
  adaptive band take injected locks/stores rather than module state.
- **Streaming is mandatory.** `query()` returns an async iterator; every
  provider delta is forwarded immediately as a `TurnEvent`. Do not buffer a
  whole turn before emitting.
- **No backward compatibility.** Dev-version project — break freely, delete
  dead code, no migration shims. (Hence `compaction_thresholds.py` was deleted
  outright once `budgets.py` subsumed it.)
- **Use `Message` models, never raw dicts.** All messages flow as the Pydantic
  `Message` / `ContentBlock` union from `contracts/types.py`.
- **Do not modify the loop structure.** Customise via hooks, `QueryEngineConfig`
  injection, RC toggles, or `system_prompt_sections`.
- **Production logging = WARNING.** Use `logger.warning(...)` for operationally
  significant events; reserve lower levels for local debugging.

Repo commands: `uv sync --extra dev`, `uv run pytest .`, `uv run ruff check .`,
`uv run mypy protocore`.
