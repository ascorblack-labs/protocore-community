# Glossary

Concise definitions of the core's key terms. Each entry maps to a real symbol in
the core and stays consistent with
[`architecture.md`](architecture.md). Terms are grouped by area, not
alphabetised, so related concepts read together.

## Runtime entry points

**`QueryEngine`** (`runtime/query_engine.py`)
: One instance per active run. Owns the **mutable per-conversation state** —
  `history`, the `LoopState` machine, `CompactionState`, `TokenUsage`, plus
  `open_intents`, `usage_rows`, `lanes`, live-control queues, live
  model/thinking overrides, optional `verification`, and recovery latches —
  and persists it via `snapshot()` ↔ `resume_from_snapshot()`, so any pod can
  resume a run another pod started. Construction-time injection lives on the
  immutable `QueryEngineConfig` (including `run_mode`, `tool_preconditions`,
  and optional `provider_chain`). See
  [ReAct loop / orchestrator / query engine / loop state](architecture.md#react-loop--orchestrator--query-engine--loop-state).

**`query()`** (`runtime/query.py`)
: The ReAct turn driver: `def query(engine) -> AsyncIterator[TurnEvent]`, a
  **sync** function that resets per-turn state and **returns** an async
  iterator. It is not an async generator and it does not persist turn-start
  or turn-end snapshots (`QueryEngine.run()` does). Each inner `yield` from
  `_query_raw` is a stop-check checkpoint. Not re-exported at the top level —
  import it from `protocore.runtime.query`. See
  [ReAct loop / orchestrator / query engine / loop state](architecture.md#react-loop--orchestrator--query-engine--loop-state).

## The three run-state concepts (do not conflate)

These three names are **distinct** and live in different layers; mixing them is a
common error.

**`LoopState`** (`runtime/loop_state.py`, `StrEnum`)
: The **in-turn loop finite-state machine** held on the `QueryEngine` instance —
  the engine's live in-flight state. Seven states:
  `PENDING → RUNNING → {AWAITING | COMPACTING} → {COMPLETED | FAILED | CANCELLED}`.
  `assert_transition()` enforces the legal-edge table; `TERMINAL_STATES` have no
  outgoing edges. See
  [ReAct loop / orchestrator / query engine / loop state](architecture.md#react-loop--orchestrator--query-engine--loop-state).

**`RunStatus`** (`contracts/types.py`, `StrEnum`)
: The **durable run lifecycle** mirrored on the Postgres `runs.status` column:
  `queued | running | completed | partial | error | cancelled | incomplete |
  paused`. `partial` is functionally terminal (loop finished but accumulated tool
  errors), distinct from `completed` and `error`. This is the persisted record,
  not the in-memory loop state.

**`RunState`** (`contracts/types.py`, `BaseModel`)
: The **ephemeral hot working set** held in the Redis hash `run:{id}` — `run_id`,
  `tenant_id`, the current `RunStatus`, `current_turn`, token counters,
  `last_event_id`. A mutable model, not an enum; distinct from the durable `Run`
  record.

## Configuration & constants

**`RuntimeConstants`** (`contracts/runtime_constants.py`)
: The single mechanism for tunable values — **no inline magic numbers**. A frozen
  Pydantic snapshot (`ConfigDict(frozen=True, extra="forbid")`); every tunable is
  a default-safe field, served per-tenant by a `RuntimeConstantsProvider`.
  `extra="forbid"` means an unknown key is a validation error (rejected), not
  silently dropped, so **core and the host must deploy paired**. See
  [RuntimeConstants system](architecture.md#runtimeconstants-system) and
  [`runtime-constants.md`](runtime-constants.md).

## Extension protocols

**`IMemory`** (`contracts/memory.py`, `Protocol`)
: The contract for a **typed, scope-aware, retrieval-ranked memory** of facts the
  agent learns and re-uses (distinct from session transcripts, blobs, the search
  index, and todos). A record is addressed by `(tenant_id, scope, scope_key)`;
  the most-isolated default scope is `session`. Core never imports the
  implementation; the host provides `PgMemoryStore`. Default-off
  (`memory_enabled = False`). See
  [IMemory](architecture.md#imemory-scoped-ftsbm25-idempotent-drift-guard-injection-scan-seam).

**`IWorkspace`** (`contracts/workspace.py`, `Protocol`)
: The contract for a **session/task-scoped, searchable, atomic scratch
  workspace** — the agent dumps intermediate data once and re-reads/searches it
  many times (a dump-once / re-read-many stability lever). Backs the
  `read`/`write`/`find`/`search` verbs. Host-wired; the availability
  flag defaults on (`workspace_enabled = True`). See
  [IWorkspace + read-dedup cache](architecture.md#iworkspace--read-dedup-cache).

## Resilience & finalization

**`AdaptiveSafetyBand`** (`runtime/adaptive_safety_band.py`)
: A per-`(provider, model)` band that **learns from token-estimator drift** and
  subtracts a calibrated margin from the per-call output budget, so
  `prompt + max_tokens` stays under the provider window even when the local
  estimator misjudges (e.g. Cyrillic-in-JSON-escape inflation). When no band is
  wired, behaviour is identical to pre-band. See
  [Attempt ledger + adaptive safety band](architecture.md#attempt-ledger--adaptive-safety-band).

**`AttemptLedger`** (`contracts/attempt_ledger.py`)
: A record of what a (sub)agent **declared** it would produce
  (`DeliverableDeclaration`) and what was **actually verified**
  (`VerificationRecord`), so finalization can decide an honest outcome. Its
  `LedgerOutcome` is a neutral literal (`completed | partial | failed | unknown`),
  not a backend enum; the agent's `SelfReportedStatus` is kept but not trusted
  blindly. See
  [Attempt ledger + adaptive safety band](architecture.md#attempt-ledger--adaptive-safety-band).

Finalization gate (`runtime/finalization_gate.py`,
`runtime/finalization_contract.py`)
: The terminal-path guard that closes a finalization gap: a (sub)agent that wrote
  the user-visible artifact but ran out of iterations without calling its
  terminal tool would otherwise be scored "failed". The gate **verifies**
  declared deliverables — `verify_declared_deliverables(...)` stats each one via
  the injected `WorkspaceStatProtocol` — and **decides** a `FinalizationDecision`
  via `decide_finalization(ledger)` (success / partial / failed template) that the
  leader's final turn uses. All toggles default `False`. See
  [Finalization gate + contract](architecture.md#finalization-gate--contract).

## Grounding & terminal answers

**Grounding / references**
: The deterministic, rubric-blind discipline that the terminal `answer`'s
  citations must be a **subset of what was actually `read`**. A grounding-tracked
  `read` records its path as observed evidence; `GROUNDING_TRACKED_TOOLS`
  (`contracts/lean_tool_surface.py`) is the frozenset `{read}` — `read_silent`
  returns identical content but is not recorded. `normalize_ref(...)`
  (`contracts/references.py`) is a pure, idempotent comparison projection that
  compares refs on a canonical form, so a flat-vs-branded path mismatch is not a
  false veto; it can only remove a false veto, never add one. See
  [Terminal-answer validation + references / grounding + payload normalize](architecture.md#terminal-answer-validation--references--grounding--payload-normalize).

## Context, caching & compaction

**Prompt-cache breakpoints** (`runtime/prompt_caching.py`)
: Placement **hints only** for provider prefix-caching. `apply_system_and_3(...)`
  computes the `system_and_3` strategy: at most four `CacheBreakpoint`s — system
  at index 0 plus the last three non-system messages. The core always emits the
  hints on `LLMRequest.extra["cache_breakpoints"]`; the host adapter
  translates them to `cache_control` markers (kill-switch
  `prompt_cache_wire_enabled`, default `True`), and adapters that don't recognise
  the key ignore it. See
  [Context management and two-tier compaction](architecture.md#context-management--two-tier-compaction--session-memory--budgets--token-counting--prompt-caching--strip-thinking).

**Compaction tiers / layers** (`runtime/context/compaction.py`)
: The **two-tier** cascade that keeps the prompt under the provider context
  window across a long run. There is no Tier 3 in this module. **Tier 1**
  (`run_tier1_truncation`) truncates / blobs oversized tool results, replacing
  the body with a placeholder + blob ref. **Tier 2**
  (`run_tier2_summarisation`) replaces whole old non-system turns with a system
  summary, keeping the recent N turns. When both are exhausted,
  `CompactionExhaustedError` transitions the loop to `FAILED`. Operator
  `/compact` is a separate `CompactCheckpoint` path
  (`runtime/compact_checkpoint.py`, `compaction_manual_enabled` default
  `False`). Cross-run fold lives in `runtime/context/session_memory.py`
  (`fold_run`). Triggers and ratios are RC-driven and derived in
  `runtime/context/budgets.py`. See
  [Context management and two-tier compaction](architecture.md#context-management--two-tier-compaction--session-memory--budgets--token-counting--prompt-caching--strip-thinking).

## Intent, usage ledger, session tree, lanes, typed hooks, telemetry

These six surfaces are **default-off**. Read the live `Field(...)` default; do
not infer that shipping the code turns them on.

**`IntentRecord`** (`runtime/intent.py`)
: Per-tool-call settlement record (`operation_id`, reserved result ids,
  `replay` `never|safe`, `status` `open|settled|interrupted`). When
  `intent_settlement_enabled` is on, **every** dispatched tool commits one
  before `ToolDispatcher.dispatch` — not only mutating tools.
  `replay_policy_for` marks names in `intent_never_replay_tools` (default
  `Write,Edit,Bash,Finalize,AppendFile`) as `never`; others are `safe`. A
  crash mid-never becomes `interrupted` and is not replayed. Persisted on
  `QueryEngine.open_intents`. See
  [Intent, usage ledger, session tree, lanes](architecture.md#intent-usage-ledger-session-tree-lanes-typed-hooks-telemetry-live-control-run-work-budget).

**`UsageRow`** (`runtime/usage_ledger.py`)
: One append-only ledger line (`seq`, `kind`, token counts, `success`,
  optional `operation_id`). When `usage_ledger_enabled` is on,
  `correctness_bind.commit_usage` records `inference` / `retry` / `compaction`
  / `abort` / `fail`. A **tool**-kind row is written only on the
  intent-settlement dispatch path. A failed attempt plus its retry is two
  rows. Persisted on `QueryEngine.usage_rows`.

**`SessionBranch`** (`runtime/session_tree.py`)
: A forked or cloned copy of a history path (`fork_session` / `clone_session`)
  that does **not** mutate the source. Gated by `session_tree_enabled`; clone
  requires a settled source; `session_tree_max_copy_messages` (default 500)
  caps the copy. **Host-invoked** — the loop does not call these helpers.

**`Lane`** (`runtime/lanes.py`)
: A named cursor over shared history. `main` always exists; extras take
  exclusive locks (`create_lane` / `acquire_lane` / `release_lane`). Gated by
  `lanes_enabled`; `lanes_max_per_session` (default 4) includes main.
  **Host-invoked**; `QueryEngine.lanes` is snapshot-persisted.

**`PUBLISHED_HOOKS`** (`runtime/typed_hooks.py`)
: The production typed hook names: `before_run`, `before_tool`, `after_tool`,
  `transform_context`, `before_compact`, `after_compact`. Dispatched through
  `HookRegistry` when `typed_hooks_enabled` is on. Distinct from the 8 pluggy
  hookspecs and from `IHookManager`. `before_tool` / `after_tool` also require
  `intent_settlement_enabled` (they live in that dispatch branch).
  `transform_context` is fired but its `rewrite` is not applied to history.

**`Span`** (`runtime/telemetry.py`)
: A low-cardinality telemetry span. Allowed names: `run` / `turn` / `step` /
  `tool` / `compact` / `hook`. Gated by `telemetry_spans_enabled`.
  `is_prometheus_safe_label` refuses `session_id` / `lane_id` / `operation_id`
  / `run_id` as label keys. `mark_recovery` tags a span when an interrupted
  intent is resumed. Lives on `engine.spans` (in-process, **not** snapshotted).

**`IProviderChain`** (`contracts/llm.py`)
: Ordered remaining providers plus a one-way `advance()` cursor. Injected on
  `QueryEngine(..., provider_chain=...)` so a mid-stream provider failure can
  step to the next rung without un-publishing already streamed deltas.

## Skill files

**`SkillFileRef`** (`contracts/skills.py`)
: One file in a multi-file skill bundle: bundle-relative `path`, `size_bytes`,
  `mime_type`, `content_hash` (lowercase hex SHA-256). Every bundle has at
  least the canonical `SKILL_ENTRY_PATH` row (`SKILL.md`, MIME
  `text/markdown`). Bytes come from `ISkillStore.load_file`; the core loop
  does **not** call `list_files` / `load_file` (those are for hosts that
  expose helper files). Catalog rendering uses `store.list` and emits
  `Skill(skill="{name}")` call shapes, not file paths. Store reads key on
  `QueryEngineConfig.account_id`, not `tenant_id`. See
  [Skills routing / surfacing](architecture.md#skills-routing--surfacing).

## Tool surface

**Lean tool surface (7 verbs)** (`contracts/lean_tool_surface.py`)
: A small, universal agent-facing tool pool so a capable model hand-composes
  operations instead of filling a dozen bespoke schemas. Exactly **seven
  canonical verbs** (`LEAN_TOOL_NAMES`): `exec`, `read`, `read_silent`, `write`,
  `find`, `search`, `answer`. `exec` is a registered-binary runner
  (`{path, args, stdin}`), explicitly **not** a `/bin/sh`. Core owns only the
  contracts and names; the host binds each name to a concrete backend. Profile
  selection is RC-driven (`tool_surface_profile`, default `"legacy"`). See
  [Lean tool surface](architecture.md#lean-tool-surface) and
  [`tools.md`](tools.md).
