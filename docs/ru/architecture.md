# Protocore Core — архитектура

> Аудитория: инженер, осваивающий **чистое ядро** (`protocore/`).
> Область: этот документ описывает только текущую библиотеку ядра (`protocore/`).
> Адаптеры хоста, сервис FastAPI, фронтенды и
> развёртывание живут в соседних репозиториях и упоминаются здесь только на границе.

---

## Обзор

`protocore` — это **чистое ядро** агентного рантайма Protocore. Это
библиотека на Python 3.12+, состоящая из **контрактов (протоколы + типизированные модели)** и
**protocol-first ReAct-рантайма**, который ведёт по одному ходу агента за раз.

По замыслу это **универсальное ядро продукта**, а не стенд для бенчмарков:

- **Ноль импортов вверх.** Ядро никогда не импортирует пакет, который стоит
  над ним, — ничего с его же именем и подчёркиванием после (`protocore_*`). У него
  нет драйвера базы данных, нет HTTP-эндпоинта, нет логики Kubernetes. Всё, что
  обращено наружу, — это `Protocol`, который реализует хост. Гарантируется
  тестом `tests/test_core_import_boundary.py`.
- **Универсальность / мультитенантность.** Ни в одном исполняемом пути нет логики,
  завязанной на отдельную задачу, tenant-id, промпт или
  скорер/рубрику. Каждый метод scoped по тенанту; политика тенанта инъецируется
  (через `RuntimeConstants` и `ToolContext.metadata`) и никогда не зашита в код.
- **Всё конфигурируемо через `RuntimeConstants` и безопасно по умолчанию.** Настраиваемые
  значения проходят через `RuntimeConstants` (замороженный Pydantic-снимок) или
  `constants.py` (лимиты безопасности по памяти). Новые возможности по умолчанию **выключены** или принимают
  значение, воспроизводящее прежнее поведение, так что тенант подключает их осознанно.
- **Безопасно при горизонтальном масштабировании.** Никаких словарей на уровне модуля, никаких блокировок `asyncio`,
  удерживаемых как состояние модуля, никакой авторитетности на уровне отдельного пода. Долговременное состояние — это Postgres;
  эфемерное межподовое состояние — это Redis (оба предоставляются через границу). Состояние, влияющее на
  корректность, живёт пер-ран на экземпляре `QueryEngine`.

### Направление зависимостей

```
protocore (чистое ядро, ноль импортов вверх)
  └─> хост-дистрибутив (адаптеры, сервисный слой, HTTP API)
        ├─> фронтенды (только HTTP/SSE)
        └─> бэкенд исполнения (только контракты сервисного API)
```

`protocore` — это корень. Он никогда не должен импортировать вверх. Сторожевой
тест утверждает, что импорт любого модуля `protocore.*` подтягивает **ноль**
символов из слоёв над ним.

### Public API (`protocore/__init__.py`)

Публичная поверхность **contract-first**: реэкспортируются 11 интерфейсных
`Protocol`-ов store / service плюс ABC `Tool` — `IAgentDispatch`,
`IBlobStore`, `IEventStream`, `IHookManager`, `ILLMProvider`, `IRunStore`,
`ISearchIndex`, `ISessionStore`, `ISkillStore`, `IToolRegistry`, `ITodoStorage`
и `Tool`. (`IMemory`, `IWorkspace`, `IToolTransport` и
`IPromptTemplateProvider` живут в своих контрактных модулях — `contracts/memory.py`,
`contracts/workspace.py`, `contracts/resilience.py`, `contracts/prompts.py`, — но
**не** реэкспортируются на верхнем уровне.) Поверхность также реэкспортирует основную систему
типов (`Message`, `ToolCall`, `ToolResult`, `Event`, `Run`, `Session`, объединение
`ContentBlock`, …), `RuntimeConstants` + `RuntimeConstantsProvider`,
`EventBus`/`EventName`, pluggy-`HookManager`, `DefaultShellSafetyPolicy`,
декоратор `@tool`, утилиты envelope/JSON и помощники подсчёта токенов
(`LanguageProfile`, `chars_per_token`, `detect_profile`, `estimate_tokens`). Она
**не** реэкспортирует `derive_budgets`, `retrieve_tools` или `bm25_score` — они
импортируются напрямую из своих рантайм-модулей
(`runtime/context/budgets.py`, `runtime/tool_retrieval.py`).

Машинерия цикла (`runtime/query.py` + `runtime/query_engine.py`) — это сердце
рантайма. Точки входа в цикл импортируются напрямую из
`protocore.runtime.query` / `protocore.runtime.query_engine`; они **не**
реэкспортируются на верхнем уровне.

---

## Архитектурные диаграммы

### Слоевая структура

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
│              │ │              │ │  ledger ·    │ │              │ │ hooks/       │
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

### Поток данных одного хода агента

`query(engine)` — **синхронная** функция: она вызывает `_reset_per_turn_state()`
и возвращает асинхронный итератор. Каждый внутренний `yield` из `_query_raw` —
контрольная точка stop-check; исполнитель транслирует выпущенные `TurnEvent`-ы
наружу по SSE (Redis pub/sub на уровне хоста). `query()` **не** сохраняет
снимки начала и конца хода — `QueryEngine.run()` добавляет пользовательское
сообщение, снимает snapshot, затем итерирует `_query_raw` (не `query()`).

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

Где подсистемы встраиваются:

- **Memory** инъецирует автоматически вспомненные факты перед вызовом LLM (шаг 4) и
  читается/пишется инструментами вида `read`/`write`/`recall` во время диспетчеризации.
- **Workspace** обслуживает `read`/`write`/`find`/`search` во время диспетчеризации; процесс-локальный
  **read-dedup cache** замыкает накоротко повторный `read`
  того же пути/содержимого.
- **Grounding** записывает ссылку-цитату всякий раз, когда срабатывает отслеживаемый для grounding `read`;
  **finalization gate / terminal-answer validation** потребляют этот
  реестр ссылок, когда формируется терминальный `answer`.
- **Resilience** оборачивает бюджет исходящего вызова LLM (AdaptiveSafetyBand) и
  доступен как универсальная обёртка `IToolTransport` для вызовов инструментов/VM
  (хост привязывает её).
- **Hooks/events** срабатывают в каждой точке жизненного цикла (UserPromptSubmit, pre/post
  tool, pre/post compact), и каждая дельта провайдера становится `TurnEvent`.
  Когда `typed_hooks_enabled` включён, `correctness_bind.fire_typed_hook`
  запускает `before_run` / `transform_context` / `before_compact` /
  `after_compact`. `before_tool` / `after_tool` также требуют
  `intent_settlement_enabled`. `transform_context` срабатывает, но его
  `rewrite` к истории не применяется.
- **Intent settlement + usage ledger** (оба выключены по умолчанию): когда
  `intent_settlement_enabled` включён, **каждый** диспетчеризуемый инструмент
  коммитит `IntentRecord` (never-replay vs safe по `intent_never_replay_tools`);
  прерванные never-replay намерения помечаются в начале хода. Usage-строки для
  `inference` / `retry` / `compaction` / `abort` / `fail` идут через
  `commit_usage`, когда `usage_ledger_enabled` включён; строка вида **tool**
  пишется только на пути intent-settlement.
- **Live control** держит очереди steer / follow-up и живые переопределения
  model/thinking; **CompactCheckpoint** — операторский путь `/compact` (не
  третий ярус компакции).

---

## Инвентаризация технологий

По одной строке на каждую технологию ядра. **Wired into loop?** = на технологию ссылается основной
рантайм-цикл (`query.py` / `query_engine.py`); подсистемы, подключённые только
адаптером хоста, помечены соответственно. **RC toggle(s) + default** фиксирует
управляющее поле (поля) `RuntimeConstants` и его безопасный/выключенный default.

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

Сквозной факт: **большинство новых возможностей выключены по умолчанию** и не
задействованы на тенанте по умолчанию, поэтому их *включённые* пути покрываются модульными
тестами, а не живыми прогонами. Это сделано намеренно — это опциональные продуктовые
возможности.

---

## Секции по технологиям

### ReAct loop / orchestrator / query engine / loop state

**Что и зачем.** Это сердце рантайма: ReAct-цикл (reason→act→observe), который
выполняет **по одному ходу ассистента за раз** и выдаёт потоковые события. Он
разделён на состояние и поведение, так что любой под может возобновить ран после
краха другого. Общий цикл ассистента **не** является единственным неизменяемым
путём: `QueryEngineConfig.run_mode` выбирает `DirectStrategy` или `DeepStrategy`
в `runtime/loop_strategies.py` до этого общего цикла.

**Ключевые классы/файлы.**

- `runtime/query_engine.py`
  - `QueryEngine` — один экземпляр на активный ран. Владеет **изменяемым
    состоянием на разговор**: `history` (список `Message`), машиной `LoopState`,
    `CompactionState`, `TokenUsage`, плюс `open_intents`, `usage_rows`,
    `lanes`, очередями live-control (`_steer_queue` / `_follow_up_queue`),
    живыми переопределениями model/thinking (`_live_model_name` /
    `_live_thinking_enabled` / `_live_reasoning_effort`), опциональным
    `verification` (`VerificationLifecycle`) и защёлками восстановления
    (terminal-only, guaranteed-terminal, self-verify, circuit-breaker,
    pending-reads, longfile, индекс tool-precondition, …). Персистентность —
    `snapshot()` ↔ `resume_from_snapshot()`; `run()` снимает snapshot в начале
    хода и в `finally`. Снимок также пишет `open_intents`, `usage_rows`,
    `lanes`, живые поля `live_*`, очереди steer/follow-up, `verification`
    (когда не default) и эти защёлки восстановления.
  - `QueryEngineConfig` — **неизменяемая поверхность инъекции**, привязываемая при
    построении движка: `run_id`/`tenant_id`/`session_id`/`model_name`,
    `system_prompt_sections`, `tool_visibility_policy`, снимок `rc`,
    `run_mode` (`"direct"` | `"deep"`, по умолчанию `"direct"`),
    `execution_profile`, `thinking_enabled` / `reasoning_effort`,
    `expected_terminal_tool`, `tool_preconditions` (run-level forcer;
    пустой default), опциональный `cache_observer`, опциональный
    `verification_delivery` и два **поставляемых хостом триггерных
    колбэка** (`pre_terminal_self_verify_trigger`,
    `pre_dispatch_terminal_verify_trigger`) — оба по умолчанию `None`, так что
    машинерия pre-dispatch veto / self-verify мертва, пока хост не
    инъецирует колбэк *и* не переключит соответствующий RC.
  - `QueryEngine.__init__` принимает опциональный `provider_chain: IProviderChain`
    для mid-stream failover провайдера. `None` (у каждого вызывающего, кто не
    настроил список приоритетов) оставляет существующее восстановление
    нетронутым.
  - `QueryEngine.run(initial_message)` — драйвер-**асинхронный генератор**: он
    добавляет пользовательское сообщение (или продолжает по уже существующей
    истории, заканчивающейся user-сообщением), увеличивает `turn_count`,
    сбрасывает состояние хода, ставит часы запуска, сохраняет снимок начала
    хода, привязывает `_current_turn_task`, затем итерирует `_query_raw` (не
    `query()`). Снимок конца хода попадает в `finally`.
- `runtime/query.py` (11517 строк) — `query(engine)` — **синхронная** функция.
  Это сознательно не асинхронный генератор: она вызывает
  `_reset_per_turn_state()` в точке вызова и **возвращает**
  `_projected_turn_events`, который итерирует `_query_raw` и применяет
  публичную границу доставки. Ход, прогнанный через `query()`, не имеет
  межподовой точки возобновления и не привязывает `_current_turn_task`.
  `_query_raw` реализует жизненный цикл хода: stop check → resume прерванных
  намерений → типизированный `before_run` → опциональный `/compact` через
  `CompactCheckpoint` → compaction check → UserPromptSubmit hook → build
  context → `select_strategy(run_mode).prepare_turn` →
  `_stream_one_assistant_message` (рекурсивно на tool_use) → dispatch →
  finalize. Восстановление шире набора 413 / max-output / thinking-trap /
  empty-nudge / idle-watchdog: ход также возобновляет прерванные намерения,
  стреляет типизированный `before_run`, обрабатывает `/compact` через
  `CompactCheckpoint` и привязывает usage/hooks через
  `runtime/correctness_bind.py` (`commit_usage`, `fire_typed_hook`,
  `mark_intent_recovery`, `persist_correctness`). Более старые ветви
  восстановления остаются модель-агностичными и закрытыми RC-гейтами.
- `runtime/loop_strategies.py` — `select_strategy(run_mode)` — единственная
  точка ветвления. `DirectStrategy` не вносит pre-action шага (auto-tool
  цикл). `DeepStrategy` запускает принудительный инструмент `Plan` (нативный
  `tool_choice` + CoT, ограниченный `reasoning_effort`), эмитирует ровно одно
  событие `REASONING_STEP`, затем общий цикл ассистента ведёт реальное
  действие с полной поверхностью.
- `runtime/loop_state.py` — `LoopState` — это чистая машина из 7 состояний:
  `PENDING → RUNNING → {AWAITING | COMPACTING} → {COMPLETED | FAILED |
  CANCELLED}`. `assert_transition()` обеспечивает таблицу легальных рёбер;
  `TERMINAL_STATES` не имеют исходящих рёбер. **Отлична** от
  `RunStatus` (долговременное зеркало PG-строки) и `RunState` (горячая запись в Redis-хеше)
  — `LoopState` — это in-flight-состояние экземпляра движка.

**Как вызывается/подключается.** Исполнитель хоста конструирует `QueryEngine` при
допуске рана, затем обычно `async for evt in engine.run(message)` на каждый ход
(или `async for evt in query(engine)` после того, как вызывающий уже засеял
историю). Каждый `TurnEvent` пробрасывается в SSE-мост. Цикл — единственный
потребитель любой другой подсистемы.

**Конфигурируемость через RC.** `max_turns_per_run`, `agent_max_seconds` (дедлайн по
настенным часам; `<= 0` = инертен), таймауты idle/stall watchdog и каждый
переключатель восстановления — это поля RC. `model_name` обязателен (нет вшитого default).
`agent_loop_default_mode` — тенантный default для `run_mode`.

**Протокол расширения.** **Не** редактируйте структуру цикла. Кастомизируйте через (а)
хуки (включая типизированные `PUBLISHED_HOOKS`), (б) инъецированные через
`QueryEngineConfig` колбэки/наблюдатели/`run_mode`/`tool_preconditions`/`provider_chain`,
(в) переключатели RC, (г) `system_prompt_sections`.

**Заметки о терминальной классификации.** Стоит выделить три поведения терминальной
классификации: (1) `query()` перепроверяет `stop_requested` после стриминга и
маршрутизирует отменённый ран в CANCELLED (а не в чистый end-turn); (2)
`_synthesize_missing_tool_results` вызывается на каждой контрольной точке teardown, так что
персистированный снимок всегда валиден по парности (см. секцию *Починка парности tool_use /
tool_result*); (3) выход по `max_turns`
классифицируется как терминальное **исчерпание** ресурса — он сохраняет
`stop_reason=max_turns` на проводе и трактуется как класс ошибки/неуспеха,
а не как чистый `COMPLETED`.

### Lean tool surface

**Что и зачем.** Маленький, универсальный пул инструментов, обращённый к агенту, чтобы
способная модель сама компоновала операции вместо заполнения дюжины кастомных типизированных схем.
Структурное разделение `read` против `read_silent` — доминирующий рычаг grounding, закодированный
здесь как контракт, чтобы любой бэкенд реализовывал ту же дисциплину.

**Ключевые классы/файлы.** `contracts/lean_tool_surface.py` определяет ровно **семь
канонических глаголов**: `exec`, `read`, `read_silent`, `write`, `find`, `search`,
`answer` (константы `LEAN_TOOL_*`). `GROUNDING_TRACKED_TOOLS` — это frozenset
`{read}` — `read` записывает свой путь как наблюдаемое свидетельство; `read_silent`
возвращает идентичное содержимое, но *не* записывается, позволяя модели просматривать
без засорения её набора цитат. `exec` — это **запускатель зарегистрированных бинарей**
(`{path, args, stdin}`), явно **не** `/bin/sh`. Поле `outcome` глагола `answer` —
это **строка свободной формы**, чьи допустимые значения определяются собственным
answer-контрактом бэкенда — ядро владеет только *формой* поля.
`tools/decorator.py` предоставляет декоратор `@tool`, который выводит
`ToolDefinition` (name, description, JSON-Schema params, флаг approval,
category) из сигнатуры функции + докстринга.

**Как вызывается/подключается.** Ядро владеет только *контрактами и именами* — оно не регистрирует
здесь конкретный `Tool`. Хост привязывает каждое имя к бэкенду (адаптер ConnectRPC,
K8s-песочница, …). Выбор профиля управляется RC через
`select_tool_surface_profile(...)`.

**Конфигурируемость через RC.** `tool_surface_profile` (`"legacy"` | `"lean"`, по умолчанию
`"legacy"`) плюс по-инструментные флаги `tool_surface_*_enabled`.

**Протокол расширения.** Реализуйте ABC `Tool` (`contracts/tools.py`) или используйте
`@tool`; зарегистрируйте в `IToolRegistry`. Принимайте канонические видимые агенту
имена ради кросс-бэкендной единообразности.

### Tool dispatch + gating

**Что и зачем.** Выполняет один вызов инструмента, переводя каждый сбой в
блок `tool_result(success=false)`, чтобы модель могла восстановиться следующим ходом — ран
никогда не падает жёстко на ошибке инструмента.

**Ключевые классы/файлы.** `runtime/tool_dispatch.py`:

- `ToolDispatcher.dispatch(...)` — единственная точка входа диспетчеризации. Стадии:
  (1) **registry lookup** — `unknown_tool`, если имя не зарегистрировано;
  (2) **schema / JSON validation** — инвариант byte-cap / JSON-сериализуемости
  на входном словаре; (3) **`ToolPermissionGate.check(...)`** (4 стадии,
  см. секцию *Permission gate*) — хук `pre_tool_use` И ЕСТЬ финальная стадия
  `hook` гейта, **а не** отдельный pre-gate-шаг; (4) **preconditions**
  (DAG; см. секцию *Tool preconditions*) — замаскированный инструмент замыкается накоротко,
  проверяется **после** гейта; (5) выполнение `tool.invoke(ctx)`;
  (6) **post_tool_use hook**.
- `DispatchOutcome` (frozen) — `success`, `content`, `is_error`, `error_kind`,
  `approval_required`/`approval_token` (→ цикл переходит в AWAITING),
  `ask_user_required`/`ask_user_payload` (human-in-the-loop), `duration_ms`,
  `metadata`.
- `DispatchErrorKind` — `validation | permission | execution | timeout |
  rate_limit | unknown_tool | consecutive_error_cap`. Последний — это страж:
  как только серия идентичных подряд ошибок на ран превышает
  `tool_dispatch_consecutive_error_cap`, диспетчер переписывает сбой
  в этот вид.
- Инструмент может прикрепить машиночитаемый словарь **`structured_error`** (напр.
  `{"finalization_recommended": True, "reason": ...}`) к выброшенному исключению;
  except-ветвь диспетчеризации пробрасывает его в `DispatchOutcome.metadata`, и
  цикл показывает модели подсказку о финализации.

#### Permission gate

`runtime/tool_permission.py` — `ToolPermissionGate.check(...)` возвращает
`ToolPermissionDecision` (значение `ToolPermissionOutcome` из `allow` / `deny` /
`require_approval`, опциональные переписанные args, опциональный approval-токен), разрешаемое
по четырём `PermissionStage` в порядке: `whitelist` → `safety_policy` →
`rate_limit` → `hook`. (`default` — это no-op-значение ALLOW, которое несёт решение,
когда ни одна стадия не возражает, а не отдельная стадия конвейера.) Стадия `whitelist`
обеспечивает тенантную `ToolVisibilityPolicy` (и опциональный scope субагента); стадия
`rate_limit` — это no-op-шов в ядре, в который хост встраивает
Redis-бакет через `register_policy`; стадия `hook` срабатывает
`pre_tool_use` последней и является точкой переопределения с наивысшим рычагом. Политики безопасности
реализуют `IToolSafetyPolicy` (`applies_to(side_effect_class)` + `evaluate(...)`):

- `ShellSafetyPolicyAdapter` — оборачивает `DefaultShellSafetyPolicy` (единственная политика
  в стеке по умолчанию; инспектирует `arguments['command']`).
- `HttpDnsAllowlistPolicy`, `WorkspacePathPolicy` — доступны, но **не** в
  стеке по умолчанию; хост регистрирует их через `register_policy` (core API
  остаётся замороженным).

**RC/расширение.** Гейт всегда включён; новые политики регистрируются в рантайме. Хук
`pre_tool_use` — точка с наивысшим рычагом для гейта LLM-как-политика.

### Tool retrieval / pool / registry / 3-layer surface

**Что и зачем.** Держит список инструментов для LLM маленьким и релевантным для меньших локальных
моделей, сохраняя при этом стабильный байтовый порядок ради переиспользования KV-prefix-кэша.

**Ключевые классы/файлы.**

- `runtime/tool_registry.py` — `ToolRegistry(IToolRegistry)`. Его
  `compute_effective_surface(tenant_id, policy, query, top_k)` — это **3-слойный
  фильтр**, вызываемый циклом на шаге 4:
  1. **Policy** — `ToolVisibilityPolicy` (visible / blocked / pinned).
  2. **Clipping** — если `top_k is None` или видимый набор ≤ `top_k`, вернуть его
     отсортированным по имени (без retrieval).
  3. **Progressive discovery** — иначе BM25-ранжирование по `query`; запинённые инструменты
     всегда включены; top-K по баллу. Финальный порядок всегда **name ASC**
     (cache-stable), хотя порядок retrieval управляет отбором.
- `runtime/tool_retrieval.py` — мультиязычный BM25: `retrieve_tools`,
  `bm25_score`, `compute_idf`/`compute_avgdl`, `build_candidate`,
  `reduce_query`.
- `runtime/tool_pool.py` — `assemble_tool_pool` / `assemble_tool_pool_from_concrete`.
  **Заметка:** это параллельный ассемблер, который **не подключён** в цикл
  (цикл использует `compute_effective_surface`); у него есть вызывающие только в тестах +
  реэкспорт.

**Конфигурируемость через RC.** `tool_retrieval_top_k` — порог отсечения, передаваемый
циклом.

**Протокол расширения.** Регистрируйте `ToolDefinition` в реестре; задавайте
видимость/пин через `ToolVisibilityPolicy`.

### Tool preconditions (DAG + action spec + run-level forcer)

**Что и зачем.** Три различных механизма обеспечивают «этот инструмент ещё нельзя
запускать» или «этот инструмент должен выполниться первым». Они никогда не
взаимодействуют.

**Ключевые классы/файлы.**

1. **Per-tool DAG** (`runtime/tool_preconditions.py`, гейтится
   `tool_preconditions_enabled`, по умолчанию `False`) —
   `check_preconditions`, `resolve_precondition`, `record_satisfaction`,
   `compute_masked_tools`, `load_satisfied_set`/`store_satisfied_set`
   (удовлетворённый набор гоняется туда-обратно через helper-bag движка, так
   что он переживает snapshot/resume). Инструмент, требующий предшествующего
   наблюдения, **маскируется**, пока его `ToolDefinition.preconditions` не
   удовлетворены (напр. `FinalizeFile` после `AppendFile`). Диспетчер
   замыкает замаскированный инструмент накоротко **после** permission gate.
2. **Action-precondition spec** (`contracts/tool_action_preconditions.py`) —
   типизированная спецификация правил со строковой нагрузкой
   (`ToolActionPreconditionRule`/`Spec`/`Predicate`, включая
   `PREDICATE_KIND_DOC_OBSERVED`). Подключена только как RC-типы + вычислитель
   хост внутри `run` инструмента — никакой код цикла ядра не читает
   спецификацию напрямую. Гейтится `tool_action_preconditions_enabled` (по
   умолчанию `False`) и `tool_action_preconditions_mode`
   (`off | shadow | block`, по умолчанию `off`).
3. **Run-level forcer** (`runtime/run_tool_preconditions.py` +
   `QueryEngineConfig.tool_preconditions`) — упорядоченный кортеж записей
   `ToolPrecondition`, которые этот ран **должен** вызвать, прежде чем агент
   свободен отвечать. Пока запись outstanding, цикл называет её инструмент в
   `LLMRequest.extra['forced_tool_choice']`, так что нативный `tool_choice`
   провайдера — а не формулировка промпта — решает, что модель вызовет первой.
   Пустой кортеж (default) делает каждую точку входа no-op. Прогресс — индекс
   в кортеж (дубликаты осмысленны); попытки ограничены
   `run_tool_precondition_max_attempts`. Исчерпание **валит** ран, называя
   инструмент. Это ПРИНУЖДАЕТ инструмент, который модель не выбирала; DAG
   БЛОКИРУЕТ инструмент, который модель уже выбрала.

**Как подключается.** Помощники DAG потребляются путём диспетчеризации.
Декларативная action spec — только хост. Run-level forcer потребляется
`_query_raw` / `_stream_one_assistant_message`, и его счётчики
(`_tool_precondition_index`, `_tool_precondition_calls`,
`_tool_precondition_attempts`, `_tool_precondition_last_error`) живут на
`snapshot()`.

**Конфигурируемость через RC.** `tool_preconditions_enabled` (по умолчанию `False`);
`tool_action_preconditions_mode` (`off | shadow | block`, по умолчанию `off`);
`run_tool_precondition_max_entries` / `run_tool_precondition_max_calls` /
`run_tool_precondition_max_attempts` ограничивают forcer.

### Universal resilience layer

**Что и зачем.** Универсальный слой **classify-then-act**, покрывающий как вызовы LLM-
провайдера, так и транспорт инструментов/VM, обобщающий собственную
форензику транспортных штормов Protocore. Базовая философия: классифицировать сбой в маленькую
нейтральную таксономию, затем выбрать маленькое нейтральное действие восстановления; бюджетировать ретраи, чтобы
сбоящий хост гасился, а не усиливался; никогда не повышать дедлайн; для мутирующих
операций — classify-don't-retry + read-back (без слепого переотправления).

**Ключевые классы/файлы.**

- `contracts/resilience.py` — контракты:
  - `ResilienceErrorClass` (таксономия): `transient_retryable`, `rate_limited`,
    плюс классы structural/deterministic/auth/billing/context-overflow/timeout-rebuild.
  - `ResilienceAction` (набор стратегий) + `ResilienceDecision` (один вердикт).
  - `RetryBudgetState` — чистая модель token-bucket.
  - `IToolTransport` / `TransportCallSpec` — инъецируемый транспортный хук
    (хост привязывает свой конкретный транспорт).
  - `ToolTransportError` (несёт опциональный `retry_after_seconds`),
    `ToolTransportTimeout`, `ToolTransportRetryBudgetExhausted`
    (`finalization_recommended=True`).
- `runtime/resilience.py` — рантайм: `ResiliencePolicy.decide(...)`,
  `classify_transport_error`, `transport_retry_after_seconds`,
  `decorrelated_jitter_backoff`, `TokenBucketRetryBudget` (принимает инъецированную
  блокировку — без состояния модуля), `deadline_finalization_reserve_ok`,
  `_maybe_rebuild_transport` (хук перестроения) и
  обёртка `resilient_transport_call`.

**Как подключается.** **Транспортная обёртка** — это контракт, которым владеет ядро, потребляемый
через границу слоем resilience хост (закрыт гейтом
`resilience_enabled`); у него нет вызывающего внутри `protocore/runtime/`.
**AdaptiveSafetyBand** и **AttemptLedger** (см. секцию *Attempt ledger + adaptive
safety band*) *действительно* подключены в цикл.

**Конфигурируемость через RC.** `resilience_enabled` (по умолчанию `False`),
`resilience_transport_max_attempts` (по умолчанию `1`, т.е. single-shot), backoff
base/cap/jitter ratios, параметры token-bucket, секунды deadline-reserve.

**Протокол расширения.** Реализуйте `IToolTransport` (и опционально хук
`rebuild()`); отобразите ошибки своего wire-формата на `ResilienceErrorClass`;
прикрепляйте `retry_after_seconds`, когда хост сообщает о сбросе. Политика учитывает
`retry_after_seconds`, а `_maybe_rebuild_transport` предоставляет опциональный
хук перестроения транспорта.

### Attempt ledger + adaptive safety band

**Что и зачем.** **Attempt ledger** записывает, что (суб)агент заявил, что
произведёт, и что было фактически верифицировано, чтобы финализация могла решить честный
исход. **Adaptive safety band** вычитает откалиброванный запас дрейфа
из по-вызовного бюджета вывода, чтобы `prompt + max_tokens` оставался под
окном провайдера, даже когда локальный оценщик токенов ошибается (напр., раздувание
Cyrillic-in-JSON-escape).

**Ключевые классы/файлы.**

- `contracts/attempt_ledger.py` — `AttemptLedger`, `DeliverableDeclaration`
  (`path`, `kind`, `required`, `min_size_bytes?`, `sha256_expected?`),
  `VerificationRecord` и литерал `LedgerOutcome`
  (`completed | partial | failed | unknown` — нейтральный исход, **не**
  какой-либо enum бэкенда). `SelfReportedStatus` (самоклассификация агента) сохраняется, но
  ей не доверяют слепо; `RuntimeAttemptStatus` — это ортогональная
  авто-классификация рантайма.
- `runtime/adaptive_safety_band.py` — `AdaptiveSafetyBand`,
  `AdaptiveBandSnapshot`, `AdaptiveBandStore` (Protocol) +
  `NullAdaptiveBandStore`. Band — пер-`(provider, model)`; по умолчанию
  персистентность только in-process (`NullAdaptiveBandStore`) — калибровке для
  горизонтального масштаба нужен Redis-backed store от хоста.

**Как подключается.** Band питает `max_output_tokens` в
`_stream_one_assistant_message` (`_resolve_safety_band_value`); когда band не
подключён, помощник возвращает 0 и поведение идентично pre-band. Ledger
потребляется finalization gate (см. секцию *Finalization gate + contract*).

### Finalization gate + contract

**Что и зачем.** Закрывает брешь финализации: (суб)агент, который успешно записал
видимый пользователю артефакт, но исчерпал итерации, не вызвав свой
терминальный инструмент, иначе был бы оценён как "failed", и лидер извинился бы,
хотя артефакт уже на диске. Гейт **верифицирует** заявленные deliverables
(stat каждый в workspace, записывает `VerificationRecord`) и **решает**
`FinalizationDecision` (шаблон success / partial / failed), который использует финальный
ход лидера.

**Ключевые классы/файлы.**

- `runtime/finalization_gate.py` — `FinalizationDecision`,
  `WorkspaceStatProtocol` / `WorkspaceStatResult` (stat-only-фасад, инъецируемый
  оркестратором, чтобы ядро оставалось свободным от импортов workspace-рантайма) и
  двуязычные (RU+EN) заголовки промпта финализации.
- `runtime/finalization_contract.py` — `build_finalization_contract_block()`
  (блок инструкций, который получает лидер) и `parse_finalization_contract`
  (разобрать заявленные deliverables обратно).

**Как подключается.** Гейт подключён в терминальный путь цикла. **Парсер**
контракта подключён на стороне хоста (не в `query.py`).

**Конфигурируемость через RC.** `terminal_tool_nudge_enabled`,
`finalize_prose_gate_enabled`, `pre_terminal_self_verify_enabled`,
`pre_dispatch_terminal_verify_enabled`, `terminal_candidate_preserve_enabled`,
`resilience_post_tool_empty_nudge_enabled`, `finalization_contract_persona_enabled`
— все по умолчанию `False`.

### Terminal-answer validation + references / grounding + payload normalize

**Что и зачем.** Детерминированная, слепая к рубрике дисциплина grounding: цитаты
терминального `answer` должны быть подмножеством того, что было фактически `read`, сравниваемые по
**канонической форме**, так что несовпадение flat-vs-branded путей — не ложное вето.

**Ключевые классы/файлы.**

- `contracts/references.py` — `normalize_ref(...)`: чистая, только-stdlib,
  RFC-3986-уровня **проекция для сравнения** (канонизация path/percent, снятие
  кавычек, опциональное снятие sublocator/extension). Она **идемпотентна** и
  **поведенчески нейтральна** — она может только *убрать* ложное вето, но никогда добавить его,
  и она никогда не мутирует выпущенную/сохранённую моделью ссылку (она канонизирует только
  ключ принадлежности). Она намеренно **не** делает HTML-entity-decode.
- `contracts/terminal_answer_validation.py` — спецификация правил в виде данных:
  `TerminalAnswerRefRule`, `TerminalAnswerValidationSpec`,
  `TerminalAnswerValidationResult` (строко-типизирована, прямо-совместима; не держит
  enum бэкенда как код).
- `runtime/terminal_payload_normalize.py` — `normalize_terminal_text(...)`: чистый
  закрытый RC-гейтом `html.unescape` текста ответа (не байт-сохраняющий —
  корректно выключен по умолчанию).

**Как подключается.** Потребляется на терминальном ходе (post-submit-валидация и, с
триггером self-verify от хоста, pre-dispatch). Нижняя граница размера
`min_size_bytes` на deliverable сегодня никогда не задаётся шаблоном лидера (латентна).

**Конфигурируемость через RC.** `terminal_answer_validation_enabled`,
`observed_ref_normalize_enabled`, `terminal_answer_entity_normalize_enabled`,
`finalization_accept_inline_artifact_when_substantive` — все по умолчанию `False`.

### IMemory (scoped, FTS/BM25, idempotent, drift-guard, injection-scan seam)

**Что и зачем.** Greenfield-универсальная **типизированная, осознающая scope, ранжируемая по
retrieval память** фактов, которые агент усваивает и переиспользует — возможность, отличная от
транскриптов сессий, blob-ов, общего поискового индекса и todo.

**Ключевые классы/файлы.** `contracts/memory.py`:

- `IMemory` (Protocol) — контракт; ядро никогда не импортирует реализацию.
- `MemoryScope` — `global | user | project | session | agent | custom`; адрес
  записи — это `(tenant_id, scope, scope_key)`. **Самый изолированный
  default — `session`**; полная грамматика управляется RC, а не зашита в код.
  `DEFAULT_RECALL_SCOPES` — это веер recall по умолчанию.
- `MemoryRecord` (с зарезервированным `embedding` под апгрейд hybrid-vector v2),
  `MemoryWriteDecision`/`MemoryWriteResult` (двухстадийная идемпотентная запись
  сообщает CREATE / MERGE / SKIP, так что рантайм никогда не пишет молча дважды),
  `MemoryHit`, `ScopeRef` и ошибки (`MemoryConflictError` для оптимистичного
  drift-guard; `MemoryStoreUnavailableError` **нефатальна** — медленная/сбойная
  операция памяти деградирует до "нет памяти в этот ход").
- `tools/memory.py` — обращённые к агенту инструменты памяти.

Контракт документирует, что `text` записи — это **недоверенный ввод** и ДОЛЖЕН
сканироваться на prompt-injection при записи *и* перед инъекцией — шов —
`IMemoryContentScanner` (хост наполняет его; ограда `<memory-context>`
— это defense-in-depth, а не единственный контроль).

**Как подключается.** Память **подключена со стороны хоста** (`build_memory_tools`,
`_maybe_run_memory_auto_recall`, `PgMemoryStore`, admin API), закрыта RC-гейтами.
Цикл не вызывает инструменты напрямую — память это контракт ядра, удерживаемый
инструментами.

**RC/расширение.** `memory_enabled` + `memory_auto_recall_enabled` (оба по умолчанию
`False`); scope-ы recall, бюджет и политика управляются RC. Реализуйте `IMemory`
(v1: Postgres FTS/BM25 — `tsvector` + `ts_rank` / `pg_trgm`; BM25 обязателен для
точных SKU/ID/путей); апгрейд hybrid-vector + decay/reinforce + rerank — это
неломающая drop-in-замена.

### IWorkspace + read-dedup cache

**Что и зачем.** Scoped по сессии/задаче, поисковое, атомарное, привязанное к жизненному циклу
**локальное черновое рабочее пространство**: агент сбрасывает промежуточные данные (результирующий
набор SQL, обнаруженную схему, вывод `jq`, заметки) один раз и пере-читает/ищет в них много
раз, не перезапрашивая ненадёжный удалённый ресурс — рычаг стабильности (dump-once /
re-read-many), а не просто эргономика.

**Ключевые классы/файлы.** `contracts/workspace.py`:

- `IWorkspace` (Protocol). `WorkspaceScope` — `session | task | project`
  (чистое подмножество `MemoryScope` ради одной согласованной модели scoping); по умолчанию
  `session`. `WorkspaceLifecycle` — `scratch` (GC-eligible) | `durable`.
  `WorkspaceUnit` (типизированный манифест + ограниченное тело + `version` + зарезервированный
  `embedding`), `WorkspaceWriteOutcome` (CREATED / REPLACED — атомарно, идемпотентно
  пер-`path`), `WorkspaceHit` и ошибки (`WorkspaceConflictError` на оптимистичном
  version-drift-guard; `WorkspaceStoreUnavailableError` нефатальна). Поиск использует
  **ту же форму FTS/BM25**, что и `IMemory`.
- Конкретные инструменты Write/Read/Search/List и их фабрика `build_workspace_tools` —
  **только в хосте**; ядро поставляет лишь контракт `IWorkspace` выше, без `tools/workspace.py`.
- `runtime/read_dedup_cache.py` — `ReadDedupCache` + `make_dedup_key` /
  `make_capability_fingerprint` / `tool_key`: процесс-локальный read-dedup-кэш,
  ключёванный на `tenant/session/agent/path/content_hash`, так что повторный `read`
  того же пути замыкается накоротко.

**Как подключается.** **Не подключён в основном цикле** — `ReadDedupCache` и
`clear_scope` упоминаются только в своём определяющем модуле + тестах, а
конкретная фабрика инструментов — только в хосте (ядро её не импортирует).
По замыслу адаптер *хост* инстанцирует store, строит инструменты рабочего
пространства (`build_workspace_tools`), инъецирует метаданные scope/quota/enabled,
подключает dedup-кэш в путь чтения и вызывает `clear_scope` при teardown в конце
сессии.

**RC/расширение.** `workspace_enabled` (по умолчанию `True` — подсистема доступна
backend/admin workspace API и метаданным диспетчеризации; хост по
умолчанию не выставляет устаревшие LLM-инструменты Workspace*, когда обычные
файловые инструменты покрывают те же операции), `workspace_scope` (по
умолчанию `"session"`), `workspace_search_enabled` (по умолчанию `True`),
по-scope-ные мягкие лимиты. Реализуйте `IWorkspace` (референс расширяет
существующий по-сессионный байтовый store и добавляет манифест Postgres
FTS/BM25).

**Страж оптимистичной записи.** `WorkspaceWriteTool` прокидывает `expected_version` —
`workspace_read` показывает `version` юнита, а `workspace_write` учитывает его на
REPLACE (выбрасывая `WorkspaceConflictError` на дрейфе, показанный как нефатальный
корректирующий результат), задействуя drift-guard контракта с поверхности агента.

### Context management / two-tier compaction / session memory / budgets / token counting / prompt caching / strip-thinking

**Что и зачем.** Держит промпт под окном контекста провайдера на протяжении длинного
многоходового рана, сохраняя при этом cache-дружественные префиксы.

**Ключевые классы/файлы.**

- `runtime/context/manager.py` — `ContextManager`. `build_context(...)`
  собирает слоистый бандл (`ContextBundle`: system sections, tools,
  возможно-скомпакченные сообщения, активный язык, бюджеты), а `run_compaction(...)`
  ведёт каскад. Без состояния между вызовами в части сборки (читает RC свежим —
  без кэша модуля); состояние компакции живёт на движке.
  `estimate_history_tokens` делегирует единственному исчерпывающему-по-`ContentBlock`
  оценщику `estimate_message_tokens` (покрывает `TextBlock`/`ThinkingBlock`/
  `ToolUseBlock`/`ToolResultBlock`/`ImageRefBlock` **плюс**
  `Message.reasoning_content` и сериализованный catch-all, так что ни один блок никогда не
  считается молча как 0). `detect_active_language` выбирает RU против EN по
  доле кириллицы.
- `runtime/context/compaction.py` — **двухъярусный каскад** (в этом модуле нет
  яруса 3):
  - **Tier 1** (`run_tier1_truncation`) — усечь / blob-нуть негабаритные tool-
    результаты (свыше `tool_result_truncation_threshold`), заменяя тело
    плейсхолдером компакции + blob-ссылкой.
  - **Tier 2** (`run_tier2_summarisation`) — заменить целые старые не-системные ходы
    системным резюме, сохраняя последние N ходов.
  - Исчерпание выбрасывает `CompactionExhaustedError` → цикл переходит в
    FAILED.
- `runtime/compact_checkpoint.py` — операторский `/compact` (и соответствующий
  host compact POST) строит `CompactCheckpoint`, сквозь который следующий
  LLM-запрос не читает (`build_checkpoint` / `apply_checkpoint`). Гейтится
  `compaction_manual_enabled` (по умолчанию `False`). Это **не** третий ярус
  компакции — это checkpoint с удерживаемым хвостом, а не шаг каскада.
- `runtime/context/session_memory.py` — межзапусковый fold + реестр артефактов.
  `fold_run` обновляет `SessionMemory` (бегущее резюме + `ArtifactLedger`
  `path`/`content` file-writing tool-call) из сообщений одного завершённого
  рана; `build_seed` собирает эту память плюс недавний сырой хвост для
  следующего рана. Ядро не вызывает клиент модели — хост инъецирует уже
  вычисленный текст резюме. В более старых документах безымянно; это
  персистентное итеративное резюме, отличное от внутриходового каскада
  Tier 1/2.
- `runtime/context/budgets.py` — `derive_budgets(rc)`: **единственный источник
  истины** для производных бюджетов (триггер компакции, порог tool-результата,
  по-секционные бюджеты, остаток истории), все производные от
  `model_context_window × ratio`. Чистая + детерминированная между подами.
- `runtime/token_counting.py` — `LanguageProfile`
  (`latin` / `cyrillic_prose` / `cyrillic_in_json_escape` / `cjk` / `json_struct`),
  `detect_profile`, `chars_per_token`, `estimate_tokens` — мультиязычные,
  управляемые RC-ratio.
- `runtime/prompt_caching.py` — `apply_system_and_3(messages)`: чистая
  стратегия точек разрыва `system_and_3` (≤ 4 `CacheBreakpoint`: system на индексе 0
  + последние 3 не-системных сообщения). Производит только **подсказки** размещения.
- `json_utils.py` — `strip_thinking` / `strip_thinking_tokens` и устойчивые
  потоковые JSON-парсеры, используемые для восстановления структурированных нагрузок из шумного
  вывода модели.

**Как подключается.** Компакция выполняется на шаге 2 `_query_raw`
(`needs_compaction()` — единственный pre-flight-гейт) и реактивно на ошибке
context-window-exceeded. Ручной `/compact` обрабатывается раньше в той же
функции, когда `compaction_manual_enabled` включён и последний пользовательский
текст начинается с `/compact`. Session-memory fold/seed подключён со стороны
хост (ядро только считает). Подсказки prompt-cache вычисляются один раз на
вызов провайдера в `_stream_one_assistant_message` и оседают в
`LLMRequest.extra["cache_breakpoints"]`.

**Подключение prompt-cache.** `prompt_cache_wire_enabled` (RC, **по умолчанию `True`**,
kill-switch) управляет переводом `cache_breakpoints` в маркеры `cache_control`
у OpenAI-совместимого клиента хост; ядро всегда выпускает
подсказки, а адаптеры, не распознающие ключ, игнорируют его.

**Конфигурируемость через RC.** `compaction_trigger_ratio`, `tool_result_truncation_ratio`,
по-секционные ratio, `compaction_keep_recent_turns`,
`compaction_manual_enabled` (по умолчанию `False`), ratio `chars_per_token_*`,
`llm_output_max_tokens_ratio`, `prompt_cache_wire_enabled`.

> Заметка: отдельного модуля `compaction_thresholds.py` нет — единственный живой
> derive-путь — это `budgets.py::derive_budgets`.

### Починка парности tool_use / tool_result

**Что и зачем.** Anthropic / OpenAI / vLLM все отклоняют запрос, чей ассистентский
`tool_use` не имеет парного `tool_result` (или осиротевший `tool_result`, или
дублирующиеся id) с HTTP 400. Парность должна гарантироваться на wire-границе
как defense-in-depth — а не предполагаться корректной от вышестоящих мутаторов (компакция,
resume-from-partial-batch, усечение по max_tokens, teardown).

**Ключевые функции (`runtime/query.py`).**

- `_repair_outbound_tool_pairing(messages, placeholder)` — **чистый**,
  безусловный backstop на wire-границе, выполняемый над списком исходящих сообщений прямо
  перед сборкой `LLMRequest` (до вычисления cache-breakpoint, так что
  индексы адресуют финальный список). Четыре починки: forward-fill синтетических
  `is_error`-результатов для осиротевших `tool_use`, репозиционирование каждого реального результата
  прямо после его `tool_use`, reverse-strip осиротевших результатов, дедупликация
  дублирующихся id.
- `_synthesize_missing_tool_results(history, error_content)` — мутирует историю на
  месте на **каждой контрольной точке teardown** (stop-before-start, compaction-failed,
  hook-deny, stop-after-stream, dispatch-cancel, LLM-error terminal), так что
  персистированный снимок остаётся валидным по парности и упорядоченным для resume на другом
  поде. Идемпотентна.

**Как подключается.** `_repair_outbound_tool_pairing` выполняется на каждом исходящем запросе;
`_synthesize_missing_tool_results` выполняется на всех путях аномального выхода.

**RC.** `tool_result_pairing_repair_placeholder`,
`tool_result_interrupted_placeholder`.

### Skills routing / surfacing

**Что и зачем.** Выносит маленький **каталог** доступных скиллов в
системный промпт каждый ход и загружает полное тело скилла по требованию, когда пользователь
ссылается на него — так доменная возможность добавляется как данные, а не как по-задачные
подсказки в промпте. Каталог — это компактный, отсортированный по алфавиту блок
`<system-reminder>`, собираемый один раз за ран (кэшируется на
`engine._skill_catalog_block`) и помещаемый в статический префикс
промпта, так что он остаётся байт-стабильным между ходами (сохраняя кэш промпта).
Это **не** BM25- и не top-K-ранжирование.

**Ключевые классы/файлы.** `runtime/skill_index.py` — `render_skills_catalog`
выдаёт `SYSTEM_REMINDER_HEADER` («Skills are tools, not files… call exactly
`Skill(skill="<name>")`») плюс одну строку
`Skill(skill="{name}") — {description}` на каждый включённый скилл, по
алфавиту имён. Сверх токен-бюджета блок деградирует до одних форм вызова
(`Skill(skill="{name}")`). `derive_skill_index_budget_tokens` — это
`model_context_window × skill_index_budget_ratio` (по умолчанию 1%).
`contracts/skills.py` — `ISkillStore`, `SkillIndexEntry`, `SkillBundle`,
`SkillFileRef`, `SKILL_ENTRY_PATH` (`SKILL.md`). `list_files` / `load_file` —
обязательные методы протокола для многофайловых бандлов (как минимум
каноническая строка `SKILL.md`; устаревший однофайловый скилл может
синтезировать её из `body_md`). **Цикл ядра никогда не вызывает**
`list_files` / `load_file` — каталог строится через `list` +
`list_enabled_subset`, тело по триггеру — через `load` / `list_subset`.
Хосты, которые отдают вспомогательные файлы, сами пользуются файловым API.

**Как подключается.** Шаг 4 вызывает `_ensure_run_skill_catalog(engine)`.
Чтения skill-store ключуются по `QueryEngineConfig.account_id` (банк на
аккаунт), **не** по `tenant_id`. Когда `engine.skills is None` или store
пуст → пустой блок нулевой стоимости. Сбои изолируются с WARNING; ран
продолжается. На каждый ход `<command-name>NAME</command-name>` в последнем
пользовательском тексте загружает совпавший `SkillBundle.body` как блок
Layer-3, с потолком `max_skills_per_run` (по умолчанию 4). Пины проекта
(`pinned_skill_names`) подмешиваются через `list_enabled_subset`, так что
выключенный скилл не попадает в каталог.

**RC/расширение.** Surfacing управляется данными (пустой store = нет блока).
`skills_hot_reload_enabled` (по умолчанию `False`) пропускает кэш на ран и
пересобирает каталог на каждый вызов `_ensure_run_skill_catalog`. Реализуйте
`ISkillStore`; ranker реализовывать не нужно.

### Hooks (pluggy) + injection / scratchpad + context_bootstrap

**Что и зачем.** Шов расширяемости: deny/modify/observe в каждой точке жизненного
цикла, не трогая цикл. Плюс опциональный turn-1 **context bootstrap**,
который читает собственные contract/readme-документы окружения и предваряет замороженным
ориентирующим сообщением `<environment_context>`.

**Ключевые классы/файлы.** `hooks/specs.py` — `AgentHookSpecs`: **8 pluggy
hookspec-ов** (`pre_tool_use`, `post_tool_use`, `user_prompt_submit`,
`session_start`, `session_end`, `pre_compact`, `post_compact`, `file_changed`).
`hooks/manager.py` — `HookManager` (in-process pluggy-реестр + агрегатор).
`contracts/hooks.py` — межподовый контракт `IHookManager`, `HookResult`,
`HookActionKind`, `HookSpec`.
`runtime/typed_hooks.py` — `PUBLISHED_HOOKS` (`before_run`, `before_tool`,
`after_tool`, `transform_context`, `before_compact`, `after_compact`) плюс
`HookRegistry` / `dispatch_hook`. Хост реэкспортирует этот опубликованный
набор (например, маршрут session-correctness перечисляет `PUBLISHED_HOOKS`).
`runtime/correctness_bind.py` — клей, который стреляет типизированные хуки и
коммитит usage из `_query_raw`.

**Как подключается (важно).** **Pluggy-`HookManager` ядра экспортируется, но не
ведёт цикл.** `engine.hooks` цикла типизирован `IHookManager` и
вызывает **3-аргументный** `invoke(event, payload, tenant_id)` — продакшен-хуки выполняются через
адаптер `IHookManager` хоста. Pluggy-менеджер конструируется в основном
в тестах. Также отметьте брешь в контракте: `HookEvent` перечисляет 10 событий, но
`AgentHookSpecs` объявляет только 8 (нет `subagent_start`/`subagent_stop`), так что эти
два никогда не могут сработать через pluggy-менеджер ядра.

Типизированные хуки — **вторая** боевая поверхность: когда
`typed_hooks_enabled` включён и задан `engine.typed_hook_registry`,
`fire_typed_hook` запускает соответствующий опубликованный обработчик.
Выключено по умолчанию — нет реестра, no-op allow.
`before_run`, `transform_context`, `before_compact` и `after_compact`
срабатывают от одного этого флага. `before_tool` и `after_tool` вложены
в ветку диспетчеризации `intent_settlement_enabled` — они не бегут, если
включён только флаг типизированных хуков. `transform_context` срабатывает
(и может быть выдан `hook_fired`), но его исход `rewrite` к истории
**не** применяется.

**RC/расширение.** `judge_failure_mode` (хук LLM-судьи fail-open/closed),
`judge_timeout_ms`; `context_bootstrap_enabled` (по умолчанию `False`),
`context_bootstrap_docs`, `context_bootstrap_tree_depth`;
`typed_hooks_enabled` (по умолчанию `False`), `typed_hooks_timeout_ms`.
Зарегистрируйте реализацию `IHookManager` (хост), pluggy-`hookimpl`-ы
или обработчики на `HookRegistry` для опубликованных типизированных имён.

### Events / observability / streaming

**Что и зачем.** Стриминг обязателен — каждая дельта провайдера становится типизированным
`TurnEvent`, проброшенным немедленно. Две различные поверхности событий:

- `events.py` — `EventBus` + `EventName` (~70 имён): **in-process** типизированный
  pub/sub для сигналинга между sibling-обработчиками внутри пода (используется HookManager,
  ContextManager, …). Отличен от межподового `IEventStream` (Redis
  Streams), используемого для SSE reconnect/replay.
- `runtime/events/types.py` — `EventType`: **по-ходовая потоковая** таксономия
  (выровнена под Anthropic: `message_*`, `content_block_*`, `tool_use_*`,
  `tool_result`, `error`, плюс расширения Protocore
  `sandbox_*`/`subagent_*`/`hook_fired`/`tool_call_pending`/`state_changed` и
  жизненный цикл цикла `run_started`/`heartbeat`/`compaction_*`). Поздние
  добавления включают `reasoning_step` (план Deep-режима), `intent_committed`,
  `usage_committed`, `session_forked`, `lane_locked`, `recovery_marked`,
  `compact_checkpoint`, события steer/follow-up/очередей (`steer_queued`,
  `follow_up_queued`, `queue_update`), live-control
  `model_changed`/`thinking_changed` и события верификации кандидата
  (`candidate_ready`, `verification_started`, `verification_reported`,
  `repair_requested`, `release_decided`, `candidate_released`). Каждое значение —
  строка `event:`, показываемая SSE-клиентам.
- `runtime/events/envelope.py` — `TurnEvent` (замороженный wire-envelope).
- `runtime/llm/delta_bridge.py` — переводит поток провайдера в
  `ProviderDelta` → `TurnEvent` (`_normalise_finish_reason`, `is_block_end`,
  …).
- `contracts/observability.py` — `CacheObserverProtocol` (опциональный
  сток hit-rate prompt-cache, инъецируемый через `QueryEngineConfig.cache_observer`).

**Как подключается.** `query()` выдаёт `TurnEvent`-ы повсюду; usage-дельта питает
cache observer. Стоки трейсинга/observability инъецируются через
границу.

### Safety (shell policy + chain parser + path isolation + approvals)

**Что и зачем.** Валидировать составленные моделью shell-команды перед исполнением, с
паттернами deny/approval на основе capability, и изолировать пути workspace.

**Ключевые классы/файлы.**

- `safety/shell.py` — `DefaultShellSafetyPolicy` + `_DENY_PATTERNS`
  (деструктивный `rm -rf /`, SUID, base64/dd, ANSI-C `$'...'` и locale `$"..."`
  кавычки, `$IFS`/`${...IFS}` инъекция через word-split, …). Возвращает
  `ShellPolicyDecision` (allow / deny / require-approval).
- `runtime/chain_parser.py` — `parse_chain(...)`: маленькая shell-грамматика, которая
  разбивает команду на `;`/`|`/`&&` на `CommandSegment`-ы и показывает `$()` /
  backtick **тела подстановок** (собираются даже внутри двойных кавычек, не
  одинарных), так что по-сегментные deny-паттерны перевзводятся на телах подстановок.
- Path-isolation + approval-политики живут в `tool_permission.py`
  (`WorkspacePathPolicy`), а approval-поток — это стадия `require_approval` гейта
  (цикл → AWAITING → resume).

> Заметка: `DefaultShellSafetyPolicy` **fails open** на несовпадении (нет
> fail-closed/ambiguous-эскалации), а `HttpDnsAllowlistPolicy` /
> `WorkspacePathPolicy` не в стеке по умолчанию — хост должен зарегистрировать
> их через `register_policy`.

### RuntimeConstants system

**Что и зачем.** Единственный механизм для настраиваемых значений — **никаких inline
magic numbers**. Каждое настраиваемое значение — это поле на замороженном Pydantic-снимке, безопасное
по умолчанию и конфигурируемое из дашборда.

**Ключевые классы/файлы.**

- `contracts/runtime_constants.py` (7423 строки) — `RuntimeConstants`
  (`model_config = ConfigDict(frozen=True, extra="forbid")`) и
  Protocol `RuntimeConstantsProvider` (`async get(tenant_id) -> RuntimeConstants`).
  `extra="forbid"` означает, что **core и хост должны деплоиться парно** (неизвестное
  поле отклоняется). Снимок включает выключенные по умолчанию поверхности
  `intent_settlement_enabled`, `usage_ledger_enabled`,
  `session_tree_enabled`, `lanes_enabled`, `typed_hooks_enabled`,
  `telemetry_spans_enabled` (и `compaction_manual_enabled`,
  `steer_follow_up_enabled`). `workspace_enabled` по умолчанию `True`.
- `runtime/runtime_constants.py` — `StaticRuntimeConstantsProvider` +
  `default_runtime_constants(**overrides)` (тесты + in-memory smoke-рантайм;
  продакшен-поды поставляют Postgres-backed провайдер с Redis-кэшем).
- `constants.py` (~70 строк) — лимиты безопасности по памяти на уровне модуля (`MAX_ARTIFACTS`,
  `MAX_TOOL_CALL_ARGUMENT_BYTES`, `PROTOCOL_VERSION`, `DEFAULT_MODEL`, …).

**Правило 3 правок.** Добавление настраиваемого значения: (1) Pydantic-поле ядра (default
safe/off) + (2) identity-запись в хосте `_FIELD_MAP` + (3) seed в каталоге
миграций. Страница Constants в дашборде после этого рендерит переключатель бесплатно.

### Intent, usage ledger, session tree, lanes, typed hooks, telemetry, live control, run work budget

Модули, которые сидят рядом с общим ReAct-циклом. Каждый выключен по умолчанию,
пока соответствующее поле RC не скажет иное.

- `runtime/intent.py` — `IntentRecord` / `commit_intent` / `settle_intent` /
  `resume_open_intents` / `replay_policy_for`. Когда
  `intent_settlement_enabled` включён, **каждый** диспетчеризуемый вызов
  инструмента коммитит `IntentRecord` с зарезервированными result id до
  `ToolDispatcher.dispatch`. `replay_policy_for` ставит `replay="never"`,
  когда имя инструмента в `intent_never_replay_tools` (по умолчанию
  `Write,Edit,Bash,Finalize,AppendFile`); любое другое имя — `"safe"`.
  Краш mid-flight: never-replay намерения становятся `interrupted`
  (синтетическая ошибка, без реплея); safe остаются `open`.
  `should_skip_never_replay` коротко замыкает возобновлённый interrupted
  never-replay вызов. Поле снимка: `open_intents`.
  `before_tool` / `after_tool` живут в той же ветке
  `if intent_settlement_enabled`.
- `runtime/usage_ledger.py` — append-only список `UsageRow`. Когда
  `usage_ledger_enabled` включён, `correctness_bind.commit_usage` дописывает
  строку. `_query_raw` пишет `inference` / `retry` / `compaction` /
  `abort` / `fail` независимо от intent. Строка вида **tool** пишется только
  на пути intent-settlement (тот же блок `if intent_settlement_enabled`,
  который settle-ит намерение). Неудачная попытка плюс её retry — две
  строки. Поле снимка: `usage_rows`.
- `runtime/session_tree.py` — `fork_session` / `clone_session` копируют путь
  истории в новую `SessionBranch`, не мутируя источник. Гейтится
  `session_tree_enabled`; clone требует settled-источник;
  `session_tree_max_copy_messages` ограничивает копию.
- `runtime/lanes.py` — именованные lanes над общей историей. Main всегда
  существует; дополнительные берут эксклюзивные блокировки (`acquire_lane` /
  `release_lane`). Гейтится `lanes_enabled`; `lanes_max_per_session` включает
  main.
- `runtime/typed_hooks.py` — `PUBLISHED_HOOKS` + `HookRegistry`. См. секцию
  [Hooks](#hooks-pluggy--injection--scratchpad--context_bootstrap).
- `runtime/telemetry.py` — spans низкой кардинальности (`run` / `turn` / `step` /
  `tool` / `compact` / `hook`). Гейтится `telemetry_spans_enabled`.
  Высококардинальные id остаются атрибутами; `is_prometheus_safe_label`
  отказывается принимать `session_id` / `lane_id` / `operation_id` / `run_id`
  как ключи меток. `mark_recovery` помечает span, когда возобновляется
  прерванное намерение.
- `runtime/correctness_bind.py` — клей, чтобы intent, ledger, типизированные
  хуки и recovery выполнялись внутри `_query_raw` (`commit_usage`,
  `fire_typed_hook`, `mark_intent_recovery`, `persist_correctness`).
- `runtime/live_control.py` — очереди steer / follow-up (`QueuedPrompt`,
  `enqueue`, `place_items`), живые переопределения model/thinking и settled-
  помощник. Гейтится `steer_follow_up_enabled` (по умолчанию `False`).
- `runtime/run_work_budget.py` — кумулятивный бюджет жизни дерева
  (`max_subagent_runs_per_tree`, `max_total_tokens_per_tree`) для одного
  корневого рана и всего, что он порождает. Сиблинг `SubagentTreeBudget`
  (permits), а не его расширение. Исчерпание отказывает в **делегировании**,
  никогда в самом ране.

---

## Точки расширения (протоколы, которые реализует хост)

Ядро — это набор `Protocol`-ов; хост предоставляет конкретные адаптеры.
Полная поверхность интерфейсов живёт в `contracts/` (каждый в своём модуле — единого
`protocols.py` **нет**; старый монолитный файл был разбит по доменам).
Основные точки расширения:

| Protocol | Module | What the host provides |
|---|---|---|
| `ILLMProvider` | `contracts/llm.py` | LLM completions: `stream_with_tools`, `complete_structured`, `complete_text` и `count_tokens`; универсальный LiteLLM/OpenAI-совместимый адаптер (OpenRouter / vLLM / OpenAI). |
| `IProviderChain` | `contracts/llm.py` | Упорядоченные оставшиеся провайдеры плюс односторонний курсор `advance()`. Внедряется на `QueryEngine(..., provider_chain=...)` для mid-stream failover; `None` оставляет существующее восстановление нетронутым. |
| `RuntimeConstantsProvider` | `contracts/runtime_constants.py` | Per-tenant `RuntimeConstants` backed by Postgres + Redis cache. |
| `ISessionStore` | `contracts/session.py` | Session/transcript persistence (Postgres). |
| `IRunStore` | `contracts/run.py` | Run record create/list/read (Postgres + Redis hot record). |
| `IToolRegistry` | `contracts/tool_registry.py` | The concrete `ToolRegistry` is in core; the host registers concrete `Tool`s + visibility policy. |
| `Tool` (ABC) / `@tool` | `contracts/tools.py`, `tools/decorator.py` | Concrete tool implementations (sandbox-backed exec/file tools, the lean verbs). |
| `IToolTransport` | `contracts/resilience.py` | The tool/VM transport (e.g. ConnectRPC) the resilience wrapper wraps; optional `rebuild()` hook. |
| `IMemory` | `contracts/memory.py` | `PgMemoryStore` (Postgres FTS/BM25, two-stage idempotent write, drift-guard) + an `IMemoryContentScanner`. |
| `IWorkspace` | `contracts/workspace.py` | Durable byte store + Postgres FTS/BM25 manifest, atomic write, per-scope GC. |
| `ISkillStore` | `contracts/skills.py` | Хранение и поиск skill-бандлов **и** многофайловые `list_files` / `load_file` (`SkillFileRef`). Цикл каталогизирует через `list` / `list_enabled_subset` и грузит тела через `load` / `list_subset`; `list_files` / `load_file` — файловый API хоста. Рендерер каталога живёт в ядре, `runtime/skill_index.py`. |
| `IHookManager` | `contracts/hooks.py` | Боевой 3-аргументный диспетчер хуков (именно он ведёт цикл, а не pluggy `HookManager`). Типизированные `PUBLISHED_HOOKS` — отдельная выключенная по умолчанию поверхность в `runtime/typed_hooks.py`. |
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

## Соглашения

- **Граница импортов.** Ядро никогда не импортирует пакет, стоящий над ним
  (`protocore_*`). Добавляйте
  поведение через контракты / адаптеры / RC, а не импортом вверх. Страж:
  `tests/test_core_import_boundary.py`.
- **Никаких inline magic numbers.** Каждое настраиваемое значение — это поле `RuntimeConstants`
  (frozen, `extra="forbid"`) или лимит из `constants.py`. Код рантайма читает из
  снимка RC, а не из зашитого литерала. Добавление одного — это правило 3 правок
  (поле ядра + хост `_FIELD_MAP` + seed миграции).
- **Безопасно при горизонтальном масштабировании.** Никаких словарей на уровне модуля, никаких удерживаемых модулем блокировок, никакой
  in-memory-авторитетности на уровне пода. Состояние, влияющее на корректность, живёт пер-ран на
  `QueryEngine` (snapshot/resume); эфемерное межподовое состояние — это Redis, долговременное
  состояние — это Postgres (оба инъецируются через границу). Token-bucket и
  adaptive band принимают инъецированные блокировки/store-ы, а не состояние модуля.
- **Стриминг обязателен.** `query()` возвращает асинхронный итератор; каждая дельта
  провайдера пробрасывается немедленно как `TurnEvent`. Не буферизуйте целый ход
  перед выпуском.
- **Никакой обратной совместимости.** Dev-версия проекта — ломайте свободно, удаляйте
  мёртвый код, никаких migration shim. (Поэтому `compaction_thresholds.py` был удалён
  начисто, как только `budgets.py` поглотил его.)
- **Используйте модели `Message`, а не сырые dict-ы.** Все сообщения текут как Pydantic-
  объединение `Message` / `ContentBlock` из `contracts/types.py`.
- **Не модифицируйте структуру цикла.** Кастомизируйте через хуки, инъекцию
  `QueryEngineConfig`, переключатели RC или `system_prompt_sections`.
- **Продакшен-логирование = WARNING.** Используйте `logger.warning(...)` для
  операционно значимых событий; более низкие уровни оставьте для локальной
  отладки.

Команды репозитория: `uv sync --extra dev`, `uv run pytest .`, `uv run ruff check .`,
`uv run mypy protocore`.

> Перевод английского оригинала `docs/architecture.md` (коммит `54b6543`). При изменении оригинала обновите перевод.
