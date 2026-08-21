# Контракты — граница ядра

> Аудитория: инженер, подключающий хост-приложение (или любой другой
> хост) к чистому ядру, либо любой, кому нужно точно знать, где заканчивается
> ядро и начинается внешний мир. Область: библиотека ядра `protocore/`.

`protocore` — это набор **контрактов** (Python `Protocol`-ы + типизированные
Pydantic-модели) и protocol-first ReAct-рантайм. Всё, что обращено наружу, —
это `Protocol`, который реализует хост; ядро **не** поставляет ни драйвера базы
данных, ни HTTP-эндпойнта, ни LLM-клиента. Этот документ — каталог этой
границы: интерфейсные протоколы, которые предоставляет хост, система
типов ядра, которая через них проходит, и соглашения, удерживающие поверхность
стабильной.

Для более глубокого взгляда «как это собирается воедино» — цикл, подсистемы,
поток данных одного хода — читайте [`architecture.md`](architecture.md),
структурный источник, который индексирует эта страница.

---

## Пакет контрактов: никакого монолитного `protocols.py`

Поверхность интерфейсов живёт в `protocore/contracts/`, **по одному модулю на
домен**. Здесь сознательно **нет единого `protocols.py`** — старый монолитный
файл был разбит так, чтобы каждая область ответственности (LLM, запуски,
сессии, память, рабочее пространство, …) владела самодостаточным модулем с её
протоколом, её типизированными моделями и её ошибками вместе.

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

### Принцип реэкспорта contract-first

Реэкспорты управляются через явные списки `__all__`, а граница —
**contract-first**: публичная поверхность начинается с интерфейсных протоколов,
затем идут типизированные модели, которые через них проходят. Значимы два
списка `__all__`:

- **`protocore/contracts/__init__.py`** — поверхность контрактов, реэкспортируемая
  как явный список `__all__` (не каждый символ, который определяют доменные
  модули). Это и есть список, из которого нужно импортировать при реализации
  адаптера. Некоторые символы, определённые в их contract-модулях, сознательно
  **не** входят в этот `__all__` — например, `AttemptLedger`
  (`contracts/attempt_ledger.py`), `ToolActionPreconditionSpec`
  (`contracts/tool_action_preconditions.py`), `TerminalAnswerValidationSpec`
  (`contracts/terminal_answer_validation.py`), `LLMTimeoutError`
  (`contracts/llm.py`) и `SkillNotFoundError` (`contracts/skills.py`) —
  импортируйте их из их именованного модуля.
- **`protocore/__init__.py`** (пакет верхнего уровня) — **курируемое подмножество**
  той же поверхности для типового случая. Он реэкспортирует 11 интерфейсов
  хранилищ/сервисов (`I*`) плюс `Tool`, систему типов ядра, `RuntimeConstants` и
  горстку утилит рантайма.

> **Обратите внимание:** `IMemory`, `IWorkspace`, `IToolTransport`,
> `IPromptTemplateProvider` и `CacheObserverProtocol` экспортируются из
> `protocore.contracts` (и их именованных модулей), но **не** входят в `__all__`
> пакета верхнего уровня `protocore`. Импортируйте их из `protocore.contracts`
> (или из конкретного модуля), а не из пакета верхнего уровня. `IProviderChain`
> и `SkillFileRef` также не входят в `protocore.contracts.__all__` —
> импортируйте их из `protocore.contracts.llm` и `protocore.contracts.skills`.

Две точки входа в цикл (`QueryEngine`, `query()`) **не** реэкспортируются ни на
одном из уровней — импортируйте их напрямую из
`protocore.runtime.query_engine` / `protocore.runtime.query`. См. раздел
[Public API](architecture.md#public-api-protocore__init__py).

---

## Интерфейсные протоколы (что предоставляет хост)

Это швы. Ядро объявляет `Protocol` (или ABC); хост привязывает конкретный
адаптер. Каждая строка — это контракт и то, что поставляет хост.

| Protocol | Module | What the host provides |
|---|---|---|
| `ILLMProvider` | `contracts/llm.py` | LLM-завершения: `stream_with_tools`, `complete_structured` (после цикла, JSON-схема), `complete_text` (после цикла, свободный документ) и `count_tokens`; универсальный LiteLLM/OpenAI-совместимый адаптер (OpenRouter / vLLM / OpenAI). |
| `IProviderChain` | `contracts/llm.py` | Упорядоченные оставшиеся провайдеры плюс односторонний курсор `advance()`. `QueryEngine` внедряет его как `provider_chain` для mid-stream failover; `None` оставляет существующее восстановление нетронутым. Не входит в `protocore.contracts.__all__` — импортируйте из `protocore.contracts.llm`. |
| `RuntimeConstantsProvider` | `contracts/runtime_constants.py` | Потенантные `RuntimeConstants` (`async get(tenant_id)`), на базе Postgres с Redis-кэшем. |
| `ISessionStore` | `contracts/session.py` | Хранение сессий / транскриптов. |
| `IRunStore` | `contracts/run.py` | Создание / список / чтение записей запусков (долговечная строка + горячая запись). |
| `IToolRegistry` | `contracts/tool_registry.py` | Конкретный `ToolRegistry` поставляется в ядре; хост регистрирует конкретные `Tool`-ы и `ToolVisibilityPolicy`. |
| `Tool` (ABC) / `@tool` | `contracts/tools.py`, `tools/decorator.py` | Конкретные реализации инструментов, привязанные к каноническим именам-глаголам. |
| `IToolTransport` | `contracts/resilience.py` | Транспорт инструмента / VM, который оборачивает обёртка отказоустойчивости; опциональный хук `rebuild()`. |
| `IMemory` | `contracts/memory.py` | Scope-aware FTS/BM25-хранилище памяти + `IMemoryContentScanner` для сканирования инъекций. |
| `IWorkspace` | `contracts/workspace.py` | Долговечное байтовое хранилище + FTS/BM25-манифест, атомарная запись, GC по скоупам. |
| `ISkillStore` | `contracts/skills.py` | Хранение и поиск skill-бандлов **и** многофайловый API: `list_files` / `load_file`, возвращающие строки `SkillFileRef` (как минимум каноническая запись `SKILL.md` / `SKILL_ENTRY_PATH`). **Цикл ядра никогда не вызывает** `list_files` / `load_file` — каталог строится через `list` / `list_enabled_subset`, тело по триггеру — через `load` / `list_subset`. Чтения store ключуются по `QueryEngineConfig.account_id`, не по `tenant_id`. Каталог рендерит `render_skills_catalog` (+ `derive_skill_index_budget_tokens`) в `runtime/skill_index.py` строками `Skill(skill="{name}")`, а не путями файлов. |
| `IHookManager` | `contracts/hooks.py` | Боевой диспетчер хуков (`invoke(event, payload, tenant_id)`) — именно он управляет циклом, а не внутрипроцессный pluggy `HookManager`. |
| `IEventStream` | `contracts/events.py` | Кросс-подовый долговечный поток событий для переподключения / повтора SSE. |
| `IBlobStore` | `contracts/blob.py` | Content-addressed хранилище блобов, используемое компакцией Tier-1. |
| `ISearchIndex` | `contracts/search.py` | Универсальный лексический поисковый индекс. |
| `ITodoStorage` | `contracts/todo.py` | Посессионное хранение todo. |
| `IAgentDispatch` | `contracts/agent_dispatch.py` | Диспетчеризация / поиск субагентов. |
| `IPromptTemplateProvider` | `contracts/prompts.py` | Рендеринг шаблона системного промпта. |
| `CacheObserverProtocol` | `contracts/observability.py` | Приёмник доли попаданий в кэш промптов, внедряемый через `QueryEngineConfig.cache_observer`. |

**15 интерфейсных протоколов**, с которых начинается публичный API ядра, остаются
`ILLMProvider`, `IRunStore`, `ISessionStore`, `IBlobStore`, `ISearchIndex`,
`ITodoStorage`, `IToolRegistry`, `ISkillStore`, `IAgentDispatch`,
`IEventStream`, `IHookManager`, плюс `IMemory`, `IWorkspace`, `IToolTransport`
и `IPromptTemplateProvider`. Ещё три протокола дополняют набор швов, но не
учитываются в этом заголовочном счёте: `RuntimeConstantsProvider` (потенантный
источник конфигурации), `IProviderChain` (mid-stream failover провайдера; не
реэкспорт верхнего уровня и не в `contracts.__all__`) и `IToolSafetyPolicy`
(дополнительные политики разрешений, регистрируемые в рантайме; см. раздел
[Permission gate](architecture.md#permission-gate)). `Tool` — это ABC, а не
`Protocol` — хост наследуется от него (или использует `@tool`).

> `IBlobStore` объявлен как ABC; остальные интерфейсы хранилищ/сервисов — это
> `Protocol`-ы. В любом случае правило одно — ядро зависит только от объявленной
> формы и никогда от конкретной реализации.

---

## Система типов ядра

Каждый примитив диалога проходит как одна из этих Pydantic-моделей (соглашение
«используйте модели `Message`, никогда не сырые dict-ы»). Все они живут в
`contracts/types.py`, если не указано иное. Большинство — замороженные
value-объекты.

### Сообщения и контент

- **`Message`** — единственный примитив диалога (с привязкой к роли). Ходы
  ассистента несут `content_blocks` (text + tool_use + thinking вперемешку).
- **`MessageRole`** (`StrEnum`) — `system` · `user` · `assistant` · `tool`.
- **`ContentBlock`** — это **union-тип**, а не класс:
  `TextBlock | ThinkingBlock | ImageRefBlock | ToolUseBlock | ToolResultBlock`.
- **`ContentBlockKind`** (`StrEnum`) — дискриминант: `text` · `thinking` ·
  `image_ref` · `tool_use` · `tool_result`.
- **`TextBlock`** / **`ThinkingBlock`** — простой текст и рассуждения модели
  (последнее обычно вырезается перед сохранением).
- **`ToolUseBlock`** — вызов инструмента, эмитированный ассистентом
  (`tool_call_id`, `name`, `arguments_json`; байты аргументов ограничены).
- **`ToolResultBlock`** — результат вызова инструмента, возвращаемый модели
  (`tool_call_id`, `content`, `is_error`, `metadata`).
- **`ImageRefBlock`** — ссылка на изображение, чьи байты живут в `IBlobStore`.

### Вызовы инструментов и результаты

- **`ToolCall`** — вызов, эмитированный LLM и выставляемый в `Tool.invoke`
  (`id`, `name`, `arguments`). Несёт флаги усечения
  (`truncated_by_output_cap`, `args_partial_truncated`), которые цикл использует
  для обнаружения усечённого посередине потока JSON аргументов.
- **`ToolResult`** — результат одного вызова
  (`tool_call_id`, `content`, `is_error`, `metadata`).
- **`ToolContext`** — контекст одного вызова, передаваемый инструменту
  (скоуп тенанта + метаданные).
- **`ToolDefinition`** — запись реестра (name, description, params schema,
  approval flag, category), которую производит функция `@tool` или подкласс
  `Tool`.
- **`ToolParameterSchema`** — форма параметров инструмента в виде JSON-Schema.
- **`ToolError`** (а также `ToolInvocationError`, `ToolPolicyDenied`,
  в `contracts/tools.py`) — иерархия ошибок инструментов.

### Запуски, сессии, события

Три различные формы запусков — не путайте их (см.
[`architecture.md`](architecture.md)):

- **`RunStatus`** (`StrEnum`) — **долговечный** жизненный цикл запуска,
  отражённый в персистентной колонке `runs.status`: `queued` · `running` ·
  `completed` · `partial` · `error` · `cancelled` · `incomplete` · `paused`.
  `partial` — функционально терминальный статус для запуска, который завершил
  свой цикл, но накопил ошибки диспетчеризации инструментов.
- **`Run`** — долговечная запись запуска (`id`, `tenant_id`, `session_id`,
  `status`, метки времени, опциональная ссылка на detail-блоб).
- **`RunState`** — **эфемерный** горячий рабочий набор (хранится хостом в
  Redis-хэше): `current_turn`, счётчики токенов, `last_event_id`. (Снова
  отличается от `LoopState`, FSM движка в полёте в `runtime/loop_state.py`,
  который *не* является контрактным типом.)
- **`Session`** — корень многоходового диалога (долговечный, никогда не
  удаляется).
- **`Event`** — конверт события в полёте (`run_id`, `name`, `payload`),
  эмитируемый через `IEventStream` / внутрипроцессный `EventBus`.
- **`StopReason`** (`StrEnum`) — почему ход завершился: `end_turn` · `tool_use` ·
  `max_tokens` · `max_turns` · `stop_sequence` · `error` · `cancelled`.
- **`ExecutionReport`** — ограниченная сводка телеметрии по запуску (события,
  записи вызовов инструментов, записи LLM-вызовов, предупреждения, запуски
  субагентов, артефакты и опциональный снимок `AttemptLedger`) со структурными
  ограничениями из `protocore.constants`.

### LLM-запрос / ответ

- **`LLMRequest`** — запрос, который цикл собирает для `ILLMProvider`
  (`messages`, `tools`, `max_tokens`, `extra` — включая подсказки кэша промптов
  `cache_breakpoints`).
- **`LLMResponse`** — форма непотокового ответа, которую возвращают
  `complete_structured` и `complete_text`.
- **`LLMObservabilityContext`** — контекст наблюдаемости на один вызов,
  прикреплённый к LLM-запросу.
- **`IProviderChain`** — не тип запроса/ответа; курсор failover, который
  `QueryEngine` перепривязывает на `self.llm`, когда mid-stream провайдер падает.

### Входящий трафик, блобы, компакция

- **`AgentEnvelope`** — единственный кросс-компонентный контракт входящего
  трафика (`kind`, `payload`, `metadata`; размер payload ограничен).
  Парсится/сериализуется через `parse_envelope` / `serialize_envelope`.
- **`EnvelopeKind`** (`StrEnum`) — `task` · `control` · `result` · `error`.
- **`BlobMetadata`** — запись индекса блоба (`ref`, `content_type`, `size_bytes`,
  `sha256`).
- **`CompactionSourceRef`** — указатель на сжатый блоб результата инструмента,
  сохраняемый как плейсхолдер в формате провода во время компакции Tier-1.

### Верификация и chunking

- **`VerificationLifecycle`** / **`VerificationDelivery`** /
  **`CandidateBundle`** / **`ReleaseDecision`** (`contracts/verification.py`) —
  жизненный цикл верификации кандидата, который `QueryEngine` снимает как
  `verification` и использует, чтобы гейтить публичную доставку читателю.
  Реэкспортируются из `protocore.contracts`.
- **`is_chunkable_content_mutation`** (`contracts/tool_chunking.py`) — единственный
  предикат восстановления усечения Write→AppendFile→FinalizeFile. Импортируется
  `query()`. Не входит в `protocore.contracts.__all__` — импортируйте из
  именованного модуля.

### Хуки

- **`HookEvent`** (`StrEnum`) — **10** событий жизненного цикла: `pre_tool_use` ·
  `post_tool_use` · `user_prompt_submit` · `session_start` · `session_end` ·
  `pre_compact` · `post_compact` · `file_changed` · `subagent_start` ·
  `subagent_stop`.
- **`HookResult`** — вердикт хука (allow / deny / modify, через
  `HookActionKind`).
- **`HookSpec`** — декларативная спецификация зарегистрированного хука.

### Навыки, субагенты, todo

- **`SkillManifest`** / **`SkillIndexEntry`** / **`SkillBundle`** /
  **`SkillFileRef`** — формы каталога навыков. `SkillFileRef` — это строка
  индекса многофайлового бандла (`path`, `size_bytes`, `mime_type`,
  `content_hash`); байты забираются через `ISkillStore.load_file`. У каждого
  бандла есть как минимум `SKILL_ENTRY_PATH` (`SKILL.md`). Устаревший
  однофайловый скилл может синтезировать эту одну строку из `body_md`.
  `SkillFileRef` **не** входит в `protocore.contracts.__all__` — импортируйте
  его из `protocore.contracts.skills`. Каталог цикла — формы вызова
  `Skill(skill="{name}")`, не эти пути. Цикл **не** вызывает `list_files` /
  `load_file`; чтения ключуются по `QueryEngineConfig.account_id`.
- **`SubagentDef`** / **`SubagentTask`** / **`SubagentResult`** — формы
  диспетчеризации субагентов, используемые `IAgentDispatch`.
- **`Todo`** / **`TodoStatus`** (`StrEnum`) — форма посессионного хранения todo.

---

## Реализация адаптера

Чтобы привязать ядро к хосту:

1. Реализуйте нужные вам интерфейсные протоколы из `protocore.contracts` (вам не
   нужны все — память по умолчанию выключена (`memory_enabled = False`);
   `workspace_enabled` по умолчанию `True`, но конкретные инструменты рабочего
   пространства по-прежнему живут в хосте).
2. Принимайте и возвращайте модели системы типов ядра — никогда не сырые dict-ы
   на границе.
3. Внедряйте конфигурацию через `RuntimeConstants` (замороженный снимок) и
   `ToolContext.metadata`; никогда не зашивайте политику тенанта в код.
4. Сконструируйте `QueryEngine` со своими адаптерами и управляйте им через
   `async for evt in query(engine)` (`query()` — синхронная функция, которая
   возвращает этот итератор) или `async for evt in engine.run(message)`.

Механика расширения рантайма — какой шов выбрать (протокол vs хук vs RC-тоггл vs
секция промпта) и жёсткое правило «не модифицировать структуру цикла» — описана
в `extending.md` и [`architecture.md`](architecture.md). Граница импортов (ядро
никогда не импортирует хост) обеспечивается
`tests/test_core_import_boundary.py`.

> Перевод английского оригинала `docs/contracts.md` (коммит `54b6543`). При изменении оригинала обновите перевод.
