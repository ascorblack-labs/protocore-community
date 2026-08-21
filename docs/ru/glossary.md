# Глоссарий

Краткие определения ключевых терминов ядра. Каждая статья соответствует
реальному символу в ядре и согласована с
[`architecture.md`](architecture.md). Термины сгруппированы по областям, а не по
алфавиту, чтобы связанные понятия читались вместе.

## Точки входа рантайма

**`QueryEngine`** (`runtime/query_engine.py`)
: Один экземпляр на активный запуск. Владеет **изменяемым состоянием в разрезе
  диалога** — `history`, конечным автоматом `LoopState`, `CompactionState`,
  `TokenUsage`, плюс `open_intents`, `usage_rows`, `lanes`, очередями
  live-control, живыми переопределениями model/thinking, опциональным
  `verification` и защёлками восстановления — и сохраняет его через
  `snapshot()` ↔ `resume_from_snapshot()`, так что любой под может возобновить
  запуск, начатый другим подом. Внедрение на этапе конструирования живёт на
  неизменяемом `QueryEngineConfig` (включая `run_mode`, `tool_preconditions` и
  опциональный `provider_chain`). См.
  [Цикл ReAct / оркестратор / query engine / состояние цикла](architecture.md#react-loop--orchestrator--query-engine--loop-state).

**`query()`** (`runtime/query.py`)
: Драйвер хода ReAct: `def query(engine) -> AsyncIterator[TurnEvent]`,
  **синхронная** функция, которая сбрасывает состояние хода и **возвращает**
  асинхронный итератор. Это не асинхронный генератор и она не сохраняет снимки
  начала и конца хода (это делает `QueryEngine.run()`). Каждый внутренний
  `yield` из `_query_raw` — контрольная точка проверки на остановку. Не
  реэкспортируется на верхнем уровне — импортируйте его из
  `protocore.runtime.query`. См.
  [Цикл ReAct / оркестратор / query engine / состояние цикла](architecture.md#react-loop--orchestrator--query-engine--loop-state).

## Три понятия состояния запуска (не смешивать)

Эти три имени **различны** и живут в разных слоях; их смешение — частая ошибка.

**`LoopState`** (`runtime/loop_state.py`, `StrEnum`)
: **Конечный автомат цикла в пределах хода**, удерживаемый на экземпляре
  `QueryEngine`, — живое текущее состояние движка. Семь состояний:
  `PENDING → RUNNING → {AWAITING | COMPACTING} → {COMPLETED | FAILED | CANCELLED}`.
  `assert_transition()` принудительно применяет таблицу допустимых переходов; у
  `TERMINAL_STATES` нет исходящих рёбер. См.
  [Цикл ReAct / оркестратор / query engine / состояние цикла](architecture.md#react-loop--orchestrator--query-engine--loop-state).

**`RunStatus`** (`contracts/types.py`, `StrEnum`)
: **Долговечный жизненный цикл запуска**, зеркалируемый в колонке Postgres
  `runs.status`: `queued | running | completed | partial | error | cancelled |
  incomplete | paused`. `partial` функционально терминально (цикл завершился, но
  накопил ошибки инструментов), отличается от `completed` и `error`. Это
  сохранённая запись, а не состояние цикла в памяти.

**`RunState`** (`contracts/types.py`, `BaseModel`)
: **Эфемерный «горячий» рабочий набор**, удерживаемый в хеше Redis `run:{id}` —
  `run_id`, `tenant_id`, текущий `RunStatus`, `current_turn`, счётчики токенов,
  `last_event_id`. Изменяемая модель, а не enum; отличается от долговечной записи
  `Run`.

## Конфигурация и константы

**`RuntimeConstants`** (`contracts/runtime_constants.py`)
: Единственный механизм для настраиваемых значений — **никаких inline-магических
  чисел**. Замороженный Pydantic-снимок (`ConfigDict(frozen=True,
  extra="forbid")`); каждая настройка — поле с безопасным значением по умолчанию,
  отдаваемое в разрезе тенанта через `RuntimeConstantsProvider`.
  `extra="forbid"` означает, что неизвестный ключ — ошибка валидации
  (отклоняется), а не молчаливое отбрасывание, поэтому **ядро и хост
  должны деплоиться парно**. См.
  [Система RuntimeConstants](architecture.md#runtimeconstants-system) и
  [`runtime-constants.md`](runtime-constants.md).

## Протоколы расширения

**`IMemory`** (`contracts/memory.py`, `Protocol`)
: Контракт для **типизированной, осведомлённой о scope, ранжируемой при
  извлечении памяти** фактов, которые агент усваивает и переиспользует (отличной
  от транскриптов сессий, блобов, поискового индекса и todo). Запись адресуется
  через `(tenant_id, scope, scope_key)`; наиболее изолированный scope по
  умолчанию — `session`. Ядро никогда не импортирует реализацию; хост
  предоставляет `PgMemoryStore`. По умолчанию выключено
  (`memory_enabled = False`). См.
  [IMemory](architecture.md#imemory-scoped-ftsbm25-idempotent-drift-guard-injection-scan-seam).

**`IWorkspace`** (`contracts/workspace.py`, `Protocol`)
: Контракт для **искомого, атомарного черновикового рабочего пространства в
  разрезе сессии/задачи** — агент один раз сбрасывает промежуточные данные и
  многократно перечитывает/ищет в них (рычаг стабильности «сбросил один раз /
  перечитал много раз»). Обеспечивает работу глаголов `read`/`write`/`find`/
  `search`. Подключается в хосте; флаг доступности по умолчанию включён
  (`workspace_enabled = True`). См.
  [IWorkspace + кэш дедупликации чтения](architecture.md#iworkspace--read-dedup-cache).

## Устойчивость и финализация

**`AdaptiveSafetyBand`** (`runtime/adaptive_safety_band.py`)
: Полоса в разрезе `(provider, model)`, которая **учится на дрейфе оценщика
  токенов** и вычитает откалиброванный запас из бюджета вывода на вызов, так что
  `prompt + max_tokens` остаётся под окном провайдера даже когда локальный
  оценщик ошибается (например, раздувание из-за кириллицы в JSON-escape). Когда
  полоса не подключена, поведение идентично состоянию до её появления. См.
  [Реестр попыток + адаптивная полоса безопасности](architecture.md#attempt-ledger--adaptive-safety-band).

**`AttemptLedger`** (`contracts/attempt_ledger.py`)
: Запись того, что (суб)агент **заявил**, что произведёт
  (`DeliverableDeclaration`), и что было **фактически проверено**
  (`VerificationRecord`), чтобы финализация могла принять честный исход. Его
  `LedgerOutcome` — нейтральный литерал (`completed | partial | failed |
  unknown`), а не enum бэкенда; `SelfReportedStatus` агента сохраняется, но не
  принимается на веру слепо. См.
  [Реестр попыток + адаптивная полоса безопасности](architecture.md#attempt-ledger--adaptive-safety-band).

Ворота финализации (`runtime/finalization_gate.py`,
`runtime/finalization_contract.py`)
: Страж терминального пути, закрывающий пробел финализации: (суб)агент, который
  записал видимый пользователю артефакт, но исчерпал итерации, не вызвав свой
  терминальный инструмент, иначе был бы оценён как «провалившийся». Ворота
  **проверяют** заявленные результаты — `verify_declared_deliverables(...)`
  собирает статистику по каждому через внедрённый `WorkspaceStatProtocol` — и
  **принимают решение** `FinalizationDecision` через `decide_finalization(ledger)`
  (шаблон success / partial / failed), которое использует финальный ход лидера.
  Все переключатели по умолчанию `False`. См.
  [Ворота финализации + контракт](architecture.md#finalization-gate--contract).

## Заземление и терминальные ответы

**Заземление / ссылки**
: Детерминированная, слепая к рубрике дисциплина: цитаты терминального `answer`
  должны быть **подмножеством того, что было фактически `read`**. Отслеживаемый
  для заземления `read` записывает свой путь как наблюдаемое свидетельство;
  `GROUNDING_TRACKED_TOOLS` (`contracts/lean_tool_surface.py`) — это frozenset
  `{read}` — `read_silent` возвращает идентичное содержимое, но не записывается.
  `normalize_ref(...)` (`contracts/references.py`) — чистая, идемпотентная
  проекция сравнения, которая сравнивает ссылки в канонической форме, так что
  расхождение между «плоским» и «брендированным» путём не становится ложным вето;
  она может только снять ложное вето, но никогда не добавить его. См.
  [Валидация терминального ответа + ссылки / заземление + нормализация полезной нагрузки](architecture.md#terminal-answer-validation--references--grounding--payload-normalize).

## Контекст, кэширование и уплотнение

**Точки разрыва кэша промптов** (`runtime/prompt_caching.py`)
: **Только подсказки** размещения для prefix-кэширования провайдера.
  `apply_system_and_3(...)` вычисляет стратегию `system_and_3`: не более четырёх
  `CacheBreakpoint` — system на индексе 0 плюс последние три не-system-сообщения.
  Ядро всегда отдаёт подсказки в `LLMRequest.extra["cache_breakpoints"]`; адаптер
  хост транслирует их в маркеры `cache_control` (kill-switch
  `prompt_cache_wire_enabled`, по умолчанию `True`), а адаптеры, не распознающие
  ключ, его игнорируют. См.
  [Управление контекстом и двухъярусное уплотнение](architecture.md#context-management--two-tier-compaction--session-memory--budgets--token-counting--prompt-caching--strip-thinking).

**Ярусы / слои уплотнения** (`runtime/context/compaction.py`)
: **Двухъярусный** каскад, удерживающий промпт под окном контекста провайдера на
  протяжении длинного запуска. В этом модуле нет яруса 3. **Ярус 1**
  (`run_tier1_truncation`) усекает / отправляет в блобы негабаритные результаты
  инструментов, заменяя тело заглушкой + ссылкой на блоб. **Ярус 2**
  (`run_tier2_summarisation`) заменяет целые старые не-system-ходы системной
  сводкой, сохраняя последние N ходов. Когда оба исчерпаны,
  `CompactionExhaustedError` переводит цикл в `FAILED`. Операторский `/compact`
  — отдельный путь `CompactCheckpoint` (`runtime/compact_checkpoint.py`,
  `compaction_manual_enabled` по умолчанию `False`). Межзапусковый fold живёт в
  `runtime/context/session_memory.py` (`fold_run`). Триггеры и соотношения
  управляются RC и выводятся в `runtime/context/budgets.py`. См.
  [Управление контекстом и двухъярусное уплотнение](architecture.md#context-management--two-tier-compaction--session-memory--budgets--token-counting--prompt-caching--strip-thinking).

## Намерения, usage ledger, дерево сессий, lanes, типизированные хуки, телеметрия

Эти шесть поверхностей **выключены по умолчанию**. Читайте живой `Field(...)`
default; не выводите, что поставка кода их включает.

**`IntentRecord`** (`runtime/intent.py`)
: Запись settlement на вызов инструмента (`operation_id`, зарезервированные
  result id, `replay` `never|safe`, `status` `open|settled|interrupted`). Когда
  `intent_settlement_enabled` включён, **каждый** диспетчеризуемый инструмент
  коммитит запись до `ToolDispatcher.dispatch` — не только мутирующие.
  `replay_policy_for` помечает имена из `intent_never_replay_tools` (по
  умолчанию `Write,Edit,Bash,Finalize,AppendFile`) как `never`; остальные —
  `safe`. Краш mid-never становится `interrupted` и не реплеится. Персистируется
  в `QueryEngine.open_intents`. См.
  [Intent, usage ledger, session tree, lanes](architecture.md#intent-usage-ledger-session-tree-lanes-typed-hooks-telemetry-live-control-run-work-budget).

**`UsageRow`** (`runtime/usage_ledger.py`)
: Одна строка append-only журнала (`seq`, `kind`, счётчики токенов, `success`,
  опциональный `operation_id`). Когда `usage_ledger_enabled` включён,
  `correctness_bind.commit_usage` пишет `inference` / `retry` / `compaction` /
  `abort` / `fail`. Строка вида **tool** пишется только на пути
  intent-settlement. Неудачная попытка плюс её retry — две строки.
  Персистируется в `QueryEngine.usage_rows`.

**`SessionBranch`** (`runtime/session_tree.py`)
: Форкнутая или клонированная копия пути истории (`fork_session` /
  `clone_session`), которая **не** мутирует источник. Гейтится
  `session_tree_enabled`; clone требует settled-источник;
  `session_tree_max_copy_messages` (по умолчанию 500) ограничивает копию.
  **Вызывается хостом** — цикл эти хелперы не вызывает.

**`Lane`** (`runtime/lanes.py`)
: Именованный курсор над общей историей. `main` всегда существует;
  дополнительные берут эксклюзивные блокировки (`create_lane` / `acquire_lane` /
  `release_lane`). Гейтится `lanes_enabled`; `lanes_max_per_session` (по
  умолчанию 4) включает main. **Вызывается хостом**; `QueryEngine.lanes`
  персистируется в снимке.

**`PUBLISHED_HOOKS`** (`runtime/typed_hooks.py`)
: Боевые типизированные имена хуков: `before_run`, `before_tool`, `after_tool`,
  `transform_context`, `before_compact`, `after_compact`. Диспатчатся через
  `HookRegistry`, когда включён `typed_hooks_enabled`. Отличны от 8 pluggy
  hookspec-ов и от `IHookManager`. `before_tool` / `after_tool` также требуют
  `intent_settlement_enabled` (они живут в этой ветке диспетчеризации).
  `transform_context` срабатывает, но его `rewrite` к истории не применяется.

**`Span`** (`runtime/telemetry.py`)
: Телеметрический span низкой кардинальности. Допустимые имена: `run` / `turn` /
  `step` / `tool` / `compact` / `hook`. Гейтится `telemetry_spans_enabled`.
  `is_prometheus_safe_label` отказывается принимать `session_id` / `lane_id` /
  `operation_id` / `run_id` как ключи меток. `mark_recovery` помечает span,
  когда возобновляется прерванное намерение. Живёт на `engine.spans`
  (внутрипроцессно, **не** в снимке).

**`IProviderChain`** (`contracts/llm.py`)
: Упорядоченные оставшиеся провайдеры плюс односторонний курсор `advance()`.
  Внедряется на `QueryEngine(..., provider_chain=...)`, чтобы mid-stream сбой
  провайдера мог перейти на следующую ступень, не отзывая уже стримленные дельты.

## Файлы скиллов

**`SkillFileRef`** (`contracts/skills.py`)
: Один файл в многофайловом skill-бандле: bundle-relative `path`, `size_bytes`,
  `mime_type`, `content_hash` (hex SHA-256 в нижнем регистре). У каждого бандла
  есть как минимум каноническая строка `SKILL_ENTRY_PATH` (`SKILL.md`, MIME
  `text/markdown`). Байты приходят из `ISkillStore.load_file`; цикл ядра
  **не** вызывает `list_files` / `load_file` (это для хостов, которые отдают
  вспомогательные файлы). Рендер каталога использует `store.list` и выдаёт
  формы вызова `Skill(skill="{name}")`, а не пути файлов. Чтения store ключуются
  по `QueryEngineConfig.account_id`, не по `tenant_id`. См.
  [Маршрутизация / серфейсинг скиллов](architecture.md#skills-routing--surfacing).

## Поверхность инструментов

**Lean-поверхность инструментов (7 глаголов)** (`contracts/lean_tool_surface.py`)
: Небольшой универсальный пул инструментов, обращённый к агенту, чтобы способная
  модель собирала операции вручную, а не заполняла дюжину узкоспециальных схем.
  Ровно **семь канонических глаголов** (`LEAN_TOOL_NAMES`): `exec`, `read`,
  `read_silent`, `write`, `find`, `search`, `answer`. `exec` — это запускатель
  зарегистрированных бинарников (`{path, args, stdin}`), явно **не** `/bin/sh`.
  Ядро владеет только контрактами и именами; хост привязывает каждое имя к
  конкретному бэкенду. Выбор профиля управляется RC (`tool_surface_profile`, по
  умолчанию `"legacy"`). См.
  [Lean-поверхность инструментов](architecture.md#lean-tool-surface) и
  [`tools.md`](tools.md).

> Перевод английского оригинала `docs/glossary.md`. При изменении оригинала обновите перевод.
