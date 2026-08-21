# Первые шаги

> Аудитория: инженер, устанавливающий **чистое ядро** (`protocore/`) и впервые
> запускающий его ReAct-цикл.

Эта страница проведёт вас от чистого checkout до рабочего хода агента. Для более
широкой картины читайте [`architecture.md`](architecture.md) (глубокий
справочник) и [хаб документации](index.md).

---

## Требования

- **Python 3.12+** — ядро является библиотекой для Python ≥ 3.12.
- **[`uv`](https://docs.astral.sh/uv/)** — менеджер пакетов/виртуальных окружений
  проекта.

Ядро — это небольшая библиотека с минимумом зависимостей. Её единственные
runtime-зависимости — `pydantic`, `pluggy`, `typing-extensions` и `jinja2`
(объявлены в `pyproject.toml`). У неё **нет** ни драйвера базы данных, ни
HTTP-сервера, ни LLM SDK — всё это живёт по ту сторону границы адаптеров (см.
раздел об управлении адаптерами ниже).

---

## Установка

Из корня репозитория:

```bash
uv sync --extra dev
```

`uv sync` создаёт виртуальное окружение и устанавливает зафиксированный набор
зависимостей; группа `--extra dev` добавляет инструментарий для
тестирования/линтинга (`pytest`, `pytest-asyncio`, `pytest-cov`, `mypy`,
`ruff`). Запускайте что угодно в этом окружении через `uv run`:

```bash
uv run pytest .            # tests
uv run ruff check .        # lint
uv run mypy protocore      # type-check
```

Полную позицию по тестам/покрытию см. в [`testing.md`](testing.md).

---

## Ядро управляется адаптерами (прочитайте перед быстрым стартом)

Ядро поставляет **контракты**, а не конкретные бэкенды. Всё, что обращено
наружу — LLM, персистентность, поток событий, hooks, реализации инструментов —
это `Protocol`, который реализует *кто-то другой*. В частности, **ядро не
поставляет ни одного конкретного `ILLMProvider`**: в нём нет ни встроенного
клиента модели, ни обработки API-ключей, ни сетевого кода.

Поэтому, чтобы запустить ход, вы должны внедрить реализации зависимостей движка.
Их можно взять в двух местах:

- **Реальные, production-grade адаптеры живут в хост-дистрибутиве** —
  универсальный LiteLLM/OpenAI-совместимый `ILLMProvider` (OpenRouter / vLLM /
  OpenAI), хранилища на Postgres, поток событий на Redis, диспетчер hooks и
  реализации инструментов на базе sandbox. Подключайте их, когда хотите, чтобы
  реальная модель отвечала на реальные промпты.
- **In-memory адаптеры поставляются внутри ядра** по пути
  `protocore.tests_support.adapters` — `InMemoryLLMProvider` (по сценарию,
  офлайн), `InMemoryToolRegistry`, `InMemoryEventStream`, `InMemoryHookManager`,
  `InMemorySkillStore`, `InMemoryBlobStore`. Они реализуют те же `Protocol`-ы,
  что и реальные адаптеры, поэтому именно так правильно запускать
  самодостаточный **smoke-прогон** без внешних сервисов. Быстрый старт ниже
  использует именно их.

Полный список протоколов и того, какой репозиторий их предоставляет, — в
[`contracts.md`](contracts.md); руководство по выбору точки расширения — в
[`extending.md`](extending.md).

---

## Быстрый старт — запуск одного хода (офлайн smoke-прогон)

Рантайм разделён на две части:

- **`QueryEngine`** (`protocore.runtime.query_engine`) — один экземпляр на
  активный прогон. Он владеет изменяемым состоянием на уровне диалога (история,
  машина состояний `LoopState`, состояние уплотнения, расход токенов, плюс
  сохраняемые в снимке намерения, строки usage, lanes, очереди live-control и
  защёлки восстановления) и внедрёнными адаптерами.
- **`query(engine)`** (`protocore.runtime.query`) — **синхронная** точка входа,
  которая сразу вызывает `_reset_per_turn_state()` и **возвращает** асинхронный
  итератор `TurnEvent`. Это сознательно не асинхронный генератор: сброс
  происходит в момент вызова, а не на первом `__anext__`. `query()` не
  сохраняет снимки начала и конца хода — это делает `QueryEngine.run()`.

> **Место импорта имеет значение.** `QueryEngine`, `QueryEngineConfig` и `query`
> **не** реэкспортируются на верхнем уровне — импортируйте их из
> `protocore.runtime.query_engine` / `protocore.runtime.query`. Типы контрактов
> (`Message`, `TextBlock`, `StopReason`, `RuntimeConstants`, …) *являются*
> реэкспортами верхнего уровня из `protocore`.

`query(engine)` читает последнее пользовательское сообщение из `engine.history`
— он не добавляет его за вас — поэтому перед итерацией засейте историю одним
пользовательским `Message`.

```python
import asyncio

from protocore import (
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    default_runtime_constants,
)
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.runtime.query import query

# In-core, dependency-free adapters for an offline smoke run.
# Swap these for your host's adapters to reach a real model.
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemorySkillStore,
    InMemoryToolRegistry,
)


async def main() -> None:
    # 1. A scripted LLM provider (the core ships NO real ILLMProvider).
    #    Queue one assistant reply that ends the turn cleanly.
    llm = InMemoryLLMProvider()
    llm.queue_response(
        text="Hello from the Protocore smoke run.",
        stop_reason=StopReason.end_turn,
    )

    # 2. The immutable injection surface. run_id / tenant_id / session_id /
    #    model_name are required; `rc` is the RuntimeConstants snapshot
    #    (default-safe; see runtime-constants.md).
    config = QueryEngineConfig(
        run_id="run-1",
        tenant_id="default",
        session_id="sess-1",
        model_name="smoke-model",
        rc=default_runtime_constants(),
    )

    # 3. Construct the engine, injecting every adapter (all keyword-only).
    engine = QueryEngine(
        config=config,
        llm_provider=llm,
        tool_registry=InMemoryToolRegistry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )

    # 4. Seed history with the user turn `query()` will answer.
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="Say hello.")],
        )
    )

    # 5. Drive ONE turn. Each yielded TurnEvent is a streaming event
    #    (state changes, message/content-block deltas, tool calls, …).
    async for event in query(engine):
        print(event.type)

    print("final state:", engine.state)


asyncio.run(main())
```

При запуске печатается поток событий хода, и он завершается в
`LoopState.COMPLETED`:

```text
state_changed
hook_fired
message_start
tool_surface_advertised
content_block_start
content_block_delta
content_block_stop
message_stop
final state: completed
```

### Что только что произошло

`query(engine)` сбросил состояние хода, затем `_query_raw` один раз прогнал
жизненный цикл хода: проверку остановки, опциональное восстановление намерений /
типизированный `before_run` / `/compact` (все выключены по умолчанию, поэтому на
этом smoke-прогоне инертны), проверку уплотнения, hook `user_prompt_submit`,
сборку контекста, шаг стратегии `run_mode` (здесь `DirectStrategy`), а затем
стримил единственное сообщение ассистента от внедрённого провайдера. Поскольку
сценарный ответ не нёс вызовов инструментов и имел `stop_reason=end_turn`, цикл
дошёл до `message_stop` и перевёл движок в `COMPLETED`. Каждый `TurnEvent` — это
ровно то, что host-executor пересылает клиентам по SSE.

> `QueryEngine` также предоставляет удобный драйвер `engine.run(message)` —
> **асинхронный генератор**, который добавляет `message` в историю (или
> продолжает по уже существующей истории, заканчивающейся пользовательским
> сообщением, когда `message` равен `None`), ставит часы запуска, сохраняет
> снимок начала хода, привязывает `_current_turn_task`, чтобы `stop()` мог
> жёстко отменить ход, затем итерирует приватный генератор `_query_raw` (не
> `query(self)`). Каждый выданный event проецируется через публичную границу
> доставки; снимок конца хода сохраняется в `finally`. Быстрый старт вызывает
> `query(engine)` напрямую, чтобы две половины рантайма — движок (состояние) и
> `query` (поведение) — были явными. Вызывающая сторона, которая использует
> `query()`, наследует сброс, но **не** обязательства снимка и cancel-handle,
> которые несёт `run()`.

---

## Дальше — реальная модель

Чтобы отвечать на реальные промпты, сохраните ту же форму, но внедрите
**хост**-адаптеры вместо in-memory:

- Замените `InMemoryLLMProvider` на host-овский LiteLLM/OpenAI-совместимый
  `ILLMProvider` и задайте `config.model_name` равным модели, которую отдаёт
  провайдер.
- Замените in-memory хранилища на адаптеры на базе Postgres/Redis.
- Зарегистрируйте конкретные инструменты в реестре (лаконичные глаголы `exec` /
  `read` / `read_silent` / `write` / `find` / `search` / `answer`; см.
  [`tools.md`](tools.md)).
- Настраивайте поведение через снимок `RuntimeConstants`, который вы передаёте
  как `config.rc`, а не редактируя цикл; см.
  [`runtime-constants.md`](runtime-constants.md).

API движка и `query()` идентичны — меняются только внедрённые адаптеры. Поскольку
ядро никогда не импортирует вверх, оно не может сконструировать эти адаптеры
само: эта проводка живёт в хосте. Полная разбивка по каждому
протоколу — в [`contracts.md`](contracts.md), а правила добавления собственного
поведения (реализовать протокол, добавить hook, переключить RC, добавить секцию
промпта) — в [`extending.md`](extending.md).

---

## Следующие шаги

- [`index.md`](index.md) — хаб документации и порядок чтения.
- [`architecture.md`](architecture.md) — глубокий справочник: цикл, каждая
  подсистема и диаграммы.
- [`contracts.md`](contracts.md) — граница протоколов и система типов ядра.
- [`tools.md`](tools.md) — лаконичная поверхность инструментов и декоратор
  `@tool`.
- [`runtime-constants.md`](runtime-constants.md) — как работают настройки.
- [`extending.md`](extending.md) — куда подключать собственное поведение.

> Перевод английского оригинала `docs/getting-started.md` (коммит `54b6543`). При изменении оригинала обновите перевод.
