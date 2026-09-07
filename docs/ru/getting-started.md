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
runtime-зависимости — `pydantic`, `typing-extensions` и `jinja2`
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
тестирования/линтинга (`pytest`, `pytest-asyncio`, `pytest-cov`,
`pytest-xdist`, `mypy`, `ruff`, `bandit`). Запускайте что угодно в этом
окружении через `uv run`:

```bash
uv run pytest .            # tests
uv run ruff check .        # lint
uv run mypy --strict       # проверка типов (никогда с путём — см. testing.md)
```

Есть ещё два extra, и ни один из них не нужен для разработки самого ядра:

- `protocore[testing]` — то, что ставит **хост**, чтобы прогнать
  conformance-наборы из `protocore.conformance` против собственных адаптеров.
  Сознательно узкий: тест-раннер и ничего больше, чтобы хост не наследовал ради
  них линтер и типизацию ядра.
- `protocore[native]` — опциональный нативный оценщик токенов, отдельный
  дистрибутив, собираемый из `native/`. Ядро остаётся чисто питоновым и выбирает
  его, только если он импортируется, так что установка этого extra меняет
  скорость и ничего больше. Переменная окружения `PROTOCORE_DISABLE_NATIVE`
  принудительно оставляет чисто питоновый путь даже при установленном
  расширении — так отличают проблему скорости от проблемы корректности.

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
- **драйвы** — асинхронный итератор `TurnEvent` по одному ходу цикла. Их два, и
  какой нужен — зависит от того, начинается прогон или продолжается:
  - `engine.run(message)` открывает ход на живом движке. Он добавляет `message`
    в историю (передайте `None`, чтобы продолжить по истории, уже
    заканчивающейся пользовательским сообщением), ставит часы запуска и
    сохраняет снимок начала хода.
  - `resume(engine, snapshot, ...)` (`protocore.runtime`) поднимает прогон из
    снимка — при необходимости в другом процессе — и ведёт его. См.
    [Подъём сохранённого прогона](#Подъём-сохранённого-прогона) ниже.

  Оба привязывают ведущую задачу, поэтому `engine.stop()` может жёстко отменить
  драйв, вставший на `await`, и оба сохраняют снимок на выходе, каким бы этот
  выход ни был.

> **Место импорта имеет значение.** `QueryEngine` и `QueryEngineConfig` **не**
> реэкспортируются на верхнем уровне — импортируйте их из
> `protocore.runtime.query_engine`; `resume` и `resume_approved_tool` — из
> `protocore.runtime`. Типы контрактов
> (`Message`, `TextBlock`, `StopReason`, `LoopConstants`, …) *являются*
> реэкспортами верхнего уровня из `protocore`.

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
    #    model_name are required; `rc` is the LoopConstants snapshot
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

    # 4. Drive ONE turn on the user message that opens it. Each yielded
    #    TurnEvent is a streaming event (state changes, message/content-block
    #    deltas, tool calls, …).
    opening_message = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="Say hello.")],
    )
    async for event in engine.run(opening_message):
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

`engine.run(message)` добавил сообщение, поставил часы запуска, сохранил снимок
начала хода и сбросил состояние хода; затем драйвер один раз прогнал
жизненный цикл хода: проверку остановки, опциональное восстановление намерений,
координату `run_start` и `/compact` (все выключены по умолчанию, поэтому на
этом smoke-прогоне инертны), проверку уплотнения, hook `user_prompt_submit`,
сборку контекста, шаг стратегии `run_mode` (здесь `DirectStrategy`), а затем
стримил единственное сообщение ассистента от внедрённого провайдера. Поскольку
сценарный ответ не нёс вызовов инструментов и имел `stop_reason=end_turn`, цикл
дошёл до `message_stop` и перевёл движок в `COMPLETED`. Каждый `TurnEvent` — это
ровно то, что host-executor пересылает клиентам по SSE.

> Каждый выданный event проецируется через публичную границу доставки, прежде
> чем вы его увидите, а снимок конца хода сохраняется в `finally` — поэтому ход,
> который упал с исключением, и ход, который оператор отменил, всё равно
> оставляют после себя точку подхвата.

---

## Подъём сохранённого прогона

Прогон не обязан заканчиваться в том процессе, где начался. `engine.snapshot()`
возвращает полное состояние прогона обычным словарём; `resume()` принимает этот
словарь обратно и ведёт то, что должно случиться дальше:

```python
from protocore.runtime import resume

async for event in resume(engine, snapshot):
    print(event.type)
```

Снимок восстанавливается строго. Его схема, режим доставки и привязка
идентичности — прогон, тенант, сессия и родословная субагента — проверяются до
первой мутации, поэтому снимок чужого прогона отвергается, движок остаётся
нетронутым, и не ведётся ничего.

Какой драйв выберет `resume()`, следует из того, что остановило прогон, и
говорите об этом вы:

| Что остановило прогон | Вызов |
| --- | --- |
| Ничего особенного — ход умер на полпути | `resume(engine, snapshot)` |
| Он ждал ввода, который теперь пришёл | `resume(engine, snapshot, message=answer)` |
| Он припарковал один или несколько вызовов, и на каждый теперь есть ответ | `resume(engine, snapshot, resolutions={...})` |
| Он ждал решения по вызову инструмента, и решение — *одобрить* | `resume(engine, snapshot, approved_tool_call=call)` |
| Отвечать на припаркованные вызовы никто не будет | `resume(engine, snapshot, abandon_approval=True)` |

`resolutions` — общая форма и единственная, которой выражается батч: карта из id
прерывания в `InterruptResolution`, так что три припаркованных вместе вызова
одобряются, отклоняются и исправляются ОДНИМ подъёмом вместо трёх раундов
«стоп — вопрос — подъём», и их результаты ложатся в том порядке, в котором их
просила модель. Карта проверяется целиком до того, как что-либо исполнится:
карта, называющая прерывание, которого прогон не ждёт, отвечающая на одобрение
ответом или оставляющая одно из них без решения, отвергается, не сделав ничего,
— а `allow_partial_resolution=True` это то, чем вызывающая сторона говорит, что
остальное намеренно остаётся припаркованным. То, чего ждёт прогон, читается, а
не выводится: `PendingInterrupt` несёт вид (`approval`, `question`,
`external_call`), вызов и то, что показывают человеку, а прогон выпускает
`interrupt_parked` в момент остановки.

Форма с одобренным вызовом исполняет ровно этот вызов — сверенный с durable
ожидающим вызовом, а не принятый на веру, и ровно один раз, даже если подъём
доставлен дважды, — и останавливается, когда его результат попадает в историю.
Ответ на этот результат — новый ход, который вы открываете сами, когда он вам
всё ещё нужен.

---

## Дальше — реальная модель

Чтобы отвечать на реальные промпты, сохраните ту же форму, но внедрите
**хост**-адаптеры вместо in-memory:

- Замените `InMemoryLLMProvider` на OpenAI-совместимый `ILLMProvider` хоста и
  задайте `config.model_name` равным модели, которую отдаёт провайдер.
- Замените in-memory хранилища на долговечные адаптеры хоста и докажите каждый
  соответствующим набором из `protocore.conformance` прежде, чем это за вас
  сделает прогон (см. [`testing.md`](testing.md)).
- Зарегистрируйте в реестре конкретные инструменты, которые привязывает ваш
  бэкенд; см. [`tools.md`](tools.md).
- Настраивайте поведение через снимок `LoopConstants`, который вы передаёте
  как `config.rc`, а не редактируя цикл; см.
  [`runtime-constants.md`](runtime-constants.md).

API движка и драйвов идентичны — меняются только внедрённые адаптеры. Поскольку
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
