# Getting Started

> Audience: an engineer installing the **pure core** (`protocore/`) and running
> its ReAct loop for the first time.

This page gets you from a clean checkout to a working agent turn. For the wider
picture, read [`architecture.md`](architecture.md) (the deep reference) and the
[doc hub](index.md).

---

## Requirements

- **Python 3.12+** — the core is a Python ≥ 3.12 library.
- **[`uv`](https://docs.astral.sh/uv/)** — the project's package/venv manager.

The core is a small, dependency-light library. Its only runtime dependencies are
`pydantic`, `pluggy`, `typing-extensions`, and `jinja2` (declared in
`pyproject.toml`). It has **no** database driver, HTTP server, or LLM SDK — those
live across the adapter boundary (see the adapter-driven section below).

---

## Install

From the repository root:

```bash
uv sync --extra dev
```

`uv sync` creates the virtualenv and installs the locked dependency set; the
`--extra dev` group adds the test/lint toolchain (`pytest`, `pytest-asyncio`,
`pytest-cov`, `mypy`, `ruff`). Run anything in the environment with `uv run`:

```bash
uv run pytest .            # tests
uv run ruff check .        # lint
uv run mypy protocore      # type-check
```

See [`testing.md`](testing.md) for the full test/coverage stance.

---

## The core is adapter-driven (read this before the quickstart)

The core ships **contracts**, not concrete backends. Everything outside-facing —
the LLM, persistence, the event stream, hooks, the tool implementations — is a
`Protocol` that *someone else* implements. In particular, **the core ships no
concrete `ILLMProvider`**: there is no built-in model client, no API key
handling, no network code.

So to drive a turn you must inject implementations of the engine's dependencies.
There are two places to get them:

- **Real, production-grade adapters live in the host distribution** — a
  universal LiteLLM/OpenAI-compatible `ILLMProvider` (OpenRouter / vLLM / OpenAI),
  Postgres-backed stores, a Redis event stream, the hook dispatcher, and the
  sandbox-backed tool implementations. Wire those up when you want a real model
  answering real prompts.
- **In-memory adapters ship inside the core** at
  `protocore.tests_support.adapters` — `InMemoryLLMProvider` (scripted, offline),
  `InMemoryToolRegistry`, `InMemoryEventStream`, `InMemoryHookManager`,
  `InMemorySkillStore`, `InMemoryBlobStore`. They implement the same Protocols the
  real adapters do, so they are the right way to run a self-contained **smoke run**
  with no external services. The quickstart below uses them.

The full list of protocols and which repo provides them is in
[`contracts.md`](contracts.md); the extension-seam decision guide is in
[`extending.md`](extending.md).

---

## Quickstart — drive one turn (offline smoke run)

The runtime is split into two pieces:

- **`QueryEngine`** (`protocore.runtime.query_engine`) — one instance per active
  run. It owns the mutable per-conversation state (history, the `LoopState`
  machine, compaction state, token usage, plus snapshot-persisted intents,
  usage rows, lanes, live-control queues, and recovery latches) and the injected
  adapters.
- **`query(engine)`** (`protocore.runtime.query`) — a **sync** entry that calls
  `_reset_per_turn_state()` immediately and **returns** an async iterator of
  `TurnEvent`s. It is deliberately not an async generator: the reset happens at
  the call, not on the first `__anext__`. `query()` does not persist turn-start
  or turn-end snapshots — `QueryEngine.run()` does.

> **Import location matters.** `QueryEngine`, `QueryEngineConfig`, and `query`
> are **not** re-exported at the top level — import them from
> `protocore.runtime.query_engine` / `protocore.runtime.query`. The contract
> types (`Message`, `TextBlock`, `StopReason`, `RuntimeConstants`, …) *are*
> top-level re-exports from `protocore`.

`query(engine)` reads the latest user message off `engine.history` — it does not
append for you — so seed the history with one user `Message` before iterating.

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

Running it prints the per-turn event stream and ends in `LoopState.COMPLETED`:

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

### What just happened

`query(engine)` reset per-turn state, then `_query_raw` ran the turn lifecycle
once: a stop check, optional intent recovery / typed `before_run` / `/compact`
(all default-off, so inert on this smoke run), a compaction check, the
`user_prompt_submit` hook, context assembly, the `run_mode` strategy step
(`DirectStrategy` here), then it streamed a single assistant message from the
injected provider. Because the scripted reply carried no tool calls and
`stop_reason=end_turn`, the loop reached `message_stop` and transitioned the
engine to `COMPLETED`. Each `TurnEvent` is exactly what the host executor
forwards to clients over SSE.

> `QueryEngine` also exposes a convenience driver, `engine.run(message)` — an
> **async generator** that appends `message` to history (or continues against an
> existing user-final history when `message` is `None`), stamps the run clock,
> persists a turn-start snapshot, binds `_current_turn_task` so `stop()` can
> hard-cancel, then iterates the private `_query_raw` generator (not
> `query(self)`). Each yielded event is projected through the public delivery
> boundary; a turn-end snapshot is persisted in `finally`. The quickstart calls
> `query(engine)` directly so the two halves of the runtime — the engine (state)
> and `query` (behaviour) — are explicit. A caller that uses `query()` inherits
> the reset but **not** the snapshot or cancel-handle obligations of `run()`.

---

## Going further — a real model

To answer real prompts, keep the same shape but inject the **the host**
adapters instead of the in-memory ones:

- Replace `InMemoryLLMProvider` with the host LiteLLM/OpenAI-compatible
  `ILLMProvider` and set `config.model_name` to a model the provider serves.
- Replace the in-memory stores with the Postgres/Redis-backed adapters.
- Register concrete tools on the registry (the lean verbs `exec` / `read` /
  `read_silent` / `write` / `find` / `search` / `answer`; see
  [`tools.md`](tools.md)).
- Tune behaviour through the `RuntimeConstants` snapshot you pass as `config.rc`
  rather than editing the loop; see [`runtime-constants.md`](runtime-constants.md).

The engine and `query()` API are identical — only the injected adapters change.
Because the core never imports upward, it cannot construct those adapters itself:
that wiring lives in the host. The full per-protocol breakdown is
in [`contracts.md`](contracts.md), and the rules for adding your own behaviour
(implement a protocol, add a hook, flip an RC, add a prompt section) are in
[`extending.md`](extending.md).

---

## Next steps

- [`index.md`](index.md) — the documentation hub and reading order.
- [`architecture.md`](architecture.md) — the deep reference: the loop, every
  subsystem, and the diagrams.
- [`contracts.md`](contracts.md) — the protocol boundary and the core type
  system.
- [`tools.md`](tools.md) — the lean tool surface and the `@tool` decorator.
- [`runtime-constants.md`](runtime-constants.md) — how tunables work.
- [`extending.md`](extending.md) — where to plug in your own behaviour.
