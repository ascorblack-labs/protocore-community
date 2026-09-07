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
`pydantic`, `typing-extensions`, and `jinja2` (declared in
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
`pytest-cov`, `pytest-xdist`, `mypy`, `ruff`, `bandit`). Run anything in the
environment with `uv run`:

```bash
uv run pytest .            # tests
uv run ruff check .        # lint
uv run mypy --strict       # type-check (never with a path — see testing.md)
```

Two other extras exist, and neither is needed to develop the core:

- `protocore[testing]` — what a **host** installs to run the conformance suites
  in `protocore.conformance` against its own adapters. Deliberately narrow: a
  test runner and nothing else, so a host does not inherit the core's linting
  and typing toolchain to use them.
- `protocore[native]` — the optional native token estimator, a separate
  distribution built from `native/`. The core stays pure Python and selects it
  only when it is importable, so installing this extra changes speed and
  nothing else. Setting `PROTOCORE_DISABLE_NATIVE` in the environment forces
  the pure-Python path even when the extension is installed — which is how you
  tell a speed problem from a correctness one.

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
- **the drives** — an async iterator of `TurnEvent`s over one turn of the loop.
  There are two, and which one you want depends on whether the run is starting
  or continuing:
  - `engine.run(message)` opens a turn on a live engine. It appends `message`
    to history (pass `None` to continue against a history that already ends in
    a user message), stamps the run clock and persists a turn-start snapshot.
  - `resume(engine, snapshot, ...)` (`protocore.runtime`) picks a run back up
    from a snapshot — on a different process if need be — and drives it. See
    [Resuming a stored run](#resuming-a-stored-run) below.

  Both bind the driving task, so `engine.stop()` can hard-cancel a drive parked
  in an `await`, and both persist a snapshot on the way out however they exit.

> **Import location matters.** `QueryEngine` and `QueryEngineConfig` are **not**
> re-exported at the top level — import them from
> `protocore.runtime.query_engine`; `resume` and `resume_approved_tool` come
> from `protocore.runtime`. The contract types (`Message`, `TextBlock`,
> `StopReason`, `LoopConstants`, …) *are* top-level re-exports from
> `protocore`.

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

`engine.run(message)` appended the message, stamped the run clock, persisted a
turn-start snapshot and reset per-turn state; then the turn driver ran the
lifecycle once: a stop check, optional intent recovery, the `run_start`
coordinate and `/compact`
(all default-off, so inert on this smoke run), a compaction check, the
`user_prompt_submit` hook, context assembly, the `run_mode` strategy step
(`DirectStrategy` here), then it streamed a single assistant message from the
injected provider. Because the scripted reply carried no tool calls and
`stop_reason=end_turn`, the loop reached `message_stop` and transitioned the
engine to `COMPLETED`. Each `TurnEvent` is exactly what the host executor
forwards to clients over SSE.

> Every yielded event is projected through the public delivery boundary before
> you see it, and the turn-end snapshot is persisted in `finally` — so a turn
> that raises, or one an operator cancels, still leaves a pickup point behind.

---

## Resuming a stored run

A run does not have to finish in the process that started it. `engine.snapshot()`
returns the full run state as a plain dict; `resume()` takes that dict back and
drives whatever comes next:

```python
from protocore.runtime import resume

async for event in resume(engine, snapshot):
    print(event.type)
```

The snapshot is restored strictly. Its schema, its delivery mode and its
identity binding — run, tenant, session and the subagent lineage — are all
checked before the first mutation, so a snapshot belonging to a different run is
refused with the engine untouched and nothing is driven.

Which drive `resume()` picks follows from what stopped the run, and you say
which that was:

| What stopped the run | Call |
| --- | --- |
| Nothing in particular — the turn died mid-flight | `resume(engine, snapshot)` |
| It was waiting for input that has now arrived | `resume(engine, snapshot, message=answer)` |
| It parked one or more calls, and every one now has an answer | `resume(engine, snapshot, resolutions={...})` |
| It was waiting for a decision on a tool call, and the answer is *approve* | `resume(engine, snapshot, approved_tool_call=call)` |
| Nobody is going to answer the parked calls | `resume(engine, snapshot, abandon_approval=True)` |

`resolutions` is the general form and the only one that can answer a batch: a
map from interrupt id to `InterruptResolution`, so three calls parked together
are approved, denied and corrected in ONE resume instead of three rounds of
stop-ask-resume, and their results land in the order the model asked for them.
The map is checked in full before anything runs — a map naming an interrupt the
run is not waiting on, answering an approval with an answer, or leaving one
undecided is refused with nothing done — and `allow_partial_resolution=True` is
how a caller says it means to leave the rest parked. What a run is waiting for
is readable rather than inferred: `PendingInterrupt` carries the kind
(`approval`, `question`, `external_call`), the call and what the person is being
shown, and the run emits `interrupt_parked` at the moment it stops.

The approved-call form executes that one call — verified against the durable
pending call rather than taken on trust, and exactly once even if the resume is
delivered twice — and stops when its result lands in history. Answering that
result is a fresh turn, which you open yourself when you still want it.

---

## Going further — a real model

To answer real prompts, keep the same shape but inject the **the host**
adapters instead of the in-memory ones:

- Replace `InMemoryLLMProvider` with the host's OpenAI-compatible
  `ILLMProvider` and set `config.model_name` to a model the provider serves.
- Replace the in-memory stores with the host's durable adapters, and prove each
  of them with the matching suite from `protocore.conformance` before a run
  does it for you (see [`testing.md`](testing.md)).
- Register the concrete tools your backend binds on the registry; see
  [`tools.md`](tools.md).
- Tune behaviour through the `LoopConstants` snapshot you pass as `config.rc`
  rather than editing the loop; see [`runtime-constants.md`](runtime-constants.md).

The engine and drive API are identical — only the injected adapters change.
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
- [`tools.md`](tools.md) — the tool surface and the `@tool` decorator.
- [`runtime-constants.md`](runtime-constants.md) — how tunables work.
- [`extending.md`](extending.md) — where to plug in your own behaviour.
