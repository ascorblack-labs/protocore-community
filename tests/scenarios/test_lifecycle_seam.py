"""The lifecycle seam, seen from outside the loop.

A host places registrations on coordinates and drives the run through the same
public entry point it uses in production. Nothing here reads a private
attribute of the engine: what a registration did is visible in the tool's own
record of being called, in the requests the provider received, and in the
events the caller iterated.
"""
from __future__ import annotations

from typing import Any

from protocore.contracts.middleware import (
    LifecycleContext,
    LifecycleDecision,
    LifecycleVerdict,
    RegistrationKind,
)
from protocore.contracts.types import HookEvent
from protocore.hooks import HookManager
from protocore.runtime.events import EventType

from .conftest import ScenarioFactory, ScriptedTool, default_rc


def _lifecycle_rc(**overrides: Any) -> Any:
    return default_rc(typed_hooks_enabled=True, **overrides)


async def test_a_decide_registration_denies_a_tool_and_the_tool_never_runs(
    scenario: ScenarioFactory,
) -> None:
    """The run continues; the call it was denied does not happen."""
    registry = HookManager()
    registry.register(
        HookEvent.pre_tool_use,
        RegistrationKind.decide,
        lambda ctx: (
            LifecycleDecision(verdict=LifecycleVerdict.deny, reason="not this one")
            if ctx.payload["tool_name"] == "Note"
            else LifecycleDecision()
        ),
        owner="policy",
    )
    tool = ScriptedTool(tool_name="Note", content="written")
    run = scenario(tools=[tool], rc=_lifecycle_rc(), lifecycle_hooks=registry)
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Note", tool_input={"text": "x"}
    )
    run.llm.queue_response(text="done")

    await run.run("use the tool")

    assert tool.invocations == []
    denied = [
        evt
        for evt in run.events_of(EventType.HOOK_FIRED)
        if evt.payload.get("hook") == HookEvent.pre_tool_use.value
    ]
    assert denied and denied[0].payload["decision"] == "deny"
    assert denied[0].payload["decided_by"] == "policy"


async def test_a_transform_registration_changes_what_the_model_is_shown(
    scenario: ScenarioFactory,
) -> None:
    """A rewrite that is returned is a rewrite that is applied."""
    registry = HookManager()

    def _rewrite(ctx: LifecycleContext) -> dict[str, Any]:
        return {
            **ctx.payload,
            "system_prompt_sections": ["THE SECTION THE HOST SUBSTITUTED"],
        }

    registry.register(
        HookEvent.context_transform,
        RegistrationKind.transform,
        _rewrite,
        owner="rewriter",
    )
    run = scenario(rc=_lifecycle_rc(), lifecycle_hooks=registry)
    run.llm.queue_response(text="ok")

    await run.run("hello")

    assert any(
        "THE SECTION THE HOST SUBSTITUTED" in text for text in run.request_texts(0)
    )


async def test_a_transform_that_returns_nothing_leaves_the_context_alone(
    scenario: ScenarioFactory,
) -> None:
    registry = HookManager()
    registry.register(
        HookEvent.context_transform,
        RegistrationKind.transform,
        lambda _ctx: None,
        owner="quiet",
    )
    run = scenario(rc=_lifecycle_rc(), lifecycle_hooks=registry)
    run.llm.queue_response(text="ok")

    await run.run("hello")

    assert "ok" in "".join(run.history_texts())


async def test_an_observing_registration_that_raises_does_not_touch_the_run(
    scenario: ScenarioFactory,
) -> None:
    """The audit sink is down; the run neither stops nor changes its answer."""
    registry = HookManager()

    def _explode(_ctx: LifecycleContext) -> None:
        raise RuntimeError("audit sink unreachable")

    for point in (HookEvent.run_start, HookEvent.pre_tool_use, HookEvent.post_tool_use):
        registry.register(point, RegistrationKind.observe, _explode, owner="audit")
    tool = ScriptedTool(tool_name="Note", content="written")
    run = scenario(tools=[tool], rc=_lifecycle_rc(), lifecycle_hooks=registry)
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Note", tool_input={"text": "x"}
    )
    run.llm.queue_response(text="the answer")

    await run.run("use the tool")

    assert len(tool.invocations) == 1
    assert "the answer" in "".join(run.history_texts())
    fired = [
        evt.payload["decision"]
        for evt in run.events_of(EventType.HOOK_FIRED)
        if evt.payload.get("hook") == HookEvent.pre_tool_use.value
    ]
    assert fired == ["allow"]


async def test_a_disposed_registration_leaves_nothing_behind(
    scenario: ScenarioFactory,
) -> None:
    """Disposal is complete and idempotent: the second run sees no seam."""
    registry = HookManager()
    calls: list[str] = []
    def _deny_and_record(ctx: LifecycleContext) -> LifecycleDecision:
        calls.append(ctx.payload["tool_name"])
        return LifecycleDecision(verdict=LifecycleVerdict.deny, reason="denied")

    dispose = registry.register(
        HookEvent.pre_tool_use,
        RegistrationKind.decide,
        _deny_and_record,
        owner="temporary",
    )
    tool = ScriptedTool(tool_name="Note", content="written")
    run = scenario(tools=[tool], rc=_lifecycle_rc(), lifecycle_hooks=registry)
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Note", tool_input={"text": "x"}
    )
    run.llm.queue_response(text="first")

    await run.run("use the tool")
    assert tool.invocations == []
    assert calls == ["Note"]

    assert dispose() is True
    assert dispose() is False
    assert registry.registrations() == ()
    assert registry.owners() == ()
    assert registry.points() == ()

    # The same registry, now empty, drives a fresh run: nothing of the disposed
    # registration is left to deny anything.
    second_tool = ScriptedTool(tool_name="Note", content="written")
    second = scenario(
        tools=[second_tool], rc=_lifecycle_rc(), lifecycle_hooks=registry
    )
    second.llm.queue_tool_call_response(
        tool_call_id="call-2", tool_name="Note", tool_input={"text": "y"}
    )
    second.llm.queue_response(text="second")

    await second.run("use the tool again")

    assert len(second_tool.invocations) == 1
    assert calls == ["Note"]
    assert [
        evt
        for evt in second.events_of(EventType.HOOK_FIRED)
        if "hook" in evt.payload
    ] == []


async def test_a_decide_registration_that_raises_denies_the_call(
    scenario: ScenarioFactory,
) -> None:
    """Fail closed end to end: a broken judge does not open the tool surface."""
    registry = HookManager()

    def _explode(_ctx: LifecycleContext) -> None:
        raise RuntimeError("judge unreachable")

    registry.register(
        HookEvent.pre_tool_use, RegistrationKind.decide, _explode, owner="judge"
    )
    tool = ScriptedTool(tool_name="Note", content="written")
    run = scenario(tools=[tool], rc=_lifecycle_rc(), lifecycle_hooks=registry)
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Note", tool_input={"text": "x"}
    )
    run.llm.queue_response(text="done")

    await run.run("use the tool")

    assert tool.invocations == []


async def test_the_seam_is_silent_when_the_switch_is_off(
    scenario: ScenarioFactory,
) -> None:
    """A registration made against a disabled seam changes nothing at all."""
    registry = HookManager()
    registry.register(
        HookEvent.pre_tool_use,
        RegistrationKind.decide,
        lambda _ctx: LifecycleDecision(verdict=LifecycleVerdict.deny),
        owner="policy",
    )
    tool = ScriptedTool(tool_name="Note", content="written")
    run = scenario(
        tools=[tool],
        rc=default_rc(typed_hooks_enabled=False),
        lifecycle_hooks=registry,
    )
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Note", tool_input={"text": "x"}
    )
    run.llm.queue_response(text="done")

    await run.run("use the tool")

    assert len(tool.invocations) == 1


async def test_an_around_registration_wraps_the_tool_invocation_itself(
    scenario: ScenarioFactory,
) -> None:
    """``tool_execute`` is the coordinate that straddles the call.

    The handler is entered before the tool runs and resumes after it returned,
    which is the property no pair of ``pre``/``post`` registrations can give:
    one handler holds the call for its whole duration.
    """
    registry = HookManager()
    trail: list[str] = []
    tool = ScriptedTool(tool_name="Note", content="written")

    async def _wrap(ctx: LifecycleContext, next_: Any) -> LifecycleDecision:
        trail.append(f"before:{ctx.payload['tool_name']}")
        assert tool.invocations == []
        await next_(ctx)
        trail.append("after")
        assert len(tool.invocations) == 1
        return LifecycleDecision()

    registry.register(
        HookEvent.tool_execute, RegistrationKind.around, _wrap, owner="timer"
    )
    run = scenario(tools=[tool], rc=_lifecycle_rc(), lifecycle_hooks=registry)
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Note", tool_input={"text": "x"}
    )
    run.llm.queue_response(text="done")

    await run.run("use the tool")

    assert trail == ["before:Note", "after"]
    assert len(tool.invocations) == 1
    assert any("written" in str(block.content) for block in run.tool_results())


async def test_an_around_registration_that_skips_next_skips_the_tool(
    scenario: ScenarioFactory,
) -> None:
    """Not reaching ``next`` is a refusal, and the model is told so.

    The alternative — reporting success for a call that never happened — would
    hand the model an answer nothing produced.
    """
    registry = HookManager()

    async def _short_circuit(
        _ctx: LifecycleContext, _next: Any
    ) -> LifecycleDecision:
        return LifecycleDecision(
            verdict=LifecycleVerdict.deny, reason="not during the freeze"
        )

    registry.register(
        HookEvent.tool_execute,
        RegistrationKind.around,
        _short_circuit,
        owner="freeze",
    )
    tool = ScriptedTool(tool_name="Note", content="written")
    run = scenario(tools=[tool], rc=_lifecycle_rc(), lifecycle_hooks=registry)
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Note", tool_input={"text": "x"}
    )
    run.llm.queue_response(text="done")

    await run.run("use the tool")

    assert tool.invocations == []
    results = run.tool_results()
    assert results and any("not during the freeze" in str(b.content) for b in results)


async def test_an_around_registration_that_raises_stops_the_call(
    scenario: ScenarioFactory,
) -> None:
    """Fail closed: a wrapper that broke did not let the call through."""
    registry = HookManager()

    async def _explode(_ctx: LifecycleContext, _next: Any) -> LifecycleDecision:
        raise RuntimeError("wrapper is broken")

    registry.register(
        HookEvent.tool_execute, RegistrationKind.around, _explode, owner="broken"
    )
    tool = ScriptedTool(tool_name="Note", content="written")
    run = scenario(tools=[tool], rc=_lifecycle_rc(), lifecycle_hooks=registry)
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Note", tool_input={"text": "x"}
    )
    run.llm.queue_response(text="done")

    await run.run("use the tool")

    assert tool.invocations == []
    assert run.tool_results()


async def test_the_turn_payload_names_the_message_index_a_host_reads(
    scenario: ScenarioFactory,
) -> None:
    """The turn payload key is a contract name, not the loop's variable path.

    A registration reads ``payload["assistant_message_idx"]`` by name. When the
    loop moved that counter into its turn flags the key travelled with it, and
    nothing said so: a host handler kept compiling and stopped finding the
    index. This pins the two names that cross the seam.
    """
    registry = HookManager()
    starts: list[dict[str, Any]] = []
    ends: list[dict[str, Any]] = []
    registry.register(
        HookEvent.turn_start,
        RegistrationKind.observe,
        lambda ctx: starts.append(dict(ctx.payload)),
        owner="audit",
    )
    registry.register(
        HookEvent.turn_end,
        RegistrationKind.observe,
        lambda ctx: ends.append(dict(ctx.payload)),
        owner="audit",
    )
    run = scenario(rc=_lifecycle_rc(), lifecycle_hooks=registry)
    run.llm.queue_response(text="a long enough answer for the run to accept")

    await run.run("go")

    assert starts and ends
    assert starts[0]["assistant_message_idx"] == 1
    assert "history_len" in starts[0]
    assert ends[0]["assistant_message_idx"] == 1
    assert "finish_reason" in ends[0]
