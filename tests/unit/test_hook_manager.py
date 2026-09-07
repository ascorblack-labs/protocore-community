"""The lifecycle registry: order, scope, disposal, and the exception policy."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from protocore.contracts.middleware import (
    LifecycleContext,
    LifecycleDecision,
    LifecycleScope,
    LifecycleVerdict,
    RegistrationKind,
)
from protocore.contracts.types import HookEvent
from protocore.hooks import HookManager, refuse_lifecycle_when_disabled


def _ctx(point: HookEvent = HookEvent.pre_tool_use, **payload: Any) -> LifecycleContext:
    return LifecycleContext(
        point=point,
        run_id="run-1",
        session_id="sess-1",
        tenant_id="tenant-1",
        payload=payload,
    )


@pytest.mark.asyncio
async def test_an_empty_coordinate_allows_and_returns_the_payload_it_got() -> None:
    manager = HookManager()
    outcome = await manager.dispatch(_ctx(tool_name="Read"))
    assert outcome.allowed
    assert outcome.payload == {"tool_name": "Read"}
    assert outcome.failures == ()


@pytest.mark.asyncio
async def test_registrations_run_by_priority_then_registration_order() -> None:
    manager = HookManager()
    seen: list[str] = []
    for owner, priority in (("c", 50), ("a", 10), ("b", 10)):
        manager.register(
            HookEvent.pre_tool_use,
            RegistrationKind.observe,
            lambda _ctx, name=owner: seen.append(name),
            owner=owner,
            priority=priority,
        )
    await manager.dispatch(_ctx())
    assert seen == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_a_decide_registration_denies_and_names_its_owner() -> None:
    manager = HookManager()
    manager.register(
        HookEvent.pre_tool_use,
        RegistrationKind.decide,
        lambda _ctx: LifecycleDecision(
            verdict=LifecycleVerdict.deny, reason="not that tool"
        ),
        owner="policy",
    )
    ran_after = []
    manager.register(
        HookEvent.pre_tool_use,
        RegistrationKind.decide,
        lambda _ctx: ran_after.append(1),
        owner="second",
        priority=200,
    )
    outcome = await manager.dispatch(_ctx())
    assert outcome.verdict is LifecycleVerdict.deny
    assert outcome.decided_by == "policy"
    assert outcome.reason == "not that tool"
    assert ran_after == []


@pytest.mark.asyncio
async def test_a_decide_registration_that_raises_denies() -> None:
    """Fail closed: a seam that cannot answer has not said yes."""
    manager = HookManager()

    def _explode(_ctx: LifecycleContext) -> None:
        raise RuntimeError("executor unreachable")

    manager.register(
        HookEvent.pre_tool_use,
        RegistrationKind.decide,
        _explode,
        owner="judge",
    )
    outcome = await manager.dispatch(_ctx())
    assert outcome.verdict is LifecycleVerdict.deny
    assert outcome.decided_by == "judge"
    assert "executor unreachable" in outcome.reason
    assert [item.isolated for item in outcome.failures] == [False]


@pytest.mark.asyncio
async def test_a_decide_registration_that_overruns_denies() -> None:
    manager = HookManager()

    async def _slow(_ctx: LifecycleContext) -> LifecycleDecision:
        await asyncio.sleep(5)
        return LifecycleDecision()

    manager.register(
        HookEvent.pre_tool_use,
        RegistrationKind.decide,
        _slow,
        owner="slow-judge",
        timeout_s=0.01,
    )
    outcome = await manager.dispatch(_ctx())
    assert outcome.verdict is LifecycleVerdict.deny
    assert "timed out" in outcome.reason


@pytest.mark.asyncio
async def test_a_decide_registration_that_answers_nonsense_denies() -> None:
    manager = HookManager()
    manager.register(
        HookEvent.pre_tool_use,
        RegistrationKind.decide,
        lambda _ctx: 17,
        owner="confused",
    )
    outcome = await manager.dispatch(_ctx())
    assert outcome.verdict is LifecycleVerdict.deny
    assert "not a decision" in outcome.reason


@pytest.mark.asyncio
async def test_an_observing_registration_that_raises_changes_nothing() -> None:
    manager = HookManager()

    def _explode(_ctx: LifecycleContext) -> None:
        raise RuntimeError("audit sink down")

    manager.register(
        HookEvent.pre_tool_use, RegistrationKind.observe, _explode, owner="audit"
    )
    outcome = await manager.dispatch(_ctx(tool_name="Read"))
    assert outcome.allowed
    assert outcome.payload == {"tool_name": "Read"}
    assert [item.isolated for item in outcome.failures] == [True]
    assert outcome.failures[0].owner == "audit"


@pytest.mark.asyncio
async def test_a_notifying_registration_that_raises_changes_nothing() -> None:
    manager = HookManager()
    manager.register(
        HookEvent.post_tool_use,
        RegistrationKind.notify,
        lambda _ctx: (_ for _ in ()).throw(RuntimeError("webhook 500")),
        owner="webhook",
    )
    outcome = await manager.dispatch(_ctx(HookEvent.post_tool_use, ok=True))
    assert outcome.allowed
    assert outcome.failures[0].isolated is True


@pytest.mark.asyncio
async def test_a_transform_replaces_the_payload_for_everyone_downstream() -> None:
    manager = HookManager()
    manager.register(
        HookEvent.context_transform,
        RegistrationKind.transform,
        lambda ctx: {**ctx.payload, "sections": ["redacted"]},
        owner="redactor",
    )
    seen: list[Any] = []

    def _watch(ctx: LifecycleContext) -> dict[str, Any]:
        seen.append(ctx.payload["sections"])
        return dict(ctx.payload)

    manager.register(
        HookEvent.context_transform,
        RegistrationKind.transform,
        _watch,
        owner="second",
        priority=200,
    )
    outcome = await manager.dispatch(
        _ctx(HookEvent.context_transform, sections=["secret"])
    )
    assert outcome.payload["sections"] == ["redacted"]
    assert seen == [["redacted"]]


@pytest.mark.asyncio
async def test_a_transform_that_raises_denies_rather_than_letting_it_through() -> None:
    manager = HookManager()
    manager.register(
        HookEvent.context_transform,
        RegistrationKind.transform,
        lambda _ctx: (_ for _ in ()).throw(ValueError("bad rewrite")),
        owner="redactor",
    )
    outcome = await manager.dispatch(
        _ctx(HookEvent.context_transform, sections=["secret"])
    )
    assert outcome.verdict is LifecycleVerdict.deny
    assert outcome.payload["sections"] == ["secret"]


@pytest.mark.asyncio
async def test_scope_keeps_a_registration_off_runs_it_was_not_made_for() -> None:
    manager = HookManager()
    hits: list[str] = []
    manager.register(
        HookEvent.pre_tool_use,
        RegistrationKind.observe,
        lambda ctx: hits.append(ctx.run_id),
        owner="scoped",
        scope=LifecycleScope(run_id="run-other"),
    )
    await manager.dispatch(_ctx())
    assert hits == []
    await manager.dispatch(
        LifecycleContext(point=HookEvent.pre_tool_use, run_id="run-other")
    )
    assert hits == ["run-other"]


@pytest.mark.asyncio
async def test_the_disposer_removes_the_registration_and_is_idempotent() -> None:
    manager = HookManager()
    hits: list[int] = []
    dispose = manager.register(
        HookEvent.pre_tool_use,
        RegistrationKind.observe,
        lambda _ctx: hits.append(1),
        owner="temp",
    )
    await manager.dispatch(_ctx())
    assert hits == [1]
    assert dispose() is True
    assert dispose() is False
    assert dispose.disposed is True
    assert manager.registrations(HookEvent.pre_tool_use) == ()
    assert manager.registrations() == ()
    assert manager.points() == ()
    await manager.dispatch(_ctx())
    assert hits == [1]


@pytest.mark.asyncio
async def test_disposing_one_owner_leaves_the_others_alone() -> None:
    manager = HookManager()
    for owner in ("a", "a", "b"):
        manager.register(
            HookEvent.pre_tool_use,
            RegistrationKind.observe,
            lambda _ctx: None,
            owner=owner,
        )
    assert manager.owners() == ("a", "b")
    assert manager.dispose_owner("a") == 2
    assert manager.owners() == ("b",)


def test_a_registration_without_an_owner_is_refused() -> None:
    manager = HookManager()
    with pytest.raises(ValueError, match="registration_requires_owner"):
        manager.register(
            HookEvent.pre_tool_use,
            RegistrationKind.observe,
            lambda _ctx: None,
            owner="",
        )


@pytest.mark.asyncio
async def test_around_wraps_the_work_and_can_short_circuit_it() -> None:
    manager = HookManager()
    ran: list[str] = []

    async def _next(_ctx: LifecycleContext) -> None:
        ran.append("work")

    manager.register(
        HookEvent.pre_tool_use,
        RegistrationKind.around,
        lambda _ctx, _n: LifecycleDecision(
            verdict=LifecycleVerdict.deny, reason="blocked"
        ),
        owner="gate",
    )
    outcome = await manager.around(_ctx(), _next)
    assert ran == []
    assert outcome.verdict is LifecycleVerdict.deny
    assert outcome.decided_by == "gate"


@pytest.mark.asyncio
async def test_around_that_calls_next_lets_the_work_happen() -> None:
    manager = HookManager()
    ran: list[str] = []

    async def _next(_ctx: LifecycleContext) -> None:
        ran.append("work")

    async def _pass_through(ctx: LifecycleContext, next_: Any) -> LifecycleDecision:
        await next_(ctx)
        return LifecycleDecision()

    manager.register(
        HookEvent.pre_tool_use,
        RegistrationKind.around,
        _pass_through,
        owner="timer",
    )
    outcome = await manager.around(_ctx(), _next)
    assert ran == ["work"]
    assert outcome.allowed


@pytest.mark.asyncio
async def test_around_with_nothing_registered_still_runs_the_work() -> None:
    manager = HookManager()
    ran: list[str] = []

    async def _next(_ctx: LifecycleContext) -> None:
        ran.append("work")

    outcome = await manager.around(_ctx(), _next)
    assert ran == ["work"]
    assert outcome.allowed


@pytest.mark.asyncio
async def test_an_around_that_raises_denies_and_the_work_is_not_reported_done() -> None:
    manager = HookManager()

    async def _next(_ctx: LifecycleContext) -> None:
        return None

    manager.register(
        HookEvent.pre_tool_use,
        RegistrationKind.around,
        lambda _ctx, _n: (_ for _ in ()).throw(RuntimeError("wrapper broke")),
        owner="wrapper",
    )
    outcome = await manager.around(_ctx(), _next)
    assert outcome.verdict is LifecycleVerdict.deny
    assert outcome.decided_by == "wrapper"


@pytest.mark.asyncio
async def test_cancellation_is_never_read_as_a_denial() -> None:
    manager = HookManager()

    async def _cancelled(_ctx: LifecycleContext) -> None:
        raise asyncio.CancelledError

    manager.register(
        HookEvent.pre_tool_use, RegistrationKind.decide, _cancelled, owner="cancelled"
    )
    with pytest.raises(asyncio.CancelledError):
        await manager.dispatch(_ctx())


@pytest.mark.asyncio
async def test_a_verdict_may_be_returned_bare_or_as_its_string() -> None:
    manager = HookManager()
    manager.register(
        HookEvent.pre_tool_use,
        RegistrationKind.decide,
        lambda _ctx: "require_approval",
        owner="asker",
    )
    outcome = await manager.dispatch(_ctx())
    assert outcome.verdict is LifecycleVerdict.require_approval

    manager = HookManager()
    manager.register(
        HookEvent.pre_tool_use,
        RegistrationKind.decide,
        lambda _ctx: LifecycleVerdict.fail_run,
        owner="stopper",
    )
    outcome = await manager.dispatch(_ctx())
    assert outcome.verdict is LifecycleVerdict.fail_run


def test_the_seam_refuses_to_be_used_while_switched_off() -> None:
    with pytest.raises(ValueError, match="lifecycle_hooks_disabled"):
        refuse_lifecycle_when_disabled(False)
    refuse_lifecycle_when_disabled(True)
