"""Tests for :class:`HookManager` dispatch + isolation."""
from __future__ import annotations

from typing import Any

from protocore.contracts.hooks import HookActionKind
from protocore.contracts.types import HookEvent
from protocore.hooks import HookManager, hookimpl


class _Counter:
    """Per-hook invocation counter."""

    def __init__(self) -> None:
        self.tool_use_count = 0
        self.session_count = 0


class _GoodPlugin:
    def __init__(self, counter: _Counter) -> None:
        self._counter = counter

    @hookimpl
    async def pre_tool_use(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        self._counter.tool_use_count += 1
        return {"action": HookActionKind.ALLOW, "reason": "noted"}

    @hookimpl
    async def session_start(self, session_id: str, context: dict[str, Any]) -> dict[str, Any]:
        self._counter.session_count += 1
        return {"action": HookActionKind.ALLOW}


class _DenyPlugin:
    @hookimpl
    async def pre_tool_use(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "action": HookActionKind.DENY,
            "reason": "blocked by safety",
        }


class _BadPlugin:
    @hookimpl
    async def pre_tool_use(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        raise RuntimeError("hook explodes")


async def test_register_and_invoke_allow() -> None:
    mgr = HookManager()
    counter = _Counter()
    mgr.register(_GoodPlugin(counter), name="good")
    result = await mgr.invoke(
        HookEvent.pre_tool_use,
        {"tool_name": "Bash", "arguments": {}, "context": {}},
    )
    assert result.action == HookActionKind.ALLOW
    assert counter.tool_use_count == 1


async def test_invoke_session_start() -> None:
    mgr = HookManager()
    counter = _Counter()
    mgr.register(_GoodPlugin(counter), name="good")
    await mgr.invoke(HookEvent.session_start, {"session_id": "s1", "context": {}})
    assert counter.session_count == 1


async def test_deny_short_circuits() -> None:
    mgr = HookManager()
    counter = _Counter()
    mgr.register(_DenyPlugin(), name="deny")
    mgr.register(_GoodPlugin(counter), name="good")
    result = await mgr.invoke(
        HookEvent.pre_tool_use,
        {"tool_name": "Bash", "arguments": {}, "context": {}},
    )
    assert result.action == HookActionKind.DENY
    assert "blocked by safety" in result.reason


async def test_handler_exception_isolated() -> None:
    mgr = HookManager()
    counter = _Counter()
    mgr.register(_BadPlugin(), name="bad")
    mgr.register(_GoodPlugin(counter), name="good")
    result = await mgr.invoke(
        HookEvent.pre_tool_use,
        {"tool_name": "Bash", "arguments": {}, "context": {}},
    )
    # A crashed handler must NOT break siblings: the good plugin still
    # records and the aggregate stays ALLOW (allow-vs-deny on crash is a
    # failure_mode decision left to the host adapter).
    assert result.action == HookActionKind.ALLOW
    assert counter.tool_use_count == 1
    # : the crash must be SURFACED, not silently swallowed — an
    # error record lands in raw_results so a consumer can observe it.
    error_records = [r for r in result.raw_results if r.get("outcome") == "error"]
    assert len(error_records) == 1
    assert error_records[0]["event"] == HookEvent.pre_tool_use.value
    assert "hook explodes" in error_records[0]["error"]


async def test_crashed_handler_surfaced_even_when_no_other_handler() -> None:
    """: a lone crashing handler must still surface an error record
    (a crashed deny hook fails open but the operator can see it broke)."""
    mgr = HookManager()
    mgr.register(_BadPlugin(), name="bad")
    result = await mgr.invoke(
        HookEvent.pre_tool_use,
        {"tool_name": "Bash", "arguments": {}, "context": {}},
    )
    assert result.action == HookActionKind.ALLOW
    error_records = [r for r in result.raw_results if r.get("outcome") == "error"]
    assert len(error_records) == 1
    assert error_records[0]["outcome"] == "error"
    assert error_records[0]["event"] == HookEvent.pre_tool_use.value


def test_all_10_hook_events_in_enum() -> None:
    """10 events total: 8 base + subagent_start/stop."""
    values = {ev.value for ev in HookEvent}
    expected = {
        "pre_tool_use",
        "post_tool_use",
        "user_prompt_submit",
        "session_start",
        "session_end",
        "pre_compact",
        "post_compact",
        "file_changed",
        "subagent_start",
        "subagent_stop",
    }
    assert values == expected
    assert len(values) == 10


def test_registered_lists_names() -> None:
    mgr = HookManager()
    counter = _Counter()
    mgr.register(_GoodPlugin(counter), name="good")
    assert "good" in mgr.registered()


class _ModifyPlugin:
    @hookimpl
    async def pre_tool_use(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "action": HookActionKind.MODIFY,
            "reason": "patched args",
            "modifications": {"tool_name": "BashSafe"},
        }


class _IgnoredReturnPlugin:
    """Returns non-dict / None — should be ignored."""

    @hookimpl
    async def pre_tool_use(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:
        return "not-a-dict"


class _NoneReturnPlugin:
    @hookimpl
    async def pre_tool_use(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        return None


async def test_modify_action_aggregates_modifications() -> None:
    mgr = HookManager()
    mgr.register(_ModifyPlugin(), name="mod")
    result = await mgr.invoke(
        HookEvent.pre_tool_use,
        {"tool_name": "Bash", "arguments": {}, "context": {}},
    )
    assert result.action == HookActionKind.MODIFY
    assert result.modifications == {"tool_name": "BashSafe"}
    assert result.reason == "patched args"


async def test_deny_includes_self_in_raw_results() -> None:
    """First-resolved deny must surface itself in raw_results before exit."""
    mgr = HookManager()
    mgr.register(_DenyPlugin(), name="deny")
    mgr.register(_ModifyPlugin(), name="mod")
    result = await mgr.invoke(
        HookEvent.pre_tool_use,
        {"tool_name": "Bash", "arguments": {}, "context": {}},
    )
    assert result.action == HookActionKind.DENY
    # raw_results contains at least the denying record before short-circuit;
    # additional entries depend on pluggy LIFO order.
    assert len(result.raw_results) >= 1
    assert any(r.get("action") == HookActionKind.DENY for r in result.raw_results)


async def test_non_dict_or_none_results_ignored() -> None:
    mgr = HookManager()
    counter = _Counter()
    mgr.register(_NoneReturnPlugin(), name="none")
    mgr.register(_IgnoredReturnPlugin(), name="ignored")
    mgr.register(_GoodPlugin(counter), name="good")
    result = await mgr.invoke(
        HookEvent.pre_tool_use,
        {"tool_name": "Bash", "arguments": {}, "context": {}},
    )
    assert result.action == HookActionKind.ALLOW
    # Only the dict-returning _GoodPlugin lands in raw_results.
    assert len(result.raw_results) == 1


async def test_no_spec_for_event_returns_allow() -> None:
    """When no hookspec exists for an event, invoke must short-circuit ALLOW."""
    import pluggy

    mgr = HookManager()
    # Replace the underlying PluginManager with one that has no hookspecs;
    # this exercises the ``caller is None`` branch.
    mgr._pm = pluggy.PluginManager("empty-project")
    result = await mgr.invoke(
        HookEvent.pre_tool_use,
        {"tool_name": "Bash", "arguments": {}, "context": {}},
    )
    assert result.action == HookActionKind.ALLOW
    assert result.reason == "no spec for event"


async def test_dispatch_failure_returns_allow() -> None:
    """When the pluggy caller itself raises, manager isolates and returns ALLOW."""
    mgr = HookManager()

    class _ExplodingCaller:
        def __call__(self, **_: Any) -> Any:
            raise RuntimeError("dispatch boom")

    # Swap the caller on the pluggy hook relay so caller(...) raises.
    object.__setattr__(mgr._pm.hook, HookEvent.pre_tool_use.value, _ExplodingCaller())
    result = await mgr.invoke(
        HookEvent.pre_tool_use,
        {"tool_name": "Bash", "arguments": {}, "context": {}},
    )
    assert result.action == HookActionKind.ALLOW
    assert "dispatch failed" in result.reason
    # : a crash at dispatch time (the path a SYNCHRONOUS hookimpl's
    # exception takes through pluggy's ``caller(...)``) must also be surfaced.
    error_records = [r for r in result.raw_results if r.get("outcome") == "error"]
    assert len(error_records) == 1
    assert "dispatch boom" in error_records[0]["error"]


def test_unregister_by_name_returns_true() -> None:
    mgr = HookManager()
    counter = _Counter()
    plugin = _GoodPlugin(counter)
    mgr.register(plugin, name="good")
    assert mgr.unregister("good") is True


def test_unregister_unknown_returns_false() -> None:
    mgr = HookManager()
    assert mgr.unregister("never-registered") is False


def test_unregister_by_plugin_instance() -> None:
    mgr = HookManager()
    counter = _Counter()
    plugin = _GoodPlugin(counter)
    mgr.register(plugin, name="good")
    assert mgr.unregister(plugin) is True
