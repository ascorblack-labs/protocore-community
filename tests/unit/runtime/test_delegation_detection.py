"""Delegation is recognised by contract, not by a marker attribute.

The loop treats a delegation call differently from every other call: it may
overlap with its neighbours under a bounded semaphore, it costs the tree one or
more child runs, and — in the foreground — it blocks its caller on a whole
nested run, so the caller has to release its tree-budget slot around the join.
All of that used to hang off a boolean class attribute. An object carrying an
attribute of that name got the treatment whether or not it could deliver any of
it, and the obligations the loop actually depends on were written down nowhere.

So a tool declares delegation by implementing the two questions the loop asks:
how many child runs one call starts, and whether the caller waits for them.
"""
from __future__ import annotations

from typing import Any

from protocore.contracts.agent_dispatch import IDelegationTool
from protocore.contracts.tool_roles import ToolRole, ToolRoleMap
from protocore.contracts.types import ToolCall, ToolResult
from protocore.runtime.query import (
    _delegation_child_run_count,
    _delegation_is_background,
    _tool_is_delegation,
)
from tests._fixtures.delegation import DelegationContract

from ._tool_fixtures import MockTool


class _ContractTool(DelegationContract, MockTool):
    """A tool that implements the delegation contract."""

    is_concurrent_safe = False

    async def invoke(self, context: Any, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(tool_call_id="", content="child", is_error=False)


class _FlagOnlyTool(MockTool):
    """A tool that carries a marker attribute and implements nothing.

    The case the contract exists to reject: a name that once meant delegation,
    on an object that can answer none of the questions delegation implies.
    """

    is_parallel_delegation = True

    async def invoke(self, context: Any, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(tool_call_id="", content="not a child", is_error=False)


class _BatchTool(_ContractTool):
    @staticmethod
    def child_run_count(arguments: Any) -> int:
        tasks = arguments.get("tasks")
        return len(tasks) if isinstance(tasks, list) else 1


def _no_roles() -> ToolRoleMap:
    """A run whose host declared nothing, so only the contract can speak."""
    return ToolRoleMap.declare({})


def _register(runtime: dict[str, Any], tool: MockTool) -> None:
    runtime["tools"].register(tool)


# ── recognition ─────────────────────────────────────────────────────────────


def test_a_tool_that_implements_the_contract_is_delegation(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(tool_roles=_no_roles())
    _register(in_memory_runtime, _ContractTool(tool_name="Spawn", description="d"))

    assert _tool_is_delegation(engine, ToolCall(id="c", name="Spawn", arguments={}))


def test_a_tool_with_only_a_marker_attribute_is_not(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(tool_roles=_no_roles())
    _register(in_memory_runtime, _FlagOnlyTool(tool_name="Spawn", description="d"))

    assert not _tool_is_delegation(engine, ToolCall(id="c", name="Spawn", arguments={}))


def test_a_host_role_declaration_is_also_delegation(
    engine_factory, in_memory_runtime
) -> None:
    """How this core learns what a host's tools do, contract or not."""
    engine = engine_factory(
        tool_roles=ToolRoleMap.declare({"Spawn": [ToolRole.delegates_work]})
    )
    _register(in_memory_runtime, _FlagOnlyTool(tool_name="Spawn", description="d"))

    assert _tool_is_delegation(engine, ToolCall(id="c", name="Spawn", arguments={}))


def test_an_unregistered_name_is_not_delegation(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(tool_roles=_no_roles())

    assert not _tool_is_delegation(engine, ToolCall(id="c", name="Gone", arguments={}))


def test_the_contract_is_structural(engine_factory) -> None:
    assert isinstance(_ContractTool(tool_name="S", description="d"), IDelegationTool)
    assert not isinstance(_FlagOnlyTool(tool_name="S", description="d"), IDelegationTool)


# ── what the contract answers ───────────────────────────────────────────────


def test_the_tool_says_how_many_child_runs_a_call_starts(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(tool_roles=_no_roles())
    _register(in_memory_runtime, _BatchTool(tool_name="Spawn", description="d"))
    call = ToolCall(id="c", name="Spawn", arguments={"tasks": ["a", "b", "c"]})

    assert _delegation_child_run_count(engine, call) == 3


def test_a_role_only_delegation_reads_as_one_child_run(
    engine_factory, in_memory_runtime
) -> None:
    """It delegates, and it cannot be asked how widely, so it counts as one."""
    engine = engine_factory(
        tool_roles=ToolRoleMap.declare({"Spawn": [ToolRole.delegates_work]})
    )
    _register(in_memory_runtime, _FlagOnlyTool(tool_name="Spawn", description="d"))
    call = ToolCall(id="c", name="Spawn", arguments={"tasks": ["a", "b"]})

    assert _delegation_child_run_count(engine, call) == 1


def test_a_counter_that_answers_with_nonsense_reads_as_one(
    engine_factory, in_memory_runtime
) -> None:
    class _BrokenTool(_ContractTool):
        @staticmethod
        def child_run_count(arguments: Any) -> int:
            raise ValueError("no idea")

    engine = engine_factory(tool_roles=_no_roles())
    _register(in_memory_runtime, _BrokenTool(tool_name="Spawn", description="d"))

    assert (
        _delegation_child_run_count(engine, ToolCall(id="c", name="Spawn", arguments={}))
        == 1
    )


def test_the_tool_says_whether_the_caller_waits(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(tool_roles=_no_roles())
    _register(in_memory_runtime, _ContractTool(tool_name="Spawn", description="d"))

    assert _delegation_is_background(
        engine, ToolCall(id="c", name="Spawn", arguments={"background": True})
    )
    assert not _delegation_is_background(
        engine, ToolCall(id="c", name="Spawn", arguments={})
    )


def test_a_tool_that_cannot_say_is_read_as_waiting(
    engine_factory, in_memory_runtime
) -> None:
    """The conservative half: a slot is only released by something that said so."""
    engine = engine_factory(
        tool_roles=ToolRoleMap.declare({"Spawn": [ToolRole.delegates_work]})
    )
    _register(in_memory_runtime, _FlagOnlyTool(tool_name="Spawn", description="d"))

    assert not _delegation_is_background(
        engine, ToolCall(id="c", name="Spawn", arguments={"background": True})
    )
