"""A child run gets a subset of its parent, and the rule is applied twice.

Narrowing only. A delegated run may be given less than the run that started it
and can never be given more, because a run that could widen on the way down
would make every bound above it advisory.

The rule is a pure function so it can be asked in both places it has to be
asked: when the catalogue for the child is resolved, and again on each of the
child's tool calls. Those were separate pieces of arithmetic over separate
inputs and nothing required them to agree — so a tool withheld from a child's
surface could still be dispatched by a child that named it.
"""
from __future__ import annotations

import random
from typing import Any

import pytest

from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tool_roles import ToolRole, ToolRoleMap
from protocore.contracts.tools import ToolContext
from protocore.contracts.types import SubagentDef, ToolResult
from protocore.runtime.child_capabilities import (
    ChildCapabilities,
    ParentCapabilities,
    narrow_child_capabilities,
    stricter_permission_mode,
)
from protocore.runtime.tool_permission import (
    ToolPermissionGate,
    ToolPermissionOutcome,
)

from ._tool_fixtures import MockTool

_ROLES = ToolRoleMap.declare(
    {
        "Read": [ToolRole.reads_path],
        "Write": [ToolRole.writes_path],
        "Agent": [ToolRole.delegates_work, ToolRole.never_delegated],
        "AskUser": [ToolRole.asks_user, ToolRole.never_delegated],
    }
)

#: A host's own ladder of permission strictness, loosest first. Core ships none.
_MODES = ("open", "guarded", "sealed")


def _definition(**overrides: Any) -> SubagentDef:
    values: dict[str, Any] = {
        "id": "researcher",
        "tenant_id": "t-1",
        "name": "Researcher",
        "description": "reads things",
        "system_prompt": "research",
    }
    values.update(overrides)
    return SubagentDef(**values)


def _parent(**overrides: Any) -> ParentCapabilities:
    values: dict[str, Any] = {"tools": frozenset({"Read", "Write", "Agent"})}
    values.update(overrides)
    return ParentCapabilities(**values)


# ── the tool set only shrinks ───────────────────────────────────────────────


def test_a_child_cannot_gain_a_tool_its_parent_lacks() -> None:
    child = narrow_child_capabilities(
        ParentCapabilities(tools=frozenset({"Read"})),
        _definition(tool_whitelist=["Read", "Write", "Bash"]),
        roles=_ROLES,
    )

    assert child.tools == frozenset({"Read"})


def test_a_definition_that_declares_no_tools_inherits_the_parents() -> None:
    """Saying nothing about tools is not the same as saying none."""
    child = narrow_child_capabilities(
        ParentCapabilities(tools=frozenset({"Read", "Write"})),
        _definition(),
        roles=_ROLES,
    )

    assert child.tools == frozenset({"Read", "Write"})


def test_blocked_names_come_out_even_when_declared() -> None:
    child = narrow_child_capabilities(
        _parent(),
        _definition(tool_whitelist=["Read", "Write"], blocked_tools=["Write"]),
        roles=_ROLES,
    )

    assert child.tools == frozenset({"Read"})


def test_a_never_delegated_tool_reaches_no_child() -> None:
    child = narrow_child_capabilities(
        _parent(), _definition(tool_whitelist=["Read", "Agent"]), roles=_ROLES
    )

    assert child.tools == frozenset({"Read"})


@pytest.mark.parametrize("seed", range(25))
def test_the_result_is_always_a_subset_of_the_parents_set(seed: int) -> None:
    """The property, over random pairs of sets: narrowing, never widening."""
    rng = random.Random(seed)
    universe = [f"T{index}" for index in range(8)]
    parent_tools = frozenset(rng.sample(universe, rng.randint(0, 8)))
    declared = rng.sample(universe, rng.randint(0, 8))
    blocked = rng.sample(universe, rng.randint(0, 4))

    child = narrow_child_capabilities(
        ParentCapabilities(tools=parent_tools),
        _definition(tool_whitelist=declared, blocked_tools=blocked),
    )

    assert child.tools <= parent_tools


# ── permission mode ─────────────────────────────────────────────────────────


def test_a_childs_mode_is_never_weaker_than_its_parents() -> None:
    loosening = narrow_child_capabilities(
        _parent(permission_mode="sealed"),
        _definition(permission_mode="open"),
        permission_mode_order=_MODES,
    )
    tightening = narrow_child_capabilities(
        _parent(permission_mode="open"),
        _definition(permission_mode="sealed"),
        permission_mode_order=_MODES,
    )

    assert loosening.permission_mode == "sealed"
    assert tightening.permission_mode == "sealed"


def test_a_definition_that_says_nothing_inherits_the_mode() -> None:
    child = narrow_child_capabilities(
        _parent(permission_mode="guarded"),
        _definition(),
        permission_mode_order=_MODES,
    )

    assert child.permission_mode == "guarded"


def test_a_mode_nobody_can_rank_leaves_the_parents_in_place() -> None:
    """A mode that cannot be shown stricter has not been shown safe to adopt."""
    child = narrow_child_capabilities(
        _parent(permission_mode="guarded"),
        _definition(permission_mode="whatever"),
        permission_mode_order=_MODES,
    )

    assert child.permission_mode == "guarded"


def test_a_parent_that_declared_no_mode_takes_the_definitions() -> None:
    assert stricter_permission_mode("", "sealed", order=_MODES) == "sealed"
    assert stricter_permission_mode("guarded", "guarded", order=_MODES) == "guarded"


# ── depth ───────────────────────────────────────────────────────────────────


def test_depth_goes_up_by_one() -> None:
    child = narrow_child_capabilities(_parent(depth=2), _definition(), roles=_ROLES)

    assert child.depth == 3


def test_the_right_to_spawn_is_withdrawn_at_the_ceiling() -> None:
    at_ceiling = narrow_child_capabilities(
        ParentCapabilities(tools=frozenset({"Read", "Agent"}), depth=2),
        _definition(),
        roles=ToolRoleMap.declare({"Agent": [ToolRole.delegates_work]}),
        max_depth=3,
    )

    assert not at_ceiling.may_spawn
    assert at_ceiling.tools == frozenset({"Read"})


def test_below_the_ceiling_the_right_to_spawn_stands() -> None:
    child = narrow_child_capabilities(
        ParentCapabilities(tools=frozenset({"Read", "Agent"}), depth=0),
        _definition(),
        roles=ToolRoleMap.declare({"Agent": [ToolRole.delegates_work]}),
        max_depth=3,
    )

    assert child.may_spawn
    assert "Agent" in child.tools


def test_a_ceiling_of_zero_is_unbounded() -> None:
    child = narrow_child_capabilities(
        ParentCapabilities(tools=frozenset({"Agent"}), depth=9),
        _definition(),
        roles=ToolRoleMap.declare({"Agent": [ToolRole.delegates_work]}),
    )

    assert child.may_spawn


# ── purity ──────────────────────────────────────────────────────────────────


def test_two_calls_with_one_input_give_equal_answers() -> None:
    parent = _parent(permission_mode="guarded", depth=1)
    definition = _definition(tool_whitelist=["Read", "Write"])

    first = narrow_child_capabilities(parent, definition, roles=_ROLES, max_depth=4)
    second = narrow_child_capabilities(parent, definition, roles=_ROLES, max_depth=4)

    assert first == second
    assert isinstance(first, ChildCapabilities)
    assert parent == _parent(permission_mode="guarded", depth=1)
    assert definition == _definition(tool_whitelist=["Read", "Write"])


# ── applied twice: the catalogue, and then the call ─────────────────────────


def test_the_catalogue_withholds_a_never_delegated_tool(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(
        parent_run_id="run-parent",
        root_run_id="run-parent",
        subagent_id="researcher",
        tool_roles=_ROLES,
    )
    object.__setattr__(
        engine.config, "subagent_tool_allowlist", frozenset({"Read", "Agent"})
    )

    allowed = engine.effective_subagent_tool_allowlist
    assert allowed is not None
    assert "Read" in allowed
    assert "Agent" not in allowed
    assert "AskUser" not in allowed


async def test_the_gate_refuses_what_the_catalogue_allowed(
    engine_factory,
) -> None:
    """The second application, and the one that closes the disagreement.

    A child that declared nothing has no allow-list stage to narrow, so the
    catalogue never gets a say — and this is the only thing standing between it
    and a tool no child is ever meant to reach.
    """
    gate = ToolPermissionGate(policies=[], roles=_ROLES)
    tool = MockTool(tool_name="AskUser", description="ask")
    engine = engine_factory(tool_roles=_ROLES)
    ctx = ToolContext(
        tenant_id="t-1",
        run_id="run-child",
        session_id="s-1",
        run_state=engine.run_state,
    )

    allowed = await gate.check(
        tool=tool,
        arguments={},
        ctx=ctx,
        visibility_policy=ToolVisibilityPolicy(),
        subagent_whitelist=None,
        child_run=False,
    )
    denied = await gate.check(
        tool=tool,
        arguments={},
        ctx=ctx,
        visibility_policy=ToolVisibilityPolicy(),
        subagent_whitelist=None,
        child_run=True,
    )

    assert allowed.outcome is ToolPermissionOutcome.allow
    assert denied.outcome is ToolPermissionOutcome.deny
    assert "never offered to a delegated run" in denied.reason


async def test_the_tool_surface_floor_is_no_exception(engine_factory) -> None:
    """A name withheld from every child was never on this child's surface."""
    gate = ToolPermissionGate(policies=[], roles=_ROLES)
    tool = MockTool(tool_name="Agent", description="delegate")
    engine = engine_factory(tool_roles=_ROLES)
    ctx = ToolContext(
        tenant_id="t-1",
        run_id="run-child",
        session_id="s-1",
        run_state=engine.run_state,
    )

    decision = await gate.check(
        tool=tool,
        arguments={},
        ctx=ctx,
        visibility_policy=ToolVisibilityPolicy(forced_pinned=frozenset({"Agent"})),
        subagent_whitelist=frozenset({"Agent"}),
        child_run=True,
    )

    assert decision.outcome is ToolPermissionOutcome.deny


assert ToolResult is not None
