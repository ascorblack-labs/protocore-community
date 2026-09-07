"""Runs whose tools are not called what the core once assumed they were called.

Three mechanisms used to recognise a tool by a name spelled inside the core: the
read-back gate that makes a caller open the files a tool declared, the published
plan profile that withholds the tools which change things, and the large-file
driver (whose own scenarios live next door). An installation that named its
tools differently lost all three at once, and lost them silently — every
comparison simply stopped matching.

So the host states what its tools DO, and these scenarios wire a run with names
no core module has ever heard of and watch the mechanisms work anyway.
"""
from __future__ import annotations

from typing import Any

from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tool_roles import ToolArgumentSlot, ToolRole, ToolRoleMap
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import (
    PENDING_READS_METADATA_KEY,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
)

from .conftest import ScenarioFactory, default_rc

#: A host that named nothing the way the core used to guess.
OPEN, PUT, SHELL, DELEGATE = "OpenDoc", "PutDoc", "RunCommand", "HandOff"
REPORT = "/workspace/report.md"


class _NamedTool(Tool):
    """A tool with a host's name, a declared role, and a scripted result."""

    def __init__(
        self,
        name: str,
        *,
        content: str = "ok",
        declares: tuple[str, ...] = (),
        pinned: bool = False,
    ) -> None:
        self._name = name
        self._content = content
        self._declares = declares
        self._pinned = pinned
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=f"{self._name}",
            parameters=ToolParameterSchema(
                properties={"path": {"type": "string"}}, required=["path"]
            ),
        )

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append(dict(arguments))
        metadata: dict[str, Any] = {}
        if self._declares:
            metadata[PENDING_READS_METADATA_KEY] = list(self._declares)
        if self._pinned:
            metadata["retention"] = "pinned"
        path = arguments.get("path")
        return ToolResult(
            tool_call_id="tc",
            content=self._content,
            is_error=False,
            metadata=metadata,
            path=path if isinstance(path, str) else None,
        )


def _roles() -> ToolRoleMap:
    return ToolRoleMap.declare(
        {
            OPEN: [ToolRole.reads_path],
            PUT: [ToolRole.writes_path],
            SHELL: [ToolRole.runs_shell],
            DELEGATE: [ToolRole.delegates_work, ToolRole.never_delegated],
        },
        argument_aliases={ToolArgumentSlot.path: ["path", "file_path"]},
    )


async def test_the_read_back_gate_forces_the_host_s_own_read_tool(
    scenario: ScenarioFactory,
) -> None:
    """A tool declares a file; the next request forces the tool that reads it.

    The gate has exactly one name to force, and it is the host's. Before roles
    it forced a name of the core's own choosing, which on this run is not a
    tool at all — the provider would have rejected the request, or the gate
    would have withheld the force and let the run answer from the pointer.
    """
    delegate = _NamedTool(DELEGATE, content="wrote it", declares=(REPORT,))
    opener = _NamedTool(OPEN, content="the whole report")
    run = scenario(
        rc=default_rc(pending_reads_enabled=True),
        tools=[delegate, opener],
        tool_roles=_roles(),
    )
    run.llm.queue_tool_call_response(
        tool_call_id="t1", tool_name=DELEGATE, tool_input={"path": REPORT}
    )
    run.llm.queue_response(text="done")

    await run.run("delegate the report")

    forced = [request.extra.get("forced_tool_choice") for request in run.requests]
    assert OPEN in forced, forced


async def test_the_plan_profile_withholds_the_host_s_writing_tools(
    scenario: ScenarioFactory,
) -> None:
    """Under a plan profile, the tool that writes is not advertised.

    Which tool that is comes from the host's declaration. The profile's
    allowlist names the reading tool; everything the roles call state-changing
    is withheld, whatever it is called.
    """
    run = scenario(
        rc=default_rc(
            execution_profile_plan_enabled=True,
            execution_profile_plan_tools=f"{OPEN},{DELEGATE}",
        ),
        tools=[_NamedTool(OPEN), _NamedTool(PUT), _NamedTool(SHELL)],
        tool_roles=_roles(),
        execution_profile="plan",
        tool_visibility_policy=ToolVisibilityPolicy(),
    )
    run.llm.queue_response(text="here is the plan")

    await run.run("plan the work")

    advertised = run.advertised_tool_names(0)
    assert OPEN in advertised, advertised
    assert PUT not in advertised, advertised
    assert SHELL not in advertised, advertised


def test_a_child_is_never_offered_a_tool_the_host_withheld() -> None:
    """Narrowing takes capability away and can never hand any back."""
    from protocore.contracts.tool_roles import narrow_tool_capabilities

    offered = (OPEN, PUT, DELEGATE)
    kept = narrow_tool_capabilities(offered, roles=_roles())

    assert DELEGATE not in kept
    assert kept <= set(offered)
    # Applying it again changes nothing: there is no path by which it widens.
    assert narrow_tool_capabilities(kept, roles=_roles()) == kept


async def test_the_advertised_surface_publishes_what_each_tool_does(
    scenario: ScenarioFactory,
) -> None:
    """A reader of the stream gets the roles, not just the names.

    Everything downstream that draws a tool — the red border on an approval
    card for a command, the dimmed surface of a plan-only run — was keeping its
    own list of names. A scope that renamed its shell tool got a destructive
    command rendered as something harmless. The map is published so no reader
    has to guess.
    """
    run = scenario(
        rc=default_rc(),
        tools=[_NamedTool(OPEN), _NamedTool(SHELL)],
        tool_roles=_roles(),
    )
    run.llm.queue_response(text="nothing to do")

    await run.run("hello")

    advertised = [
        evt for evt in run.events if evt.type.value == "tool_surface_advertised"
    ]
    assert advertised, [evt.type for evt in run.events]
    by_name = {entry["name"]: entry for entry in advertised[0].payload["tools"]}
    assert by_name[SHELL]["roles"] == ["runs_shell"]
    assert by_name[OPEN]["roles"] == ["reads_path"]
    assert "path" in advertised[0].payload["argument_names"]["path"]


async def test_a_pinned_read_stops_being_served_once_the_file_is_rewritten(
    scenario: ScenarioFactory,
) -> None:
    """The run itself makes the pinned result untrue, and the pin gives way.

    A pin is a request to keep a value in front of the model. It was never a
    request to keep it after the file it describes has been rewritten — but
    that is what it did, and the model went on reading the version before the
    write from the one place it is guaranteed to look. Here the same run reads
    the file, pins the result, writes the file, and the next request no longer
    carries the old body.

    The tool that did the rewriting is called nothing the core has heard of.
    What made it count is its declared role.
    """
    old_body = "the body before the rewrite"
    run = scenario(
        rc=default_rc(result_eviction_enabled=True, result_eviction_tool_names=()),
        tools=[
            _NamedTool(OPEN, content=old_body, pinned=True),
            _NamedTool(PUT, content="written"),
        ],
        tool_roles=_roles(),
    )
    run.llm.queue_tool_call_response(
        tool_call_id="t1", tool_name=OPEN, tool_input={"path": REPORT}
    )
    run.llm.queue_tool_call_response(
        tool_call_id="t2", tool_name=PUT, tool_input={"path": REPORT}
    )
    run.llm.queue_response(text="rewritten")

    await run.run("read it, then rewrite it")

    # It was pinned while it was still true: the request made between the read
    # and the write carries the body.
    assert any(old_body in text for text in run.request_texts(1))
    # After the write, it is not served again.
    assert not any(old_body in text for text in run.request_texts(-1))
