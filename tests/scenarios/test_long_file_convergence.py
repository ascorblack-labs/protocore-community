"""A run that has to finish writing one large file, and the switch that drives it.

The failure this exists for has a shape: a weak model at a small output cap
writes a truncated header, then inspects the file over and over without
appending or finalising, until the turn budget runs out and the file is left
half-written. Neither obvious recovery works — starting again re-truncates, and
telling the model the file is safe on disk reads to it as "done".

So the run drives the completion itself. What a caller can see of that is the
forced tool name on the next request, which is exactly what these scenarios
read: the requests the provider received, and the file the tools ended up with.
The pair with the switch off is the other half of the claim — nothing is forced
when nobody asked for it.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from protocore.contracts.llm import LLMRequest, LLMStreamEvent
from protocore.contracts.tool_roles import ToolArgumentSlot, ToolRole, ToolRoleMap
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import (
    TERMINAL_TOOL_METADATA_KEY,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
)

from .conftest import ScenarioFactory, default_rc

TARGET = "/workspace/big.py"
WRITE, APPEND, FINALIZE, READ = "Write", "AppendFile", "FinalizeFile", "Read"
#: The same four tools under a host's own names. Nothing about the convergence
#: driver may depend on which of these two sets a run is wired with.
RENAMED = ("CreateDoc", "ExtendDoc", "SealDoc", "OpenDoc")
FLOOR_BYTES = 4_096
HEADER = "x" * 1_000
CHUNK = "y" * 4_000


class _FileTool(Tool):
    """A file tool that reports the bytes it moved, the way a real one does.

    The convergence driver keys on bytes actually landing, so a tool that only
    said "ok" would make every scenario here vacuous.
    """

    def __init__(
        self,
        name: str,
        files: dict[str, str],
        *,
        kind: str,
        terminal: bool = False,
    ) -> None:
        self._name = name
        self._kind = kind
        self._files = files
        self._terminal = terminal

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        properties: dict[str, Any] = {"path": {"type": "string"}}
        required = ["path"]
        if self._kind in ("write", "append"):
            properties["content"] = {"type": "string"}
            required.append("content")
        return ToolDefinition(
            name=self._name,
            description=f"{self._name} a file",
            parameters=ToolParameterSchema(properties=properties, required=required),
        )

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        path = str(arguments.get("path", TARGET))
        content = str(arguments.get("content", ""))
        if self._kind == "write":
            self._files[path] = content
            payload: dict[str, Any] = {"path": path, "bytes_written": len(content.encode())}
        elif self._kind == "append":
            self._files[path] = self._files.get(path, "") + content
            payload = {
                "path": path,
                "bytes_appended": len(content.encode()),
                "bytes_total": len(self._files[path].encode()),
            }
        elif self._kind == "finalize":
            payload = {
                "path": path,
                "bytes_total": len(self._files.get(path, "").encode()),
            }
        else:
            payload = {"ok": True}
        return ToolResult(
            tool_call_id="tc",
            content=json.dumps(payload),
            is_error=False,
            metadata={TERMINAL_TOOL_METADATA_KEY: True} if self._terminal else {},
        )


def _calls_tool(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    *,
    cut_short: bool = False,
) -> list[LLMStreamEvent]:
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(
            name="tool_use_start",
            payload={"tool_call_id": call_id, "tool_name": name},
        ),
        LLMStreamEvent(
            name="tool_use_stop",
            payload={
                "tool_call_id": call_id,
                "final_input": arguments,
                "truncated_by_output_cap": cut_short,
            },
        ),
        LLMStreamEvent(
            name="message_stop",
            payload={"stop_reason": "length" if cut_short else "tool_use"},
        ),
    ]


def _says(text: str) -> list[LLMStreamEvent]:
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
        LLMStreamEvent(name="content_block_delta", payload={"text": text}),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"}),
    ]


class _ScriptedProvider:
    """Answers each request with the next scripted stream, repeating the last."""

    def __init__(self, scripts: list[list[LLMStreamEvent]]) -> None:
        self._scripts = scripts
        self._position = 0
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        script = self._scripts[min(self._position, len(self._scripts) - 1)]
        self._position += 1
        for event in script:
            yield event

    def count_tokens(self, text: str, model: str | None = None) -> int:
        return max(1, len(text) // 4)


def _converging_rc(**overrides: object) -> object:
    values: dict[str, object] = {
        "model_context_window": 8_192,
        "longfile_convergence_enabled": True,
        "longfile_stall_turns": 2,
        "longfile_expected_floor_bytes": FLOOR_BYTES,
        "longfile_min_finalize_fraction": 1.0,
        "longfile_max_forced_appends": 8,
        "longfile_max_forced_finalizes": 2,
    }
    values.update(overrides)
    return default_rc(**values)


def _header_then_idle_script() -> list[list[LLMStreamEvent]]:
    """The failing shape: a cut-off header, then inspection instead of writing."""
    return [
        _calls_tool("t1", WRITE, {"path": TARGET, "content": HEADER}, cut_short=True),
        _calls_tool("t2", READ, {"path": TARGET}),
        _calls_tool("t3", READ, {"path": TARGET}),
        _calls_tool("t4", APPEND, {"path": TARGET, "content": CHUNK}),
        _calls_tool("t5", READ, {"path": TARGET}),
        _calls_tool("t6", READ, {"path": TARGET}),
        _calls_tool("t7", FINALIZE, {"path": TARGET}),
        _says("done"),
    ]


def _file_tools(
    files: dict[str, str], names: tuple[str, str, str, str] = (WRITE, APPEND, FINALIZE, READ)
) -> list[Tool]:
    write, append, finalize, read = names
    return [
        _FileTool(write, files, kind="write"),
        _FileTool(append, files, kind="append"),
        _FileTool(finalize, files, kind="finalize", terminal=True),
        _FileTool(read, files, kind="read"),
    ]


def _roles(
    names: tuple[str, str, str, str] = (WRITE, APPEND, FINALIZE, READ),
) -> ToolRoleMap:
    """What the host says its four file tools do — the only thing core reads."""
    write, append, finalize, read = names
    return ToolRoleMap.declare(
        {
            write: [ToolRole.writes_path],
            append: [ToolRole.appends_path],
            finalize: [ToolRole.finalizes_path],
            read: [ToolRole.reads_path],
        },
        argument_aliases={
            ToolArgumentSlot.path: ["path"],
            ToolArgumentSlot.content: ["content"],
        },
    )


def _renamed_script() -> list[list[LLMStreamEvent]]:
    """The same failing shape, played by tools with the host's own names."""
    write, append, finalize, read = RENAMED
    return [
        _calls_tool("t1", write, {"path": TARGET, "content": HEADER}, cut_short=True),
        _calls_tool("t2", read, {"path": TARGET}),
        _calls_tool("t3", read, {"path": TARGET}),
        _calls_tool("t4", append, {"path": TARGET, "content": CHUNK}),
        _calls_tool("t5", read, {"path": TARGET}),
        _calls_tool("t6", read, {"path": TARGET}),
        _calls_tool("t7", finalize, {"path": TARGET}),
        _says("done"),
    ]


async def test_a_run_that_stalls_mid_file_is_driven_to_a_finished_one(
    scenario: ScenarioFactory,
) -> None:
    """The run asks for the append, then for the seal, and the file is whole.

    Everything asserted here is what a caller sees: the tool name each request
    carried as the forced choice, and the bytes the tools were left holding.
    """
    files: dict[str, str] = {}
    provider = _ScriptedProvider(_header_then_idle_script())
    run = scenario(
        rc=_converging_rc(),
        tools=_file_tools(files),
        tool_roles=_roles(),
        llm_provider=provider,
    )

    await run.run("write the whole file")

    forced = [request.extra.get("forced_tool_choice") for request in run.requests]
    assert APPEND in forced, forced
    assert FINALIZE in forced, forced
    assert forced.index(APPEND) < forced.index(FINALIZE)
    assert len(files[TARGET].encode()) >= FLOOR_BYTES


async def test_with_convergence_off_the_run_asks_for_nothing(
    scenario: ScenarioFactory,
) -> None:
    """The same stall, the switch off: the run leaves the model to it.

    The file is left where the model left it, which is the point — the driving
    is the behaviour the switch turns on, not a property of the loop.
    """
    files: dict[str, str] = {}
    provider = _ScriptedProvider(_header_then_idle_script())
    run = scenario(
        rc=_converging_rc(longfile_convergence_enabled=False),
        tools=_file_tools(files),
        tool_roles=_roles(),
        llm_provider=provider,
    )

    await run.run("write the whole file")

    forced = [request.extra.get("forced_tool_choice") for request in run.requests]
    assert set(forced) == {None}, forced


async def test_the_driver_follows_the_host_names_not_its_own(
    scenario: ScenarioFactory,
) -> None:
    """Rename all four tools and the run converges exactly as before.

    The whole point of stating roles: an installation that calls its file tools
    something else keeps the driving. Before roles this run went quiet — every
    comparison in the driver was against a name it no longer saw — and the file
    was left half-written with nothing in the transcript saying why.
    """
    _write, append, finalize, read = RENAMED
    files: dict[str, str] = {}
    provider = _ScriptedProvider(_renamed_script())
    run = scenario(
        rc=_converging_rc(),
        tools=_file_tools(files, RENAMED),
        tool_roles=_roles(RENAMED),
        llm_provider=provider,
    )

    await run.run("write the whole file")

    forced = [request.extra.get("forced_tool_choice") for request in run.requests]
    assert append in forced, forced
    assert finalize in forced, forced
    assert forced.index(append) < forced.index(finalize)
    assert len(files[TARGET].encode()) >= FLOOR_BYTES
    # The core's own former spellings never appear: nothing was forced by name.
    assert WRITE not in forced and APPEND not in forced and FINALIZE not in forced
    assert read not in forced


async def test_a_run_whose_host_declared_no_roles_is_not_driven_silently(
    scenario: ScenarioFactory,
    caplog: Any,
) -> None:
    """No roles declared, convergence on: nothing is forced, and it is logged.

    This is the state the map exists to make visible. The run behaves as if the
    driver were off, but the log names what could not be resolved, so the
    misconfiguration is diagnosable instead of being a file that came out
    half-written for no stated reason.
    """
    files: dict[str, str] = {}
    provider = _ScriptedProvider(_header_then_idle_script())
    run = scenario(
        rc=_converging_rc(),
        tools=_file_tools(files),
        llm_provider=provider,
    )

    with caplog.at_level("WARNING"):
        await run.run("write the whole file")

    forced = [request.extra.get("forced_tool_choice") for request in run.requests]
    assert set(forced) == {None}, forced
    assert any(
        "no_byte_producing_tools" in record.getMessage() for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


async def test_the_forced_round_is_granted_the_message_it_needs(
    scenario: ScenarioFactory,
) -> None:
    """A forced round at the message budget must be given a slot to run in.

    The convergence forces the next tool at the very turn the cap would end
    the run on. Without the grant the forced round is charged and its history
    written, and then killed before the model ever sees it: the budget spent,
    the transcript mutated, and no output. What a caller can see of the grant
    is the forced tool name arriving on a request at all.
    """
    files: dict[str, str] = {}
    provider = _ScriptedProvider(_header_then_idle_script())
    run = scenario(
        rc=_converging_rc(max_turns_per_run=2, soft_stop_enabled=False),
        tools=_file_tools(files),
        tool_roles=_roles(),
        llm_provider=provider,
    )

    await run.run("write the whole file")

    forced = [request.extra.get("forced_tool_choice") for request in run.requests]
    assert APPEND in forced, forced
    # The forced round is the one past the cap: it exists because it was
    # granted, not because the budget had room for it.
    assert forced.index(APPEND) == len(forced) - 1
    assert "longfile_forced_append" in run.state_reasons()


async def test_a_prose_turn_that_forces_a_tool_restarts_the_loop(
    scenario: ScenarioFactory,
) -> None:
    """The other half of the convergence: a turn that said it was finished.

    A model that answers "the file is written, I am done" over a file that is
    not written has produced no tool call, so there is nothing for the loop to
    fall through to. The convergence sends the turn round again with the next
    tool forced and the context rebuilt — and what a caller sees of that is
    the forced tool name on a request that would not otherwise exist.
    """
    files: dict[str, str] = {}
    provider = _ScriptedProvider(
        [
            _calls_tool(
                "t1", WRITE, {"path": TARGET, "content": HEADER}, cut_short=True
            ),
            _calls_tool("t2", READ, {"path": TARGET}),
            _says("the file is written, I am done"),
            _says("still done"),
        ]
    )
    run = scenario(
        rc=_converging_rc(),
        tools=_file_tools(files),
        tool_roles=_roles(),
        llm_provider=provider,
    )

    await run.run("write the whole file")

    forced = [request.extra.get("forced_tool_choice") for request in run.requests]
    assert forced[-1] == APPEND, forced
    assert "longfile_forced_append" in run.state_reasons()
