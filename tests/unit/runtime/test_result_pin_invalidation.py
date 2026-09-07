"""A pinned result stops being kept once the run makes it untrue.

Pinning says "keep this in front of the model". It never said "keep it after
the file it describes has been rewritten", but that is what it did: pin the
result of reading a file, let the agent rewrite that file two turns later, and
the context goes on holding the version before the rewrite — in the one place
the model is guaranteed to look. Nothing lifted the pin, because nothing was
watching for the write.

These tests watch for the write. They also fix which QUESTION is asked: what a
tool DOES, declared by the host as a role, never what it happens to be called.
"""

from __future__ import annotations

import json

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_roles import ToolRole, ToolRoleMap
from protocore.contracts.types import (
    Message,
    MessageRole,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.prompts import bundled_prompt_provider
from protocore.runtime.result_eviction import (
    evict_history_for_llm,
    evictable_tool_names,
    pins_invalidated_by_writes,
)

_READING_ROLES = ToolRoleMap.declare(
    {
        "Read": [ToolRole.reads_path],
        "Write": [ToolRole.writes_path],
        "Edit": [ToolRole.edits_path],
        "Bash": [ToolRole.runs_shell],
    }
)

#: A host that spells its tools nothing like the one above. Same roles.
_RENAMED_ROLES = ToolRoleMap.declare(
    {
        "open_file": [ToolRole.reads_path],
        "put_file": [ToolRole.writes_path],
    }
)


def _rc(**overrides: object) -> LoopConstants:
    return LoopConstants(
        result_eviction_enabled=True,
        result_eviction_tool_names=(),
        **overrides,  # type: ignore[arg-type]
    )


def _call(name: str, call_id: str, path: str | None = None) -> Message:
    return Message(
        role=MessageRole.assistant,
        content_blocks=[
            ToolUseBlock(
                tool_call_id=call_id,
                name=name,
                arguments_json=json.dumps({"path": path} if path else {}),
            )
        ],
    )


def _result(
    call_id: str, content: str, *, path: str | None = None, pinned: bool = False
) -> Message:
    return Message(
        role=MessageRole.tool,
        content_blocks=[
            ToolResultBlock(
                tool_call_id=call_id,
                content=content,
                path=path,
                metadata={"retention": "pinned"} if pinned else {},
            )
        ],
    )


def _contents(history: list[Message]) -> list[str]:
    return [
        block.content
        for message in history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    ]


# ----------------------------------------------------------------------
# The rule
# ----------------------------------------------------------------------


def test_a_pinned_read_loses_its_pin_when_the_same_path_is_written() -> None:
    history = [
        _call("Read", "c1", "/w/app.py"),
        _result("c1", "the old body", path="/w/app.py", pinned=True),
        _call("Write", "c2", "/w/app.py"),
        _result("c2", "wrote /w/app.py", path="/w/app.py"),
    ]

    assert pins_invalidated_by_writes(history, roles=_READING_ROLES) == {"c1"}

    view, evicted = evict_history_for_llm(
        history, _rc(), bundled_prompt_provider(), roles=_READING_ROLES
    )

    assert evicted == ["c1"]
    assert "the old body" not in _contents(view)


def test_a_pin_on_a_different_path_is_left_alone() -> None:
    history = [
        _call("Read", "c1", "/w/other.py"),
        _result("c1", "the other body", path="/w/other.py", pinned=True),
        _call("Write", "c2", "/w/app.py"),
        _result("c2", "wrote /w/app.py", path="/w/app.py"),
    ]

    assert pins_invalidated_by_writes(history, roles=_READING_ROLES) == frozenset()

    view, evicted = evict_history_for_llm(
        history, _rc(), bundled_prompt_provider(), roles=_READING_ROLES
    )

    assert evicted == []
    assert "the other body" in _contents(view)


def test_a_pin_that_names_no_path_is_never_invalidated() -> None:
    """Nothing said this result was about a file, so no write can falsify it."""
    history = [
        _call("Read", "c1"),
        _result("c1", "a summary of something", pinned=True),
        _call("Write", "c2", "/w/app.py"),
        _result("c2", "wrote /w/app.py", path="/w/app.py"),
    ]

    assert pins_invalidated_by_writes(history, roles=_READING_ROLES) == frozenset()

    view, evicted = evict_history_for_llm(
        history, _rc(), bundled_prompt_provider(), roles=_READING_ROLES
    )

    assert evicted == [] and "a summary of something" in _contents(view)


def test_the_write_must_come_after_the_result_it_falsifies() -> None:
    """A read taken after the write is the current view, and it stays pinned.

    This is also the way back: re-read the file and the pin means something
    again. Were order ignored, re-reading would be pointless — the same write
    would keep invalidating every later view of the file forever.
    """
    history = [
        _call("Write", "c1", "/w/app.py"),
        _result("c1", "wrote /w/app.py", path="/w/app.py"),
        _call("Read", "c2", "/w/app.py"),
        _result("c2", "the new body", path="/w/app.py", pinned=True),
    ]

    assert pins_invalidated_by_writes(history, roles=_READING_ROLES) == frozenset()

    view, evicted = evict_history_for_llm(
        history, _rc(), bundled_prompt_provider(), roles=_READING_ROLES
    )

    assert evicted == [] and "the new body" in _contents(view)


def test_an_edit_falsifies_a_pinned_read_exactly_as_a_write_does() -> None:
    """Part of a file is still the file. A rule that only noticed whole-file
    writes would keep feeding the model the version before the edit."""
    history = [
        _call("Read", "c1", "/w/app.py"),
        _result("c1", "the old body", path="/w/app.py", pinned=True),
        _call("Edit", "c2", "/w/app.py"),
        _result("c2", "edited /w/app.py", path="/w/app.py"),
    ]

    assert pins_invalidated_by_writes(history, roles=_READING_ROLES) == {"c1"}


def test_a_call_that_changes_nothing_leaves_every_pin_standing() -> None:
    history = [
        _call("Read", "c1", "/w/app.py"),
        _result("c1", "the body", path="/w/app.py", pinned=True),
        _call("Bash", "c2", "/w/app.py"),
        _result("c2", "ran it", path="/w/app.py"),
    ]

    assert pins_invalidated_by_writes(history, roles=_READING_ROLES) == frozenset()


# ----------------------------------------------------------------------
# Roles, not names
# ----------------------------------------------------------------------


def test_a_host_that_names_its_tools_differently_keeps_the_behaviour() -> None:
    """The rule is about what the call DID. Names are the host's business."""
    history = [
        _call("open_file", "c1", "/w/app.py"),
        _result("c1", "the old body", path="/w/app.py", pinned=True),
        _call("put_file", "c2", "/w/app.py"),
        _result("c2", "wrote it", path="/w/app.py"),
    ]

    assert pins_invalidated_by_writes(history, roles=_RENAMED_ROLES) == {"c1"}


def test_a_host_that_declared_no_roles_invalidates_nothing() -> None:
    """No declaration is not a licence to guess. It is an installation whose
    runtime was told nothing, and inventing a rule for it is how the core
    ended up knowing tool names in the first place."""
    history = [
        _call("Read", "c1", "/w/app.py"),
        _result("c1", "the old body", path="/w/app.py", pinned=True),
        _call("Write", "c2", "/w/app.py"),
    ]

    assert pins_invalidated_by_writes(history) == frozenset()


# ----------------------------------------------------------------------
# Matching one file under two spellings
# ----------------------------------------------------------------------


def test_two_spellings_of_one_path_are_one_file() -> None:
    history = [
        _call("Read", "c1", "./w/app.py"),
        _result("c1", "the old body", path="./w/app.py", pinned=True),
        _call("Write", "c2", "w/app.py"),
    ]

    assert pins_invalidated_by_writes(history, roles=_READING_ROLES) == {"c1"}


def test_a_result_that_states_no_path_inherits_the_one_its_call_named() -> None:
    """A tool that did not fill the field still made a call that named a file,
    and the transcript kept the call. Reading only the result's own field would
    exempt every tool that has not been taught to state one."""
    history = [
        _call("Read", "c1", "/w/app.py"),
        _result("c1", "the old body", pinned=True),
        _call("Write", "c2", "/w/app.py"),
    ]

    assert pins_invalidated_by_writes(history, roles=_READING_ROLES) == {"c1"}


def test_an_invalidated_pin_loses_to_every_way_of_asking_for_it() -> None:
    """The engine's own pin set is not a stronger claim than the block's mark;
    both are requests to keep a value that is no longer true."""
    history = [
        _call("Read", "c1", "/w/app.py"),
        _result("c1", "the old body", path="/w/app.py"),
        _call("Write", "c2", "/w/app.py"),
    ]

    _, evicted = evict_history_for_llm(
        history, _rc(), bundled_prompt_provider(), pinned_ids=["c1"], roles=_READING_ROLES
    )

    assert evicted == ["c1"]


def _parallel_calls(*calls: tuple[str, str, str]) -> Message:
    """One assistant message asking for several tools at once."""
    return Message(
        role=MessageRole.assistant,
        content_blocks=[
            ToolUseBlock(
                tool_call_id=call_id,
                name=name,
                arguments_json=json.dumps({"path": path}),
            )
            for name, call_id, path in calls
        ],
    )


def _parallel_results(*results: tuple[str, str, bool]) -> Message:
    """One tool-role message answering several tool_use turns."""
    return Message(
        role=MessageRole.tool,
        content_blocks=[
            ToolResultBlock(
                tool_call_id=call_id,
                content=content,
                path="/w/app.py",
                is_error=is_error,
                metadata={"retention": "pinned"},
            )
            for call_id, content, is_error in results
        ],
    )


def test_a_write_beside_the_read_in_one_message_still_lifts_the_pin() -> None:
    """The model may ask for both at once; the read is still the older version."""
    history = [
        _parallel_calls(("Read", "c1", "/w/app.py"), ("Write", "c2", "/w/app.py")),
        _parallel_results(("c1", "the old body", False), ("c2", "written", False)),
    ]

    assert pins_invalidated_by_writes(history, roles=_READING_ROLES) == {"c1"}


def test_a_read_taken_after_the_write_in_one_message_is_kept() -> None:
    history = [
        _parallel_calls(("Write", "c1", "/w/app.py"), ("Read", "c2", "/w/app.py")),
        _parallel_results(("c1", "written", False), ("c2", "the new body", False)),
    ]

    assert pins_invalidated_by_writes(history, roles=_READING_ROLES) == frozenset()


def test_a_write_that_failed_lifts_nothing() -> None:
    """It changed no file, so the read it would have falsified is still true."""
    history = [
        _call("Read", "c1", "/w/app.py"),
        _result("c1", "the body", path="/w/app.py", pinned=True),
        _call("Write", "c2", "/w/app.py"),
        _result("c2", "permission denied", path="/w/app.py"),
    ]
    history[-1] = Message(
        role=MessageRole.tool,
        content_blocks=[
            ToolResultBlock(
                tool_call_id="c2",
                content="permission denied",
                path="/w/app.py",
                is_error=True,
            )
        ],
    )

    assert pins_invalidated_by_writes(history, roles=_READING_ROLES) == frozenset()


# ── which tools the rule applies to ─────────────────────────────────────────


def test_names_given_outright_are_the_whole_set() -> None:
    rc = LoopConstants(result_eviction_tool_names=("Fetch",))
    roles = ToolRoleMap.declare({"Open": [ToolRole.reads_path]})

    assert evictable_tool_names(rc, roles) == frozenset({"Fetch"})


def test_naming_nothing_falls_back_to_the_roles_and_widens_the_set() -> None:
    """An empty list says nothing about which tools, so the roles answer.

    It reads like a switch and is not one: the fallback is usually WIDER than
    the names it replaces, so an operator who emptied the list to stop eviction
    got more of it. Turning eviction off is its own constant.
    """
    rc = LoopConstants(result_eviction_tool_names=())
    roles = ToolRoleMap.declare(
        {
            "Open": [ToolRole.reads_path],
            "Sift": [ToolRole.searches_workspace],
            "Put": [ToolRole.writes_path],
        }
    )

    assert evictable_tool_names(rc, roles) == frozenset({"Open", "Sift"})
