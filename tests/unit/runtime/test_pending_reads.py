"""The declared-file read-back gate — the pure driver.

A tool result may declare paths the caller must open before it continues
(``PENDING_READS_METADATA_KEY``). While one is unread the driver names the
workspace read tool for the forced ``tool_choice`` slot; the moment the last
one is read it stops. These tests cover the decision logic in isolation — the
loop-level property (an engine cannot answer until it has read) lives in
``test_pending_reads_forced_read_gate.py``.
"""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import PENDING_READS_METADATA_KEY, ToolCall
from protocore.runtime import pending_reads as _pending_reads


def _delegation(paths: list[str]) -> tuple[ToolCall, dict[str, object]]:
    """A tool call + result metadata declaring ``paths`` as read-back debt."""
    return (
        ToolCall(id="call-1", name="Agent", arguments={"subagent_type": "worker"}),
        {PENDING_READS_METADATA_KEY: paths},
    )


def _read(path: str, *, call_id: str = "read-1") -> ToolCall:
    return ToolCall(id=call_id, name="Read", arguments={"file_path": path})


def _observe(engine, tool_call: ToolCall, metadata=None, *, is_error: bool = False):
    _pending_reads.observe_tool_result(
        engine, tool_call, metadata, is_error=is_error
    )


def test_declared_paths_engage_and_force_the_read_tool(engine_factory) -> None:
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=True)
    )
    call, metadata = _delegation(["reports/a.md", "reports/b.md"])

    _observe(engine, call, metadata)

    assert _pending_reads.pending_paths(engine) == ("reports/a.md", "reports/b.md")
    assert _pending_reads.peek_forced_tool(engine) == "Read"


def test_gate_releases_itself_when_the_last_path_is_read(engine_factory) -> None:
    """The whole point: no operator action, no timeout — reading is the release."""
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=True)
    )
    call, metadata = _delegation(["reports/a.md", "reports/b.md"])
    _observe(engine, call, metadata)

    _observe(engine, _read("reports/a.md"))
    assert _pending_reads.pending_paths(engine) == ("reports/b.md",)
    assert _pending_reads.peek_forced_tool(engine) == "Read"

    _observe(engine, _read("reports/b.md", call_id="read-2"))
    assert _pending_reads.pending_paths(engine) == ()
    assert _pending_reads.peek_forced_tool(engine) is None


def test_several_declarations_accumulate(engine_factory) -> None:
    """A fan-out of delegations owes every set of files, not just the last."""
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=True)
    )
    _observe(
        engine,
        ToolCall(id="c1", name="Agent", arguments={}),
        {PENDING_READS_METADATA_KEY: ["one.md"]},
    )
    _observe(
        engine,
        ToolCall(id="c2", name="Agent", arguments={}),
        {PENDING_READS_METADATA_KEY: ["two.md", "three.md"]},
    )

    assert _pending_reads.pending_paths(engine) == ("one.md", "two.md", "three.md")


def test_a_path_read_earlier_in_the_run_is_never_forced(engine_factory) -> None:
    """Re-reading a file the agent already opened is a wasted turn, not a gate."""
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=True)
    )
    _observe(engine, _read("reports/a.md"))

    call, metadata = _delegation(["reports/a.md", "reports/b.md"])
    _observe(engine, call, metadata)

    assert _pending_reads.pending_paths(engine) == ("reports/b.md",)


def test_a_read_of_the_same_file_under_another_prefix_releases(engine_factory) -> None:
    """The tool declares what it wrote, the model types what it reads; the two
    forms differ by a workspace prefix and mean the same file."""
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=True)
    )
    call, metadata = _delegation(["reports/audit.md"])
    _observe(engine, call, metadata)

    _observe(engine, _read("/workspace/reports/audit.md"))

    assert _pending_reads.pending_paths(engine) == ()


def test_a_same_named_file_in_another_directory_does_not_release(
    engine_factory,
) -> None:
    """Matching on the bare filename would call two different files the same
    one — precisely the confusion the gate exists to prevent."""
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=True)
    )
    call, metadata = _delegation(["alpha/audit.md"])
    _observe(engine, call, metadata)

    _observe(engine, _read("beta/audit.md"))

    assert _pending_reads.pending_paths(engine) == ("alpha/audit.md",)


def test_duplicate_declarations_are_not_double_tracked(engine_factory) -> None:
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=True)
    )
    call, metadata = _delegation(["a.md", "a.md", "./a.md"])
    _observe(engine, call, metadata)

    assert _pending_reads.pending_paths(engine) == ("a.md",)


def test_an_errored_tool_result_never_engages_the_gate(engine_factory) -> None:
    """A tool that failed may name files it never finished writing; forcing
    reads of those spends the whole budget on files that cannot exist."""
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=True)
    )
    call, metadata = _delegation(["never-written.md"])

    _observe(engine, call, metadata, is_error=True)

    assert _pending_reads.pending_paths(engine) == ()
    assert _pending_reads.peek_forced_tool(engine) is None


def test_an_errored_read_does_not_release_the_path(engine_factory) -> None:
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=True)
    )
    call, metadata = _delegation(["missing.md"])
    _observe(engine, call, metadata)

    _observe(engine, _read("missing.md"), is_error=True)

    assert _pending_reads.pending_paths(engine) == ("missing.md",)


def test_malformed_declarations_are_ignored_not_raised(engine_factory) -> None:
    """The value crosses a tool boundary; a third-party tool returning the
    wrong shape must not be able to fail the run."""
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=True)
    )
    call = ToolCall(id="c1", name="Agent", arguments={})

    for bad in ("a string", 42, {"path": "a.md"}, None, ["", "   ", 7]):
        _observe(engine, call, {PENDING_READS_METADATA_KEY: bad})

    assert _pending_reads.pending_paths(engine) == ()


def test_the_pending_set_is_capped(engine_factory) -> None:
    """The set rides on every run snapshot, so a pathological list cannot grow
    it without limit."""
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_max_paths=3, pending_reads_enabled=True)
    )
    call, metadata = _delegation([f"file-{i}.md" for i in range(10)])
    _observe(engine, call, metadata)

    assert _pending_reads.pending_paths(engine) == (
        "file-0.md",
        "file-1.md",
        "file-2.md",
    )


def test_the_attempt_bound_releases_rather_than_wedging(engine_factory) -> None:
    """A file that cannot be read costs a bounded number of turns and then
    nothing at all — the agent gets its whole surface back."""
    engine = engine_factory(
        rc=RuntimeConstants(
            model_context_window=4_096, pending_reads_max_forced_attempts=2
        , pending_reads_enabled=True)
    )
    call, metadata = _delegation(["missing.md"])
    _observe(engine, call, metadata)

    assert _pending_reads.release_exhausted(engine) == ()
    _pending_reads.charge_forced_attempt(engine)
    assert _pending_reads.peek_forced_tool(engine) == "Read"
    _pending_reads.charge_forced_attempt(engine)

    # Budget spent — the driver stops asking for the slot and the gate opens.
    assert _pending_reads.peek_forced_tool(engine) is None
    assert _pending_reads.release_exhausted(engine) == ("missing.md",)
    assert _pending_reads.pending_paths(engine) == ()
    assert _pending_reads.peek_forced_tool(engine) is None


def test_a_released_path_is_never_forced_again(engine_factory) -> None:
    """A tool that keeps re-declaring an unreadable file cannot re-engage."""
    engine = engine_factory(
        rc=RuntimeConstants(
            model_context_window=4_096, pending_reads_max_forced_attempts=1
        , pending_reads_enabled=True)
    )
    call, metadata = _delegation(["missing.md"])
    _observe(engine, call, metadata)
    _pending_reads.charge_forced_attempt(engine)
    assert _pending_reads.release_exhausted(engine) == ("missing.md",)

    _observe(engine, call, metadata)

    assert _pending_reads.pending_paths(engine) == ()


def test_a_later_declaration_of_another_file_still_engages(engine_factory) -> None:
    """The budget bounds one unproductive streak, not the mechanism for the
    rest of the run."""
    engine = engine_factory(
        rc=RuntimeConstants(
            model_context_window=4_096, pending_reads_max_forced_attempts=1
        , pending_reads_enabled=True)
    )
    first_call, first_metadata = _delegation(["missing.md"])
    _observe(engine, first_call, first_metadata)
    _pending_reads.charge_forced_attempt(engine)
    _pending_reads.release_exhausted(engine)

    second_call, second_metadata = _delegation(["written.md"])
    _observe(engine, second_call, second_metadata)

    assert _pending_reads.pending_paths(engine) == ("written.md",)
    assert _pending_reads.peek_forced_tool(engine) == "Read"


def test_a_productive_read_buys_back_the_whole_budget(engine_factory) -> None:
    """A caller working through five declared files is never starved by its own
    progress — what is bounded is the run of turns that cleared NOTHING."""
    engine = engine_factory(
        rc=RuntimeConstants(
            model_context_window=4_096, pending_reads_max_forced_attempts=2
        , pending_reads_enabled=True)
    )
    call, metadata = _delegation(["a.md", "b.md"])
    _observe(engine, call, metadata)

    _pending_reads.charge_forced_attempt(engine)
    _observe(engine, _read("a.md"))
    _pending_reads.charge_forced_attempt(engine)

    assert _pending_reads.peek_forced_tool(engine) == "Read"
    assert _pending_reads.release_exhausted(engine) == ()


def test_kill_switch_off_is_a_full_no_op(engine_factory) -> None:
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=False)
    )
    call, metadata = _delegation(["a.md", "b.md"])

    _observe(engine, call, metadata)
    _observe(engine, _read("a.md"))

    assert _pending_reads.pending_paths(engine) == ()
    assert _pending_reads.peek_forced_tool(engine) is None
    assert _pending_reads.release_exhausted(engine) == ()


@pytest.mark.asyncio
async def test_pending_state_survives_a_snapshot_round_trip(engine_factory) -> None:
    """A run re-driven on another pod still owes the reads it owed, still knows
    what it has opened, and gets no fresh attempt budget."""
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=True)
    )
    call, metadata = _delegation(["a.md", "b.md"])
    _observe(engine, call, metadata)
    _observe(engine, _read("a.md"))
    _pending_reads.charge_forced_attempt(engine)

    snapshot = engine.snapshot()
    resumed = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=True)
    )
    await resumed.resume_from_snapshot(snapshot)

    assert _pending_reads.pending_paths(resumed) == ("b.md",)
    assert _pending_reads.peek_forced_tool(resumed) == "Read"
    # The already-read path is not re-forced after the resume either.
    _observe(resumed, call, metadata)
    assert _pending_reads.pending_paths(resumed) == ("b.md",)


@pytest.mark.parametrize(
    ("declared", "read_path", "expected"),
    [
        ("a.md", "a.md", ()),
        ("./a.md", "a.md", ()),
        ("a.md", "./a.md", ()),
        ("dir/a.md", "dir\\a.md", ()),
        ("dir/a.md", "a.md", ()),
        ("a.md", "other.md", ("a.md",)),
    ],
)
def test_path_form_equivalence(
    engine_factory, declared: str, read_path: str, expected: tuple[str, ...]
) -> None:
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=True)
    )
    _observe(
        engine,
        ToolCall(id="c1", name="Agent", arguments={}),
        {PENDING_READS_METADATA_KEY: [declared]},
    )
    _observe(engine, _read(read_path))
    assert _pending_reads.pending_paths(engine) == expected
