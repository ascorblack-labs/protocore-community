"""Unit tests for the DAG-precondition mechanism.

Covers the pure functions in
:mod:`protocore.runtime.tool_preconditions`:

* :func:`check_preconditions` — bare-name, parameterised and prefix patterns.
* :func:`record_satisfaction` — bare-name + path-keyed entries.
* :func:`resolve_precondition` — path normalisation.
* :func:`compute_masked_tools` — only bare-name patterns are masked
 pre-emptively.
* the run state's satisfied set — the cross-call round trip a dispatch makes.
"""

from __future__ import annotations

from typing import Any

from protocore.contracts.run_state import RunScopedState
from protocore.runtime.tool_preconditions import (
    check_preconditions,
    compute_masked_tools,
    derive_satisfied_from_messages,
    record_satisfaction,
    resolve_precondition,
)

# ----------------------------------------------------------------------
# check_preconditions
# ----------------------------------------------------------------------


def test_check_preconditions_empty_returns_none() -> None:
    """No preconditions → always satisfied."""
    assert check_preconditions(preconditions=[], arguments={}, satisfied=set()) is None


def test_check_preconditions_bare_satisfied() -> None:
    """A bare ``"tool"`` precondition is satisfied if the tool is in the set."""
    assert (
        check_preconditions(
            preconditions=["AppendFile"],
            arguments={"path": "x.py"},
            satisfied={"AppendFile"},
        )
        is None
    )


def test_check_preconditions_bare_unsatisfied() -> None:
    """A missing bare precondition surfaces a denial reason."""
    reason = check_preconditions(
        preconditions=["AppendFile"],
        arguments={"path": "x.py"},
        satisfied=set(),
    )
    assert reason is not None
    assert "AppendFile" in reason
    assert "Required tool must be called first" in reason


def test_check_preconditions_parameterised_satisfied() -> None:
    """``"tool:{path}"`` substitutes from current call arguments."""
    assert (
        check_preconditions(
            preconditions=["AppendFile:{path}"],
            arguments={"path": "src/big.py"},
            satisfied={"AppendFile:src/big.py"},
        )
        is None
    )


def test_check_preconditions_parameterised_unsatisfied() -> None:
    """Mismatched paths produce a denial."""
    reason = check_preconditions(
        preconditions=["AppendFile:{path}"],
        arguments={"path": "src/big.py"},
        satisfied={"AppendFile:src/other.py"},
    )
    assert reason is not None
    assert "src/big.py" in reason


def test_check_preconditions_prefix_satisfied() -> None:
    """Prefix patterns match any satisfied entry beginning with the prefix."""
    assert (
        check_preconditions(
            preconditions=["AppendFile:src/*"],
            arguments={},
            satisfied={"AppendFile:src/deep/module.py"},
        )
        is None
    )


def test_check_preconditions_prefix_unsatisfied() -> None:
    """Prefix patterns fail when no entry starts with the prefix."""
    reason = check_preconditions(
        preconditions=["AppendFile:src/*"],
        arguments={},
        satisfied={"AppendFile:tests/x.py"},
    )
    assert reason is not None


def test_check_preconditions_path_normalisation() -> None:
    """``src/./big.py`` and ``src/big.py`` are equivalent."""
    assert (
        check_preconditions(
            preconditions=["AppendFile:{path}"],
            arguments={"path": "src/./big.py"},
            satisfied={"AppendFile:src/big.py"},
        )
        is None
    )


def test_check_preconditions_unresolved_placeholder_treated_literally() -> None:
    """Missing argument keeps the literal ``{key}`` so it can never match."""
    reason = check_preconditions(
        preconditions=["AppendFile:{path}"],
        arguments={},  # no `path` key
        satisfied={"AppendFile:foo.py"},
    )
    assert reason is not None
    assert "{path}" in reason


# ----------------------------------------------------------------------
# record_satisfaction
# ----------------------------------------------------------------------


def test_record_satisfaction_bare_and_path() -> None:
    """Recording adds both bare tool name and tool:path entry."""
    satisfied: set[str] = set()
    record_satisfaction(
        tool_name="AppendFile",
        arguments={"path": "src/big.py"},
        satisfied=satisfied,
    )
    assert "AppendFile" in satisfied
    assert "AppendFile:src/big.py" in satisfied


def test_record_satisfaction_no_path_fields() -> None:
    """Arguments without a recognised path field only record bare name."""
    satisfied: set[str] = set()
    record_satisfaction(
        tool_name="Write",
        arguments={"content": "hello"},
        satisfied=satisfied,
    )
    assert satisfied == {"Write"}


def test_record_satisfaction_multiple_path_fields() -> None:
    """Each recognised path field records its own tool:path entry."""
    satisfied: set[str] = set()
    record_satisfaction(
        tool_name="copy_path",
        arguments={"source_path": "src/a.py", "destination_path": "dst/a.py"},
        satisfied=satisfied,
    )
    # copy_path has a custom suffix mapping → only destination is recorded.
    assert "copy_path" in satisfied
    assert "copy_path:dst/a.py" in satisfied
    assert "copy_path:src/a.py" not in satisfied


def test_record_satisfaction_explicit_path_fields_override() -> None:
    """Passing ``path_fields`` overrides the default list."""
    satisfied: set[str] = set()
    record_satisfaction(
        tool_name="Custom",
        arguments={"path": "x", "custom_path": "y"},
        satisfied=satisfied,
        path_fields=["custom_path"],
    )
    assert "Custom" in satisfied
    assert "Custom:y" in satisfied
    assert "Custom:x" not in satisfied


def test_record_then_check_round_trip() -> None:
    """End-to-end: record → check returns satisfied."""
    satisfied: set[str] = set()
    record_satisfaction(
        tool_name="AppendFile",
        arguments={"path": "/workspace/big.py"},
        satisfied=satisfied,
    )
    assert (
        check_preconditions(
            preconditions=["AppendFile:{path}"],
            arguments={"path": "/workspace/big.py"},
            satisfied=satisfied,
        )
        is None
    )


# ----------------------------------------------------------------------
# resolve_precondition
# ----------------------------------------------------------------------


def test_resolve_precondition_substitutes_param() -> None:
    assert (
        resolve_precondition("AppendFile:{path}", {"path": "x.py"})
        == "AppendFile:x.py"
    )


def test_resolve_precondition_normalises_path() -> None:
    """Path-field values are run through ``posixpath.normpath``."""
    assert (
        resolve_precondition("AppendFile:{path}", {"path": "src/./big.py"})
        == "AppendFile:src/big.py"
    )


def test_resolve_precondition_keeps_unresolved_placeholder() -> None:
    """Missing argument leaves the literal placeholder in place."""
    assert (
        resolve_precondition("AppendFile:{path}", {})
        == "AppendFile:{path}"
    )


# ----------------------------------------------------------------------
# compute_masked_tools
# ----------------------------------------------------------------------


def test_compute_masked_tools_bare_only() -> None:
    """Only bare-name preconditions are checked pre-emptively."""

    class FakeTool:
        def __init__(self, name: str, preconditions: list[str] | None) -> None:
            self.name = name
            self.preconditions = preconditions

    tools = [
        FakeTool("FinalizeFile", ["AppendFile"]),
        FakeTool("RecallArtifact", ["AppendFile:{path}"]),  # parameterised
        FakeTool("Write", None),
    ]
    masked = compute_masked_tools(tool_definitions=tools, satisfied=set())
    # FinalizeFile is masked because AppendFile is missing.
    assert "FinalizeFile" in masked
    # RecallArtifact has only a parameterised pattern → not masked.
    assert "RecallArtifact" not in masked
    # Write has no preconditions → not masked.
    assert "Write" not in masked


def test_compute_masked_tools_unmasks_when_satisfied() -> None:
    """Tools become available once their bare prerequisites are recorded."""

    class FakeTool:
        def __init__(self, name: str, preconditions: list[str] | None) -> None:
            self.name = name
            self.preconditions = preconditions

    tools = [FakeTool("FinalizeFile", ["AppendFile"])]
    assert "FinalizeFile" not in compute_masked_tools(
        tool_definitions=tools, satisfied={"AppendFile"}
    )


# ----------------------------------------------------------------------
# The run state's satisfied set
# ----------------------------------------------------------------------


def test_a_fresh_run_has_satisfied_nothing() -> None:
    assert RunScopedState().satisfied_preconditions == set()


def test_satisfaction_round_trip() -> None:
    """Read → record → write back preserves and extends the set."""
    state = RunScopedState()
    satisfied = set(state.satisfied_preconditions)
    record_satisfaction(
        tool_name="AppendFile",
        arguments={"path": "x.py"},
        satisfied=satisfied,
    )
    state.satisfied_preconditions = satisfied

    assert state.satisfied_preconditions == {"AppendFile", "AppendFile:x.py"}

    reloaded = set(state.satisfied_preconditions)
    record_satisfaction(
        tool_name="AppendFile",
        arguments={"path": "y.py"},
        satisfied=reloaded,
    )
    state.satisfied_preconditions = reloaded

    assert state.satisfied_preconditions == {
        "AppendFile",
        "AppendFile:x.py",
        "AppendFile:y.py",
    }


# ----------------------------------------------------------------------
# derive_satisfied_from_messages
# ----------------------------------------------------------------------


def _make_assistant_tool_use(
    tool_name: str,
    tool_call_id: str,
    arguments: dict[str, Any],
    *,
    seeded: bool = False,
) -> Any:
    """Build a minimal ``Message`` carrying a single ``ToolUseBlock``.

    Avoids importing the full :class:`Message` machinery at the
    test-fixture level — we only need the attributes the
    rebuilder actually reads (``role``, ``content_blocks``,
    ``metadata``, ``block.kind``, ``block.name``,
    ``block.arguments_json``).

    ``seeded`` tags the turn as a PRIOR RUN's, the way cross-run history
    seeding does, so a test can assert the engine-side rehydrator refuses to
    replay another run's completed calls into this run's satisfied set.
    """
    import json
    from types import SimpleNamespace

    from protocore.contracts.types import (
        SESSION_HISTORY_SEED_METADATA_KEY,
        ContentBlockKind,
        MessageRole,
        ToolUseBlock,
    )

    block = ToolUseBlock(
        kind=ContentBlockKind.tool_use,
        tool_call_id=tool_call_id,
        name=tool_name,
        arguments_json=json.dumps(arguments),
    )
    return SimpleNamespace(
        role=MessageRole.assistant,
        content_blocks=[block],
        metadata={SESSION_HISTORY_SEED_METADATA_KEY: True} if seeded else {},
    )


def test_derive_satisfied_from_messages_replays_assistant_tool_use() -> None:
    """assistant ``tool_use`` blocks populate the set."""
    messages = [
        _make_assistant_tool_use("AppendFile", "c1", {"path": "x.py"}),
        _make_assistant_tool_use("Write", "c2", {"path": "y.py", "content": "hi"}),
    ]
    rebuilt = derive_satisfied_from_messages(messages)
    assert rebuilt == {"AppendFile", "AppendFile:x.py", "Write", "Write:y.py"}


def test_derive_satisfied_from_messages_ignores_non_assistant_roles() -> None:
    """only ``assistant``-role messages contribute.

    ``user``/``system``/``tool`` messages may carry content blocks
    but none of those blocks are ``ToolUseBlock``-shaped; if a
    future tool role ever gains one we still want the rebuilder to
    skip it. Pin the contract by injecting a synthetic non-assistant
    message and confirming it has no effect.
    """
    from types import SimpleNamespace

    from protocore.contracts.types import ContentBlockKind, MessageRole, ToolUseBlock

    tool_msg = SimpleNamespace(
        role=MessageRole.tool,
        content_blocks=[
            ToolUseBlock(
                kind=ContentBlockKind.tool_use,
                tool_call_id="c0",
                name="AppendFile",
                arguments_json='{"path": "should_not_be_picked_up.py"}',
            )
        ],
    )
    messages = [
        _make_assistant_tool_use("Write", "c1", {"path": "y.py", "content": "ok"}),
        tool_msg,
    ]
    rebuilt = derive_satisfied_from_messages(messages)
    assert rebuilt == {"Write", "Write:y.py"}
    # The synthetic tool-role entry must NOT leak in.
    assert "AppendFile:should_not_be_picked_up.py" not in rebuilt


def test_derive_satisfied_from_messages_handles_undecodable_arguments() -> None:
    """a malformed ``arguments_json`` yields the bare-name entry only.

    A live call would have raised an error long before the
    transcript was persisted, so a malformed block in the
    transcript is a defensive case. The rebuilder must NOT crash
    AND must NOT lose the bare-name satisfaction (the model DID
    call the tool — only the path is unknowable).
    """
    from types import SimpleNamespace

    from protocore.contracts.types import ContentBlockKind, MessageRole, ToolUseBlock

    bad_block = ToolUseBlock(
        kind=ContentBlockKind.tool_use,
        tool_call_id="c1",
        name="AppendFile",
        arguments_json="not-json-{",
    )
    messages = [SimpleNamespace(role=MessageRole.assistant, content_blocks=[bad_block])]
    rebuilt = derive_satisfied_from_messages(messages)
    assert rebuilt == {"AppendFile"}


def test_rehydrate_satisfied_from_history_seeds_an_empty_set() -> None:
    """The engine-side rehydrator seeds a run state that has satisfied nothing.

    The run's state is composed fresh per process; on a cross-process re-drive
    ``engine.history`` carries completed tool results but the satisfied set is
    empty. The rehydrator must replay every successful completed call into it so
    a precondition check in the new process sees the same set live recording
    would have produced.
    """
    from types import SimpleNamespace

    # Synthetic engine: a real QueryEngine is overkill for the two attributes
    # the rehydrator reads (``engine.history`` and ``engine.run_state``).
    from protocore.contracts.types import MessageRole, ToolResultBlock
    from protocore.runtime.query import _rehydrate_satisfied_from_history

    state = RunScopedState()
    engine = SimpleNamespace(
        run_state=state,
        history=[
            _make_assistant_tool_use("AppendFile", "c1", {"path": "x.py"}),
            _make_assistant_tool_use("Write", "c2", {"path": "y.py", "content": "ok"}),
            SimpleNamespace(
                role=MessageRole.tool,
                content_blocks=[
                    ToolResultBlock(tool_call_id="c1", content="ok"),
                    ToolResultBlock(tool_call_id="c2", content="ok"),
                ],
                metadata={},
            ),
        ]
    )
    _rehydrate_satisfied_from_history(engine)  # type: ignore[arg-type]
    assert state.satisfied_preconditions == {
        "AppendFile",
        "AppendFile:x.py",
        "Write",
        "Write:y.py",
    }


def test_rehydrate_satisfied_from_history_excludes_failed_tool_results() -> None:
    """A failed historical call never authorizes a later precondition."""
    from types import SimpleNamespace

    from protocore.contracts.types import MessageRole, ToolResultBlock
    from protocore.runtime.query import _rehydrate_satisfied_from_history

    state = RunScopedState()
    engine = SimpleNamespace(
        run_state=state,
        history=[
            _make_assistant_tool_use("AppendFile", "failed", {"path": "x.py"}),
            SimpleNamespace(
                role=MessageRole.tool,
                content_blocks=[
                    ToolResultBlock(tool_call_id="failed", content="failed", is_error=True)
                ],
                metadata={},
            ),
        ]
    )
    _rehydrate_satisfied_from_history(engine)  # type: ignore[arg-type]
    assert state.satisfied_preconditions == set()


def test_rehydrate_satisfied_from_history_excludes_prior_run_seeded_calls() -> None:
    """A PRIOR run's completed call never authorizes THIS run's dependent call.

    ``engine.history`` is a session transcript: cross-run history seeding
    prepends the earlier runs of the same session verbatim, tool calls and
    results included. Replaying those into the satisfied set would let a run
    that has appended nothing call ``FinalizeFile`` on a path some earlier run
    appended to — the same class of error as a FAILED call authorizing a
    dependent one, arriving from a different direction.
    """
    from types import SimpleNamespace

    from protocore.contracts.types import (
        SESSION_HISTORY_SEED_METADATA_KEY,
        MessageRole,
        ToolResultBlock,
    )
    from protocore.runtime.query import _rehydrate_satisfied_from_history

    state = RunScopedState()
    engine = SimpleNamespace(
        run_state=state,
        history=[
            # An earlier run of the session appended to the file and succeeded.
            _make_assistant_tool_use(
                "AppendFile", "prior", {"path": "report.md"}, seeded=True
            ),
            SimpleNamespace(
                role=MessageRole.tool,
                content_blocks=[ToolResultBlock(tool_call_id="prior", content="ok")],
                metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
            ),
            # THIS run has written a different file, and nothing else.
            _make_assistant_tool_use("Write", "here", {"path": "notes.md", "content": "hi"}),
            SimpleNamespace(
                role=MessageRole.tool,
                content_blocks=[ToolResultBlock(tool_call_id="here", content="ok")],
                metadata={},
            ),
        ]
    )
    _rehydrate_satisfied_from_history(engine)  # type: ignore[arg-type]
    assert state.satisfied_preconditions == {"Write", "Write:notes.md"}


def test_rehydrate_satisfied_from_history_preserves_existing_set() -> None:
    """an in-process populated set always wins over the replay.

    A run that has been dispatching in-process has already recorded
    the live satisfaction entries on the run's state. The
    rehydrator must NOT clobber them (the live entries are a
    strict superset of the history-replay entries — live
    recording has already added the satisfaction of the
    in-flight call).
    """
    from types import SimpleNamespace

    from protocore.runtime.query import _rehydrate_satisfied_from_history

    # Live recording has already recorded the in-flight call's satisfaction.
    state = RunScopedState(
        satisfied_preconditions={"AppendFile", "AppendFile:x.py"}
    )
    # History is shorter than the live set (e.g. the in-flight
    # call hasn't been appended yet) — the rehydrator must not
    # drop the live entries to match the history.
    engine = SimpleNamespace(
        run_state=state,
        history=[_make_assistant_tool_use("Write", "c1", {"path": "y.py"})],
    )
    _rehydrate_satisfied_from_history(engine)  # type: ignore[arg-type]
    assert state.satisfied_preconditions == {"AppendFile", "AppendFile:x.py"}


def test_rehydrate_satisfied_from_history_with_no_history_is_noop() -> None:
    """An empty transcript leaves the run having satisfied nothing."""
    from types import SimpleNamespace

    from protocore.runtime.query import _rehydrate_satisfied_from_history

    state = RunScopedState()
    _rehydrate_satisfied_from_history(  # type: ignore[arg-type]
        SimpleNamespace(run_state=state, history=[])
    )
    assert state.satisfied_preconditions == set()
