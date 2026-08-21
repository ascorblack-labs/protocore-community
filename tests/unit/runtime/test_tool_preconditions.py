"""Unit tests for the DAG-precondition mechanism.

Covers the pure functions in
:mod:`protocore.runtime.tool_preconditions`:

* :func:`check_preconditions` — bare-name, parameterised and prefix patterns.
* :func:`record_satisfaction` — bare-name + path-keyed entries.
* :func:`resolve_precondition` — path normalisation.
* :func:`compute_masked_tools` — only bare-name patterns are masked
 pre-emptively.
* :func:`load_satisfied_set` / :func:`store_satisfied_set` — helper-bag
 round-trip (adaptation for cross-call satisfaction state).
"""

from __future__ import annotations

from typing import Any

from protocore.runtime.tool_preconditions import (
    SATISFIED_PRECONDITIONS_KEY,
    check_preconditions,
    compute_masked_tools,
    derive_satisfied_from_messages,
    load_satisfied_set,
    record_satisfaction,
    resolve_precondition,
    store_satisfied_set,
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
# load_satisfied_set / store_satisfied_set
# ----------------------------------------------------------------------


def test_load_satisfied_set_missing_helpers_returns_empty() -> None:
    assert load_satisfied_set(None) == set()
    assert load_satisfied_set({}) == set()


def test_load_satisfied_set_handles_list_storage() -> None:
    """Stored as a sorted list, hydrated as a set."""
    helpers: dict[str, Any] = {SATISFIED_PRECONDITIONS_KEY: ["A", "B:x"]}
    assert load_satisfied_set(helpers) == {"A", "B:x"}


def test_load_satisfied_set_handles_set_storage() -> None:
    helpers: dict[str, Any] = {SATISFIED_PRECONDITIONS_KEY: {"A", "B:x"}}
    assert load_satisfied_set(helpers) == {"A", "B:x"}


def test_store_satisfied_set_persists_sorted_list() -> None:
    helpers: dict[str, Any] = {}
    store_satisfied_set(helpers, {"B:x", "A"})
    assert helpers[SATISFIED_PRECONDITIONS_KEY] == ["A", "B:x"]


def test_store_satisfied_set_no_helpers_is_noop() -> None:
    # Should not raise.
    store_satisfied_set(None, {"A"})


def test_helper_bag_round_trip() -> None:
    """Persist → reload → mutate → persist preserves and extends the set."""
    helpers: dict[str, Any] = {}
    satisfied = load_satisfied_set(helpers)
    record_satisfaction(
        tool_name="AppendFile",
        arguments={"path": "x.py"},
        satisfied=satisfied,
    )
    store_satisfied_set(helpers, satisfied)
    reloaded = load_satisfied_set(helpers)
    assert reloaded == {"AppendFile", "AppendFile:x.py"}
    record_satisfaction(
        tool_name="AppendFile",
        arguments={"path": "y.py"},
        satisfied=reloaded,
    )
    store_satisfied_set(helpers, reloaded)
    final = load_satisfied_set(helpers)
    assert final == {"AppendFile", "AppendFile:x.py", "AppendFile:y.py"}


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


def test_rehydrate_satisfied_from_history_seeds_empty_bag() -> None:
    """the engine-side rehydrator seeds an empty helper bag.

    The helper bag is built fresh per pod; on a cross-pod re-drive
    ``engine.history`` carries completed tool results but the helper bag is
    empty. The rehydrator must replay every successful completed call into the
    bag's
    :data:`SATISFIED_PRECONDITIONS_KEY` so a precondition check
    on the new pod sees the same set live recording would have
    produced.
    """
    from types import SimpleNamespace

    from protocore.runtime.query import _rehydrate_satisfied_from_history

    # Empty helper bag = cross-pod re-drive baseline.
    helpers: dict[str, Any] = {}
    # Synthetic engine: a real QueryEngine is overkill for the
    # attribute the rehydrator reads (``engine.history``).
    from protocore.contracts.types import MessageRole, ToolResultBlock

    engine = SimpleNamespace(
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
    _rehydrate_satisfied_from_history(helpers, engine)  # type: ignore[arg-type]
    assert helpers.get(SATISFIED_PRECONDITIONS_KEY) == [
        "AppendFile",
        "AppendFile:x.py",
        "Write",
        "Write:y.py",
    ]


def test_rehydrate_satisfied_from_history_excludes_failed_tool_results() -> None:
    """A failed historical call never authorizes a later precondition."""
    from types import SimpleNamespace

    from protocore.contracts.types import MessageRole, ToolResultBlock
    from protocore.runtime.query import _rehydrate_satisfied_from_history

    helpers: dict[str, Any] = {}
    engine = SimpleNamespace(
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
    _rehydrate_satisfied_from_history(helpers, engine)  # type: ignore[arg-type]
    assert SATISFIED_PRECONDITIONS_KEY not in helpers


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

    helpers: dict[str, Any] = {}
    engine = SimpleNamespace(
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
    _rehydrate_satisfied_from_history(helpers, engine)  # type: ignore[arg-type]
    assert helpers.get(SATISFIED_PRECONDITIONS_KEY) == ["Write", "Write:notes.md"]


def test_rehydrate_satisfied_from_history_preserves_existing_set() -> None:
    """an in-process populated set always wins over the replay.

    A run that has been dispatching in-process has already recorded
    the live satisfaction entries on the helper bag. The
    rehydrator must NOT clobber them (the live entries are a
    strict superset of the history-replay entries — live
    recording has already added the satisfaction of the
    in-flight call).
    """
    from types import SimpleNamespace

    from protocore.runtime.query import _rehydrate_satisfied_from_history

    # Live recording has already populated the bag with the
    # in-flight call's satisfaction.
    helpers: dict[str, Any] = {
        SATISFIED_PRECONDITIONS_KEY: ["AppendFile", "AppendFile:x.py"],
    }
    # History is shorter than the live set (e.g. the in-flight
    # call hasn't been appended yet) — the rehydrator must not
    # drop the live entries to match the history.
    engine = SimpleNamespace(
        history=[_make_assistant_tool_use("Write", "c1", {"path": "y.py"})]
    )
    _rehydrate_satisfied_from_history(helpers, engine)  # type: ignore[arg-type]
    # The existing entries are preserved verbatim (sorted list).
    assert helpers.get(SATISFIED_PRECONDITIONS_KEY) == [
        "AppendFile",
        "AppendFile:x.py",
    ]


def test_rehydrate_satisfied_from_history_no_helpers_is_noop() -> None:
    """None / empty helpers is a no-op (legacy test wiring)."""
    from types import SimpleNamespace

    from protocore.runtime.query import _rehydrate_satisfied_from_history

    engine = SimpleNamespace(history=[])
    _rehydrate_satisfied_from_history(None, engine)  # type: ignore[arg-type]
    _rehydrate_satisfied_from_history({}, engine)  # type: ignore[arg-type]
