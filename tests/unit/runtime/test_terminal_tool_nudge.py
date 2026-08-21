# ruff: noqa: RUF001 - Bilingual RU+EN + math-notation test fixtures.
"""Unit tests for the universal terminal-tool nudge mechanism.

The terminal-tool nudge routes through ``_terminal_tool_nudge_required``,
gated on ``QueryEngineConfig.expected_terminal_tool`` AND
``rc.terminal_tool_nudge_enabled``. Tests cover the universal path plus the
negative cases (kill-switch off, no tool configured).
"""
from __future__ import annotations

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.tools import Tool
from protocore.contracts.types import (
    SYNTHETIC_RECOVERY_METADATA_KEY,
    SYNTHETIC_RECOVERY_TERMINAL_TOOL_NUDGE,
    TERMINAL_TOOL_METADATA_KEY,
    Message,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.query import (
    _append_terminal_tool_nudge,
    _history_has_file_write_result,
    _history_has_terminal_tool_result,
    _resolved_terminal_tool_name,
    _resolved_terminal_tool_nudge_text,
    _suppress_terminal_only_meta_text,
    _terminal_only_blocks,
    _terminal_only_enforced,
    _terminal_only_error_message,
    _terminal_tool_nudge_required,
)
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemorySkillStore,
    InMemoryToolRegistry,
)


def _build_engine(
    *,
    rc: RuntimeConstants,
    expected_terminal_tool: str | None = None,
) -> QueryEngine:
    return QueryEngine(
        config=QueryEngineConfig(
            run_id="run-test",
            tenant_id="tenant-test",
            session_id="sess-test",
            model_name="qwen3.6-35b-a3b",
            rc=rc,
            expected_terminal_tool=expected_terminal_tool,
        ),
        llm_provider=InMemoryLLMProvider(),
        tool_registry=InMemoryToolRegistry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


def _append_terminal_result(
    engine: QueryEngine,
    *,
    tool_name: str,
    tool_call_id: str = "toolu_term",
) -> None:
    """Mimic a successful terminal tool round-trip in history."""

    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    arguments_json="{}",
                )
            ],
        )
    )
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(
                    tool_call_id=tool_call_id,
                    content="submitted",
                    is_error=False,
                    metadata={TERMINAL_TOOL_METADATA_KEY: True},
                )
            ],
        )
    )


# ---------------------------------------------------------------------------
# Universal terminal-tool nudge — per-tenant expected_terminal_tool
# ---------------------------------------------------------------------------


def test_terminal_tool_nudge_fires_for_configured_tool() -> None:
    """Universal path — a tenant declaring a terminal tool and flipping
    ``terminal_tool_nudge_enabled`` MUST trigger the nudge predicate when
    the run is about to finish without that tool's result."""
    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
    )
    engine = _build_engine(rc=rc, expected_terminal_tool="final_answer")
    assert _terminal_tool_nudge_required(engine) is True
    assert _resolved_terminal_tool_name(engine) == "final_answer"


def test_terminal_tool_nudge_message_is_marked_synthetic() -> None:
    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
    )
    engine = _build_engine(rc=rc, expected_terminal_tool="final_answer")

    _append_terminal_tool_nudge(engine)

    assert engine.history[-1].role is MessageRole.user
    assert engine.history[-1].metadata[SYNTHETIC_RECOVERY_METADATA_KEY] == (
        SYNTHETIC_RECOVERY_TERMINAL_TOOL_NUDGE
    )


def test_terminal_tool_nudge_skipped_when_disabled() -> None:
    """RC kill-switch ``terminal_tool_nudge_enabled=False`` blocks the
    nudge even when ``expected_terminal_tool`` is configured."""
    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=False,
    )
    engine = _build_engine(rc=rc, expected_terminal_tool="final_answer")
    assert _terminal_tool_nudge_required(engine) is False


def test_terminal_tool_nudge_skipped_when_no_tool_configured() -> None:
    """Default tenant (no ``expected_terminal_tool``) MUST NOT trigger the
    nudge — universal-core philosophy: opt-in only."""
    rc = RuntimeConstants(model_context_window=4_096)
    engine = _build_engine(rc=rc, expected_terminal_tool=None)
    assert _terminal_tool_nudge_required(engine) is False
    assert _resolved_terminal_tool_name(engine) is None


def test_history_has_terminal_tool_result_per_tool_name() -> None:
    """When ``expected_terminal_tool`` is set, the history check rejects
    a foreign tool's terminal-metadata-flagged result. A matching tool
    name satisfies the check."""

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
    )

    # Case A — engine configured for ``final_answer``, only ``final_answer``
    # in history: the predicate sees a satisfied finalisation and the nudge
    # does not fire.
    engine_a = _build_engine(rc=rc, expected_terminal_tool="final_answer")
    _append_terminal_result(engine_a, tool_name="final_answer")
    assert _history_has_terminal_tool_result(engine_a) is True
    assert _terminal_tool_nudge_required(engine_a) is False

    # Case B — engine configured for ``final_answer``, only a foreign tool in
    # history: the foreign tool's metadata flag does NOT satisfy the
    # configured contract, so the predicate still requires the nudge.
    engine_mismatch = _build_engine(
        rc=rc, expected_terminal_tool="final_answer"
    )
    _append_terminal_result(engine_mismatch, tool_name="other_answer")
    assert _history_has_terminal_tool_result(engine_mismatch) is False
    assert _terminal_tool_nudge_required(engine_mismatch) is True


# ---------------------------------------------------------------------------
# Nudge text resolution
# ---------------------------------------------------------------------------


def test_terminal_tool_nudge_text_universal_wins() -> None:
    """Operator-supplied universal text takes precedence over the templated
    fallback. ``terminal_tool_nudge_write_first_enabled=False`` isolates the
    universal-text precedence from the write-first prefix (default on),
    which would otherwise prepend the deliverable-write instruction."""

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
        terminal_tool_nudge_text="Call final_answer now.",
        terminal_tool_nudge_write_first_enabled=False,
    )
    engine = _build_engine(rc=rc, expected_terminal_tool="final_answer")
    assert _resolved_terminal_tool_nudge_text(engine) == "Call final_answer now."


def test_terminal_tool_nudge_text_generic_fallback() -> None:
    """Tenant flips ``terminal_tool_nudge_enabled`` but leaves the text
    empty — the templated message keys on the declared tool name."""

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
    )
    engine = _build_engine(rc=rc, expected_terminal_tool="final_answer")
    text = _resolved_terminal_tool_nudge_text(engine)
    assert "final_answer" in text


# ---------------------------------------------------------------------------
# Write-first generalisation of the terminal nudge
# ---------------------------------------------------------------------------


def _append_file_write_result(
    engine: QueryEngine,
    *,
    tool_name: str,
    tool_call_id: str = "toolu_write",
    is_error: bool = False,
) -> None:
    """Mimic a (successful by default) Write/AppendFile tool round-trip."""

    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    arguments_json="{}",
                )
            ],
        )
    )
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(
                    tool_call_id=tool_call_id,
                    content="written",
                    is_error=is_error,
                    metadata={},
                )
            ],
        )
    )


def test_history_has_file_write_result_detects_write_and_appendfile() -> None:
    """The history check is keyed on the RC tuple (default Write/AppendFile)
    and recognises a successful write result; an errored write does not
    count."""

    rc = RuntimeConstants(model_context_window=4_096)

    engine_none = _build_engine(rc=rc, expected_terminal_tool="Finalize")
    assert _history_has_file_write_result(engine_none) is False

    engine_write = _build_engine(rc=rc, expected_terminal_tool="Finalize")
    _append_file_write_result(engine_write, tool_name="Write")
    assert _history_has_file_write_result(engine_write) is True

    engine_append = _build_engine(rc=rc, expected_terminal_tool="Finalize")
    _append_file_write_result(engine_append, tool_name="AppendFile")
    assert _history_has_file_write_result(engine_append) is True

    engine_err = _build_engine(rc=rc, expected_terminal_tool="Finalize")
    _append_file_write_result(engine_err, tool_name="Write", is_error=True)
    assert _history_has_file_write_result(engine_err) is False


def test_no_tool_end_on_file_deliverable_triggers_forced_write_nudge() -> None:
    """A no-tool end on a file-deliverable task (no Write/AppendFile
    in history) prepends the write-first instruction to the nudge so the model
    is steered to the ACTUAL deliverable write tool, not just the terminal
    tool. Default RC arms this (write_first_enabled defaults True)."""

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
    )
    engine = _build_engine(rc=rc, expected_terminal_tool="Finalize")

    text = _resolved_terminal_tool_nudge_text(engine)
    # The write-first prefix fires: it names the actual write tool(s), not only
    # the terminal tool.
    assert "Write" in text
    assert "AppendFile" in text
    assert rc.terminal_tool_nudge_write_first_text.split("\n")[0] in text


def test_write_first_prefix_text_is_conditional_not_false_premise() -> None:
    """Review the write-first prefix fires on ANY no-tool end with no
    file in history (incl. Q&A/coding runs under the now-default Finalize
    contract). Its text MUST therefore be CONDITIONAL ('if the task asked … then
    finish with the terminal tool'), never a declarative false premise ('you
    declared work that produces a file'), which would (a) be factually wrong on
    a Q&A run and (b) contradict the terminal-tool nudge body. Bilingual."""

    default_text = RuntimeConstants().terminal_tool_nudge_write_first_text
    lowered = default_text.lower()
    # Conditional framing present (EN + RU).
    assert "if the task asked" in lowered
    assert "если задача просила" in lowered
    # Complements the terminal tool rather than contradicting it.
    assert "terminal tool" in lowered
    assert "терминальным инструментом" in lowered
    # The old false-premise wording is GONE (regression guard).
    assert "you declared work that produces a file" not in lowered
    assert "вы заявили работу" not in lowered


def test_full_resolved_nudge_is_internal_control_framed_no_echo_wording() -> None:
    """The FULL resolved nudge with default RC and NO file-write result (so
    BOTH the write-first prefix AND the fallback body render) must be
    self-evidently INTERNAL-control framed end-to-end, so a weak model that
    paraphrases it cannot leak reply-looking text. Both the prefix and the body
    must open with the ``[internal control — …]`` marker (EN+RU), and the old
    user-facing imperatives the model used to echo MUST be gone."""

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
    )
    engine = _build_engine(rc=rc, expected_terminal_tool="Finalize")
    # No file-write result in history → the write-first prefix fires.
    assert not _history_has_file_write_result(engine)

    text = _resolved_terminal_tool_nudge_text(engine)
    lowered = text.lower()

    # The resolved nudge STARTS with the internal-control marker (prefix branch),
    # and the marker also appears for the fallback body — both EN and RU framings
    # are present so a verbatim echo of any part reads as internal control.
    assert text.startswith("[internal control — not part of the reply]")
    assert "[внутреннее управление — не часть ответа]" in text
    # Two internal-control markers minimum (prefix + fallback body).
    assert text.count("[internal control — not part of the reply]") >= 2

    # The OLD echo-prone user-facing wording is GONE (regression guard).
    assert "do not answer in prose" not in lowered
    assert "не отвечайте прозой" not in lowered
    assert "do not write normal final text" not in lowered
    assert "you have not called" not in lowered

    # Functional intent preserved: still steers to the terminal tool + the
    # actual write tools, and keeps the phrase the forced-backstop test keys on.
    assert "Finish now by calling" in text
    assert "Finalize" in text
    assert "Write" in text
    assert "AppendFile" in text


def test_write_first_prefix_suppressed_once_file_written() -> None:
    """A run that already wrote its deliverable does NOT get the write-first
    prefix — strong-model no-op path (the plain terminal nudge stands)."""

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
    )
    engine = _build_engine(rc=rc, expected_terminal_tool="Finalize")
    _append_file_write_result(engine, tool_name="Write")

    text = _resolved_terminal_tool_nudge_text(engine)
    assert rc.terminal_tool_nudge_write_first_text.split("\n")[0] not in text
    # The plain terminal nudge body is preserved.
    assert "Finalize" in text


def test_write_first_disabled_by_rc_kill_switch() -> None:
    """``terminal_tool_nudge_write_first_enabled=False`` restores the plain
    terminal nudge even with no file written."""

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
        terminal_tool_nudge_write_first_enabled=False,
    )
    engine = _build_engine(rc=rc, expected_terminal_tool="Finalize")
    text = _resolved_terminal_tool_nudge_text(engine)
    assert rc.terminal_tool_nudge_write_first_text.split("\n")[0] not in text


# ---------------------------------------------------------------------------
# Terminal-only dispatch guard — armed by the run wind-down, and by nothing
# else. It used to key on a deadline-specific latch, so the strictness followed
# one of the five ways a run can be cut short; all five are the same wind-down
# now.
# ---------------------------------------------------------------------------


def test_terminal_only_enforces_once_the_wind_down_has_withdrawn_the_tools() -> None:
    """A blocked call gets an actionable error, not a bare denial.

    The narrowed surface is what actually stops the call — a withdrawn tool is
    neither advertised nor admitted by the permission gate. This guard is what
    the model reads when it tries one anyway off a stale schema in its context,
    and it names the tool it should be calling instead.
    """
    from protocore.runtime import soft_stop as _soft_stop

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
    )
    engine = _build_engine(rc=rc, expected_terminal_tool="final_answer")
    assert _resolved_terminal_tool_name(engine) == "final_answer"

    non_terminal = ToolCall(name="remote_tree", arguments={"root": "/"})
    terminal = ToolCall(name="final_answer", arguments={})

    # Nothing withdrawn yet -> nothing blocked (exploration).
    assert _terminal_only_enforced(engine) is False
    assert _terminal_only_blocks(engine, non_terminal) is False

    _soft_stop.enter(engine, cause_name=_soft_stop.CAUSE_DEADLINE)
    engine._terminal_only_active = True

    assert _terminal_only_enforced(engine) is True
    assert _terminal_only_blocks(engine, non_terminal) is True
    message = _terminal_only_error_message(engine, non_terminal.name)
    assert "remote_tree" in message  # names the blocked tool
    assert "final_answer" in message  # names the terminal tool to call now
    assert "terminal-only mode" in message  # actionable

    # The resolved terminal tool itself is the single allowed tool.
    assert _terminal_only_blocks(engine, terminal) is False

    # Once a successful terminal result lands, the guard releases.
    _append_terminal_result(engine, tool_name="final_answer")
    assert _terminal_only_blocks(engine, non_terminal) is False


def test_terminal_only_is_not_enforced_by_the_per_turn_latch_alone() -> None:
    """``_terminal_only_active`` is a turn fact; the wind-down is a run fact.

    The voluntary-finish contract repair sets the per-turn latch on a turn the
    model is still free to work in. Strict-blocking that turn would break the
    repair it exists to perform.
    """
    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
    )
    engine = _build_engine(rc=rc, expected_terminal_tool="final_answer")
    engine._terminal_only_active = True
    assert _terminal_only_enforced(engine) is False
    non_terminal = ToolCall(name="remote_read", arguments={"path": "/x"})
    assert _terminal_only_blocks(engine, non_terminal) is False


def test_terminal_only_inert_for_default_tenant() -> None:
    """Negative — a default tenant (no ``expected_terminal_tool``, no generic
    RC) MUST never block, even if both latch fields are somehow set.
    Universal-core opt-in only — there is no resolvable terminal tool to
    force."""

    from protocore.runtime import soft_stop as _soft_stop

    rc = RuntimeConstants(model_context_window=4_096)
    engine = _build_engine(rc=rc, expected_terminal_tool=None)
    _soft_stop.enter(engine, cause_name=_soft_stop.CAUSE_DEADLINE)
    engine._terminal_only_active = True
    # Enforce latch is set, but with no resolvable terminal tool the guard
    # still cannot block.
    assert _resolved_terminal_tool_name(engine) is None
    non_terminal = ToolCall(name="remote_tree", arguments={})
    assert _terminal_only_blocks(engine, non_terminal) is False


# ---------------------------------------------------------------------------
# The nudge ALWAYS fires (write-first recovery + Finalize depend on it).
# Instead, the post-nudge terminal-only turn's redundant META TEXT is
# suppressed at the stream when a prior answer exists + background terminal.
# ``_suppress_terminal_only_meta_text`` is the decision predicate; the REAL
# end-to-end loop behaviour is covered in test_query_async_gen.py.
# ---------------------------------------------------------------------------


class _BackgroundFinalizeTool(Tool):
    """A ``Finalize``-shaped BACKGROUND terminal: its schema carries NO
    answer-bearing field, so the user-facing answer can only be the model's
    prose. Suppression applies ONLY to this class of terminal tool."""

    @property
    def name(self) -> str:
        return "Finalize"

    @property
    def definition(self):  # type: ignore[no-untyped-def]
        from protocore.contracts.types import ToolDefinition, ToolParameterSchema

        return ToolDefinition(
            name="Finalize",
            description="Background finalize gate",
            parameters=ToolParameterSchema(
                properties={"declared_deliverables": {"type": "array"}},
                required=[],
            ),
        )

    async def invoke(self, context, arguments):  # type: ignore[no-untyped-def]
        from protocore.contracts.types import TERMINAL_TOOL_METADATA_KEY, ToolResult

        return ToolResult(
            tool_call_id="",
            content="{}",
            is_error=False,
            metadata={TERMINAL_TOOL_METADATA_KEY: True},
        )


class _MessageCarryingTerminalTool(Tool):
    """A ``pcm_answer``-shaped MESSAGE-CARRYING terminal: its schema declares a
    ``message`` field, so the answer flows through the tool call (not prose).
    Suppression must NOT engage for this class (the nudge keeps firing)."""

    @property
    def name(self) -> str:
        return "pcm_answer"

    @property
    def definition(self):  # type: ignore[no-untyped-def]
        from protocore.contracts.types import ToolDefinition, ToolParameterSchema

        return ToolDefinition(
            name="pcm_answer",
            description="Submit the final answer",
            parameters=ToolParameterSchema(
                properties={
                    "message": {"type": "string"},
                    "outcome": {"type": "string"},
                    "refs": {"type": "array"},
                },
                required=["message", "outcome"],
            ),
        )

    async def invoke(self, context, arguments):  # type: ignore[no-untyped-def]
        from protocore.contracts.types import TERMINAL_TOOL_METADATA_KEY, ToolResult

        return ToolResult(
            tool_call_id="",
            content="submitted",
            is_error=False,
            metadata={TERMINAL_TOOL_METADATA_KEY: True},
        )


def _build_finalize_engine(rc: RuntimeConstants) -> QueryEngine:
    """An engine whose terminal tool is a registered BACKGROUND ``Finalize`` —
    the live shape under ``agent_finalize_tool_as_terminal=True``."""

    engine = _build_engine(rc=rc, expected_terminal_tool="Finalize")
    engine.tools.register(_BackgroundFinalizeTool())  # type: ignore[attr-defined]
    return engine


def _append_assistant_text(engine: QueryEngine, text: str) -> None:
    engine.history.append(
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=text)])
    )


def _append_work_tool(
    engine: QueryEngine,
    *,
    tool_name: str = "Bash",
    tool_call_id: str = "toolu_work",
) -> None:
    """A non-terminal work tool round-trip (real work, not the gate)."""

    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id=tool_call_id, name=tool_name, arguments_json="{}"
                )
            ],
        )
    )
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(
                    tool_call_id=tool_call_id,
                    content="ok",
                    is_error=False,
                    metadata={"tool_name": tool_name},
                )
            ],
        )
    )


def test_meta_text_suppressed_when_substantive_answer_already_exists() -> None:
    """Dominant Direct-mode shape: a substantive prose answer already
    exists, and we are in the post-nudge terminal-only turn (``_terminal_only_
    active``). The turn's visible TEXT is the redundant META narration → suppress
    it. The nudge itself STILL fired (``_terminal_tool_nudge_required`` is True)."""

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
        finalize_prose_gate_min_chars=1,
    )
    engine = _build_finalize_engine(rc)
    _append_assistant_text(engine, "12 × 12 = 144")
    engine._terminal_only_active = True  # the nudge fired; we are in turn-2

    # The nudge predicate fired (no Finalize result yet) …
    assert _terminal_tool_nudge_required(engine) is True
    # … and the terminal-only turn's text is the redundant meta → suppressed.
    assert _suppress_terminal_only_meta_text(engine) is True


def test_meta_text_not_suppressed_before_nudge_fires() -> None:
    """On the FIRST answer turn (``_terminal_only_active`` False — the
    nudge has not fired yet) the real answer's TEXT MUST stream normally. The
    suppression predicate only engages on the post-nudge terminal-only turn."""

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
        finalize_prose_gate_min_chars=1,
    )
    engine = _build_finalize_engine(rc)
    # No prior answer yet, terminal-only NOT active → first answer streams.
    assert getattr(engine, "_terminal_only_active", False) is False
    assert _suppress_terminal_only_meta_text(engine) is False


def test_meta_text_suppressed_for_terse_answer_min_chars_one() -> None:
    """Regression guard: a terse-but-complete prior answer (``144``) counts
    as substantive under ``finalize_prose_gate_min_chars=1``, so the terminal-only
    turn's text is suppressed (the duplicate-fix floor of 1 is preserved)."""

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
        finalize_prose_gate_min_chars=1,
    )
    engine = _build_finalize_engine(rc)
    _append_assistant_text(engine, "144")
    engine._terminal_only_active = True
    assert _suppress_terminal_only_meta_text(engine) is True


def test_meta_text_suppressed_after_work_then_answer() -> None:
    """Work tool, then a prior prose answer AFTER the work: the prose is
    the real answer, so the terminal-only turn's text is suppressed (the
    Remember/Read-then-answer shape from the forensics)."""

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
        finalize_prose_gate_min_chars=1,
    )
    engine = _build_finalize_engine(rc)
    _append_work_tool(engine, tool_name="Remember")
    _append_assistant_text(engine, "Запомнил: синий и 7.")
    engine._terminal_only_active = True
    assert _suppress_terminal_only_meta_text(engine) is True


def test_meta_text_not_suppressed_when_no_prior_answer() -> None:
    """When NO prior substantive answer exists, the
    terminal-only turn's text IS the answer (the genuinely-empty / write-first
    'Done, I created the file'→then-real-Write case where the model finally
    answers in the terminal turn), so it MUST stay visible. Here only progress
    narration exists BEFORE the work, with no answer after it."""

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
        finalize_prose_gate_min_chars=1,
    )
    engine = _build_finalize_engine(rc)
    _append_assistant_text(engine, "Let me check the file.")  # progress, pre-work
    _append_work_tool(engine, tool_name="Read")  # latest work AFTER the prose
    engine._terminal_only_active = True
    # No answer after the work → text NOT suppressed (it would be the answer).
    assert _suppress_terminal_only_meta_text(engine) is False


def test_meta_text_not_suppressed_for_empty_run() -> None:
    """An empty run (no prior assistant prose at all) is not suppressed;
    the terminal-only turn's text would be the run's only answer."""

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
        finalize_prose_gate_min_chars=1,
    )
    engine = _build_finalize_engine(rc)
    engine._terminal_only_active = True
    assert _suppress_terminal_only_meta_text(engine) is False


def test_meta_text_not_suppressed_for_message_carrying_terminal() -> None:
    """A MESSAGE-CARRYING terminal (``pcm_answer``, schema declares
    ``message``) submits its answer via the tool call; the visible text IS the
    real answer surface. The suppression MUST NOT engage even with a prior answer
    + terminal-only active. (The integration test
    ``test_terminal_nudge_recovers_plain_text_final`` depends on the text + nudge
    surviving for ``pcm_answer``.)"""

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
        finalize_prose_gate_min_chars=1,
    )
    engine = _build_engine(rc=rc, expected_terminal_tool="pcm_answer")
    engine.tools.register(_MessageCarryingTerminalTool())  # type: ignore[attr-defined]
    _append_assistant_text(engine, "14")
    engine._terminal_only_active = True
    assert _suppress_terminal_only_meta_text(engine) is False


def test_meta_text_not_suppressed_when_terminal_tool_unknown_to_core() -> None:
    """Fail-safe: when the terminal tool is not registered (core cannot
    introspect its schema), the suppression EXEMPTS it (returns False, keeps the
    text), matching the prose-gate's multi-tenant fail-safe."""

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
        finalize_prose_gate_min_chars=1,
    )
    engine = _build_engine(rc=rc, expected_terminal_tool="backend_answer")
    _append_assistant_text(engine, "some answer")
    engine._terminal_only_active = True
    assert _suppress_terminal_only_meta_text(engine) is False


def test_meta_text_suppression_inert_for_default_tenant() -> None:
    """With no resolvable terminal tool the suppression is inert."""

    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
        finalize_prose_gate_min_chars=1,
    )
    engine = _build_engine(rc=rc, expected_terminal_tool=None)
    _append_assistant_text(engine, "Some answer.")
    engine._terminal_only_active = True
    assert _resolved_terminal_tool_name(engine) is None
    assert _suppress_terminal_only_meta_text(engine) is False
