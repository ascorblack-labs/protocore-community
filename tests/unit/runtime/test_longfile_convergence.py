"""unit tests for the large-file convergence driver.

Covers the spec verification points against the pure decision module
:mod:`protocore.runtime.longfile_convergence`:

* — the stall counter increments on non-byte turns / resets on a
 byte-adding mutation; size tracked from the tool RESULT payloads;
* — a stall forces ``AppendFile`` and respects ``longfile_max_forced_appends``;
* — a byte-plateau / done-with-content forces ``FinalizeFile`` and respects
 ``longfile_max_forced_finalizes``;
* — the HARD empty/below-floor finalize guard BLOCKS a forced FinalizeFile
 on an empty / below-floor file (the validated edge);
* — the continue message is INCOMPLETE + tail anchor, RU+EN, never "safe";
* — a model that adds bytes every turn triggers NOTHING (no-op); the
 RC kill-switch makes every entry point bit-identical to pre-FEAT;
* snapshot/resume preserves all new convergence latches (cross-pod safe).
"""
from __future__ import annotations

import json

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import (
    Message,
    MessageRole,
    ToolCall,
    ToolUseBlock,
)
from protocore.runtime import longfile_convergence as lfc


# ── helpers ────────────────────────────────────────────────────────────────
def _rc(**overrides: object) -> RuntimeConstants:
    base: dict[str, object] = {
        "model_context_window": 4_096,
        "longfile_convergence_enabled": True,
        "longfile_stall_turns": 2,
        "longfile_max_forced_appends": 8,
        "longfile_max_forced_finalizes": 2,
        "longfile_plateau_delta_fraction": 0.25,
        "longfile_plateau_min_mutations": 2,
        "longfile_expected_floor_bytes": 4096,
        "longfile_min_finalize_fraction": 1.0,
        "longfile_tail_anchor_chars": 200,
    }
    base.update(overrides)
    return RuntimeConstants(**base)


def _write_call(path: str, content: str = "x") -> ToolCall:
    return ToolCall(id="tc", name="Write", arguments={"path": path, "content": content})


def _append_call(path: str, content: str = "x") -> ToolCall:
    return ToolCall(
        id="tc", name="AppendFile", arguments={"path": path, "content": content}
    )


def _read_call(path: str = "/workspace/big.py") -> ToolCall:
    return ToolCall(id="tc", name="Read", arguments={"path": path})


def _write_result(bytes_written: int, path: str = "/workspace/big.py") -> str:
    return json.dumps({"path": path, "bytes_written": bytes_written})


def _append_result(
    bytes_appended: int,
    bytes_total: int,
    *,
    path: str = "/workspace/big.py",
    line_count_total: int | None = None,
) -> str:
    payload: dict[str, object] = {
        "path": path,
        "bytes_appended": bytes_appended,
        "bytes_total": bytes_total,
    }
    if line_count_total is not None:
        payload["line_count_total"] = line_count_total
    return json.dumps(payload)


# ── stall counter ───────────────────────────────────────────────────────
def test_stall_counter_resets_on_byte_adding_mutation(engine_factory) -> None:
    engine = engine_factory(rc=_rc())
    # A Write that grew the file resets the counter and binds the active path.
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(2000), is_error=False
    )
    lfc.register_turn_byte_production(engine, added_bytes=True)
    assert engine._turns_since_last_byte_adding_mutation == 0
    assert engine._longfile_active_path == "/workspace/big.py"
    assert engine._longfile_active_file_bytes == 2000

    # A Read turn (no bytes) increments the counter.
    lfc.observe_tool_result(engine, _read_call(), json.dumps({"ok": True}), is_error=False)
    lfc.register_turn_byte_production(engine, added_bytes=False)
    assert engine._turns_since_last_byte_adding_mutation == 1

    # A prose-only turn (no tool result observed) increments again.
    lfc.register_turn_byte_production(engine, added_bytes=False)
    assert engine._turns_since_last_byte_adding_mutation == 2

    # An AppendFile that grew the file resets to 0 and updates the running size.
    lfc.observe_tool_result(
        engine, _append_call("/workspace/big.py"), _append_result(1500, 3500),
        is_error=False,
    )
    lfc.register_turn_byte_production(engine, added_bytes=True)
    assert engine._turns_since_last_byte_adding_mutation == 0
    assert engine._longfile_active_file_bytes == 3500
    assert engine._longfile_mutation_deltas == [2000, 1500]


def test_zero_byte_write_does_not_reset_stall(engine_factory) -> None:
    engine = engine_factory(rc=_rc())
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(5000), is_error=False
    )
    engine._turns_since_last_byte_adding_mutation = 1
    # A zero-byte Write result is NOT a byte-adding mutation.
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(0), is_error=False
    )
    lfc.register_turn_byte_production(engine, added_bytes=False)
    assert engine._turns_since_last_byte_adding_mutation == 2


def test_error_result_does_not_count_as_byte_production(engine_factory) -> None:
    engine = engine_factory(rc=_rc())
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(5000), is_error=True
    )
    assert engine._longfile_active_path is None
    assert engine._longfile_active_file_bytes == 0


def test_off_path_write_does_not_corrupt_active_file_tracking(engine_factory) -> None:
    """A byte-adding write to a DIFFERENT path than the bound active file is
    FULLY IGNORED for convergence tracking.

    Fails before the active-path guard (active bytes get clobbered to 50, the
    delta list grows to ``[2000, 50]``, and the stall clock is reset by the
    off-path write); passes after.
    """
    engine = engine_factory(rc=_rc())
    # Write the active file (binds the active path).
    lfc.observe_tool_result(
        engine,
        _write_call("/workspace/big.py"),
        _write_result(2000, path="/workspace/big.py"),
        is_error=False,
    )
    lfc.register_turn_byte_production(engine, added_bytes=True)
    assert engine._longfile_active_path == "/workspace/big.py"
    assert engine._longfile_active_file_bytes == 2000
    assert engine._longfile_mutation_deltas == [2000]

    # Advance the stall clock on an idle turn.
    lfc.register_turn_byte_production(engine, added_bytes=False)
    assert engine._turns_since_last_byte_adding_mutation == 1

    # Now write a SMALL, DIFFERENT file (a side note). It must be ignored:
    # the tracked size/deltas/stall-clock for big.py stay untouched.
    lfc.observe_tool_result(
        engine,
        _write_call("/workspace/notes.md", content="just a note\n"),
        _write_result(50, path="/workspace/notes.md"),
        is_error=False,
    )
    assert engine._longfile_active_path == "/workspace/big.py"
    assert engine._longfile_active_file_bytes == 2000  # NOT 50
    assert engine._longfile_mutation_deltas == [2000]  # NOT [2000, 50]
    assert engine._turns_since_last_byte_adding_mutation == 1  # clock untouched


def test_appendfile_missing_total_uses_running_size_plus_delta(engine_factory) -> None:
    """A-an AppendFile result lacking ``bytes_total`` derives the new
    total from running-size + delta, NOT the bare delta (which would freeze the
    size counter at the prior maximum)."""
    engine = engine_factory(rc=_rc())
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(5000), is_error=False
    )
    assert engine._longfile_active_file_bytes == 5000
    # AppendFile result with bytes_appended but NO bytes_total.
    append_no_total = json.dumps(
        {"path": "/workspace/big.py", "bytes_appended": 200}
    )
    lfc.observe_tool_result(
        engine, _append_call("/workspace/big.py"), append_no_total, is_error=False
    )
    # Fallback total = 5000 + 200 = 5200, NOT max(5000, 200) == 5000.
    assert engine._longfile_active_file_bytes == 5200
    assert engine._longfile_mutation_deltas == [5000, 200]


def test_continue_message_reports_real_full_file_line_count(engine_factory) -> None:
    """B-``file_lines`` in the continue message is the REAL full-file
    line count (from a Write's content lines / AppendFile ``line_count_total``),
    not the line count of the 200-char tail."""
    engine = engine_factory(rc=_rc())
    # A Write whose content is 120 lines; bytes below floor so the stall path
    # forces an AppendFile (a continue message is built).
    content = "\n".join(f"line {i}" for i in range(120))  # 120 lines
    lfc.observe_tool_result(
        engine,
        _write_call("/workspace/big.py", content=content),
        _write_result(2000),
        is_error=False,
    )
    assert engine._longfile_active_file_lines == 120
    msg = lfc.build_continue_message(engine)
    assert "120 lines" in msg
    assert "120 строк" in msg

    # An AppendFile reports the cumulative full-file line count directly.
    lfc.observe_tool_result(
        engine,
        _append_call("/workspace/big.py"),
        _append_result(500, 2500, line_count_total=145),
        is_error=False,
    )
    assert engine._longfile_active_file_lines == 145
    assert "145 lines" in lfc.build_continue_message(engine)


# ── Zero-collateral: driver inert on a NON-truncated below-floor file ──────────
def test_driver_inert_on_untruncated_below_floor_file(engine_factory) -> None:
    """REGRESSION-GUARD (forensic S2): a small file written WITHOUT a truncation
    event, idle past the stall threshold, must force ZERO tools.

    The engage condition was originally 'below-floor + idle' (no truncation gate),
    which forced an AppendFile causing a self-loop. The engage decision is now
    gated on a per-path truncation latch, so an untruncated path NEVER engages.
    """
    engine = engine_factory(rc=_rc(longfile_stall_turns=2))
    # A clean (NON-truncated) below-floor write — exactly the alpha.txt/calc.py
    # shape: bytes land, file below floor, NO truncation event ever occurred.
    lfc.observe_tool_result(
        engine, _write_call("/workspace/notes.md"), _write_result(14), is_error=False
    )
    lfc.register_turn_byte_production(engine, added_bytes=True)
    # Idle past the stall threshold (the model reads/lists/thinks).
    for _ in range(engine.config.rc.longfile_stall_turns + 1):
        lfc.register_turn_byte_production(engine, added_bytes=False)
    # No truncation was ever seen on this path → the driver MUST stay inert.
    assert engine._longfile_last_mutation_truncated is False
    assert lfc.active_path_truncation_seen(engine) is False
    assert lfc.decide_next_forced_tool(engine) is None


def test_driver_engages_only_after_truncation_seen_on_active_path(engine_factory) -> None:
    """The flip-side of the regression-guard: the SAME below-floor + idle state
    DOES engage once a truncation event is recorded for the active path."""
    engine = engine_factory(rc=_rc(longfile_stall_turns=2))
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(1000), is_error=False
    )
    lfc.register_turn_byte_production(engine, added_bytes=True)
    # Inert until a truncation is recorded for the active path.
    for _ in range(engine.config.rc.longfile_stall_turns):
        lfc.register_turn_byte_production(engine, added_bytes=False)
    assert lfc.decide_next_forced_tool(engine) is None
    # Record a truncation on the active path → the driver may now engage.
    lfc.note_truncated_mutation(engine, "/workspace/big.py")
    assert lfc.active_path_truncation_seen(engine) is True
    assert lfc.decide_next_forced_tool(engine) == "AppendFile"


# ── forced AppendFile on stall + cap ─────────────────────────────────────
def _drive_to_stall_below_floor(engine) -> None:
    """One small (below-floor) TRUNCATED header write, then enough idle turns to
    stall — the canonical truncated-large-file shape the driver engages on."""
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(1000), is_error=False
    )
    # The engage gate requires a truncation event on the active path.
    lfc.note_truncated_mutation(engine, "/workspace/big.py")
    lfc.register_turn_byte_production(engine, added_bytes=True)
    # idle for longfile_stall_turns turns
    for _ in range(engine.config.rc.longfile_stall_turns):
        lfc.register_turn_byte_production(engine, added_bytes=False)


def test_stall_below_floor_forces_appendfile(engine_factory) -> None:
    engine = engine_factory(rc=_rc(longfile_stall_turns=2))
    _drive_to_stall_below_floor(engine)
    assert lfc.decide_next_forced_tool(engine) == "AppendFile"


def test_no_force_before_stall_threshold(engine_factory) -> None:
    engine = engine_factory(rc=_rc(longfile_stall_turns=2))
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(1000), is_error=False
    )
    lfc.register_turn_byte_production(engine, added_bytes=True)
    lfc.register_turn_byte_production(engine, added_bytes=False)  # only 1 idle turn
    assert engine._turns_since_last_byte_adding_mutation == 1
    assert lfc.decide_next_forced_tool(engine) is None


def test_forced_appendfile_respects_cap(engine_factory) -> None:
    engine = engine_factory(rc=_rc(longfile_stall_turns=1, longfile_max_forced_appends=3))
    # Write below floor so finalize is never permitted — only appends fire.
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(1000), is_error=False
    )
    lfc.note_truncated_mutation(engine, "/workspace/big.py")  # Truncation engage gate
    lfc.register_turn_byte_production(engine, added_bytes=True)
    for _ in range(3):
        lfc.register_turn_byte_production(engine, added_bytes=False)
        assert lfc.decide_next_forced_tool(engine) == "AppendFile"
        lfc.commit_forced_append(engine)
    # Budget exhausted, file still below floor → no further append; finalize is
    # blocked by the empty-finalize guard → no action at all.
    lfc.register_turn_byte_production(engine, added_bytes=False)
    assert engine._longfile_forced_appends == 3
    assert lfc.decide_next_forced_tool(engine) is None


# ── forced FinalizeFile on plateau / done ───────────────────────────────
def test_plateau_forces_finalize(engine_factory) -> None:
    engine = engine_factory(rc=_rc())
    # Two big mutations that bring the file well above floor, then a tiny delta
    # (a plateau) → forced finalize. The first header truncated (engage gate);
    # the sticky truncated-path latch survives the later clean appends.
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(4000), is_error=False
    )
    lfc.note_truncated_mutation(engine, "/workspace/big.py")  # Truncation engage gate
    lfc.register_turn_byte_production(engine, added_bytes=True)
    lfc.observe_tool_result(
        engine, _append_call("/workspace/big.py"), _append_result(4000, 8000),
        is_error=False,
    )
    lfc.register_turn_byte_production(engine, added_bytes=True)
    # A tiny third append → plateau (recent delta << mean), file at/above floor.
    lfc.observe_tool_result(
        engine, _append_call("/workspace/big.py"), _append_result(50, 8050),
        is_error=False,
    )
    lfc.register_turn_byte_production(engine, added_bytes=True)
    assert lfc.plausibly_complete(engine) is True
    assert lfc.decide_next_forced_tool(engine) == "FinalizeFile"


def test_stall_at_floor_forces_finalize(engine_factory) -> None:
    """A model that produced a complete (>= floor, clean) file then idles →
    forced FinalizeFile (it is done producing, just won't seal)."""
    engine = engine_factory(rc=_rc(longfile_stall_turns=2))
    # Header truncated early (engage gate + sets the latch); then a CLEAN append
    # brings the file above floor and clears the transient truncated-tail flag
    # while the sticky path latch survives → plausibly complete, idle → seal.
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(3000), is_error=False
    )
    lfc.note_truncated_mutation(engine, "/workspace/big.py")  # Truncation engage gate
    lfc.register_turn_byte_production(engine, added_bytes=True)
    lfc.observe_tool_result(
        engine, _append_call("/workspace/big.py"), _append_result(3000, 6000),
        is_error=False,
    )
    lfc.register_turn_byte_production(engine, added_bytes=True)
    for _ in range(2):
        lfc.register_turn_byte_production(engine, added_bytes=False)
    assert lfc.plausibly_complete(engine) is True
    assert lfc.active_path_truncation_seen(engine) is True
    assert lfc.decide_next_forced_tool(engine) == "FinalizeFile"


def test_terminal_seal_after_forced_appends_when_truncating(engine_factory) -> None:
    """When forced appends keep truncating (body grows past floor but the tail
    stays mid-content), the terminal seal fires once the append budget is spent."""
    engine = engine_factory(rc=_rc(longfile_max_forced_appends=2, longfile_stall_turns=1))
    lfc.observe_tool_result(
        engine, _append_call("/workspace/big.py"), _append_result(9000, 9000),
        is_error=False,
    )
    lfc.register_turn_byte_production(engine, added_bytes=True)
    # Mark the tail truncated (mid-content) so it is NOT plausibly complete.
    lfc.note_truncated_mutation(engine, "/workspace/big.py")
    engine._longfile_forced_appends = 2  # append budget already exhausted
    lfc.register_turn_byte_production(engine, added_bytes=False)
    # Above floor (9000 >= 4096) so finalize is permitted; append budget spent;
    # not plausibly complete → terminal seal.
    assert lfc.plausibly_complete(engine) is False
    assert lfc.finalize_permitted(engine) is True
    assert lfc.decide_next_forced_tool(engine) == "FinalizeFile"


def test_forced_finalize_respects_cap(engine_factory) -> None:
    engine = engine_factory(rc=_rc(longfile_stall_turns=2, longfile_max_forced_finalizes=1))
    # Truncated header (engage gate) then a clean append above floor (clears the
    # transient flag, keeps the sticky latch) → plausibly complete, idle → seal.
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(3000), is_error=False
    )
    lfc.note_truncated_mutation(engine, "/workspace/big.py")  # Truncation engage gate
    lfc.register_turn_byte_production(engine, added_bytes=True)
    lfc.observe_tool_result(
        engine, _append_call("/workspace/big.py"), _append_result(3000, 6000),
        is_error=False,
    )
    lfc.register_turn_byte_production(engine, added_bytes=True)
    for _ in range(2):
        lfc.register_turn_byte_production(engine, added_bytes=False)
    assert lfc.decide_next_forced_tool(engine) == "FinalizeFile"
    lfc.commit_forced_finalize(engine)
    # Budget of 1 exhausted → no further finalize.
    lfc.register_turn_byte_production(engine, added_bytes=False)
    assert lfc.decide_next_forced_tool(engine) is None


# ── HARD empty / below-floor finalize guard (the validated edge) ─────────
def test_never_force_finalize_on_empty_file(engine_factory) -> None:
    """The validated edge: a 0-byte file MUST NEVER be force-finalized."""
    engine = engine_factory(rc=_rc(longfile_stall_turns=1))
    # No byte-adding mutation ever landed → file is empty, no active path.
    for _ in range(5):
        lfc.register_turn_byte_production(engine, added_bytes=False)
    assert lfc.active_file_bytes(engine) == 0
    assert lfc.finalize_permitted(engine) is False
    assert lfc.decide_next_forced_tool(engine) is None


def test_never_force_finalize_below_floor(engine_factory) -> None:
    """A below-floor file (1000 < 4096) is never sealed — only appends drive."""
    engine = engine_factory(rc=_rc(longfile_stall_turns=1, longfile_max_forced_appends=8))
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(1000), is_error=False
    )
    lfc.note_truncated_mutation(engine, "/workspace/big.py")  # Truncation engage gate
    lfc.register_turn_byte_production(engine, added_bytes=True)
    lfc.register_turn_byte_production(engine, added_bytes=False)
    # Below floor → finalize is BLOCKED by the empty-finalize guard, so the
    # ONLY action available is to drive more content via AppendFile (never a
    # premature seal of an under-floor file).
    assert lfc.finalize_permitted(engine) is False
    assert lfc.decide_next_forced_tool(engine) == "AppendFile"


def test_never_force_finalize_below_floor_when_append_exhausted(engine_factory) -> None:
    """Below-floor + no append budget left → NO action (never seal under-floor)."""
    engine = engine_factory(rc=_rc(longfile_stall_turns=1, longfile_max_forced_appends=2))
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(1000), is_error=False
    )
    lfc.note_truncated_mutation(engine, "/workspace/big.py")  # Truncation engage gate
    lfc.register_turn_byte_production(engine, added_bytes=True)
    engine._longfile_forced_appends = 2  # append budget exhausted
    lfc.register_turn_byte_production(engine, added_bytes=False)
    # Below floor, no append budget — finalize is BLOCKED → no action at all.
    assert lfc.finalize_permitted(engine) is False
    assert lfc.decide_next_forced_tool(engine) is None


def test_finalize_permitted_exactly_at_floor(engine_factory) -> None:
    engine = engine_factory(rc=_rc(longfile_expected_floor_bytes=4096, longfile_min_finalize_fraction=1.0))
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(4096), is_error=False
    )
    assert lfc.finalize_permitted(engine) is True
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(4095), is_error=False
    )
    assert lfc.finalize_permitted(engine) is False


def test_min_finalize_fraction_below_one_lowers_guard(engine_factory) -> None:
    engine = engine_factory(
        rc=_rc(longfile_expected_floor_bytes=4000, longfile_min_finalize_fraction=0.5)
    )
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(2000), is_error=False
    )
    # 2000 >= max(1, 4000*0.5=2000) → permitted; but still below the expected
    # floor so it is NOT plausibly_complete (only the explicit seal paths use it).
    assert lfc.finalize_permitted(engine) is True
    assert lfc.plausibly_complete(engine) is False


# ── INCOMPLETE continue message + tail anchor ───────────────────────────
def test_continue_message_is_incomplete_with_tail(engine_factory) -> None:
    engine = engine_factory(rc=_rc(longfile_tail_anchor_chars=20))
    tail_marker = "LAST_TWENTY_CHARS_END!"
    body = "header line\n" * 50 + tail_marker
    # Record the model's own write into history so the tail anchor is found.
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="tc",
                    name="Write",
                    arguments_json=json.dumps(
                        {"path": "/workspace/big.py", "content": body}
                    ),
                )
            ],
        )
    )
    lfc.observe_tool_result(
        engine,
        _write_call("/workspace/big.py", content=body),
        _write_result(len(body.encode())),
        is_error=False,
    )
    msg = lfc.build_continue_message(engine)
    low = msg.lower()
    # (a) byte counts present
    assert str(len(body.encode())) in msg
    # (b) tail anchor present (last N chars of the model's write)
    assert tail_marker[-20:] in msg
    # (c) says INCOMPLETE / continue (EN + RU)
    assert "incomplete" in low
    assert "неполная" in low
    assert "appendfile" in low
    # (d) NEVER "safe on disk" / done
    assert "safe" not in low
    assert "готов" not in low


def test_continue_message_without_history_tail_still_valid(engine_factory) -> None:
    engine = engine_factory(rc=_rc())
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(1234), is_error=False
    )
    msg = lfc.build_continue_message(engine)
    assert "1234" in msg
    assert "/workspace/big.py" in msg
    assert "safe" not in msg.lower()


def test_continue_message_has_no_byte_target(engine_factory) -> None:
    """The continue message states the file's CURRENT size but NO byte
    target (an earlier draft said 'the target is about N bytes', which caused
    models to pad legitimately small files). The expected-floor value must not
    appear and the 'target ... bytes' phrasing must be gone."""
    floor = 4096
    engine = engine_factory(rc=_rc(longfile_expected_floor_bytes=floor))
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(1234), is_error=False
    )
    msg = lfc.build_continue_message(engine)
    low = msg.lower()
    # The floor value is NOT quoted as a target anywhere in the message.
    assert str(floor) not in msg
    # No "target ... bytes" / "целевой размер" framing (EN + RU).
    assert "target is about" not in low
    assert "целевой размер" not in low
    # Still INCOMPLETE + continue + finish (EN + RU), never a byte goal.
    assert "incomplete" in low
    assert "неполная" in low
    assert "finish the file" in low
    assert "допишите файл" in low
    # The current size IS reported (it is informative, not a target).
    assert "1234" in msg
    # Formatting must not leave a dangling placeholder.
    assert "{expected_floor_bytes}" not in msg


def test_continue_message_tolerates_legacy_floor_placeholder_override(engine_factory) -> None:
    """A tenant DB override that STILL contains the old ``{expected_floor_bytes}``
    placeholder must keep formatting (the field is kept as a tolerated format
    key; a missing kwarg would raise)."""
    engine = engine_factory(
        rc=_rc(
            longfile_expected_floor_bytes=4096,
            longfile_continue_message_en=(
                "INCOMPLETE: {file_bytes} bytes, target {expected_floor_bytes}. "
                "Tail: {tail}. Continue {path}."
            ),
            longfile_continue_message_ru=(
                "НЕПОЛНАЯ: {file_bytes} байт, цель {expected_floor_bytes}. "
                "Хвост: {tail}. Продолжайте {path}."
            ),
        )
    )
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(1234), is_error=False
    )
    # Must NOT raise a KeyError despite the legacy placeholder.
    msg = lfc.build_continue_message(engine)
    assert "1234" in msg
    assert "4096" in msg  # the legacy override DID ask for it → it formats


def test_continue_message_tolerates_unknown_placeholder_in_tenant_override(
    engine_factory, caplog
) -> None:
    """a tenant DB override that carries an UNKNOWN placeholder typo
    (e.g. ``{bytes}`` instead of ``{file_bytes}``) must NOT raise ``KeyError``
    inside ``_maybe_drive_longfile_convergence`` — that path is the run's
    end-of-turn convergence step and a KeyError here surfaces as a
    run-terminal error the generic stream-loop catch-all swallows, killing
    the run instead of converging.

    The fix renders the unknown placeholder LITERALLY (e.g. ``{bytes}`` stays
    as ``{bytes}``) so the typo is visible in the produced message, and emits
    a WARNING so prod operators see the bad override in run logs. Known keys
    (path / file_bytes / file_lines / tail / expected_floor_bytes) must still
    format correctly in the SAME message.
    """
    engine = engine_factory(
        rc=_rc(
            longfile_continue_message_en=(
                "INCOMPLETE {path} has {file_bytes} bytes and {bytes} more. "
                "Tail: {tail}."
            ),
            longfile_continue_message_ru=(
                "НЕПОЛНАЯ {path} — {file_bytes} байт и {bytes} ещё. "
                "Хвост: {tail}."
            ),
        )
    )
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(1234), is_error=False
    )
    # Must NOT raise a KeyError on the unknown {bytes} placeholder.
    with caplog.at_level("WARNING", logger="protocore.runtime.longfile_convergence"):
        msg = lfc.build_continue_message(engine)
    # Known placeholders formatted exactly as before.
    assert "/workspace/big.py" in msg
    assert "1234" in msg
    # The unknown placeholder is rendered LITERALLY (so the typo is visible).
    assert "{bytes}" in msg
    # A warning was emitted so prod operators see the bad override.
    assert any(
        "unknown placeholder" in rec.message and "bytes" in rec.message
        for rec in caplog.records
    ), f"expected a warning naming the unknown placeholder, got: {[r.message for r in caplog.records]}"


# ── Per-path append circuit-breaker ──────────────────────────────────────────
def test_per_path_append_breaker(engine_factory) -> None:
    """Total appends (forced + voluntary) to one path are capped at
    ``longfile_max_appends_per_path``. Beyond the cap the driver stops forcing
    appends (it seals if past floor, else stops driving). Bounds the 195/295
    self-loop even on a genuinely-truncated path."""
    engine = engine_factory(
        rc=_rc(
            longfile_stall_turns=1,
            longfile_max_appends_per_path=3,
            longfile_max_forced_appends=100,  # large so the breaker, not this, binds
            longfile_expected_floor_bytes=4096,
        )
    )
    # Truncated header (engage gate), below floor so only appends ever fire.
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(1000), is_error=False
    )
    lfc.note_truncated_mutation(engine, "/workspace/big.py")
    lfc.register_turn_byte_production(engine, added_bytes=True)
    # Three appends (each below floor) — the breaker fires on the 3rd.
    total = 1000
    for i in range(3):
        lfc.register_turn_byte_production(engine, added_bytes=False)
        assert lfc.decide_next_forced_tool(engine) == "AppendFile", f"append {i}"
        total += 500
        lfc.observe_tool_result(
            engine, _append_call("/workspace/big.py"), _append_result(500, total),
            is_error=False,
        )
        lfc.register_turn_byte_production(engine, added_bytes=True)
    # 3 appends recorded → breaker tripped; below floor → no seal, no action.
    assert engine._longfile_appends_per_path["/workspace/big.py"] == 3
    assert lfc.append_breaker_tripped(engine) is True
    lfc.register_turn_byte_production(engine, added_bytes=False)
    assert lfc.decide_next_forced_tool(engine) is None


def test_append_breaker_seals_when_past_floor(engine_factory) -> None:
    """When the breaker trips on a file already past floor, the driver
    seals it (rather than leaving a complete-but-unsealed file)."""
    engine = engine_factory(
        rc=_rc(
            longfile_stall_turns=1,
            longfile_max_appends_per_path=2,
            longfile_max_forced_appends=100,
            longfile_max_forced_finalizes=2,
            longfile_expected_floor_bytes=4096,
        )
    )
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(2000), is_error=False
    )
    lfc.note_truncated_mutation(engine, "/workspace/big.py")
    lfc.register_turn_byte_production(engine, added_bytes=True)
    # Two clean appends push past floor (and hit the per-path cap of 2). The
    # clean appends clear the transient truncated-tail flag → plausibly complete.
    for total in (5000, 8000):
        lfc.observe_tool_result(
            engine, _append_call("/workspace/big.py"),
            _append_result(3000, total), is_error=False,
        )
        lfc.register_turn_byte_production(engine, added_bytes=True)
    assert lfc.append_breaker_tripped(engine) is True
    lfc.register_turn_byte_production(engine, added_bytes=False)
    # Breaker tripped + past floor + finalize permitted → seal.
    assert lfc.decide_next_forced_tool(engine) == "FinalizeFile"


def test_append_breaker_round_trips_through_snapshot(engine_factory) -> None:
    """The per-path truncation latch + append counter survive a
    cross-pod snapshot/resume."""
    import asyncio

    engine = engine_factory(rc=_rc())
    lfc.observe_tool_result(
        engine, _append_call("/workspace/big.py"), _append_result(500, 500),
        is_error=False,
    )
    lfc.note_truncated_mutation(engine, "/workspace/big.py")
    engine._longfile_salvage_seq = 3  # salvage id counter
    snap = engine.snapshot()
    assert snap["longfile_truncated_paths"] == ["/workspace/big.py"]
    assert snap["longfile_appends_per_path"] == {"/workspace/big.py": 1}
    assert snap["longfile_salvage_seq"] == 3
    fresh = engine_factory(rc=_rc())
    asyncio.run(fresh.resume_from_snapshot(snap))
    assert fresh._longfile_truncated_paths == {"/workspace/big.py"}
    assert fresh._longfile_appends_per_path == {"/workspace/big.py": 1}
    assert fresh._longfile_salvage_seq == 3
    assert lfc.active_path_truncation_seen(fresh) is True


# ── no-op for strong models ──────────────────────────────────────────────
def test_happy_path_no_forcing(engine_factory) -> None:
    """A model that adds bytes every turn never stalls → nothing fires."""
    engine = engine_factory(rc=_rc(longfile_stall_turns=2))
    total = 0
    for delta in (3000, 3000, 3000, 3000):
        total += delta
        lfc.observe_tool_result(
            engine,
            _append_call("/workspace/big.py"),
            _append_result(delta, total),
            is_error=False,
        )
        lfc.register_turn_byte_production(engine, added_bytes=True)
        # Every turn produced bytes → never a stall, never a forced action.
        assert engine._turns_since_last_byte_adding_mutation == 0
        assert lfc.decide_next_forced_tool(engine) in (None,)
    assert engine._longfile_forced_appends == 0
    assert engine._longfile_forced_finalizes == 0


def test_no_force_when_no_file_in_flight(engine_factory) -> None:
    """A run that never wrote a file (pure Q&A) never triggers the driver."""
    engine = engine_factory(rc=_rc(longfile_stall_turns=1))
    for _ in range(5):
        lfc.observe_tool_result(engine, _read_call(), json.dumps({"ok": True}), is_error=False)
        lfc.register_turn_byte_production(engine, added_bytes=False)
    assert lfc.decide_next_forced_tool(engine) is None


# ── RC kill-switch — bit-identical to pre-FEAT ──────────────────────────
def test_disabled_rc_is_bit_identical(engine_factory) -> None:
    engine = engine_factory(rc=_rc(longfile_convergence_enabled=False, longfile_stall_turns=1))
    # Every entry point is a no-op when disabled.
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(6000), is_error=False
    )
    lfc.register_turn_byte_production(engine, added_bytes=False)
    lfc.register_turn_byte_production(engine, added_bytes=False)
    lfc.note_truncated_mutation(engine, "/workspace/big.py")
    # State stays pristine: no path bound, no size, no counter advance.
    assert engine._longfile_active_path is None
    assert engine._longfile_active_file_bytes == 0
    assert engine._turns_since_last_byte_adding_mutation == 0
    assert engine._longfile_last_mutation_truncated is False
    assert lfc.decide_next_forced_tool(engine) is None
    assert lfc.is_enabled(engine) is False


# ── snapshot/resume preserves the convergence latches ────────────────────────
def test_snapshot_resume_preserves_convergence_state(engine_factory) -> None:
    engine = engine_factory(rc=_rc())
    lfc.observe_tool_result(
        engine, _append_call("/workspace/big.py"), _append_result(5000, 5000),
        is_error=False,
    )
    engine._turns_since_last_byte_adding_mutation = 2
    engine._longfile_forced_appends = 3
    engine._longfile_forced_finalizes = 1
    lfc.note_truncated_mutation(engine, "/workspace/big.py")

    snap = engine.snapshot()
    assert snap["turns_since_last_byte_adding_mutation"] == 2
    assert snap["longfile_forced_appends"] == 3
    assert snap["longfile_forced_finalizes"] == 1
    assert snap["longfile_active_path"] == "/workspace/big.py"
    assert snap["longfile_active_file_bytes"] == 5000
    assert snap["longfile_mutation_deltas"] == [5000]
    assert snap["longfile_last_mutation_truncated"] is True

    fresh = engine_factory(rc=_rc())
    import asyncio

    asyncio.run(fresh.resume_from_snapshot(snap))
    assert fresh._turns_since_last_byte_adding_mutation == 2
    assert fresh._longfile_forced_appends == 3
    assert fresh._longfile_forced_finalizes == 1
    assert fresh._longfile_active_path == "/workspace/big.py"
    assert fresh._longfile_active_file_bytes == 5000
    assert fresh._longfile_mutation_deltas == [5000]
    assert fresh._longfile_last_mutation_truncated is True


def test_force_next_tool_round_trips(engine_factory) -> None:
    engine = engine_factory(rc=_rc())
    assert lfc.take_force_next_tool(engine) is None
    lfc.set_force_next_tool(engine, "AppendFile")
    assert lfc.take_force_next_tool(engine) == "AppendFile"
    # Consumed once.
    assert lfc.take_force_next_tool(engine) is None


def test_peek_force_next_tool_does_not_consume(engine_factory) -> None:
    """``peek_force_next_tool`` is a non-destructive
    read so the stream builder can gate the pop on the surface including the
    forced tool. A peek must NOT clear the hint, so a subsequent
    ``take_force_next_tool`` still returns the same value.
    """
    engine = engine_factory(rc=_rc())
    assert lfc.peek_force_next_tool(engine) is None
    lfc.set_force_next_tool(engine, "AppendFile")
    # Peek does NOT consume.
    assert lfc.peek_force_next_tool(engine) == "AppendFile"
    assert lfc.peek_force_next_tool(engine) == "AppendFile"
    # The hint is still live for a downstream consumer.
    assert lfc.take_force_next_tool(engine) == "AppendFile"


def test_peek_force_next_tool_ignores_non_string(engine_factory) -> None:
    """A non-string sentinel (e.g. an int set via
    ``setattr`` by a buggy caller) is reported as ``None`` by both peek and
    take, matching the pre-existing type-guard in ``take_force_next_tool``.
    """
    engine = engine_factory(rc=_rc())
    setattr(engine, lfc.FORCE_NEXT_TOOL_ATTR, 42)
    assert lfc.peek_force_next_tool(engine) is None
    # The bogus value is still parked there until something else clears it;
    # a take still returns None (the type guard).
    assert lfc.take_force_next_tool(engine) is None


@pytest.mark.parametrize("enabled", [True, False])
def test_is_enabled_reflects_rc(engine_factory, enabled: bool) -> None:
    engine = engine_factory(rc=_rc(longfile_convergence_enabled=enabled))
    assert lfc.is_enabled(engine) is enabled


def test_finalized_file_stops_the_driver(engine_factory) -> None:
    """A successful FinalizeFile seals the file → no further forced action."""
    engine = engine_factory(rc=_rc(longfile_stall_turns=2))
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(6000), is_error=False
    )
    lfc.register_turn_byte_production(engine, added_bytes=True)
    # The model finalizes (success).
    finalize = ToolCall(id="tc", name="FinalizeFile", arguments={"path": "/workspace/big.py"})
    lfc.observe_tool_result(engine, finalize, json.dumps({"path": "/workspace/big.py"}), is_error=False)
    assert engine._longfile_finalized is True
    # Even after idling past the stall threshold, the driver does nothing.
    for _ in range(3):
        lfc.register_turn_byte_production(engine, added_bytes=False)
    assert lfc.decide_next_forced_tool(engine) is None


def test_side_file_finalize_does_not_seal_active_artifact(engine_factory) -> None:
    """A voluntary FinalizeFile of a SMALL side file must NOT flip the run-global
    ``_longfile_finalized`` latch while the truncation-latched large artifact is
    still in flight (path-aware seal).

    Trigger: report.md (big) truncates once (engage latch) and is the active
    path; the model then Write->AppendFile->FinalizeFile's summary.md (its own
    chunk-protocol side use); report.md then stalls. A path-blind seal would
    flip the latch on summary.md's finalize, making ``decide_next_forced_tool``
    and ``terminal_seal_required`` both go inert so report.md ends unconverged
    and unsealed. The path-aware seal ignores the off-path finalize.
    """
    engine = engine_factory(rc=_rc(longfile_stall_turns=2))
    # report.md (big) — header truncates → engaged + active path bound, below
    # floor so it must keep being driven.
    lfc.observe_tool_result(
        engine, _write_call("/workspace/report.md"), _write_result(3000), is_error=False
    )
    lfc.note_truncated_mutation(engine, "/workspace/report.md")  # engage latch
    lfc.register_turn_byte_production(engine, added_bytes=True)
    assert engine._longfile_active_path == "/workspace/report.md"

    # The model finalizes a DIFFERENT, small side file (summary.md). This must be
    # ignored — it is off-path for the tracked convergence artifact.
    side_finalize = ToolCall(
        id="tc", name="FinalizeFile", arguments={"path": "/workspace/summary.md"}
    )
    lfc.observe_tool_result(
        engine, side_finalize, json.dumps({"path": "/workspace/summary.md"}), is_error=False
    )
    assert engine._longfile_finalized is False  # the side-file seal is ignored

    # report.md now stalls (no byte-adding mutation for stall_turns) below floor:
    # the driver must still engage (force AppendFile), NOT stay inert.
    for _ in range(2):
        lfc.register_turn_byte_production(engine, added_bytes=False)
    assert lfc.decide_next_forced_tool(engine) == "AppendFile"

    # And had report.md been driven past floor + self-continued, the run-end
    # terminal seal would still be eligible (the latch was never falsely flipped).
    for delta in (2000, 2000):
        lfc.observe_tool_result(
            engine,
            _append_call("/workspace/report.md"),
            _append_result(delta, lfc.active_file_bytes(engine) + delta),
            is_error=False,
        )
        lfc.register_turn_byte_production(engine, added_bytes=True)
    assert lfc.finalize_permitted(engine) is True
    assert lfc.terminal_seal_required(engine) is True


def test_active_path_finalize_seals_the_driver(engine_factory) -> None:
    """The companion to the off-path case: a FinalizeFile of the ACTIVE path
    still seals the latch (the fix is scoped to off-path / no-artifact only)."""
    engine = engine_factory(rc=_rc())
    lfc.observe_tool_result(
        engine, _write_call("/workspace/report.md"), _write_result(6000), is_error=False
    )
    lfc.register_turn_byte_production(engine, added_bytes=True)
    on_path_finalize = ToolCall(
        id="tc", name="FinalizeFile", arguments={"path": "/workspace/report.md"}
    )
    lfc.observe_tool_result(
        engine, on_path_finalize, json.dumps({"path": "/workspace/report.md"}), is_error=False
    )
    assert engine._longfile_finalized is True


def test_finalized_flag_round_trips_through_snapshot(engine_factory) -> None:
    engine = engine_factory(rc=_rc())
    engine._longfile_finalized = True
    snap = engine.snapshot()
    assert snap["longfile_finalized"] is True
    fresh = engine_factory(rc=_rc())
    import asyncio

    asyncio.run(fresh.resume_from_snapshot(snap))
    assert fresh._longfile_finalized is True


# ── TERMINAL SEAL — pure predicate eligibility ───────────────────────────────
def _drive_to_unsealed_complete_truncation_gated(engine) -> None:
    """A truncation-gated file driven past floor that the model self-continued
    (steady byte progress, NO stall) and never sealed — the run-end gap shape.

    The header truncated (engage latch) then clean appends grew the file well
    past floor, every turn adding bytes (so the stall clock never advanced).
    The file is complete-enough but UNSEALED — exactly the state the terminal
    seal must catch at turn-budget exhaustion.
    """
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(3000), is_error=False
    )
    lfc.note_truncated_mutation(engine, "/workspace/big.py")  # engage latch
    lfc.register_turn_byte_production(engine, added_bytes=True)
    total = 3000
    for delta in (4000, 4000, 4000):
        total += delta
        lfc.observe_tool_result(
            engine,
            _append_call("/workspace/big.py"),
            _append_result(delta, total),
            is_error=False,
        )
        lfc.register_turn_byte_production(engine, added_bytes=True)


def test_terminal_seal_required_on_unsealed_truncation_gated_complete_file(
    engine_factory,
) -> None:
    """The core predicate: a complete-enough, unsealed, truncation-gated file is
    eligible for the run-end seal (all gates pass)."""
    engine = engine_factory(rc=_rc())
    _drive_to_unsealed_complete_truncation_gated(engine)
    assert lfc.active_path_truncation_seen(engine) is True
    assert engine._longfile_finalized is False
    assert lfc.finalize_permitted(engine) is True
    assert lfc.terminal_seal_required(engine) is True


def test_terminal_seal_required_inert_without_truncation(engine_factory) -> None:
    """Zero-collateral: a file driven past floor WITHOUT any truncation event is
    NEVER eligible for the terminal seal (requires truncation event)."""
    engine = engine_factory(rc=_rc())
    total = 0
    for delta in (4000, 4000, 4000):
        total += delta
        lfc.observe_tool_result(
            engine,
            _append_call("/workspace/big.py"),
            _append_result(delta, total),
            is_error=False,
        )
        lfc.register_turn_byte_production(engine, added_bytes=True)
    assert lfc.active_file_bytes(engine) >= engine.config.rc.longfile_expected_floor_bytes
    assert lfc.active_path_truncation_seen(engine) is False
    assert lfc.terminal_seal_required(engine) is False


def test_terminal_seal_required_blocked_below_floor(engine_factory) -> None:
    """A truncated but below-floor file is NEVER sealed (the empty/floor guard)."""
    engine = engine_factory(rc=_rc(longfile_expected_floor_bytes=4096))
    lfc.observe_tool_result(
        engine, _write_call("/workspace/big.py"), _write_result(1000), is_error=False
    )
    lfc.note_truncated_mutation(engine, "/workspace/big.py")
    lfc.register_turn_byte_production(engine, added_bytes=True)
    assert lfc.active_path_truncation_seen(engine) is True
    assert lfc.finalize_permitted(engine) is False
    assert lfc.terminal_seal_required(engine) is False


def test_terminal_seal_required_blocked_when_already_finalized(engine_factory) -> None:
    """An already-sealed file is never re-sealed."""
    engine = engine_factory(rc=_rc())
    _drive_to_unsealed_complete_truncation_gated(engine)
    engine._longfile_finalized = True
    assert lfc.terminal_seal_required(engine) is False


def test_terminal_seal_required_respects_finalize_budget(engine_factory) -> None:
    """The seal reuses the finalize budget — exhausted budget ⇒ not eligible."""
    engine = engine_factory(rc=_rc(longfile_max_forced_finalizes=2))
    _drive_to_unsealed_complete_truncation_gated(engine)
    engine._longfile_forced_finalizes = 2  # budget exhausted
    assert lfc.terminal_seal_required(engine) is False
    engine._longfile_forced_finalizes = 1  # budget remaining
    assert lfc.terminal_seal_required(engine) is True


def test_terminal_seal_required_disabled_rc_is_noop(engine_factory) -> None:
    """The RC kill-switch makes the predicate a bit-identical no-op (False)."""
    engine = engine_factory(rc=_rc(longfile_convergence_enabled=False))
    # Force the eligibility state directly (the observe path is itself disabled).
    engine._longfile_active_path = "/workspace/big.py"
    engine._longfile_active_file_bytes = 12000
    engine._longfile_truncated_paths.add("/workspace/big.py")
    assert lfc.terminal_seal_required(engine) is False


def test_voluntary_seal_used_latch_round_trips_through_snapshot(engine_factory) -> None:
    """The one-shot VOLUNTARY-seal latch (separate from the max-turns
    terminal-seal latch) is snapshot-persisted and restored with a conservative
    default (False)."""
    engine = engine_factory(rc=_rc())
    assert engine._longfile_voluntary_seal_used is False
    assert engine.snapshot()["longfile_voluntary_seal_used"] is False

    engine._longfile_voluntary_seal_used = True
    snap = engine.snapshot()
    assert snap["longfile_voluntary_seal_used"] is True

    fresh = engine_factory(rc=_rc())
    import asyncio

    asyncio.run(fresh.resume_from_snapshot(snap))
    assert fresh._longfile_voluntary_seal_used is True

    # Pre-FEAT snapshot (key absent) restores the safe default.
    pre_feat = engine_factory(rc=_rc())
    legacy = dict(snap)
    del legacy["longfile_voluntary_seal_used"]
    asyncio.run(pre_feat.resume_from_snapshot(legacy))
    assert pre_feat._longfile_voluntary_seal_used is False
