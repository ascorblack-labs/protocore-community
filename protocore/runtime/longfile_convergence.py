"""runtime-driven, stall-aware large-file convergence.

The pure decision logic for driving a large-file write to completion on a weak
local model at a small per-call output cap. The model's dominant failure shape
is NON-CONVERGENCE: it writes one header (often truncated), then idle-inspects
via non-mutation tool calls (Read/Grep/Bash/List) or prose, never appending or
finalizing, until the turn cap. Both naive recoveries fail — "discard & redo"
re-truncates/spirals, and "the file is SAFE on disk, continue" reads to the
model as "task done" so it stops producing.

The validated fix (empirically confirmed; forcing is the active ingredient)
is for the
RUNTIME to drive completion:

* a **stall detector** keyed to *turns-since-last-BYTE-ADDING-mutation* (a
  Write/AppendFile that actually grew the file) while the file is below an
  expected-complete floor — NOT keyed to append-count and NOT the prose path
  (both are bypassed by the header-then-idle shape);
* on a stall while the file is below its plausible-complete floor → force
  ``tool_choice=AppendFile`` ("continue now");
* on a byte-plateau (recent delta shrank) OR a stall while the file is already
  at/above floor → force ``tool_choice=FinalizeFile``;
* a **HARD empty-finalize guard** (the validated edge): NEVER force
  FinalizeFile on an empty / below-floor file (the probe's one weak task was a
  forced-finalize firing on a 0-byte file);
* everything bounded by per-run forced-round caps (``longfile_max_forced_*``),
  subordinate to ``max_turns_per_run`` — the driver can NEVER spin.

This module is PURE: it reads/updates the engine's per-run convergence counters
(all snapshot-persisted on :class:`~protocore.runtime.query_engine.QueryEngine`)
and returns a decision string; the query loop performs the actual forced-tool
wiring + message injection. It never touches IO, the LLM, or module-level
state. RC-gated by ``longfile_convergence_enabled`` — when disabled every entry
point is a no-op, so behaviour is bit-identical to pre-FEAT.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Final, Literal

from protocore.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from protocore.contracts.runtime_constants import RuntimeConstants
    from protocore.contracts.types import ToolCall
    from protocore.runtime.query_engine import QueryEngine

_logger = get_logger(__name__)


class _SafeFormatDict(dict[str, Any]):
    """``format_map`` mapping that tolerates unknown placeholders.

    A tenant-overridable continue message (RC: ``longfile_continue_message_en``
    / ``_ru``) may carry a typo'd placeholder (e.g. ``{bytes}`` instead of
    ``{file_bytes}``). A bare ``str.format(**fields)`` raises :class:`KeyError`
    for any unknown name, which surfaces from
    :func:`_maybe_drive_longfile_convergence` as a run-terminal error the
    generic stream-loop catch-all swallows — the run terminates instead of
    converging. ``format_map`` lets us intercept the lookup: any missing key
    is rendered as its literal placeholder text (e.g. ``{bytes}``) so the
    typo is visible in the message AND a ``logger.warning`` is emitted so
    prod operators see the bad override. Known keys (path/file_bytes/
    file_lines/expected_floor_bytes/tail) format exactly as before.
    """

    def __missing__(self, key: str) -> str:
        # Render the unknown placeholder literally so the message still ships;
        # log once per unknown name so a dashboard override typo is visible
        # in the run's prod logs without flooding.
        _logger.warning(
            "longfile continue message references unknown placeholder %r; "
            "known keys are path, file_bytes, file_lines, expected_floor_bytes, tail",
            key,
        )
        return "{" + key + "}"

# The runtime's own chunkable file-write tools. The stall detector keys on byte
# production reported by THESE tools' results; a per-tenant flagged content tool
# drives the truncation-recovery wording but its append-resume byte semantics
# are tool-specific, so it is not tracked here (mirrors
# ``query._record_chunk_write_success`` / ``CHUNKABLE_CONTENT_MUTATION_ALLOWLIST``).
_BYTE_MUTATION_TOOLS: frozenset[str] = frozenset({"Write", "AppendFile"})

ForcedTool = Literal["AppendFile", "FinalizeFile"]

# The tool that seals an in-flight artifact. Named here because the wind-down
# has to keep it on an otherwise-emptied tool surface: a run cut short while a
# long file is being written has that file on disk, unsealed, and removing the
# only tool able to close it would throw the work away in the name of stopping
# cleanly.
FINALIZE_FILE_TOOL_NAME: Final[str] = "FinalizeFile"

# The forced-tool the convergence driver wants on the NEXT assistant stream.
# Stored as a transient engine attribute (consumed + cleared by the stream
# builder). A within-loop hint — NOT snapshot-persisted (a crash before the
# stream merely re-derives the stall on the next turn from the persisted
# counters, which is safe).
FORCE_NEXT_TOOL_ATTR = "_longfile_force_next_tool"

# Transient per-turn flag: set True by :func:`observe_tool_result` when a
# byte-adding mutation landed this turn; read + cleared by
# :func:`register_completed_turn` at the turn boundary so the stall clock is
# advanced exactly once per turn. Not snapshot-persisted (within-turn only).
_TURN_ADDED_BYTES_ATTR = "_longfile_turn_added_bytes"


def is_enabled(engine: QueryEngine) -> bool:
    """Return True iff the convergence driver is enabled for this engine."""
    return bool(engine.config.rc.longfile_convergence_enabled)


def _int_field(payload: dict[str, Any], key: str) -> int | None:
    """Read a strictly-positive int field from a result payload, else None."""
    raw = payload.get(key)
    if not isinstance(raw, int) or isinstance(raw, bool):
        return None
    return raw if raw > 0 else None


def _parse_byte_result(
    tool_call: ToolCall, content: str
) -> tuple[int, int | None, int | None] | None:
    """Parse a Write/AppendFile result into ``(delta_bytes, total_bytes, lines)``.

 The tool result content is ``BaseToolOutput.model_dump_json`` (the JSON the
 model sees). ``WriteOutput`` carries ``bytes_written`` — a Write OVERWRITES,
 so that IS both the delta and the new total; it carries NO line count, so
 the caller derives the full-file line count from the Write ``content`` arg.
 ``AppendFileOutput`` carries ``bytes_appended`` (delta), ``bytes_total``
 (cumulative size) and ``line_count_total`` (full-file lines).

 Returns ``(delta, total, lines)`` where ``total``/``lines`` are ``None`` when
 the payload does not carry them (the CALLER applies the engine-aware fallback
never the bare delta, see A-). Returns None entirely for a non-byte
 tool, a non-JSON / error body, or a non-positive delta. NEVER raises — a
 parse failure is treated as "no bytes" (the stall counter then advances, the
 safe direction: drive MORE production, never a premature seal).
 """
    if tool_call.name not in _BYTE_MUTATION_TOOLS:
        return None
    try:
        payload: Any = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if tool_call.name == "Write":
        written = _int_field(payload, "bytes_written")
        if written is None:
            return None
        # A Write overwrites: bytes_written IS the new total. No line count in
        # WriteOutput — the caller counts lines from the Write ``content`` arg.
        return (written, written, None)
    # AppendFile — the explicit delta is required; total + line_count_total are
    # reported but tolerated-absent (the caller falls back to running-size+delta
    # for the total, never the bare delta — see A-).
    appended = _int_field(payload, "bytes_appended")
    if appended is None:
        return None
    total = _int_field(payload, "bytes_total")
    lines = _int_field(payload, "line_count_total")
    return (appended, total, lines)


def _resolve_write_path(tool_call: ToolCall) -> str | None:
    """Resolve the file path a Write/AppendFile call targets, or None.

    Accepts the canonical ``path`` plus the ``file_path`` alias (mirrors
    the host write-tool ``AliasChoices``). A call with no resolvable path is
    not tracked (returns None) — the stall detector then keeps using the
    previously-bound active path.
    """
    args = tool_call.arguments
    if not isinstance(args, dict):
        return None
    return _resolve_path_from_args(args)


def observe_tool_result(
    engine: QueryEngine,
    tool_call: ToolCall,
    content: str,
    *,
    is_error: bool,
    keep_truncated: bool = False,
) -> None:
    """Record one tool result's effect on byte production .

 Called from the successful-dispatch path for EVERY tool result. When the
 driver is disabled this is a no-op. A Write/AppendFile result that reports a
 positive byte delta:

 * resets ``_turns_since_last_byte_adding_mutation`` to 0;
 * binds ``_longfile_active_path`` to the call's path on the FIRST byte-adding
 mutation (so the stall/force/floor logic follows the file that exists);
 * appends the delta to ``_longfile_mutation_deltas`` (plateau read);
 * clears ``_longfile_last_mutation_truncated`` (a clean byte-adding write
 means the tail is no longer mid-content) — unless ``keep_truncated`` is
 True, which signals a SYNTHETIC RECOVERY dispatch
 (longfile salvage) where the synthetic write IS
 itself recovering a truncated write and the tail is semantically still
 mid-content. The caller's re-assert (or pre-set) of the flag is part of
 the SAME snapshot persist as the dispatch result, so a pod kill
 between the dispatch and the re-assert cannot land on a half-file
 with the truncated-tail flag already cleared.

 A non-byte result (an error, a non-mutation tool, or a zero-byte write) does
 NOT touch the stall clock here — the clock is advanced once per turn by
 :func:`register_turn_byte_production` after all of a turn's results land, so
 a turn with one Read and one byte-adding Write correctly counts as
 production (clock reset), not a stall.

 A byte-adding write to a DIFFERENT path than the bound active path (e.g. the
 model writes a small side-note file mid-run) is FULLY IGNORED for
 convergence tracking — it never updates the tracked size/lines, the plateau
 deltas, the stall clock, or the per-turn added-bytes flag.
 Otherwise a multi-file run corrupts the tracked size for the real artifact
 → false stall/plateau → spurious forced actions (universality violation).

 The same per-path discipline applies to FinalizeFile: a successful finalize
 seals the run-global ``_longfile_finalized`` latch ONLY when it targets the
 bound active path. A voluntary finalize of a small SIDE file (its own
 chunk-protocol use) while the truncation-latched large artifact is still in
 flight is IGNORED — otherwise it flips the run-global latch, the stall driver
 and the terminal seal both go inert, and the real artifact ends unconverged
 and unsealed (the exact gap the terminal seal shipped to close).
 """
    if not is_enabled(engine):
        return
    if is_error:
        return
    # A successful FinalizeFile seals the file — the driver stops (no point
    # re-forcing a finalize on an already-sealed file, even when FinalizeFile
    # is not the tenant's terminal tool and the loop continues). Path-aware:
    # only flip the run-global latch when the finalize targets the bound active
    # path. A finalize of a side file, or one with no active artifact in flight,
    # must NOT seal the latch — sealing the wrong path would let the
    # truncation-latched artifact end unconverged.
    if tool_call.name == "FinalizeFile":
        active_path = engine._longfile_active_path
        if active_path is not None and _resolve_write_path(tool_call) == active_path:
            engine._longfile_finalized = True
        return
    parsed = _parse_byte_result(tool_call, content)
    if parsed is None:
        return
    delta, total, lines = parsed
    call_path = _resolve_write_path(tool_call)
    if engine._longfile_active_path is None:
        if call_path is not None:
            engine._longfile_active_path = call_path
    elif call_path is not None and call_path != engine._longfile_active_path:
        # A byte-adding write to a path that is NOT the tracked active file.
        # Ignore it ENTIRELY: do not corrupt the active file's size/lines,
        # plateau deltas, stall clock, or per-turn flag.
        return
    # Resolve the running total. ``WriteOutput.bytes_written`` and
    # ``AppendFileOutput.bytes_total`` are authoritative when present; a
    # tolerated-absent AppendFile total falls back to running-size + delta
    # (A-) — NEVER the bare delta, which would freeze the size counter.
    if total is None:
        total = engine._longfile_active_file_bytes + delta
    # Track the running file SIZE from the result. A Write overwrites (total ==
    # bytes_written); an AppendFile reports the cumulative bytes_total. Take the
    # reported/derived total as authoritative (it already accounts for the
    # delta). The ``max()`` guards a malformed under-report.
    engine._longfile_active_file_bytes = max(
        engine._longfile_active_file_bytes if tool_call.name == "AppendFile" else 0,
        total,
    )
    # Track the REAL full-file line count (B-) so the continue
    # message reports the file's actual lines, not the tail's lines. AppendFile
    # reports ``line_count_total`` directly; a Write overwrites, so its full-file
    # line count is the line count of its ``content`` arg.
    engine._longfile_active_file_lines = _resolve_file_lines(
        tool_call, lines, fallback=engine._longfile_active_file_lines
    )
    engine._longfile_mutation_deltas.append(delta)
    # Do NOT clear the truncated-tail flag on a
    # synthetic-recovery dispatch (longfile salvage):
    # the synthetic Write / AppendFile is RECOVERING a truncated write, so the
    # tail is semantically still mid-content. The caller sets the flag BEFORE
    # dispatch so the in-``_dispatch_tool`` persist captures it, and
    # ``keep_truncated`` here prevents the clear. A subsequent GENUINE clean
    # (non-truncated) AppendFile does NOT pass ``keep_truncated`` → the flag
    # is cleared → the finalize path is reachable (anti-wedge).
    if not keep_truncated:
        engine._longfile_last_mutation_truncated = False
    engine._turns_since_last_byte_adding_mutation = 0
    setattr(engine, _TURN_ADDED_BYTES_ATTR, True)
    # Count EVERY successful AppendFile to the active path
    # (forced + voluntary) so the per-path append circuit-breaker can cap a
    # self-loop. A Write is the file's (re)creation, not an append, so it is not
    # counted. The path key is the bound active path (the off-path guard above
    # already returned for a side-file).
    if tool_call.name == "AppendFile" and engine._longfile_active_path is not None:
        active = engine._longfile_active_path
        engine._longfile_appends_per_path[active] = (
            engine._longfile_appends_per_path.get(active, 0) + 1
        )


def _resolve_file_lines(
    tool_call: ToolCall, reported_lines: int | None, *, fallback: int
) -> int:
    """Resolve the full-file line count after a byte-adding mutation (B-).

    AppendFile reports ``line_count_total`` (the cumulative full-file lines) —
    used directly when present. A Write OVERWRITES, so its full-file line count
    is the line count of its ``content`` argument (WriteOutput carries no line
    count). When neither is available, keep the prior ``fallback`` count rather
    than reporting a wrong number.
    """
    if reported_lines is not None:
        return reported_lines
    if tool_call.name == "Write":
        args = tool_call.arguments
        if isinstance(args, dict):
            content = args.get("content")
            if isinstance(content, str) and content:
                return content.count("\n") + 1
    return fallback


def note_truncated_mutation(engine: QueryEngine, path: str | None) -> None:
    """Mark the active file's tail as mid-content + record the truncation latch.

    Called when a chunkable content write to ``path`` is detected truncated at
    the output cap. Sets ``_longfile_last_mutation_truncated`` so the file is
    treated as NOT plausibly complete (the stall detector keeps driving appends)
    no matter how many bytes are on disk — a 14 KB truncated Write ends inside a
    string. Also binds the active path if not yet bound (the truncated write is
    salvaged to disk, so it IS the file being produced).

    Additionally records ``path`` in the STICKY ``_longfile_truncated_paths``
    latch (the load-bearing engage gate). The driver may force a tool for the
    active path ONLY if its path is in this set, so a file written WITHOUT a
    truncation never engages the driver. The latch is never cleared by a clean
    append (unlike the transient ``_longfile_last_mutation_truncated``).

    Multi-file handoff: the active path is first-byte-bound, so in a multi-file
    run a small CLEAN file written first would bind the active path and a later
    GENUINELY-truncated large file would be ignored as off-path → the driver
    would never engage on the real target. So a truncation on a REAL path
    ``path`` (re)binds the active convergence path to it WHENEVER the currently-
    bound active path has NOT itself truncated (it is a clean side-file, never a
    large-file-in-progress). Switching resets the path-keyed size/line/delta
    counters so the new artifact is tracked from scratch. This is zero-collateral:
    a rebind only ever happens on an actual output-cap truncation. ``path`` MUST
    be a real resolvable path — the caller never passes the ``"the target file"``
    display placeholder; a ``None`` path only flips the transient tail flag
    (no latch, no bind).
    """
    if not is_enabled(engine):
        return
    if not path:
        # Unresolvable target — only mark the tail mid-content; never latch or
        # bind a placeholder path (that would poison the active-path tracking).
        engine._longfile_last_mutation_truncated = True
        return
    current = engine._longfile_active_path
    if current is None:
        engine._longfile_active_path = path
    elif current != path and current not in engine._longfile_truncated_paths:
        # The bound active path is a clean side-file (never truncated); hand the
        # active convergence target over to the truncated large-file path and
        # reset its per-path size/line/delta tracking.
        engine._longfile_active_path = path
        engine._longfile_active_file_bytes = 0
        engine._longfile_active_file_lines = 0
        engine._longfile_mutation_deltas = []
    engine._longfile_last_mutation_truncated = True
    engine._longfile_truncated_paths.add(path)


def register_turn_byte_production(engine: QueryEngine, *, added_bytes: bool) -> None:
    """Advance the stall clock once per completed assistant turn .

    ``added_bytes`` is True iff this turn produced at least one byte-adding
    mutation (already reset the clock to 0 in :func:`observe_tool_result`). A
    turn that added NO bytes — prose-only OR a non-mutation tool call — advances
    the stall counter. No-op when the driver is disabled.
    """
    if not is_enabled(engine):
        return
    if not added_bytes:
        engine._turns_since_last_byte_adding_mutation += 1


def register_completed_turn(engine: QueryEngine) -> None:
    """Advance the stall clock for one completed assistant turn .

    Reads + clears the transient per-turn ``_longfile_turn_added_bytes`` flag
    that :func:`observe_tool_result` sets when a byte-adding mutation landed
    this turn, then advances the stall counter via
    :func:`register_turn_byte_production`. Call EXACTLY ONCE per completed
    assistant turn (both the tool-call-turn-end and the prose-turn seams). The
    flag is reset whether or not the driver is enabled, so a later enable does
    not see a stale flag. No-op (beyond the flag reset) when disabled.
    """
    added_bytes = bool(getattr(engine, _TURN_ADDED_BYTES_ATTR, False))
    setattr(engine, _TURN_ADDED_BYTES_ATTR, False)
    register_turn_byte_production(engine, added_bytes=added_bytes)


def active_file_bytes(engine: QueryEngine) -> int:
    """Bytes currently written for the active large-file artifact .

    Read straight from the engine's running size counter, which is fed by the
    tool RESULTS (``WriteOutput.bytes_written`` / ``AppendFileOutput.bytes_total``)
    in :func:`observe_tool_result`. Core has no direct workspace read, so the
    result payloads are the universal size signal — no disk dependency, no
    the host hook. Returns 0 until the first byte-adding mutation lands.
    """
    if engine._longfile_active_path is None:
        return 0
    return max(0, engine._longfile_active_file_bytes)


def _finalize_floor(rc: RuntimeConstants) -> int:
    """The minimum bytes required before a forced FinalizeFile is permitted.

 The HARD empty-finalize guard : ``max(1, floor * min_fraction)``.
 Never below 1 byte, so a 0-byte file can NEVER be force-finalized.
 """
    floor = rc.longfile_expected_floor_bytes * rc.longfile_min_finalize_fraction
    return max(1, int(floor))


def finalize_permitted(engine: QueryEngine) -> bool:
    """True iff the active file is large enough to permit a forced FinalizeFile.

    The empty-finalize guard. ``file_bytes >= max(1, floor*min_fraction)``.
    A below-floor or empty file is NEVER eligible (the validated edge).
    """
    return active_file_bytes(engine) >= _finalize_floor(engine.config.rc)


def _below_expected_floor(engine: QueryEngine) -> bool:
    """True iff the active file is below its expected-complete floor ."""
    return active_file_bytes(engine) < engine.config.rc.longfile_expected_floor_bytes


def plausibly_complete(engine: QueryEngine) -> bool:
    """A file is plausibly complete only when it is BOTH at/above the expected
 floor AND not sitting on a truncated (mid-content) tail .

 Below-floor OR last-truncated ⇒ NOT complete ⇒ the stall detector is allowed
 to drive more appends. This is the gate that separates "big" from
 "complete" — a big-but-truncated file is NOT done.
 """
    return (not _below_expected_floor(engine)) and (
        not engine._longfile_last_mutation_truncated
    )


def _plateau_reached(engine: QueryEngine) -> bool:
    """True iff the body has stopped growing across the recent mutations .

    Requires at least ``longfile_plateau_min_mutations`` successful byte-adding
    mutations AND the most recent delta below
    ``longfile_plateau_delta_fraction`` * the running-mean delta. Only meaningful
    once the file is plausibly complete (checked by the caller).
    """
    rc = engine.config.rc
    deltas = engine._longfile_mutation_deltas
    if len(deltas) < rc.longfile_plateau_min_mutations:
        return False
    mean_delta = sum(deltas) / len(deltas)
    if mean_delta <= 0:
        return False
    return deltas[-1] < rc.longfile_plateau_delta_fraction * mean_delta


def active_path_truncation_seen(engine: QueryEngine) -> bool:
    """Load-bearing engage gate.

    True iff the currently-bound active file path has had an output-cap
    truncation event recorded this run (``note_truncated_mutation`` added it to
    the sticky ``_longfile_truncated_paths`` latch). A file written WITHOUT a
    truncation (an ordinary small write, a dialog task) is NEVER in the set, so
    the driver is provably INERT on it — this is the zero-collateral guarantee
    that reverts the file_ops / long_dialog regression. Returns False when no
    active path is bound.
    """
    path = engine._longfile_active_path
    if path is None:
        return False
    return path in engine._longfile_truncated_paths


def append_breaker_tripped(engine: QueryEngine) -> bool:
    """True iff the active path has hit the per-path forced-append circuit-breaker
    (``longfile_max_appends_per_path``).

    The counter tallies ALL appends to the path (forced + voluntary); when it
    trips the DRIVER stops FORCING appends (the loop then takes its normal
    terminal path). This bounds the forced-driver contribution to a self-loop;
    it does NOT block the model's own voluntary AppendFile dispatch (the
    truncation gate removes the small-file voluntary flood at the root). Returns
    False when no active path is bound or the breaker is not yet reached.
    """
    path = engine._longfile_active_path
    if path is None:
        return False
    return (
        engine._longfile_appends_per_path.get(path, 0)
        >= engine.config.rc.longfile_max_appends_per_path
    )


def _stall_detected(engine: QueryEngine) -> bool:
    """True iff the run has stalled : no byte-adding mutation for
    ``longfile_stall_turns`` turns while a large-file artifact is in flight.

    A stall requires an active artifact with at least one byte on disk (a run
    that has not started any file is not a large-file run) — the driver never
    fires on an empty workspace.

    Additionally requires that the active path has had a truncation event
    (:func:`active_path_truncation_seen`). Below-floor + idle ALONE never counts
    as a stall: the driver only ever engages on a file the model actually tried
    to produce large and got CUT OFF. This is the gate that makes the driver
    inert on every non-truncated file.
    """
    if engine._longfile_active_path is None:
        return False
    if active_file_bytes(engine) <= 0:
        return False
    if not active_path_truncation_seen(engine):
        return False
    return (
        engine._turns_since_last_byte_adding_mutation
        >= engine.config.rc.longfile_stall_turns
    )


def decide_next_forced_tool(engine: QueryEngine) -> ForcedTool | None:
    """Decide the forced ``tool_choice`` for the NEXT assistant stream (-5.5).

    Mirrors the validated probe decision (``cases/probe_longfile_converge.py``):

    1. **Plateau finalize** — file plausibly complete AND the body has stopped
       growing → force FinalizeFile (subject to budget + the empty-finalize
       guard).
    2. **Terminal seal** — file comfortably past floor AND enough forced appends
       already fired (the model keeps dumping truncated chunks so the body never
       cleanly plateaus) → force FinalizeFile (subject to budget + guard). This
       is the terminal driver the plateau trigger cannot reach when forced
       appends keep truncating at the cap.
    3. **Stall** — no byte-adding mutation for ``longfile_stall_turns`` while a
       file is in flight:
         * NOT plausibly complete (below floor OR truncated tail) AND
           forced-append budget remaining → force AppendFile (drive content);
         * else (plausibly complete, or append budget exhausted) AND finalize is
           PERMITTED (the empty-finalize guard passes) AND finalize budget
           remaining → force FinalizeFile (the model has stopped producing; stop
           fighting it with endless appends).

    Returns the forced tool name, or None when the driver should not act
    (disabled / no stall / no budget / guard blocks finalize). The HARD
    empty-finalize guard (:func:`finalize_permitted`) gates EVERY FinalizeFile
    decision so a 0-byte / below-floor file is NEVER sealed — the one validated
    edge to protect.

    This function does NOT mutate engine state or charge budgets. The caller
    commits the decision via :func:`commit_forced_append` /
    :func:`commit_forced_finalize` only when it actually issues the forced tool.
    """
    if not is_enabled(engine):
        return None
    rc = engine.config.rc

    # Already sealed → the driver is done for this run (never re-force a
    # finalize / append on a file the model has already FinalizeFile'd).
    if engine._longfile_finalized:
        return None

    # No file in flight → never act (a strong model that self-completes, or a
    # run that produced no file, never reaches a stall on a tracked artifact).
    if engine._longfile_active_path is None or active_file_bytes(engine) <= 0:
        return None

    # Load-bearing zero-collateral gate. The driver may force a tool for the
    # active path ONLY if a truncation was actually detected on it. A
    # non-truncated below-floor file never passes this gate → the driver is
    # provably inert on it.
    if not active_path_truncation_seen(engine):
        return None

    finalize_budget = engine._longfile_forced_finalizes < rc.longfile_max_forced_finalizes
    append_budget = engine._longfile_forced_appends < rc.longfile_max_forced_appends

    # 1. Plateau finalize — the body stopped growing and the file is complete.
    if (
        finalize_budget
        and plausibly_complete(engine)
        and finalize_permitted(engine)
        and _plateau_reached(engine)
    ):
        return "FinalizeFile"

    # 2. Terminal seal — enough forced content; the body keeps truncating so it
    # never cleanly plateaus, but the file is comfortably past its floor.
    if (
        finalize_budget
        and finalize_permitted(engine)
        and engine._longfile_forced_appends >= rc.longfile_max_forced_appends
        and not plausibly_complete(engine)
    ):
        return "FinalizeFile"

    # Per-path forced-append circuit-breaker. Once the active path has hit
    # ``longfile_max_appends_per_path`` total appends (forced + voluntary) the
    # DRIVER stops FORCING appends: either seal it (if past floor) or stop
    # driving entirely. Bounds the forced-driver contribution to a self-loop on a
    # genuinely truncated path (voluntary appends are not dispatch-blocked here;
    # the truncation gate removes the small-file voluntary flood at the root).
    if append_breaker_tripped(engine):
        if finalize_budget and finalize_permitted(engine):
            return "FinalizeFile"
        return None

    # 3. Stall handling — only when the stall threshold is actually met.
    if not _stall_detected(engine):
        return None

    if not plausibly_complete(engine) and append_budget:
        return "AppendFile"

    # Plausibly complete (model is done producing, just won't seal) OR append
    # budget exhausted → force the seal, but ONLY if the empty-finalize guard
    # permits it. Below-floor + no append budget left → no action (let the loop
    # take its normal terminal path; we never seal an under-floor file).
    if finalize_budget and finalize_permitted(engine):
        return "FinalizeFile"
    return None


def terminal_seal_required(engine: QueryEngine) -> bool:
    """Is there an artifact worth sealing? — the "an artifact is open" question.

    The probe-confirmed gap (9-rep real-model run, core ``9404e7c8``): after a
    truncation the model often SELF-CONTINUES, producing AppendFile chunks with
    steady byte progress — so no stall ever registers, the stall-keyed driver
    correctly stays silent, and the run burns its entire turn budget ending
    UNSEALED (``state=failed`` at max-turns) with a giant, parseable, GOOD file
    on disk (71-126 KB observed). The stall/plateau forced FinalizeFile is
    proven; the missing piece is sealing at run-END.

    This predicate is True iff, at turn-budget exhaustion, the runtime should
    grant ONE extra forced-FinalizeFile turn for a complete-enough but unsealed
    truncation-gated file. It is PURE (no mutation, no budget charge) — the
    caller commits via :func:`commit_forced_finalize` only when it issues the
    forced seal. Gated identically to the stall-driver's finalize path so it
    introduces NO new tunable:

    * :func:`is_enabled` — the RC kill-switch; disabled ⇒ bit-identical no-op;
    * :func:`active_path_truncation_seen` — the zero-collateral gate;
      NEVER fires for a non-truncated file (an ordinary small write, a dialog
      task) — the seal is provably inert on it;
    * NOT already finalized (``_longfile_finalized``);
    * :func:`finalize_permitted` — the HARD empty/below-floor guard; an empty /
      below-floor file is NEVER sealed (the validated edge — never seal junk);
    * forced-finalize budget remaining
      (``_longfile_forced_finalizes < longfile_max_forced_finalizes``).

    Returns False on any failed gate. Pure — no mutation, no budget charge.

    Two callers ask it, and they ask the same question. The voluntary-finish
    seal dispatches a synthetic FinalizeFile when the model ends the run with a
    file still open. The run wind-down keeps FinalizeFile on the otherwise
    emptied tool surface for the same reason: a run cut short mid-file should
    not have that file thrown away because the stop removed the only tool that
    could close it.
    """
    if not is_enabled(engine):
        return False
    if not active_path_truncation_seen(engine):
        return False
    if engine._longfile_finalized:
        return False
    if not finalize_permitted(engine):
        return False
    return (
        engine._longfile_forced_finalizes
        < engine.config.rc.longfile_max_forced_finalizes
    )


def commit_forced_append(engine: QueryEngine) -> None:
    """Charge one forced AppendFile against the per-run budget .

    Increments the forced-append counter and resets the stall clock (a forced
    turn is given a fresh chance to produce). Snapshot-persisted via the engine
    counters; the caller persists the snapshot.
    """
    engine._longfile_forced_appends += 1
    # The reset-to-0 is safe at this point in the call stack:
    # ``register_completed_turn`` has ALREADY advanced the stall clock for the
    # just-completed turn earlier this iteration (in
    # ``_maybe_drive_longfile_convergence``, before the decision), so this reset
    # only grants the upcoming FORCED turn a fresh window — it never swallows an
    # unaccounted-for stall turn (B-MEDIUM-2; no behaviour change).
    engine._turns_since_last_byte_adding_mutation = 0


def commit_forced_finalize(engine: QueryEngine) -> None:
    """Charge one forced FinalizeFile against the per-run budget ."""
    engine._longfile_forced_finalizes += 1


def set_force_next_tool(engine: QueryEngine, tool_name: ForcedTool) -> None:
    """Record the forced ``tool_choice`` for the next assistant stream.

    Stored as a transient engine attribute consumed (and cleared) by the stream
    builder. Within-loop hint only — not snapshot-persisted (the persisted stall
    counters re-derive the decision on resume if a crash drops the hint).
    """
    setattr(engine, FORCE_NEXT_TOOL_ATTR, tool_name)


def peek_force_next_tool(engine: QueryEngine) -> str | None:
    """Read the pending forced ``tool_choice`` WITHOUT consuming it.

    The stream builder needs to inspect the hint to decide whether the
    forced tool is on the per-turn surface BEFORE popping it — a
    compacted / BM25-clipped surface may not include the tool (e.g.
    ``AppendFile`` / ``FinalizeFile`` are not in
    :attr:`RuntimeConstants.tool_surface_forced_pins` by default), in which
    case :func:`take_force_next_tool` must NOT consume the hint: the
    forcing is the active ingredient, and burning the budget on a stream
    the model never sees offered is the exact failure that drops the force
    silently while the continue message + ``commit_forced_*`` charge are
    already settled.
    """
    value = getattr(engine, FORCE_NEXT_TOOL_ATTR, None)
    return value if isinstance(value, str) else None


def take_force_next_tool(engine: QueryEngine) -> str | None:
    """Pop the pending forced ``tool_choice`` (consumed once).

    Callers that need to gate the pop on the tool being on the surface
    (the usual case in :func:`_drive_one_stream`) must use
    :func:`peek_force_next_tool` to inspect first, then call this only
    when the surface includes the tool — otherwise the hint is dropped
    while the model is never offered the forced tool. The unconditional
    pop here is preserved for back-compat with any caller that genuinely
    wants to consume regardless of the surface (there are none today).
    """
    value = getattr(engine, FORCE_NEXT_TOOL_ATTR, None)
    if value is not None:
        setattr(engine, FORCE_NEXT_TOOL_ATTR, None)
    return value if isinstance(value, str) else None


def build_continue_message(engine: QueryEngine) -> str:
    """Build the bilingual INCOMPLETE continue message + tail anchor .

 The tail anchor is read from the model's OWN most-recent ``Write``/
 ``AppendFile`` history entry for the active path via :func:`_active_file_tail`
core has NO direct workspace read, so the history scan is the universal
 source (Deviation #1; no disk read, an empty tail on a compacted history).
 ``file_lines`` is the REAL full-file line count tracked from the tool-result
 payloads (``AppendFileOutput.line_count_total`` / a Write's content lines),
 NOT a count of the 200-char tail. States the file's current bytes/lines
 (NO byte target — stating a target made a weak model pad a small file),
 FORBIDS stopping/declaring done, and NEVER says "safe on disk". EN first,
 RU second (Multilingual mandatory).
 """
    rc = engine.config.rc
    path = engine._longfile_active_path or "the target file"
    file_bytes = active_file_bytes(engine)
    tail = _active_file_tail(engine, rc.longfile_tail_anchor_chars)
    file_lines = engine._longfile_active_file_lines
    # The bundled default messages state the file's current bytes/lines but NO
    # byte target (stating a target made a weak model pad a legitimately small
    # file). ``expected_floor_bytes`` is KEPT in the format dict purely as a
    # TOLERATED key so an existing tenant DB override that still carries the old
    # ``{expected_floor_bytes}`` placeholder keeps formatting (an extra kwarg is
    # harmless; a missing one would raise).
    # wrap the dict so an UNKNOWN placeholder in a tenant override
    # (e.g. ``{bytes}`` typo for ``{file_bytes}``) does NOT raise KeyError
    # inside _maybe_drive_longfile_convergence; the typo is rendered literally
    # and logged. See ``_SafeFormatDict`` above.
    fields = _SafeFormatDict(
        path=path,
        file_bytes=file_bytes,
        file_lines=file_lines,
        expected_floor_bytes=rc.longfile_expected_floor_bytes,
        tail=tail,
    )
    en = rc.longfile_continue_message_en.format_map(fields)
    ru = rc.longfile_continue_message_ru.format_map(fields)
    return f"{en}\n\n{ru}"


def _active_file_tail(engine: QueryEngine, chars: int) -> str:
    """Tail of the active file: the last ``chars`` chars the model last wrote.

    Core has no direct workspace read, so the universal tail-anchor source is
    the model's OWN most-recent content-mutation call to the active path, found
    by walking ``engine.history`` backward for the last ``Write``/``AppendFile``
    ``ToolUseBlock`` whose resolved path matches the active artifact. Its
    ``content`` argument is exactly what was just appended to the file, so its
    tail is the correct "continue from here" anchor — it tells the model where
    it left off without core needing disk access. Returns '' when no such call
    is found (the message still carries bytes-written/expected + the INCOMPLETE
    directive). NEVER raises.

    A chunked write compacts the content body out of history
    (``query._apply_deferred_tool_history``), so the tail is read from the
    LATEST mutation that still carries a real body;
    a compacted-only history yields '' (acceptable — the byte count + directive
    remain).
    """
    path = engine._longfile_active_path
    if chars <= 0 or not path:
        return ""
    for message in reversed(engine.history):
        for block in reversed(message.content_blocks):
            tool_use = _tool_use_block(block)
            if tool_use is None:
                continue
            name, args = tool_use
            if name not in _BYTE_MUTATION_TOOLS:
                continue
            block_path = _resolve_path_from_args(args)
            # Require an EXPLICIT path match: a block whose path is unparseable
            # (``block_path is None``) must be SKIPPED, never wildcard-matched to
            # the active path — reading the tail from an unknown-path write would
            # hand the model the wrong continuation anchor (B-/ ).
            if block_path != path:
                continue
            content = args.get("content")
            if isinstance(content, str) and content:
                return content[-chars:]
    return ""


def _tool_use_block(block: Any) -> tuple[str, dict[str, Any]] | None:
    """Return ``(tool_name, arguments_dict)`` for a ToolUseBlock, else None.

    Tolerant of the block shape: a ``ToolUseBlock`` carries ``name`` plus the
    args as a JSON string (``arguments_json``) or a dict (``arguments``).
    """
    name = getattr(block, "name", None)
    if not isinstance(name, str):
        return None
    raw_args = getattr(block, "arguments", None)
    if isinstance(raw_args, dict):
        return name, raw_args
    raw_json = getattr(block, "arguments_json", None)
    if isinstance(raw_json, str):
        try:
            parsed = json.loads(raw_json)
        except (TypeError, ValueError):
            return None
        if isinstance(parsed, dict):
            return name, parsed
    return None


def _resolve_path_from_args(args: dict[str, Any]) -> str | None:
    """Resolve a file path from a tool-call args dict (``path``/``file_path``)."""
    for key in ("path", "file_path"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return None
