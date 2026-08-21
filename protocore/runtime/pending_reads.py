"""Forced read-back of the files a tool result declares.

A tool whose real output is FILES — a delegation that writes its findings to
disk, a job that renders an export — returns a short body naming the paths it
produced. Measured on a live stand, the caller then answers from that one-line
pointer: a leader handed two files holding 22-37 sourced findings answered with
zero citations, and in another shape relayed "the review has been saved to
<path>" as its entire final answer, which for a chat user is an empty reply.
The files were correct and complete; nobody opened them.

Asking for the read in the system prompt produces a tendency. This produces a
property, the same way :mod:`protocore.runtime.run_tool_preconditions` does:
any tool may set :data:`~protocore.contracts.types.PENDING_READS_METADATA_KEY`
on its result to a list of paths meaning "the caller must open these before it
relies on this". While one of those paths is unread the loop names the
workspace read tool in ``LLMRequest.extra['forced_tool_choice']``, which the
provider adapter translates into a native ``tool_choice`` — so the caller
cannot emit a final answer at all until it has read them. The moment the last
pending path is read the driver stops forcing and the entire surface is the
agent's again.

Nothing is ever taken away. The mechanism engages on a signal the caller
produced, narrows the agent only while it owes a read, and releases itself the
instant the debt is paid — no operator action, no prompt instruction, no
timeout on the happy path.

The metadata key is a general PROTOCOL, not a subagent feature: this module
reads a list of strings and knows nothing about who produced it, which is what
lets it live in core while the tool that populates it lives in the host
service (core never imports upward).

Bounded in three directions so it can never wedge a run:

* a path is forced at most until ``pending_reads_max_forced_attempts``
  CONSECUTIVE forced turns have failed to clear anything — a file that does not
  exist, or a model that reads something else every time, releases the gate
  rather than trapping the run;
* a released path is never forced again, so a tool that keeps re-declaring an
  unreadable file cannot re-engage on it;
* the pending set is capped at ``pending_reads_max_paths`` entries, so a tool
  returning a pathological list cannot grow the run snapshot without limit.

Every state the gate can be in leaves a ``DIAG pending_reads.*`` WARNING behind
— forced, not_forced, release_exhausted, cap_reached — because production keeps
WARNING and nothing below it. The one that matters most is ``not_forced``: a
turn the gate wanted and could not take charges nothing and changes nothing, so
on the wire it is identical to a run in which no tool ever declared a read-back,
and the two need opposite fixes.

This module is PURE: it reads/updates the engine's per-run pending-read state
(all snapshot-persisted on
:class:`~protocore.runtime.query_engine.QueryEngine`) and returns decisions;
the query loop performs the actual forced-tool wiring. It never touches IO, the
LLM, or module-level state. RC-gated by ``pending_reads_enabled`` — when
disabled every entry point is a no-op, so behaviour is bit-identical to a build
without the driver.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from protocore.contracts.types import PENDING_READS_METADATA_KEY
from protocore.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from protocore.contracts.types import ToolCall
    from protocore.runtime.query_engine import QueryEngine

_logger = get_logger(__name__)

# The tool the driver forces, and the tool whose successful result RELEASES a
# pending path. One name, because ``tool_choice`` names exactly one tool per
# request and the read-back obligation is satisfied by reading — a Grep or a
# List over the file is a different act with a different result shape, and a
# caller that greps a report has still not read it.
FORCED_TOOL_NAME: Final[str] = "Read"


def is_enabled(engine: QueryEngine) -> bool:
    """Return True iff the read-back gate is enabled for this engine."""
    return bool(engine.config.rc.pending_reads_enabled)


def _resolve_path_from_args(args: dict[str, Any]) -> str | None:
    """Resolve a file path from a tool-call args dict (``file_path``/``path``).

    Accepts both spellings because the read tool's canonical field is
    ``file_path`` with ``path`` as a validation alias, and the model reaches
    for either (mirrors the same helper in
    :mod:`protocore.runtime.longfile_convergence`).
    """
    for key in ("file_path", "path"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _normalise(path: str) -> str:
    """Normalise a path for comparison: POSIX separators, no ``./`` prefix.

    Deliberately NOT ``os.path.abspath`` — core has no workspace root and the
    paths on both sides of the comparison come from two different actors (the
    tool declares them, the model types them into its read call), so resolving
    against the runtime's own cwd would be meaningless.
    """
    text = path.strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/")


def _same_file(declared: str, read: str) -> bool:
    """True iff a read of ``read`` satisfies a pending declaration of ``declared``.

    Exact match after normalisation, or one is the other with a directory
    prefix in front (``/workspace/reports/audit.md`` reads as a satisfaction of
    a declared ``reports/audit.md``). The prefix must end at a path-component
    boundary, so a declared ``a/audit.md`` is NOT satisfied by reading
    ``b/audit.md`` — the loose "same basename" rule would call two different
    files the same one, which is exactly the confusion the gate exists to
    prevent.
    """
    left = _normalise(declared)
    right = _normalise(read)
    if not left or not right:
        return False
    if left == right:
        return True
    return left.endswith("/" + right) or right.endswith("/" + left)


def _already_read(engine: QueryEngine, path: str) -> bool:
    """True iff this run has already read ``path`` (or an equivalent form)."""
    return any(_same_file(path, seen) for seen in engine._pending_reads_satisfied)


def _abandoned(engine: QueryEngine, path: str) -> bool:
    """True iff the gate already gave up on ``path`` this run."""
    return any(_same_file(path, gone) for gone in engine._pending_reads_abandoned)


def _declared_paths(metadata: dict[str, Any] | None) -> tuple[str, ...]:
    """Extract the declared read-back paths from a tool result's metadata.

    Tolerant by construction: the value crosses a tool boundary, so anything
    that is not a list of non-empty strings is ignored rather than raised on. A
    malformed declaration must cost the run nothing — the alternative is a
    third-party tool being able to fail a run by returning the wrong shape.
    """
    if not metadata:
        return ()
    raw = metadata.get(PENDING_READS_METADATA_KEY)
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(
        item.strip() for item in raw if isinstance(item, str) and item.strip()
    )


def observe_tool_result(
    engine: QueryEngine,
    tool_call: ToolCall,
    metadata: dict[str, Any] | None,
    *,
    is_error: bool,
) -> None:
    """Fold one tool result into the pending read-back set.

    Called from the successful-dispatch path for EVERY tool result (both the
    serial and the parallel-batch path). No-op when the driver is disabled.
    Two independent effects, in this order:

    * **Release.** A successful read of a path clears every pending entry it
      satisfies and resets the consecutive-attempt counter, and is remembered
      for the rest of the run so a path read BEFORE a tool declared it is never
      forced at all. The release is recorded for any successful read, forced or
      voluntary — the contract is that the file was opened, not that the
      runtime coerced it.
    * **Engage.** A successful result carrying
      :data:`~protocore.contracts.types.PENDING_READS_METADATA_KEY` adds its
      paths to the pending set, skipping ones already read this run, ones the
      gate already gave up on, and anything past the
      ``pending_reads_max_paths`` cap. SEVERAL declarations in one turn
      accumulate — a fan-out of three delegations owes three sets of reads, so
      this is a set and not a slot.

    An ERROR result never engages the gate: a tool that failed may name files
    it never finished writing, and forcing reads of those spends the whole
    attempt budget on files that cannot exist. The release half still runs for
    an errored read (it clears nothing, since only a successful read counts).
    """
    if not is_enabled(engine):
        return
    if tool_call.name == FORCED_TOOL_NAME and not is_error:
        args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
        read_path = _resolve_path_from_args(args)
        if read_path:
            _record_read(engine, read_path)
    if is_error:
        return
    declared = _declared_paths(metadata)
    if declared:
        _record_declarations(engine, declared)


def _record_read(engine: QueryEngine, read_path: str) -> None:
    """Remember a successful read and clear whatever it satisfies."""
    engine._pending_reads_satisfied.add(_normalise(read_path))
    remaining = [
        pending
        for pending in engine._pending_read_paths
        if not _same_file(pending, read_path)
    ]
    if len(remaining) != len(engine._pending_read_paths):
        engine._pending_read_paths = remaining
        # A productive forced turn buys back the whole budget: what is bounded
        # is the run of CONSECUTIVE turns that cleared nothing, so a caller
        # working through five declared files is never starved by its own
        # progress (mirrors ``run_tool_preconditions.observe_tool_result``).
        engine._pending_reads_forced_attempts = 0


def _record_declarations(engine: QueryEngine, declared: tuple[str, ...]) -> None:
    """Add newly-declared paths to the pending set, subject to the cap."""
    cap = engine.config.rc.pending_reads_max_paths
    for path in declared:
        if len(engine._pending_read_paths) >= cap:
            _logger.warning(
                "DIAG pending_reads.cap_reached run=%s cap=%d dropped=%s",
                engine.config.run_id,
                cap,
                path,
            )
            return
        if _already_read(engine, path) or _abandoned(engine, path):
            continue
        if any(_same_file(existing, path) for existing in engine._pending_read_paths):
            continue
        engine._pending_read_paths.append(path)


def pending_paths(engine: QueryEngine) -> tuple[str, ...]:
    """The paths the caller still owes a read, in declaration order."""
    return tuple(engine._pending_read_paths)


def peek_forced_tool(engine: QueryEngine) -> str | None:
    """The tool to force on this stream, or None to force nothing.

    Non-mutating by design, exactly like
    :func:`longfile_convergence.peek_force_next_tool`: the stream builder has
    to know whether the read tool is on THIS turn's advertised surface before
    it commits to anything. A forced choice naming an unadvertised tool makes
    the provider reject the whole request, and a BM25-clipped surface can drop
    the read tool, so the caller peeks, checks the surface, and only then
    charges an attempt. The pending set is untouched either way, so a later
    stream whose surface does include the tool still forces it.

    Returns None once the attempt budget is spent; the caller releases the
    gate via :func:`release_exhausted` rather than forcing forever.
    """
    if not is_enabled(engine):
        return None
    if not engine._pending_read_paths:
        return None
    if _attempts_exhausted(engine):
        return None
    return FORCED_TOOL_NAME


def _attempts_exhausted(engine: QueryEngine) -> bool:
    """True when the current pending set has burnt its whole attempt budget."""
    return (
        engine._pending_reads_forced_attempts
        >= engine.config.rc.pending_reads_max_forced_attempts
    )


def charge_forced_attempt(engine: QueryEngine) -> None:
    """Spend one attempt on the pending set.

    Called ONLY when the read tool was actually named in the request's
    ``forced_tool_choice``. A turn where the tool could not be forced (it was
    missing from that turn's surface) is NOT charged — unlike a tool
    precondition, an unforceable read-back is not a promise broken to the
    caller, and charging for a turn the model was never offered the tool on
    would spend the budget on streams that could not possibly have paid it.

    The charge lands BEFORE the stream rather than after its result, because
    the turn that matters most for the bound is the one where the model ignores
    the forced choice and returns no call at all: there is no result to
    observe, and without a charge here the run would force forever.

    This is also the ONE point at which the gate provably engaged, so it is
    where the engagement is recorded. A mechanism that narrows the model's
    surface and leaves no trace can be neither confirmed nor refuted from a
    production log: whether it ran at all becomes a matter of inference, and a
    failure that has nothing to do with it reads just as well as one that does.
    """
    engine._pending_reads_forced_attempts += 1
    _logger.warning(
        "DIAG pending_reads.forced run=%s attempt=%d/%d pending=%d paths=%s",
        engine.config.run_id,
        engine._pending_reads_forced_attempts,
        engine.config.rc.pending_reads_max_forced_attempts,
        len(engine._pending_read_paths),
        ", ".join(engine._pending_read_paths),
    )


def note_not_forced(engine: QueryEngine, *, reason: str) -> None:
    """Record a turn on which the gate WANTED to force a read and could not.

    Called by the query loop when :func:`peek_forced_tool` named a tool but the
    stream could not carry it — the read tool was missing from this turn's
    advertised surface, or a strict terminal-only latch permits exactly one
    dispatch. Neither turn is charged (see :func:`charge_forced_attempt`), so
    without this line the run looks EXACTLY like one in which the gate never
    engaged at all: same absent ``forced_tool_choice``, same untouched counter,
    same silence. Those two states demand opposite fixes — one is a surface
    problem, the other a declaration problem — and telling them apart after the
    fact is the whole point of the line.

    Purely diagnostic: it reads state and writes a log record, and no caller
    branches on it.
    """
    if not is_enabled(engine):
        return
    _logger.warning(
        "DIAG pending_reads.not_forced run=%s reason=%s attempt=%d/%d "
        "pending=%d paths=%s",
        engine.config.run_id,
        reason,
        engine._pending_reads_forced_attempts,
        engine.config.rc.pending_reads_max_forced_attempts,
        len(engine._pending_read_paths),
        ", ".join(engine._pending_read_paths),
    )


def release_exhausted(engine: QueryEngine) -> tuple[str, ...]:
    """Give up on the pending set when its attempt budget is spent.

    Returns the abandoned paths (empty when there is nothing to release), so
    the caller can say so in the run's event stream. The paths are remembered
    as abandoned and never forced again, the counter is reset, and the agent
    gets its full freedom back immediately — a file that does not exist, or a
    model that will not read the one it was given, costs a bounded number of
    turns and then nothing at all.

    A LATER declaration of a DIFFERENT file still engages the gate normally:
    the budget bounds one unproductive streak, not the mechanism for the rest
    of the run.
    """
    if not is_enabled(engine):
        return ()
    if not engine._pending_read_paths or not _attempts_exhausted(engine):
        return ()
    abandoned = tuple(engine._pending_read_paths)
    engine._pending_reads_abandoned.update(_normalise(path) for path in abandoned)
    engine._pending_read_paths = []
    engine._pending_reads_forced_attempts = 0
    _logger.warning(
        "DIAG pending_reads.release_exhausted run=%s attempts=%d abandoned=%s",
        engine.config.run_id,
        engine.config.rc.pending_reads_max_forced_attempts,
        ", ".join(abandoned),
    )
    return abandoned
