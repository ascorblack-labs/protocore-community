"""Run-level tool preconditions — tools a run MUST call before it may answer.

A run may carry an ordered tuple of
:class:`~protocore.contracts.types.ToolPrecondition` entries on
:attr:`~protocore.runtime.query_engine.QueryEngineConfig.tool_preconditions`.
While an entry is outstanding the query loop names its tool in
``LLMRequest.extra['forced_tool_choice']``, which the provider adapter
translates into a native ``tool_choice``. Asking the model in the system prompt
to search before answering produces a tendency; this produces a property.

NOT :mod:`protocore.runtime.tool_preconditions`, despite the near-identical
name. That module is the per-tool dependency DAG
(:attr:`ToolDefinition.preconditions` — "FinalizeFile may not run before
AppendFile"), which BLOCKS a tool the model chose. This one is a RUN-level
obligation the caller states up front, and it FORCES a tool the model did not
choose. The two never interact.

Three facts shape everything here:

* ``tool_choice`` names exactly ONE tool per request, so entries can only be
  satisfied in SEQUENCE. Progress is therefore an index into the tuple, never a
  set of satisfied names — a repeated tool (``[A, B, A]``) has to be forced
  again the second time.
* A forced choice may only name a tool the request actually advertises; naming
  an unadvertised one makes the provider reject the whole request. The
  caller-facing layer validates the names against the run's resolved surface,
  but a per-turn surface can still be clipped, so the loop re-checks and this
  module counts "could not be forced" as a spent attempt rather than a silent
  skip.
* Counts are satisfied by SUCCESSFUL calls only — a tool that errored did not
  run. A voluntary successful call counts exactly as much as a forced one: the
  contract is that the tool ran, not that the runtime coerced it.

Attempts are bounded by ``rc.run_tool_precondition_max_attempts`` so a tool
that can never succeed cannot loop the run. On exhaustion the run FAILS naming
the tool and its last error — a caller who asked for a precondition and did not
get one has been lied to, so the run never quietly continues without it.

Pure: reads the config, reads/updates the engine's snapshot-persisted
precondition counters. No IO, no LLM, no module-level state. An empty
``tool_preconditions`` (the default) makes every entry point here a no-op.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from protocore.contracts.types import ToolCall, ToolPrecondition
    from protocore.runtime.query_engine import QueryEngine


def current_entry(engine: QueryEngine) -> ToolPrecondition | None:
    """Return the outstanding precondition entry, or ``None`` when there is none.

    ``None`` covers both "this run has no preconditions" and "every entry is
    satisfied". From the loop's point of view those are the same state: force
    nothing, the whole tool surface is the agent's again.
    """
    entries = engine.config.tool_preconditions
    index = engine._tool_precondition_index
    if index >= len(entries):
        return None
    return entries[index]


def outstanding_tool(engine: QueryEngine) -> str | None:
    """Return the tool name to force on this turn, or ``None`` to force nothing."""
    entry = current_entry(engine)
    return None if entry is None else entry.tool


def charge_attempt(engine: QueryEngine, *, error: str | None = None) -> None:
    """Spend one attempt on the current entry.

    Called once per assistant stream the current entry drives — whether its
    tool was forced onto the request or could not be (missing from that turn's
    advertised surface, in which case the caller passes ``error``). The charge
    lands BEFORE the stream rather than after its result, because the turn that
    matters most for the bound is the one where the model ignores the forced
    choice and returns no call at all: there is no result to observe, and
    without a charge here the run would force forever.

    A successful call resets the counter (see :func:`observe_tool_result`), so
    what is bounded is the run of CONSECUTIVE unproductive turns for one entry
    — an entry asking for several calls is never starved by its own successes.
    """
    if current_entry(engine) is None:
        return
    engine._tool_precondition_attempts += 1
    if error:
        engine._tool_precondition_last_error = _clip_error(engine, error)


def observe_tool_result(
    engine: QueryEngine,
    tool_call: ToolCall,
    content: str,
    *,
    is_error: bool,
) -> None:
    """Fold one tool result into precondition progress.

    Only results for the CURRENT entry's tool matter; whatever else the model
    called alongside it is neither progress nor failure. A successful call
    advances the entry and clears the attempt counter; an error result leaves
    progress untouched and is retained so an exhaustion failure can name it.
    """
    entry = current_entry(engine)
    if entry is None or tool_call.name != entry.tool:
        return
    if is_error:
        engine._tool_precondition_last_error = _clip_error(engine, content)
        return
    engine._tool_precondition_calls += 1
    engine._tool_precondition_attempts = 0
    engine._tool_precondition_last_error = None
    if engine._tool_precondition_calls >= entry.calls:
        # Entry satisfied — advance with a fresh budget. When this was the last
        # entry the index runs past the end of the tuple, and from then on
        # nothing is forced again for the rest of the run.
        engine._tool_precondition_index += 1
        engine._tool_precondition_calls = 0


def is_exhausted(engine: QueryEngine) -> bool:
    """True when the current entry has burnt its whole attempt budget."""
    if current_entry(engine) is None:
        return False
    return (
        engine._tool_precondition_attempts
        >= engine.config.rc.run_tool_precondition_max_attempts
    )


def failure_message(engine: QueryEngine) -> str:
    """Reason for an exhausted entry, naming the tool and its last error.

    Reads the CURRENT entry, so it is only meaningful while :func:`is_exhausted`
    holds.
    """
    entry = current_entry(engine)
    if entry is None:
        return "tool precondition unsatisfied"
    last_error = engine._tool_precondition_last_error
    detail = (
        f"last error: {last_error}"
        if last_error
        else "the tool was never called successfully and reported no error"
    )
    return (
        f"tool precondition {entry.tool!r} was not satisfied after "
        f"{engine._tool_precondition_attempts} forced attempt(s) "
        f"({engine._tool_precondition_calls} of {entry.calls} successful "
        f"call(s)); {detail}"
    )


def _clip_error(engine: QueryEngine, error: str) -> str:
    """Clip error text to the RC bound before it is retained on the engine.

    A tool result is unbounded, and this string travels on the run's terminal
    error frame and into every subsequent snapshot.
    """
    limit = engine.config.rc.run_tool_precondition_error_max_chars
    text = error.strip()
    return text if len(text) <= limit else text[:limit]
