# ruff: noqa: RUF001 — Bilingual RU+EN runtime nudge strings intentionally use Cyrillic characters.
"""``query`` — public entry that drives ONE turn of the agent loop.

md` (full ASCII sequence).

The function is **pure** w.r.t. global state — all mutations flow through
the injected :class:`QueryEngine` (history, state, compaction state). It
is the **single place in core that knows about** :class:`ProviderDelta`;
outside this function the loop deals in :class:`TurnEvent`.

Lifecycle (one invocation = one turn):

 1. Stop-check + state → RUNNING (PENDING → RUNNING transition).
 2. Compaction check + run if needed.
 3. UserPromptSubmit hook fire.
 4. Build context bundle (system prompt + tools + history).
 5. message_start event.
 6. Stream provider deltas → translate to TurnEvent.
 7. Collect tool_calls; dispatch each with hooks.
 8. Recurse (open new LLM stream w/ tool_results) until no tool calls.
 9. message_stop + state → COMPLETED (or AWAITING / CANCELLED / FAILED).
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final

from protocore.constants import MAX_DATA_NESTING_DEPTH
from protocore.contracts.hooks import HookActionKind
from protocore.contracts.llm import (
    LLMContextWindowExceeded,
    LLMError,
    LLMObservabilityContext,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMStreamEvent,
    LLMStreamIdleError,
    LLMTimeoutError,
    MaxOutputTokensExhausted,
    ProviderDelta,
    ProviderDeltaKind,
)
from protocore.contracts.skills import (
    ISkillStore,
    SkillBundle,
    SkillIndexEntry,
    SkillNotFoundError,
)
from protocore.contracts.tool_chunking import (
    CHUNKABLE_CONTENT_FIELD,
    CHUNKABLE_CONTENT_MUTATION_ALLOWLIST,
    is_chunkable_content_mutation,
)
from protocore.contracts.tools import (
    SUBAGENT_DISPATCH_GROUP_METADATA_KEY,
    SUBAGENT_DISPATCH_ORDER_METADATA_KEY,
    SUBAGENT_TREE_PERMIT_METADATA_KEY,
    ToolContext,
)
from protocore.contracts.types import (
    PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
    SYNTHETIC_RECOVERY_CIRCUIT_BREAKER,
    SYNTHETIC_RECOVERY_GUARANTEED_TERMINAL,
    SYNTHETIC_RECOVERY_LONGFILE_CONTINUE,
    SYNTHETIC_RECOVERY_LONGFILE_SALVAGE,
    SYNTHETIC_RECOVERY_LONGFILE_TERMINAL_SEAL,
    SYNTHETIC_RECOVERY_MAX_OUTPUT_CONTINUE,
    SYNTHETIC_RECOVERY_METADATA_KEY,
    SYNTHETIC_RECOVERY_POST_TOOL_EMPTY_NUDGE,
    SYNTHETIC_RECOVERY_PRE_DISPATCH_TERMINAL_VERIFY,
    SYNTHETIC_RECOVERY_PRE_TERMINAL_SELF_VERIFY,
    SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR,
    SYNTHETIC_RECOVERY_TERMINAL_REPAIR,
    SYNTHETIC_RECOVERY_TERMINAL_TOOL_NUDGE,
    SYNTHETIC_RECOVERY_THINKING_CONTINUE,
    SYNTHETIC_RECOVERY_TRUNCATION_CONTINUE,
    TERMINAL_TOOL_METADATA_KEY,
    TOOL_RESULT_CONSECUTIVE_CAP_ELIGIBLE_METADATA_KEY,
    ContentBlock,
    HookEvent,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.logging_utils import get_logger
from protocore.runtime import longfile_convergence as _longfile
from protocore.runtime import pending_reads as _pending_reads
from protocore.runtime import run_tool_preconditions as _preconditions
from protocore.runtime import soft_stop as _soft_stop
from protocore.runtime.answer_narration import leading_narration_span
from protocore.runtime.context.compaction import (
    CompactionExhaustedError,
    current_tool_batch_protect_index,
)
from protocore.runtime.context.manager import ContextBundle
from protocore.runtime.events import BlockVisibility, EventType, TurnEvent
from protocore.runtime.live_control import (
    QueuedPrompt,
    place_items,
    placed_as_user_message,
    restore_queued_prompts,
)
from protocore.runtime.llm.delta_bridge import (
    delta_to_turn_events,
    stream_events_to_provider_deltas,
)
from protocore.runtime.loop_guard import (
    canonical_tool_fingerprint,
    identical_tool_should_block,
    inspect_stream_repeat,
)
from protocore.runtime.loop_state import LoopState
from protocore.runtime.loop_strategies import select_strategy
from protocore.runtime.prompt_caching import apply_system_and_3
from protocore.runtime.result_eviction import evict_history_for_llm
from protocore.runtime.run_work_budget import (
    RunWorkLedger,
    resolve_run_work_ledger,
)
from protocore.runtime.skill_index import (
    derive_skill_index_budget_tokens,
    render_skills_catalog,
)
from protocore.runtime.subagent_budget import SubagentTreeBudget, SubagentTreePermit
from protocore.runtime.tool_dispatch import (
    DISPATCH_POST_TOOL_OUTPUT_MODIFIED_METADATA_KEY,
    DISPATCH_REPLAY_ERROR_KIND_METADATA_KEY,
    DISPATCH_REPLAY_ERROR_MESSAGE_METADATA_KEY,
    DISPATCH_STRUCTURED_ERROR_METADATA_KEY,
    HELPER_SUBAGENT_TREE_BUDGET_KEY,
    HELPER_SUBAGENT_TREE_PERMIT_KEY,
    STRUCTURED_ERROR_FINALIZATION_RECOMMENDED_KEY,
    STRUCTURED_ERROR_REASON_KEY,
    DispatchErrorKind,
    DispatchOutcome,
    ToolDispatcher,
    _annotate_tool_result_event,
    _append_soft_cap_warning_to_content,
    _merge_soft_cap_warning_metadata,
    _record_tool_call_soft_cap_warning,
)
from protocore.runtime.tool_permission import ToolPermissionGate

if TYPE_CHECKING:
    from protocore.contracts.hooks import HookResult
    from protocore.contracts.runtime_constants import RuntimeConstants
    from protocore.runtime.query_engine import QueryEngine


_logger = get_logger(__name__)


def _enter_soft_stop(engine: QueryEngine, *, cause: str) -> list[TurnEvent]:
    """Begin the run wind-down for ``cause``. The ONE entry point.

    Returns the events the caller must forward, or an empty list when the
    wind-down does not apply — it is disabled, or it is already running. An
    empty list is the caller's signal to take its own terminal path, which is
    exactly what a second bound reached DURING the wind-down should do: the run
    was already told to stop and is now out of the turns it was given to do it.

    Every bound routes here — the tool-call budget, the turn cap, the
    output-token budget, the deadline, an upstream that stopped answering.
    Before, each ended somewhere different: one appended a paragraph of advice
    to a tool result and let the agent keep working, one nudged then forged a
    synthetic answer, one granted a best-effort turn, one just stopped. The
    caller's remaining job after this returns is mechanical and the same
    everywhere: grant the wind-down its turns, persist, rebuild the context so
    the next stream sees the narrowed surface, and continue.
    """
    if not _soft_stop.is_enabled(engine):
        return []
    if _soft_stop.is_armed(engine):
        return []
    events = _soft_stop.enter(engine, cause_name=cause)
    if events:
        # The terminal-only guard shares this latch with the voluntary-finish
        # contract repair: it is what makes a blocked non-terminal dispatch
        # answer with an instruction to finalise rather than a bare denial.
        engine._terminal_only_active = True
    return events


def _soft_stop_turn_budget(engine: QueryEngine, assistant_message_idx: int) -> int:
    """Turns the wind-down gets, counted from where the run actually is."""
    return assistant_message_idx + engine.config.rc.soft_stop_max_turns


async def _emit_voluntary_completion(
    engine: QueryEngine,
) -> AsyncIterator[TurnEvent]:
    """Close a run the model finished of its own accord.

    Names the wind-down when one is running: ``stop_reason="soft_stop"`` rather
    than ``end_turn``, preceded by the ``soft_stop_finalized`` state change. The
    two are different facts and a consumer needs both — the run DID finish, with
    an answer, and it finished because it was made to.
    """
    finalized = _soft_stop.finalize(engine)
    if finalized is not None:
        yield finalized
    yield TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "stop_reason": (
                _soft_stop.STOP_REASON
                if _soft_stop.is_armed(engine)
                else "end_turn"
            ),
            "tokens_used": _tokens_used_payload(engine),
            "cache_hit_rate": engine.total_usage.this_turn_cache_hit_rate(),
        },
    )


def _tool_call_budget_reached(engine: QueryEngine) -> bool:
    """True once this run has dispatched its whole cumulative tool-call budget.

    Counted off the tool-call ledger's ordinal, which advances once per
    dispatched call in transcript order on both the serial and the parallel
    path, and is persisted — so a run re-driven on another pod does not get a
    fresh budget. The leader and each subagent run in separate engines with
    separate ledgers, so a subagent's internal calls are its own.
    """
    cap = (
        engine.config.rc.subagent_tool_call_soft_cap
        if engine.config.parent_run_id is not None
        else engine.config.rc.leader_tool_call_soft_cap
    )
    if cap <= 0:
        return False
    return engine._tool_call_ledger_seq >= cap


INTERNAL_ERROR_KIND: Final[str] = "internal_error"
"""Terminal ``kind`` for a failure that is this process, not the upstream.

Reported on :class:`~protocore.runtime.events.EventType.ERROR` and carried into
``runs.error_class`` by the executor. Every exception outside the
:class:`~protocore.contracts.llm.LLMError` family lands here — a parser bug, a
``RecursionError``, an ``AttributeError`` in the loop. Kept distinct from
``llm_provider_error`` because the two demand opposite responses: a provider
failure is a reason to try a different endpoint, and this is a reason to read a
traceback. Conflating them made the provider-failure metric count our own bugs,
and made the one record of a real crash say the upstream had failed.

``stop_reason`` is unaffected — an internal error still ends the run in
``error``.
"""


def _llm_history(engine: QueryEngine) -> tuple[list[Message], list[str]]:
    """History view for the next LLM request (eviction never mutates persist)."""
    from protocore.runtime.compact_checkpoint import apply_checkpoint

    view, evicted = evict_history_for_llm(
        engine.history,
        engine.config.rc,
        engine._pinned_tool_result_ids,
    )
    view = apply_checkpoint(view, getattr(engine, "compact_checkpoint", None))
    if engine.config.rc.tool_result_split_enabled:
        from protocore.contracts.types import ToolResultBlock
        from protocore.runtime.tool_result_split import split_result

        split_view: list[Message] = []
        for message in view:
            new_blocks: list[ContentBlock] = []
            changed = False
            for block in message.content_blocks:
                if isinstance(block, ToolResultBlock):
                    content, details = split_result(block.content, rc=engine.config.rc)
                    if content != block.content:
                        meta = dict(block.metadata)
                        if details:
                            meta["ui_details"] = details
                        new_blocks.append(
                            block.model_copy(
                                update={"content": content, "metadata": meta}
                            )
                        )
                        changed = True
                        continue
                new_blocks.append(block)
            split_view.append(
                message.model_copy(update={"content_blocks": new_blocks})
                if changed
                else message
            )
        view = split_view
    return view, evicted


def _maybe_run_settled_event(engine: QueryEngine) -> TurnEvent | None:
    if not engine.config.rc.run_settled_enabled or engine._run_settled_emitted:
        return None
    if engine.state is LoopState.COMPACTING:
        return None
    engine._run_settled_emitted = True
    return TurnEvent(
        type=EventType.RUN_SETTLED,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "phase_was": engine.state.value,
            "will_continue": False,
            # The two facts about the run that neither ``status`` nor
            # ``stop_reason`` carries: was the user answered, and what did the
            # run actually call. Both ride the settle event because it fires
            # once per run, unlike ``message_stop`` which fires per round.
            "has_final_answer": engine.has_final_answer,
            "tool_calls": engine.tool_call_ledger,
            "tool_calls_truncated": engine.tool_call_ledger_truncated,
        },
    )


def _apply_stream_loop_guard(
    engine: QueryEngine,
    stream_result: Any,
) -> TurnEvent | None:
    new_text, new_reason, hit = inspect_stream_repeat(
        stream_result.text_buffer,
        stream_result.reasoning_buffer,
        engine.config.rc,
    )
    if hit is None:
        return None
    stream_result._text_fragments = [new_text] if new_text else []
    stream_result._reasoning_fragments = [new_reason] if new_reason else []
    engine._loop_guard_nudge_count += 1
    return TurnEvent(
        type=EventType.LOOP_GUARD_FIRED,
        run_id=engine.config.run_id,
        payload={
            "kind": hit.kind,
            "nudge_index": engine._loop_guard_nudge_count,
            "stripped_chars": hit.stripped_chars,
        },
    )


def _block_identical_tools(
    engine: QueryEngine,
    tool_calls: list[ToolCall],
) -> tuple[list[ToolCall], list[TurnEvent]]:
    """Split tool calls into executable vs blocked-identical, emitting results."""
    executable: list[ToolCall] = []
    events: list[TurnEvent] = []
    rc = engine.config.rc
    for call in tool_calls:
        fingerprint = canonical_tool_fingerprint(call.name, call.arguments)
        blocked = identical_tool_should_block(
            fingerprint, engine._identical_tool_counts, rc
        )
        engine._identical_tool_counts[fingerprint] = (
            engine._identical_tool_counts.get(fingerprint, 0) + 1
        )
        if not blocked:
            executable.append(call)
            continue
        engine.history.append(
            Message(
                role=MessageRole.tool,
                content_blocks=[
                    ToolResultBlock(
                        tool_call_id=call.id,
                        content=(
                            "identical tool call blocked by loop guard; "
                            "change arguments or stop repeating this call"
                        ),
                        is_error=True,
                        metadata={"loop_guard": "identical_tool"},
                    )
                ],
            )
        )
        events.append(
            TurnEvent(
                type=EventType.TOOL_RESULT,
                run_id=engine.config.run_id,
                payload={
                    "tool_call_id": call.id,
                    "content": "identical tool call blocked by loop guard",
                    "is_error": True,
                },
            )
        )
        events.append(
            TurnEvent(
                type=EventType.LOOP_GUARD_FIRED,
                run_id=engine.config.run_id,
                payload={
                    "kind": "identical_tool",
                    "nudge_index": engine._loop_guard_nudge_count,
                    "tool_call_id": call.id,
                    "tool_name": call.name,
                },
            )
        )
    return executable, events


def _rule_project_roots(engine: QueryEngine) -> tuple[str, ...]:
    raw = getattr(engine, "rule_project_roots", ()) or ()
    return tuple(str(item) for item in raw)


def _rule_file_tuples(engine: QueryEngine) -> list[tuple[str, str]]:
    files = getattr(engine, "rule_files", None)
    if not files:
        return []
    out: list[tuple[str, str]] = []
    for item in files:
        out.append((str(item[0]), str(item[1])))
    return out


async def _populate_discovered_rules(engine: QueryEngine) -> None:
    """Assign ``engine.discovered_rules`` from the workspace/project tree once."""
    if not engine.config.rc.rules_discovery_enabled:
        return
    if engine.discovered_rules:
        return
    from protocore.runtime.rules_activation import discover_agents_md

    files = _rule_file_tuples(engine)
    if not files:
        loader = getattr(engine, "list_rule_files", None)
        if callable(loader):
            loaded = loader()
            if inspect.isawaitable(loaded):
                loaded = await loaded
            files = [(str(item[0]), str(item[1])) for item in (loaded or [])]
    if not files:
        return
    engine.discovered_rules = discover_agents_md(
        files, engine.config.rc, project_roots=_rule_project_roots(engine)
    )


def _activate_rules_from_tool(engine: QueryEngine, tool_call: ToolCall) -> None:
    from protocore.runtime.rules_activation import activate_on_filesystem_touch, discover_agents_md

    if tool_call.name not in {"Read", "Write", "Edit", "Glob", "Grep"}:
        return
    args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
    path = str(args.get("path") or args.get("file_path") or args.get("pattern") or "")
    if not path:
        return
    if engine.config.rc.rules_discovery_enabled and not engine.discovered_rules:
        files = _rule_file_tuples(engine)
        if files:
            engine.discovered_rules = discover_agents_md(
                files, engine.config.rc, project_roots=_rule_project_roots(engine)
            )
    before = list(engine.active_rule_paths)
    engine.active_rule_paths = activate_on_filesystem_touch(
        touched_path=path,
        tool_name=tool_call.name,
        discovered=list(engine.discovered_rules),
        already_active=engine.active_rule_paths,
        rc=engine.config.rc,
    )
    if engine.active_rule_paths != before:
        engine._pending_rules_activated = [
            path for path in engine.active_rule_paths if path not in before
        ]


async def _maybe_place_background_wakes(engine: QueryEngine) -> list[str]:
    """Refresh the session pool and inject one batched wake turn if needed."""
    pool = getattr(engine, "background_pool", None)
    if pool is None or not engine.config.rc.background_tasks_enabled:
        return []
    for task in list(pool.list(engine.config.session_id)):
        await pool.refresh(task.id)
    ids: list[str] = list(pool.drain_wakes(engine.config.session_id))
    if not ids:
        return []
    statuses = []
    for task_id in ids:
        item = pool.get(task_id)
        if item is not None:
            statuses.append(f"{item.id} {item.status}")
    text = "background tasks finished: " + ", ".join(statuses)
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=text)],
        )
    )
    persister = getattr(engine, "persist_session_history", None)
    if callable(persister):
        persister(engine)
    return ids


async def _reload_live_control(engine: QueryEngine) -> None:
    """Pull mid-run steer / follow-up / model from the bound live store."""
    reloader = getattr(engine, "reload_live_control", None)
    if reloader is None:
        return
    await reloader(engine)


async def _persist_live_control(engine: QueryEngine) -> None:
    persister = getattr(engine, "persist_live_control", None)
    if persister is None:
        return
    await persister(engine)


def _inject_queue_into_history(
    engine: QueryEngine,
    *,
    kind: str,
    raw_queue: list[dict[str, Any]],
    mode: str,
) -> TurnEvent | None:
    if not engine.config.rc.steer_follow_up_enabled or not raw_queue:
        return None
    items = [QueuedPrompt.from_dict(raw) for raw in raw_queue]
    placed, remaining = place_items(items, mode)  # type: ignore[arg-type]
    message = placed_as_user_message(placed)
    if message is None:
        return None
    engine.history.append(message)
    remaining_dicts = [item.to_dict() for item in remaining]
    if kind == "steer":
        engine._steer_queue = remaining_dicts
    else:
        engine._follow_up_queue = remaining_dicts
    return TurnEvent(
        type=EventType.QUEUE_UPDATE,
        run_id=engine.config.run_id,
        payload={
            "placed": [item.id for item in placed],
            "kind": kind,
            "remaining": len(remaining_dicts),
        },
    )


def _inject_steer_into_history(engine: QueryEngine) -> TurnEvent | None:
    return _inject_queue_into_history(
        engine,
        kind="steer",
        raw_queue=engine._steer_queue,
        mode=engine.config.rc.steer_default_mode,
    )


def _inject_follow_up_into_history(engine: QueryEngine) -> TurnEvent | None:
    return _inject_queue_into_history(
        engine,
        kind="follow_up",
        raw_queue=engine._follow_up_queue,
        mode=engine.config.rc.follow_up_default_mode,
    )


def _pin_keep_flag(engine: QueryEngine, tool_call: ToolCall) -> None:
    parsed = tool_call.arguments
    if isinstance(parsed, dict) and parsed.get("keep") is True:
        engine.pin_tool_result(tool_call.id)


# Heartbeat observability for the PRE-DISPATCH terminal-verify gate. The gate
# ran silently on the no-veto path, which hid mis-diagnoses. The heartbeat
# logs an UNCONDITIONAL line on every gate application so "did the gate run,
# and with what verdict?" is answerable from the executor log alone.
#
# ``observed`` (the size of the per-run observed-state collection a trigger
# may compare cited refs against) lives on the OPAQUE helper bag core already
# forwards (``engine._helpers`` — see the ``protocore.helpers`` threading in
# ``_dispatch_tool``). Reading it is a plain mapping lookup, NOT a host
# import, so core stays import-boundary-pure (guard:
# ``tests/test_core_import_boundary.py``). The helper key is a runtime-facing
# convention; if its producer ever drifts, the heartbeat degrades gracefully to
# ``observed=-1``
# (sentinel for "not cheaply observable from core") — it never lies and never
# raises. ``cited`` is computed purely from the un-submitted ``ToolCall``
# arguments (the exact input the trigger reads: canonical ``refs`` slot, legacy
# ``sources`` alias), so it is always exact.
_OBSERVED_REF_LEDGER_HELPER_KEY: Final[str] = "terminal_answer_observed_refs"
# The per-run content-read collection key (the set of paths whose BODY the run
# actually read — the real, harness-returned paths the run banked). Read off
# the SAME opaque helper bag core already
# forwards (``engine._helpers``); a plain mapping lookup, NOT a host
# import, so core stays import-boundary-pure. The guaranteed-terminal backstop
# cites these REAL banked paths (never fabricated); if the key ever drifts the
# backstop degrades gracefully to a message-only answer.
_CONTENT_READ_LEDGER_HELPER_KEY: Final[str] = "terminal_answer_content_read_refs"
_TERMINAL_ANSWER_REFS_KEY: Final[str] = "refs"
_TERMINAL_ANSWER_REFS_LEGACY_ALIAS: Final[str] = "sources"
_HEARTBEAT_OBSERVED_UNAVAILABLE: Final[int] = -1


def _observability_context(
    engine: QueryEngine,
    *,
    call_purpose: str,
    call_category: str,
) -> LLMObservabilityContext:
    return LLMObservabilityContext(
        tenant_id=engine.config.tenant_id,
        run_id=engine.config.run_id,
        parent_run_id=engine.config.parent_run_id,
        session_id=engine.config.session_id,
        agent_id=engine.config.subagent_id,
        call_purpose=call_purpose,
        call_category=call_category,
    )


def _provider_call_category(engine: QueryEngine) -> str:
    if engine.config.parent_run_id is not None:
        return "subagent_call"
    return "agent_call"


def _tool_surface_advertised_payload(
    engine: QueryEngine,
    context: ContextBundle,
) -> dict[str, object]:
    """Diagnostic payload for the exact tool list sent to the LLM provider."""

    policy = engine.effective_tool_policy
    toolsearch_pins = frozenset(engine.context_manager.pinned_tool_names())
    forced_pins = frozenset(policy.forced_pinned)
    configured_pins = frozenset(policy.pinned) - toolsearch_pins
    tool_names = [tool.name for tool in context.tools]
    tools: list[dict[str, object]] = []
    for tool in context.tools:
        sources: list[str] = []
        if tool.name in toolsearch_pins:
            sources.append("toolsearch_pin")
        if tool.name in configured_pins:
            sources.append("configured_pin")
        if tool.name in forced_pins:
            sources.append("forced_pin")
        if not sources:
            sources.append("retrieved_or_visible")
        tools.append(
            {
                "name": tool.name,
                "description": tool.description,
                "sources": sources,
            }
        )
    return {
        "turn_id": engine.turn_id(),
        "tool_count": len(tool_names),
        "tool_names": tool_names,
        "toolsearch_pinned_tool_names": sorted(toolsearch_pins),
        "configured_pinned_tool_names": sorted(configured_pins),
        "forced_pinned_tool_names": sorted(forced_pins),
        "retrieval_top_k": engine.config.rc.tool_retrieval_top_k,
        "tools": tools,
    }


def query(engine: QueryEngine) -> AsyncIterator[TurnEvent]:
    """Drive one already-prepared turn through the public delivery boundary.

    ``QueryEngine.run`` owns initial-message admission — the input message, the
    turn number, the run clock and the turn-start snapshot — and then consumes
    the same private generator directly.  This lower-level public iterator
    leaves that admission to its caller, but it must never become an alternate
    route around verification-gated reader delivery, and it must never become
    one around the turn boundary either: the per-turn counters and latches,
    which a caller cannot reach because they are private to the engine, this
    entry puts back itself via
    :meth:`~protocore.runtime.query_engine.QueryEngine._reset_per_turn_state`.
    Without that, a turn opened here on an engine nudged in an earlier turn
    began already finalising and deleted its own answer as post-answer
    narration.

    That is the reset state only, and the sentence is deliberately not the
    general rule "everything private belongs to the entry".  Two private
    obligations of ``run`` are still skipped here, and a caller driving this
    entry inherits them:

    * ``_current_turn_task`` is never bound, so :meth:`QueryEngine.stop` has
      no handle to cancel through and only its cooperative flag fires;
    * no turn-start or turn-end snapshot is persisted, so a turn driven here
      has no cross-pod resume point.

    Both are reachable work rather than accepted design; they are named so the
    next reader does not take the reset guarantee for a wider one.

    Deliberately NOT an async generator.  ``run`` is one, so its reset lands on
    the first ``__anext__`` rather than at the call — harmless there because
    every live caller iterates immediately, but a caveat that stops being
    harmless once it holds at more than one entry.  Returning the generator
    instead of being one makes the reset here happen when ``query`` is called,
    so there is still exactly one place where a built-but-not-yet-iterated turn
    carries last turn's state.
    """
    engine._reset_per_turn_state()
    return _projected_turn_events(engine)


async def _projected_turn_events(engine: QueryEngine) -> AsyncIterator[TurnEvent]:
    """The body of :func:`query`, split out so the reset above stays eager."""
    async for event in _query_raw(engine):
        for projected in engine._project_public_turn_event(event):
            yield projected


async def _query_raw(engine: QueryEngine) -> AsyncIterator[TurnEvent]:
    """Drive one full turn of ``engine``. Yields :class:`TurnEvent` envelopes.

    See module docstring for the lifecycle. The function is a Python async
    generator — each ``yield`` is a stop-check checkpoint.
    """
    # ── 1. Stop check ────────────────────────────────────────────────
    if engine.stop_requested:
        # a run resumed with stop already requested may carry a
        # dangling tool_use from the interrupted turn in its rehydrated
        # history; pair it before the cancel terminal so the persisted
        # snapshot stays pairing-valid.
        _synthesize_missing_tool_results(
            engine.history,
            error_content=engine.config.rc.tool_result_interrupted_placeholder,
        )
        restored = restore_queued_prompts(engine)
        await _persist_live_control(engine)
        from protocore.runtime.correctness_bind import commit_usage

        abort_evt = commit_usage(
            engine,
            kind="abort",
            input_tokens=0,
            output_tokens=0,
            success=False,
        )
        if abort_evt is not None:
            yield abort_evt
        yield _emit_state_change(
            engine,
            engine.state,
            LoopState.CANCELLED,
            reason="stop_before_start",
        )
        engine.transition_to(LoopState.CANCELLED)
        yield TurnEvent(
            type=EventType.MESSAGE_STOP,
            run_id=engine.config.run_id,
            payload={
                "turn_id": engine.turn_id(),
                "stop_reason": StopReason.cancelled.value,
                "restored_queue_text": "\n\n".join(restored),
            },
        )
        return

    # PENDING → RUNNING transition
    if engine.state is LoopState.PENDING:
        engine.transition_to(LoopState.RUNNING)
        yield TurnEvent(
            type=EventType.STATE_CHANGED,
            run_id=engine.config.run_id,
            payload={"from": LoopState.PENDING.value, "to": LoopState.RUNNING.value},
        )

    # Per-turn block index reset
    engine.reset_block_idx()
    engine.total_usage.reset_turn()
    persister = getattr(engine, "persist_session_history", None)
    if callable(persister):
        persister(engine)
    await _populate_discovered_rules(engine)
    from protocore.runtime.intent import resume_open_intents

    if engine.config.rc.intent_settlement_enabled and engine.open_intents:
        engine.open_intents = resume_open_intents(list(engine.open_intents))
        from protocore.runtime.correctness_bind import mark_intent_recovery, persist_correctness

        for item in engine.open_intents:
            if item.status == "interrupted":
                yield TurnEvent(
                    type=EventType.INTENT_COMMITTED,
                    run_id=engine.config.run_id,
                    payload={"status": "interrupted", "operation_id": item.operation_id},
                )
                for rec_evt in mark_intent_recovery(engine, item):
                    yield rec_evt
        persist_correctness(engine)

    from protocore.runtime.correctness_bind import fire_typed_hook

    before_run, before_run_evt = fire_typed_hook(engine, "before_run", {"run_id": engine.config.run_id})
    if before_run_evt is not None:
        yield before_run_evt
    if before_run.decision == "deny":
        yield TurnEvent(
            type=EventType.MESSAGE_STOP,
            run_id=engine.config.run_id,
            payload={"stop_reason": "hook_denied", "hook": "before_run"},
        )
        return

    latest = engine.latest_user_message
    if (
        engine.config.rc.compaction_manual_enabled
        and latest is not None
        and latest.text.lstrip().startswith("/compact")
    ):
        from protocore.runtime.compact_checkpoint import build_checkpoint

        instructions = latest.text.lstrip()[len("/compact") :].strip()
        ckpt = build_checkpoint(
            engine.history,
            keep_recent_turns=engine.config.rc.compaction_keep_recent_turns,
            instructions=instructions,
            reason="manual",
            enabled=True,
            tracked_tool_names=engine.config.rc.compaction_tracked_tool_names,
        )
        if ckpt is not None:
            engine.compact_checkpoint = ckpt
            persister = getattr(engine, "persist_session_history", None)
            if callable(persister):
                persister(engine)
            from protocore.runtime.correctness_bind import commit_usage

            usage_evt = commit_usage(
                engine,
                kind="compaction",
                input_tokens=0,
                output_tokens=0,
                success=True,
            )
            if usage_evt is not None:
                yield usage_evt
            yield TurnEvent(
                type=EventType.COMPACT_CHECKPOINT,
                run_id=engine.config.run_id,
                payload=ckpt.to_dict(),
            )

    # ── 2. Compaction check ──────────────────────────────────────────
    # When history already exceeds the emergency cliff
    # (``model_context_window * compaction_emergency_ratio``) at turn-start,
    # run a proactive ``force_compaction`` (both tiers, unconditional) rather
    # than the routine gated pass, so the wire payload is aggressively shrunk
    # before the first stream. RC-gated kill-switch
    # (``compaction_emergency_proactive_enabled``, default on).
    _emergency_turn_start = (
        engine.config.rc.compaction_emergency_proactive_enabled
        and engine.needs_emergency_compaction()
    )
    if _emergency_turn_start or engine.needs_compaction():
        async for evt in _run_compaction(
            engine,
            force=_emergency_turn_start,
            reason="proactive_emergency" if _emergency_turn_start else "routine",
        ):
            yield evt
        # If compaction transitioned to FAILED, surface the terminal
        # message_stop now and bail.
        if engine.state is LoopState.FAILED:
            # a compaction-exhausted FAILED terminal can persist a
            # history whose last assistant turn (or a turn compaction kept)
            # carries a tool_use with no result; pair it before the snapshot.
            _synthesize_missing_tool_results(
                engine.history,
                error_content=engine.config.rc.tool_result_interrupted_placeholder,
            )
            yield TurnEvent(
                type=EventType.MESSAGE_STOP,
                run_id=engine.config.run_id,
                payload={
                    "turn_id": engine.turn_id(),
                    "stop_reason": StopReason.error.value,
                },
            )
            return

    # ── 3. UserPromptSubmit hook ─────────────────────────────────────
    hook_result = await _safe_hook_invoke(
        engine,
        HookEvent.user_prompt_submit,
        {
            "run_id": engine.config.run_id,
            "tenant_id": engine.config.tenant_id,
            "message": engine.latest_user_message.model_dump(mode="json") if engine.latest_user_message else None,
        },
    )
    yield TurnEvent(
        type=EventType.HOOK_FIRED,
        run_id=engine.config.run_id,
        payload={
            "hook_event": HookEvent.user_prompt_submit.value,
            "outcome": "deny" if hook_result.action == HookActionKind.DENY else "success",
        },
    )
    if hook_result.action == HookActionKind.DENY:
        # a resumed history may already carry a dangling tool_use
        # when a UserPromptSubmit hook denies the turn; pair it before the
        # FAILED snapshot so a later resume is wire-valid.
        _synthesize_missing_tool_results(
            engine.history,
            error_content=engine.config.rc.tool_result_interrupted_placeholder,
        )
        from_state = engine.state
        engine.transition_to(LoopState.FAILED)
        yield _emit_state_change(
            engine,
            from_state,
            LoopState.FAILED,
            reason=hook_result.reason or "hook_denied",
        )
        yield TurnEvent(
            type=EventType.ERROR,
            run_id=engine.config.run_id,
            payload={
                "kind": "hook_denied",
                "message": hook_result.reason or "blocked by policy",
            },
        )
        yield TurnEvent(
            type=EventType.MESSAGE_STOP,
            run_id=engine.config.run_id,
            payload={
                "turn_id": engine.turn_id(),
                "stop_reason": StopReason.error.value,
            },
        )
        return

    # ── 4. Build context bundle ──────────────────────────────────────
    # ``effective_tool_policy`` applies the RC core tool-surface floor so
    # the six core file tools survive the BM25 RU clip on the live engine path.
    tool_defs = list(
        engine.tools.compute_effective_surface(
            tenant_id=engine.config.tenant_id,
            policy=engine.effective_tool_policy,
            query=engine.latest_user_message.text if engine.latest_user_message else "",
            top_k=engine.config.rc.tool_retrieval_top_k,
        )
    )

    # Skill catalog: the account's enabled skills (plus project pins) rendered
    # into a stable ``<system-reminder>`` block, built ONCE per run (cached on
    # the engine) and placed in the static system-prompt prefix so it is
    # byte-identical across turns + the inner agent loop + every recovery
    # rebuild (keeps the prompt cache; no redundant per-turn store round-trip).
    skill_catalog_block = await _ensure_run_skill_catalog(engine)
    if engine.config.rc.rules_discovery_enabled and engine.active_rule_paths:
        from protocore.runtime.rules_activation import bodies_for_prompt

        rule_bodies = bodies_for_prompt(
            list(engine.discovered_rules),
            engine.active_rule_paths,
            engine.config.rc,
        )
        if rule_bodies:
            skill_catalog_block = (skill_catalog_block + "\n" if skill_catalog_block else "") + "\n".join(rule_bodies)
    # Loaded bundles: rebuilt EVERY turn from THIS turn's user message —
    # trigger-style `<command-name>NAME</command-name>` references load the
    # full skill body as Layer 3. Per-turn, NOT cached on the engine.
    if engine.skills is not None and engine.latest_user_message is not None:
        engine._skill_loaded_bundles = await _load_triggered_skill_bodies(
            engine, engine.skills, engine.latest_user_message.text
        )
    else:
        engine._skill_loaded_bundles = []

    llm_history, evicted_ids = _llm_history(engine)
    context = engine.context_manager.build_context(
        history=llm_history,
        tools=tool_defs,
        system_prompt_sections=engine.config.system_prompt_sections,
        skill_index_block=skill_catalog_block,
        skills_loaded=engine._skill_loaded_bundles,
    )
    if evicted_ids:
        yield TurnEvent(
            type=EventType.TOOL_RESULT_EVICTED,
            run_id=engine.config.run_id,
            payload={"tool_call_ids": evicted_ids},
        )

    # #1/#4 — open the FIRST assistant-message wire round HERE, after every
    # pre-loop terminal (``stop_before_start`` L~254, compaction-failed L~314,
    # hook-denied L~358 — all keep the LEGACY ``turn-{run}-{turn_count}`` id with
    # ``_wire_round_seq == 0``) and BEFORE the Deep loop-strategy 4b step below.
    # The Deep ``REASONING_STEP`` (``loop_strategies._reasoning_step_payload``)
    # reads ``engine.turn_id()``; the chat reducer keys the SGR plan placeholder
    # by that id and later MERGES the first ``message_start`` into it (reducer
    # ``reasoning_step`` → ``message_start``), so the plan frame and round 1's
    # ``message_start`` MUST share one id. ``begin_wire_round()`` advances the
    # round to 1 (so every frame of round 1 reads the suffixed id) and restarts
    # ``block_idx`` at 0. Subsequent rounds advance inside the assistant-message
    # loop (guarded to skip round 1). No pre-loop terminal emits a content block,
    # so the earlier ``reset_block_idx()`` + this restart are equivalent there.
    engine.begin_wire_round()

    # ── 4b. Loop-strategy pre-action step. The
    # SINGLE branch point on ``run_mode``: Direct contributes nothing (today's
    # auto-tool loop, byte-unchanged); Deep runs the stand-validated SGR step
    # (a forced ``plan`` tool with native CoT bounded by ``reasoning_effort``)
    # → emits exactly one ``REASONING_STEP`` event → records the plan into
    # history. The shared assistant-message loop below then drives the real
    # action with the FULL surface — dispatch / pairing / repair / loop
    # detection / terminal gate all stay shared. ──────────────────────────
    strategy = select_strategy(engine.config.run_mode)
    history_len_before_plan = len(engine.history)
    async for evt in strategy.prepare_turn(engine, context):
        yield evt
    if len(engine.history) != history_len_before_plan:
        # Deep recorded a planning turn — rebuild the context bundle so the
        # action stream sees it. The surface + skill blocks are unchanged
        # (same user message), so only the (now longer) history is refreshed.
        rebuilt_history, _ = _llm_history(engine)
        context = engine.context_manager.build_context(
            history=rebuilt_history,
            tools=tool_defs,
            system_prompt_sections=engine.config.system_prompt_sections,
            skill_index_block=skill_catalog_block,
            skills_loaded=engine._skill_loaded_bundles,
        )

    # ── 5-9. Stream one assistant message (recursive on tool_use) ────
    async for evt in _stream_one_assistant_message(engine, context):
        yield evt
        # NOTE: do NOT short-circuit here on ``stop_requested`` or
        # ``engine.is_terminal``. The inner generator is responsible for
        # draining its own pending events (e.g. an LLM-error path that
        # yields ``state_changed → FAILED`` followed by ``error`` and
        # ``message_stop``). Short-circuiting mid-drain would swallow
        # those events and leave the wire-format invariant broken.

    # If the inner generator has driven us to a terminal state already
    # (COMPLETED / FAILED / CANCELLED), there is nothing more to emit;
    # the engine.run() finally-block will persist the terminal snapshot.
    # Terminal states have empty outgoing edges so a stop-after-terminal
    # MUST NOT attempt another transition.
    if engine.is_terminal:
        settled = _maybe_run_settled_event(engine)
        if settled is not None:
            yield settled
        return

    # Stop requested but engine is not terminal — synthesise a
    # message_stop(cancelled) cleanly (RUNNING / AWAITING / COMPACTING
    # all allow → CANCELLED).
    if engine.stop_requested:
        # before the cancel terminal, pair any already-emitted
        # tool_use that never received a result (stop landed after the
        # tool_use block closed but before/while the dispatch loop ran).
        # The engine.run() finally-block persists this snapshot; a resume on
        # another pod must not replay a dangling tool_use into a 400.
        _synthesize_missing_tool_results(
            engine.history,
            error_content=engine.config.rc.tool_result_interrupted_placeholder,
        )
        restored = restore_queued_prompts(engine)
        await _persist_live_control(engine)
        from_state = engine.state
        engine.transition_to(LoopState.CANCELLED)
        yield _emit_state_change(
            engine,
            from_state,
            LoopState.CANCELLED,
            reason="stop_requested",
        )
        yield TurnEvent(
            type=EventType.MESSAGE_STOP,
            run_id=engine.config.run_id,
            payload={
                "turn_id": engine.turn_id(),
                "stop_reason": StopReason.cancelled.value,
                "restored_queue_text": "\n\n".join(restored),
            },
        )
        return

    if engine.state is LoopState.AWAITING:
        return

    settled = _maybe_run_settled_event(engine)
    if settled is not None:
        yield settled

    # Final snapshot is persisted by ``engine.run()`` finally-block.


# ----------------------------------------------------------------------
# Helpers — compaction + state + streaming
# ----------------------------------------------------------------------


async def _run_compaction(
    engine: QueryEngine,
    *,
    force: bool = False,
    reason: str = "routine",
    protect_tail_from_index: int | None = None,
) -> AsyncIterator[TurnEvent]:
    """Drive one compaction attempt; emit started/completed events.

    ``force`` selects ``force_compaction`` (both tiers, unconditional) over
    the routine ``run_compaction`` — used by the proactive emergency-cliff
    branch at turn-start and per-iteration. ``reason`` is surfaced in the
    started/completed event payloads for telemetry (``"routine"`` /
    ``"proactive_emergency"`` / ``"proactive_per_iteration"`` /
    ``"proactive_per_iteration_emergency"``).

    ``protect_tail_from_index`` is set ONLY by the per-iteration gate to the
    index of the current assistant ``tool_use`` turn. Everything from that
    index to the end of history (the just-executed tool batch, any size) is
    exempted from compaction this iteration on top of the keep window, so a
    >keep parallel batch's fresh, unconsumed results are never
    blobbed/summarised before the next assistant stream consumes them. The
    turn-start gate and reactive-413 path pass ``None`` (no in-flight batch).
    """
    from protocore.runtime.context.budgets import derive_budgets
    from protocore.runtime.context.manager import estimate_history_tokens

    from_state = engine.state
    engine.transition_to(LoopState.COMPACTING)
    yield _emit_state_change(
        engine,
        from_state,
        LoopState.COMPACTING,
        reason=f"compaction_triggered:{reason}",
    )

    # ``tokens_before`` MUST reflect the actual current token count in the
    # history (the value that crossed the trigger threshold). The
    # ``trigger_threshold`` field MUST surface the compaction-trigger
    # boundary (= model_context_window * compaction_trigger_ratio), NOT
    # the bare model_context_window. Previously both fields conflated to
    # the bare window which made the telemetry incoherent.
    rc = engine.context_manager._rc
    budgets = derive_budgets(rc)
    tokens_before_value = estimate_history_tokens(engine.history, rc)
    yield TurnEvent(
        type=EventType.COMPACTION_STARTED,
        run_id=engine.config.run_id,
        payload={
            "reason": reason,
            "tokens_before": tokens_before_value,
            "trigger_threshold": budgets.compaction_trigger_tokens,
            "emergency_threshold": budgets.compaction_emergency_tokens,
            "history_messages": len(engine.history),
            "holds_settled": bool(engine.config.rc.run_settled_enabled),
        },
    )

    compaction_call = (
        engine.context_manager.force_compaction
        if force
        else engine.context_manager.run_compaction
    )
    from protocore.runtime.correctness_bind import fire_typed_hook

    _before_compact, before_compact_evt = fire_typed_hook(
        engine, "before_compact", {"reason": reason}
    )
    if before_compact_evt is not None:
        yield before_compact_evt
    try:
        attempt = await compaction_call(
            history=engine.history,
            compaction_state=engine.compaction_state,
            tenant_id=engine.config.tenant_id,
            model_name=engine.config.model_name,
            observability=_observability_context(
                engine,
                call_purpose="structured",
                call_category="compaction",
            ),
            protect_tail_from_index=protect_tail_from_index,
        )
    except CompactionExhaustedError as exc:
        compacting_from = engine.state
        engine.transition_to(LoopState.FAILED)
        yield _emit_state_change(
            engine,
            compacting_from,
            LoopState.FAILED,
            reason=str(exc),
        )
        yield TurnEvent(
            type=EventType.ERROR,
            run_id=engine.config.run_id,
            payload={"kind": "compaction_exhausted", "message": str(exc)},
        )
        return

    yield TurnEvent(
        type=EventType.COMPACTION_COMPLETED,
        run_id=engine.config.run_id,
        payload={
            "reason": reason,
            "tokens_before": attempt.tokens_before,
            "tokens_after": attempt.tokens_after,
            "tier1_freed": attempt.tier1.tokens_freed if attempt.tier1 else 0,
            "tier2_summarised": attempt.tier2.turns_summarised if attempt.tier2 else 0,
            "blob_refs_created": (list(attempt.tier1.blob_refs_created) if attempt.tier1 else []),
        },
    )
    _after_compact, after_compact_evt = fire_typed_hook(
        engine, "after_compact", {"reason": reason}
    )
    if after_compact_evt is not None:
        yield after_compact_evt
    from protocore.runtime.correctness_bind import commit_usage

    compact_usage = commit_usage(
        engine,
        kind="compaction",
        input_tokens=int(getattr(attempt, "tokens_before", 0) or 0),
        output_tokens=int(getattr(attempt, "tokens_after", 0) or 0),
        success=True,
    )
    if compact_usage is not None:
        yield compact_usage

    # The last real prompt measurement now describes a pre-compaction history
    # that no longer exists. Clear it so the gate does not re-fire on a stale
    # high-water mark: the freshly-shrunk history is re-measured by the cheap
    # estimate until the next LLM call reports a new ground-truth prompt size.
    engine.last_observed_prompt_tokens = 0

    # Snapshot after compaction completion
    # — executor pod crash between compaction and the next LLM call would
    # lose the freed-up history otherwise.
    await engine._persist_snapshot()

    compacting_from = engine.state
    engine.transition_to(LoopState.RUNNING)
    yield _emit_state_change(
        engine,
        compacting_from,
        LoopState.RUNNING,
        reason="compaction_completed",
    )


def _emit_state_change(
    engine: QueryEngine,
    from_state: LoopState,
    to_state: LoopState,
    *,
    reason: str,
) -> TurnEvent:
    """Build a ``state_changed`` :class:`TurnEvent`.

    Caller passes the explicit ``from_state``/``to_state`` pair so that
    payload accuracy is independent of when ``engine.transition_to`` is
    invoked relative to event emission. (Previously the helper read
    ``engine.state`` directly which made ``from`` self-referential when the
    transition had already been applied.)
    """
    return TurnEvent(
        type=EventType.STATE_CHANGED,
        run_id=engine.config.run_id,
        payload={
            "from": from_state.value,
            "to": to_state.value,
            "reason": reason,
        },
    )


async def _emit_dispatch_cancel_teardown(
    engine: QueryEngine,
) -> AsyncIterator[TurnEvent]:
    """route a tool-dispatch turn to the CANCELLED terminal.

    Shared teardown for every ``engine.stop_requested`` checkpoint in the
    dispatch prelude/loop of :func:`_stream_one_assistant_message` (cancel
    before the dispatch loop, cancel inside the awaited hook predicate, cancel
    between two tool calls in a serial OR parallel batch). The guarantee is
    "no NEW tool dispatch once a stop is observed": tools already dispatched
    before the stop keep their real results; any already-emitted-but-
    undispatched ``tool_use`` (the pending assistant tool_use blocks were
    appended to history BEFORE the dispatch loop) is paired here via
    :func:`_synthesize_missing_tool_results` (synthetic ``is_error`` results,
    idempotent — a call already paired is skipped) so the persisted snapshot
    stays pairing-valid for a resume on another pod. Yields the
    ``state_changed`` + terminal ``message_stop(cancelled)`` envelopes; the
    caller MUST ``return`` immediately after draining it. The outer ``query()``
    finally-guard then sees ``engine.is_terminal`` and emits nothing further.
    """
    _synthesize_missing_tool_results(
        engine.history,
        error_content=engine.config.rc.tool_result_interrupted_placeholder,
    )
    from_state = engine.state
    engine.transition_to(LoopState.CANCELLED)
    yield _emit_state_change(
        engine,
        from_state,
        LoopState.CANCELLED,
        reason="stop_requested",
    )
    yield TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "stop_reason": StopReason.cancelled.value,
        },
    )


#: Failure classes a DIFFERENT provider could plausibly serve. Each names a
#: property of the row that failed — its quota, its capacity, its key, its
#: balance, its catalogue — and never a property of the request, so the next row
#: gets a real chance rather than reproducing the same failure at a second
#: vendor's expense.
#:
#: Everything absent from this set keeps the recovery it already has, and the
#: omissions are the load-bearing part. A prompt that overflows the window is
#: fixed by compaction, which advancing would skip; an oversized body is fixed
#: by compressing it; a malformed request is fixed by the structured-output
#: ladder; an invalid reasoning payload is CAUSED by a model swap, so swapping
#: again compounds it; a policy refusal re-routed to a vendor that might accept
#: the content is a compliance decision and not a default. An unclassified
#: failure is excluded on principle — advancing on an error nobody has
#: characterised turns one broken run into N provider calls and N bills.
#:
#: Compared as plain strings against the verdict the adapter layer attaches to
#: the raised error. The taxonomy itself lives above core and core must not
#: import it; a string comparison keeps the boundary intact and still fails
#: closed, since a reason spelled differently simply is not in the set.
_PROVIDER_ADVANCE_REASONS: frozenset[str] = frozenset(
    {
        "auth",
        "auth_permanent",
        "billing",
        "model_not_found",
        "overloaded",
        "rate_limit",
        "server_error",
        "timeout",
    }
)


def _classified_failure_reason(exc: BaseException) -> str:
    """The adapter's verdict on ``exc``, or ``""`` when it carries none.

    Read duck-typed. The provider adapters pin their classification onto the
    exception they raise; core reads it without naming the type, which is what
    lets the decision below stay in core beside the loop that acts on it while
    the taxonomy stays where the vendors are.
    """
    classified = getattr(exc, "classified", None)
    if classified is None:
        return ""
    reason = getattr(classified, "reason", None)
    if reason is None:
        return ""
    return str(getattr(reason, "value", reason))


def _reason_permits_provider_advance(exc: BaseException) -> bool:
    """May this failure be answered by moving to the next provider?

    :class:`LLMRateLimitError` and :class:`LLMTimeoutError` are their own
    answer — a provider adapter raises those two types only for a quota it hit
    or a stream that stopped producing, both of which belong to the endpoint and
    not to the request. They also arrive unclassified from a caller that raised
    them directly, and treating the type as the verdict keeps that working.

    Everything else is a :class:`LLMProviderError`, which is the adapters'
    catch-all: a 503 and a policy refusal reach this branch as the same Python
    type. Only the attached verdict separates them, so an unclassified one does
    not advance.
    """
    if isinstance(exc, LLMRateLimitError | LLMTimeoutError):
        return True
    return _classified_failure_reason(exc) in _PROVIDER_ADVANCE_REASONS


async def _advance_provider_chain(
    engine: QueryEngine,
    exc: BaseException,
    *,
    kind: str,
) -> str:
    """Move ``engine`` onto the next provider. Returns its name, or ``""``.

    ``""`` — the run stays where it is and the caller falls through to its
    existing recovery — covers every way the step is unavailable: no chain
    configured, a failure class this list must not answer, the advance budget
    spent, or no rung left. Collapsing them is deliberate: the caller's next
    branch is the same in all four cases, and the reason each one happened is
    already on the record via the chain's own accounting.

    The whole provider moves. Rebinding only ``config.model_name`` would leave
    the previous vendor's endpoint and key in place, holding a name that
    endpoint does not serve.
    """
    chain = engine.provider_chain
    if chain is None:
        return ""
    if not _reason_permits_provider_advance(exc):
        return ""
    if engine._provider_chain_advances >= engine.config.rc.llm_provider_chain_max_advances:
        return ""
    if not await chain.advance(reason=kind):
        return ""
    engine._provider_chain_advances += 1
    engine.llm = chain.current()
    engine.config = replace(engine.config, model_name=chain.current_model_name())
    return chain.current_model_name()


def _persist_partial_attempt_to_history(
    engine: QueryEngine,
    stream_result: _StreamAttemptResult,
) -> None:
    """Persist the partial text + completed tool calls from a failed stream
    attempt to ``engine.history`` so the durable snapshot matches what the
    SSE consumers already saw live.

    On the fallback-model retry, the ``LLMStreamIdleError`` backstop, and
    the ``LLMProviderError`` backstop / terminal paths, the failed
    attempt's deltas were forwarded to the wire before the exception
    was raised. The normal completion path appends the assistant turn
    to history at the end of the inner stream loop, but a failure exits
    the inner loop BEFORE that line — so the partial is silently
    dropped from ``engine.history``. The user sees it live; a reload
    does not. This helper closes the gap by appending the partial
    assistant turn (text + completed ``tool_calls`` +
    ``reasoning_buffer``) to history exactly the way the normal
    completion path does.

    No-op when ``stream_result`` carries no text, tool calls, or reasoning —
    preserves the pre-existing behaviour for an empty pre-fail attempt
    (e.g. provider error on the very first byte). Mirrors the
    normal-completion guard in :func:`_stream_one_assistant_message`
    (the ``if assistant_blocks:`` check that gates the history
    append).
    """
    assistant_blocks: list[ContentBlock] = []
    if stream_result.text_buffer:
        assistant_blocks.extend(
            _split_answer_text_blocks(
                stream_result.text_buffer, stream_result.narration_prefix_chars
            )
        )
    for tc in stream_result.tool_calls:
        assistant_blocks.append(
            ToolUseBlock(
                tool_call_id=tc.id,
                name=tc.name,
                arguments_json=json.dumps(tc.arguments, ensure_ascii=False),
            )
        )
    if not assistant_blocks and not stream_result.reasoning_buffer:
        return
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=assistant_blocks,
            reasoning_content=stream_result.reasoning_buffer or None,
            metadata={PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY: True},
        )
    )


def _content_block_visibility(
    kind: ProviderDeltaKind,
    *,
    tool_interleaved: bool,
    narration_prefix: bool = False,
) -> BlockVisibility:
    """Where this content block belongs in the reader's prose stream.

    A forward-only stream cannot see its own future. At the moment a text block
    OPENS we genuinely do not know whether it will turn out to be the answer or
    narration, because that is decided by whether a tool call follows — and no
    signal available at open time predicts it. Shape does not: a person asking
    about a config format gets an answer that looks exactly like a model
    thinking aloud about one, so any content heuristic eats real answers.

    So this function does not guess. It reports facts that are already settled,
    and the caller evaluates them twice — once when the block opens, once when
    it closes:

    ``tool_interleaved``
        A tool call that CONTINUES the run has already started inside this same
        assistant message. That is proof by construction, not inference: a
        non-terminal tool call must have its result fed back, so this message is
        followed by at least one more, and no text sharing a message with such a
        call can be the run's final prose. Text here is narration between tool
        calls.

    ``narration_prefix``
        This block holds a leading run of process narration that was CUT OFF
        the answer following it — the one place the runtime does read text (see
        :mod:`protocore.runtime.answer_narration` for why it is narrow enough
        to be safe, and why it applies only to a run that delegated). It is a
        settled fact by the time it reaches here: the caller has already
        measured the run, proved a substantial answer follows it, and put that
        answer in a block of its own. Nothing was removed — the two blocks
        concatenate to the text the model wrote.

    At open time ``tool_interleaved`` is only True for text that comes AFTER
    the tool call it is interleaved with. The commoner shape — narration, THEN
    the tool call — is unprovable at open and provable the instant the tool
    starts, which is also the instant the block closes. The caller therefore
    emits the settled value on ``content_block_stop``; both flags only ever move
    from False to True, so the settled value is never weaker than the one
    already sent.

    Reasoning is unconditionally ``COLLAPSED``: it is the model's thinking by
    definition and never the reply, which is also how the durable transcript
    projects it.

    Where the signal is absent or ambiguous the answer is ``PUBLIC``. The two
    errors are not symmetric — a missed ``COLLAPSED`` on narration shows the
    user one bubble too many, while a false ``COLLAPSED`` on the answer takes
    the reply away entirely — so ambiguity resolves to the visible side every
    time.
    """

    if kind is ProviderDeltaKind.thinking:
        return BlockVisibility.COLLAPSED
    if tool_interleaved:
        return BlockVisibility.COLLAPSED
    if narration_prefix:
        return BlockVisibility.COLLAPSED
    return BlockVisibility.PUBLIC


def _answer_narration_split_active(engine: QueryEngine) -> bool:
    """Whether this run's assistant text may be split at its leading narration.

    Two facts, both about the RUN rather than the text. The tenant switch, and
    whether the run has actually handed work to a subagent — the narration is a
    delegating leader's habit, measured at 11-12 of 12 delegating runs against a
    minority without delegation, so a run that dispatched no subtask is left
    exactly as it was.
    """

    rc = engine.config.rc
    return (
        rc.delegated_answer_narration_split_enabled
        and rc.delegated_answer_narration_scan_chars > 0
        and engine._run_delegated
    )


def _settled_narration_split(
    engine: QueryEngine,
    text: str,
    *,
    complete: bool,
) -> tuple[str, str] | None:
    """Split ``text`` into (leading narration, answer), once that is knowable.

    ``None`` means "not yet" and can only be returned for a block still being
    streamed: either the run of narration sentences may still grow, or it has
    settled but not enough answer has arrived to clear
    ``delegated_answer_narration_min_answer_chars``. The caller keeps buffering.
    Both waits are bounded by the scan ceiling plus that floor, so the wire is
    never held for more than a fixed number of characters.

    A ``("", text)`` result is the ordinary one: no narration, or not enough
    answer behind it to be worth splitting. Nothing about the text changes —
    the caller emits it as the single block it always was.
    """

    rc = engine.config.rc
    span = leading_narration_span(
        text,
        scan_chars=rc.delegated_answer_narration_scan_chars,
        complete=complete,
    )
    if not span.settled:
        return None
    if not span.found:
        return ("", text)
    if len(text) - span.length < rc.delegated_answer_narration_min_answer_chars:
        # An answer that is narration and little else stays whole and visible.
        # Collapsing it would hand the reader an empty bubble, which is a worse
        # failure than the narration this exists to hide.
        return ("", text) if complete else None
    return (text[: span.length], text[span.length :])


def _text_delta_events(
    engine: QueryEngine,
    text: str,
    *,
    block_idx: int,
) -> Iterator[TurnEvent]:
    """Put held-back text on the wire, through the ordinary delta mapping.

    The buffered head is released as one delta rather than replayed as the many
    the provider sent. A reader appends deltas in order, so the rendered block
    is identical; the only visible difference is that the first characters of a
    split block arrive together instead of one token at a time.
    """

    yield from delta_to_turn_events(
        ProviderDelta(kind=ProviderDeltaKind.text, content=text),
        run_id=engine.config.run_id,
        turn_id=engine.turn_id(),
        block_idx=block_idx,
    )


def _split_answer_text_blocks(text: str, narration_prefix_chars: int) -> list[ContentBlock]:
    """Build the durable text block(s) for one assistant message's prose.

    Mirrors, exactly, the split the live stream already made — it is handed the
    stream's own cut point rather than re-deciding from the finished text, so a
    message cannot render one way live and another way after a reload. The two
    blocks concatenate to ``text``; ``Message.text`` joins every
    :class:`TextBlock`, so the model's view of its own turn is unchanged.
    """

    if 0 < narration_prefix_chars < len(text):
        return [
            TextBlock(
                text=text[:narration_prefix_chars],
                visibility=BlockVisibility.COLLAPSED,
            ),
            TextBlock(text=text[narration_prefix_chars:]),
        ]
    return [TextBlock(text=text)]


def _tool_call_continues_run(engine: QueryEngine, tool_name: str | None) -> bool:
    """True iff this tool call guarantees the run has another turn to come.

    Read by the block-visibility signal, which needs "a tool call happened" to
    mean "this message cannot hold the final prose". That inference holds for
    every ordinary tool — the result must go back to the model — but NOT for the
    run's terminal tool. Under the background terminal gate the model writes its
    answer and calls ``Finalize`` in the SAME message, and the gate call is
    stripped from every reader-facing view, so treating it like any other tool
    would collapse exactly the text the user came for and would contradict the
    durable transcript, which shows that message as prose and nothing else.

    An unnamed call is treated as terminal (i.e. not proof) whenever the run
    HAS a terminal tool: some providers withhold the name until the arguments
    finish streaming, and by then the text block is long closed. Losing a
    collapse on those providers is the cheap error. With no terminal tool
    configured no call can be the terminal one, so an unnamed call is proof
    like any other.
    """

    expected_terminal = engine.config.expected_terminal_tool
    if expected_terminal is None:
        return True
    return tool_name is not None and tool_name != expected_terminal


async def _stream_one_assistant_message(
    engine: QueryEngine,
    context: ContextBundle,
) -> AsyncIterator[TurnEvent]:
    """Drive the assistant-message tool-dispatch loop for one user input.

    Iterates: open LLM stream → emit deltas → dispatch tool_calls →
    re-open LLM stream with tool_results → … until either (a) the model
    emits ``finish`` with no tool_calls, (b) approval is pending, (c) the
    cap on assistant messages (``max_turns_per_run``) is hit, or (d) the
    LLM raises.

    The loop body was previously implemented as recursion; it now uses an
    explicit ``while``-loop with a depth counter so that endless tool
    calls cannot exceed Python's recursion limit before the
    ``max_turns_per_run`` cap fires. Each iteration counts as one
    assistant message.
    """
    assistant_message_idx = 0
    max_messages = engine.config.rc.max_turns_per_run
    current_context = context
    previous_tool_results_ready_at: float | None = None
    terminal_nudge_used = False
    # When the forced backstop is armed from a typed provider/stream error,
    # the original exception + its terminal ``kind`` are stashed here. If the
    # best-effort forced turn (or any later turn) reaches a no-answer
    # completion path, the stored error is surfaced as the terminal LLM error
    # rather than silently completing with no answer. Cleared implicitly once
    # the terminal answer is produced (the success completion paths do not
    # consult it).
    stored_stream_error: tuple[BaseException, str] | None = None

    while True:
        await _reload_live_control(engine)
        wake_ids = await _maybe_place_background_wakes(engine)
        if wake_ids:
            yield TurnEvent(
                type=EventType.BACKGROUND_WAKE,
                run_id=engine.config.run_id,
                payload={"task_ids": wake_ids},
            )
        # Cumulative tool-call budget. The bound that fires first on a run that
        # is working rather than spiralling, and the one that used to do
        # nothing: it appended a paragraph of English asking the agent to wrap
        # up, said in the same breath that tools still ran, and the agent made
        # another eighteen calls. Reaching it now starts the wind-down, so the
        # NEXT turn has no tools to make a nineteenth call with.
        if _tool_call_budget_reached(engine):
            _budget_windup = _enter_soft_stop(
                engine, cause=_soft_stop.CAUSE_TOOL_CALL_BUDGET
            )
            if _budget_windup:
                for _evt in _budget_windup:
                    yield _evt
                max_messages = _soft_stop_turn_budget(engine, assistant_message_idx)
                await engine._persist_snapshot()
                current_context = await _rebuild_context_for_recovery(engine)
                continue

        # Cumulative OUTPUT-token budget guard. A spiral that re-emits a large
        # truncated tool call burns output tokens every round and, with
        # unbounded history, eventually trips the provider context-length
        # ceiling. Reaching it starts the wind-down; reaching it AGAIN, with the
        # wind-down already running and out of turns, terminates FAILED before
        # the run degrades into the provider's context-length ceiling.
        # ``run_max_output_tokens_budget=0`` disables the guard.
        rc_budget = engine.config.rc.run_max_output_tokens_budget
        if (
            rc_budget > 0
            and engine.total_usage.output_tokens > rc_budget
            and not engine.is_terminal
        ):
            _logger.warning(
                "DIAG query.run_output_token_budget_exhausted run=%s tenant=%s "
                "output_tokens=%d budget=%d turn=%d",
                engine.config.run_id,
                engine.config.tenant_id,
                engine.total_usage.output_tokens,
                rc_budget,
                assistant_message_idx,
            )
            _output_windup = _enter_soft_stop(
                engine, cause=_soft_stop.CAUSE_OUTPUT_TOKEN_BUDGET
            )
            if _output_windup:
                for _evt in _output_windup:
                    yield _evt
                max_messages = _soft_stop_turn_budget(engine, assistant_message_idx)
                await engine._persist_snapshot()
                current_context = await _rebuild_context_for_recovery(engine)
                continue
            async for evt in _emit_llm_terminal(
                engine,
                MaxOutputTokensExhausted(
                    "run_max_output_tokens_budget exhausted "
                    f"({engine.total_usage.output_tokens} > {rc_budget} output "
                    "tokens) — terminating before the context-length ceiling"
                ),
                kind="run_output_token_budget_exhausted",
            ):
                yield evt
            return

        # A run-level tool precondition that burnt its attempt budget ends the
        # run. Checked at the top of every outer iteration — i.e. BEFORE
        # another attempt is spent, and before the run can reach any
        # completion path — because a caller who asked for a precondition and
        # did not get one has been lied to, and silently answering anyway is
        # exactly the failure this mechanism exists to prevent. Inert for a run
        # that carries no preconditions.
        if _preconditions.is_exhausted(engine) and not engine.is_terminal:
            async for evt in _emit_tool_precondition_terminal(engine):
                yield evt
            return

        # Hard cap on assistant messages within a single ``query()`` turn.
        # ``turn_count`` only increments once per ``engine.run()`` — without
        # this counter, a model emitting endless tool calls would blow the
        # Python recursion stack before max_turns_per_run could fire.
        assistant_message_idx += 1
        if assistant_message_idx > max_messages:
            # The turn budget is spent. First time here, that starts the
            # wind-down: the model is told, its tools are taken away, and it
            # gets ``soft_stop_max_turns`` turns to write the answer it has the
            # evidence for. This branch used to hold three mechanisms — a forced
            # artifact seal, a terminal-tool nudge, and a synthetic answer the
            # runtime submitted on the model's behalf — each with its own latch,
            # its own RC and its own idea of what "finish now" meant. The
            # wind-down subsumes all three: the artifact sealer stays on the
            # narrowed surface while an artifact is open, the notice is the
            # nudge, and the answer is written by the model rather than
            # assembled by the runtime out of its last words.
            _turns_windup = _enter_soft_stop(engine, cause=_soft_stop.CAUSE_MAX_TURNS)
            if _turns_windup:
                for _evt in _turns_windup:
                    yield _evt
                max_messages = _soft_stop_turn_budget(engine, assistant_message_idx)
                await engine._persist_snapshot()
                current_context = await _rebuild_context_for_recovery(engine)
                continue

            # Second time here: the wind-down was given its turns and did not
            # produce an answer, or it is disabled. If a typed stream error was
            # what started it and the run is still unanswered, that error is the
            # run's outcome — surfacing it beats completing silently on a budget
            # the error is the reason we ran out of.
            # The stored error is the run's outcome ONLY if the wind-down it
            # started produced nothing. A wind-down that got the model to write
            # its answer did the job it exists for, and re-raising the upstream
            # failure over that answer would throw away the recovery and report
            # a run that answered as a run that failed.
            if (
                stored_stream_error is not None
                and not _history_has_terminal_tool_result(engine)
                and not run_has_final_answer(engine)
            ):
                _exc, _kind = stored_stream_error
                async for evt in _emit_llm_terminal(engine, _exc, kind=_kind):
                    yield evt
                return

            # Exhaustion, not a successful completion. Route to the
            # FAILURE-class terminal so downstream success/failure accounting
            # (the host _finalise_run, the terminal-signal classifier,
            # dashboards, eval rigs) does not score a budget-exhausted run
            # green. A run that got here THROUGH the wind-down says
            # ``soft_stop`` rather than ``max_turns``: it was told to stop and
            # given turns to finish in, which is a different fact about the run
            # than simply running out of them.
            #
            # The transition MUST happen BEFORE the ``MESSAGE_STOP`` yield (the
            # contract every other FAILED/CANCELLED site follows): ``query()``
            # is consumed directly by the executor, so while the consumer
            # processes the yielded event this generator is suspended — a
            # transition placed after the yield only runs on the next pull,
            # which is ``StopAsyncIteration``. The executor's terminal mirror
            # reads ``engine.state`` while handling ``MESSAGE_STOP``;
            # transitioning late made it read RUNNING and score the
            # budget-exhausted run completed.
            from_state = engine.state
            # Pair any already-appended tool_use that never received a result
            # before driving the FAILED terminal. The budget-exhausted last turn
            # may have appended an assistant ``ToolUseBlock`` the dispatch loop
            # never reached; without this the persisted snapshot carries an
            # orphan tool_use and ``_repair_outbound_tool_pairing`` forward-fills
            # an opaque synthetic on every subsequent request, which a
            # transcript consumer cannot tell from a real model output.
            _synthesize_missing_tool_results(
                engine.history,
                error_content=engine.config.rc.tool_result_interrupted_placeholder,
            )
            _soft_stopped = _soft_stop.is_armed(engine)
            _finalized = _soft_stop.finalize(engine)
            if _finalized is not None:
                yield _finalized
            engine.transition_to(LoopState.FAILED)
            yield _emit_state_change(
                engine,
                from_state,
                LoopState.FAILED,
                reason=(
                    "soft_stop_exhausted" if _soft_stopped else "max_turns_exhausted"
                ),
            )
            yield TurnEvent(
                type=EventType.MESSAGE_STOP,
                run_id=engine.config.run_id,
                payload={
                    "turn_id": engine.turn_id(),
                    "stop_reason": (
                        _soft_stop.STOP_REASON
                        if _soft_stopped
                        else StopReason.max_turns.value
                    ),
                },
            )
            return

        # Wall-clock deadline. The failure this guards: the run produced a good
        # answer and was killed by an external trial or reaper before it got
        # round to delivering it. Reaching ``agent_max_seconds`` minus the
        # configured slack starts the wind-down, which is the same wind-down the
        # turn and token budgets start — one mechanism, so "the run was cut
        # short" reads the same in the transcript whichever bound did it. Inert
        # when ``agent_max_seconds`` is 0.0. The wind-down's own arming latch is
        # durable, so a run resumed near or past its deadline does not start a
        # second one.
        if _terminal_deadline_reached(engine):
            _deadline_windup = _enter_soft_stop(
                engine, cause=_soft_stop.CAUSE_DEADLINE
            )
            if _deadline_windup:
                for _evt in _deadline_windup:
                    yield _evt
                max_messages = _soft_stop_turn_budget(engine, assistant_message_idx)
                await engine._persist_snapshot()
                current_context = await _rebuild_context_for_recovery(engine)
                continue

        # Reset per-message recovery state at every new assistant
        # message — the budget is per-message, not per-run.
        engine.reset_recovery_state()

        # #1/#4 — advance to this round's wire turn id + restart block_idx at 0
        # BEFORE emitting ``message_start``. ``engine.turn_id()`` now yields a
        # DISTINCT id per assistant-message round, and every frame this round
        # emits (``message_start`` below, the ``content_block_*`` / ``tool_use_*``
        # / ``tool_result`` frames inside ``_drive_one_stream``, and the
        # ``message_stop`` at the bottom of this iteration) reads the same id
        # because they all call ``engine.turn_id()`` — so no frame is orphaned
        # from its round. The FIRST round (``assistant_message_idx == 1``) was
        # already opened just before the 4b loop-strategy step (so the Deep
        # ``REASONING_STEP`` and this message_start share one id); only rounds 2+
        # advance here. Pre-loop terminals keep the legacy (unsuffixed) id.
        if assistant_message_idx > 1:
            engine.begin_wire_round()

        # ── message_start ───────────────────────────────────────────
        yield TurnEvent(
            type=EventType.MESSAGE_START,
            run_id=engine.config.run_id,
            payload={
                "turn_id": engine.turn_id(),
                "model": engine.effective_model_name,
                "role": "assistant",
            },
        )

        # ── Stream loop with reactive-413 recovery ────────────────────
        pending_tool_calls: list[ToolCall] = []
        history_tool_calls: list[ToolCall] = []
        text_buffer = ""
        reasoning_buffer = ""

        terminal_yielded = False
        # Forced terminal backstop — set True when an exhaustion exit
        # inside the inner stream loop armed the forced terminal backstop
        # instead of going terminal. Checked right after the inner loop so
        # the OUTER loop iterates with the injected nudge in history (one
        # final bounded turn). Default behaviour (RC off) never sets this.
        backstop_armed = False
        # An inner while-loop lets us re-stream after recovery
        # for ``LLMContextWindowExceeded`` (reactive compaction),
        # ``LLMProviderError`` with configured fallback model, or
        # ``finish_reason='length'`` within the round budget.
        while True:
            stream_result = _StreamAttemptResult()
            tool_results_ready_at = previous_tool_results_ready_at
            previous_tool_results_ready_at = None
            from protocore.runtime.correctness_bind import commit_usage, fire_typed_hook

            _transform_out, transform_evt = fire_typed_hook(
                engine, "transform_context", {"history_len": len(engine.history)}
            )
            if transform_evt is not None:
                yield transform_evt
            try:
                async for evt in _drive_one_stream(
                    engine,
                    current_context,
                    stream_result,
                    previous_tool_results_ready_at=tool_results_ready_at,
                ):
                    yield evt
                retrying = engine._transient_stream_retry_count > 0
                usage_evt = commit_usage(
                    engine,
                    kind="retry" if retrying else "inference",
                    input_tokens=engine.total_usage.this_turn_input,
                    output_tokens=engine.total_usage.this_turn_output,
                    success=True,
                )
                if usage_evt is not None:
                    yield usage_evt
            except LLMContextWindowExceeded as exc:
                # Context-window-overflow recovery.
                _persist_partial_attempt_to_history(engine, stream_result)
                async for evt in _handle_context_window_exceeded(engine, exc):
                    yield evt
                if engine.is_terminal:
                    terminal_yielded = True
                    break
                current_context = await _rebuild_context_for_recovery(engine)
                continue
            except LLMStreamIdleError as exc:
                _persist_partial_attempt_to_history(engine, stream_result)
                # The stream went quiet. A model that stalls AFTER the last
                # tool result has its evidence in hand and only needs to be
                # asked for the answer, so this starts the wind-down rather than
                # terminating. The original error is stashed first: if the
                # wind-down produces nothing, that error is what the run
                # reports, instead of a silent no-answer completion.
                _idle_windup = _enter_soft_stop(
                    engine, cause=_soft_stop.CAUSE_PROVIDER_ERROR
                )
                if _idle_windup:
                    stored_stream_error = (exc, "llm_stream_idle")
                    max_messages = _soft_stop_turn_budget(
                        engine, assistant_message_idx
                    )
                    for _evt in _idle_windup:
                        yield _evt
                    await engine._persist_snapshot()
                    current_context = await _rebuild_context_for_recovery(engine)
                    backstop_armed = True
                    break
                # Backstop disabled / already used — terminal LLM error.
                # But a transient idle stream on a harness-forced continuation
                # must NOT bury an answer the user already received: if a
                # substantive reply is already in history, complete on it
                # instead of driving the run FAILED.
                if _preserve_completed_answer_on_stream_error(engine):
                    async for evt in _complete_run_on_preserved_answer(
                        engine,
                        reason="stream_error_completed_answer_preserved",
                    ):
                        yield evt
                    return
                # Persist the partial before the terminal FAILED transition;
                # otherwise a reload after the terminal event shows no record
                # of what the user saw live.
                async for evt in _emit_llm_terminal(engine, exc, kind="llm_stream_idle"):
                    yield evt
                terminal_yielded = True
                break
            except (LLMRateLimitError, LLMTimeoutError) as exc:
                _persist_partial_attempt_to_history(engine, stream_result)
                # TRANSIENT upstream failure — a 429 rate-limit
                # (``LLMRateLimitError``) or a request/stream timeout
                # (``LLMTimeoutError``). The error classifier marks both
                # retryable / should_fallback, so — unlike an unclassified
                # crash — they must get the SAME recovery a generic provider
                # error gets, plus a bounded in-place retry. These two classes
                # are siblings of ``LLMProviderError`` (not subclasses), so
                # this branch MUST precede it to intercept them. Recovery
                # order: (a) step down the run's provider chain;
                # (b) else re-open the same stream up to
                # ``llm_transient_error_retry_max_attempts`` with a backoff;
                # (c) only after both are unavailable/exhausted go terminal —
                # and even then preserve an already-delivered answer.
                #
                # (a) before (b) because a healthy sibling provider beats
                # sleeping on a sick one: the backoff is a bet that this
                # endpoint recovers, and the chain exists precisely for the
                # runs where that bet is wrong.
                rc = engine.config.rc
                kind = (
                    "llm_rate_limit"
                    if isinstance(exc, LLMRateLimitError)
                    else "llm_timeout"
                )
                # (a) step down the provider chain — bounded by
                # ``llm_provider_chain_max_advances``, one-way, and the same
                # path the ``LLMProviderError`` branch below uses.
                advanced_to = await _advance_provider_chain(engine, exc, kind=kind)
                if advanced_to:
                    yield TurnEvent(
                        type=EventType.STATE_CHANGED,
                        run_id=engine.config.run_id,
                        payload={
                            "from": engine.state.value,
                            "to": engine.state.value,
                            "reason": "model_fallback_triggered",
                            "fallback_model_id": advanced_to,
                            "primary_error": str(exc),
                            "error_class": kind,
                        },
                    )
                    # Persist the partial the user already saw live before the
                    # swap so the next provider's stream (and a reload) carry it.
                    current_context = await _rebuild_context_for_recovery(engine)
                    continue
                # (b) bounded in-place retry with backoff. The streak counter
                # resets after any successful stream (below), so the bound is
                # per consecutive-failure streak.
                if (
                    engine._transient_stream_retry_count
                    < rc.llm_transient_error_retry_max_attempts
                ):
                    fail_evt = commit_usage(
                        engine,
                        kind="inference",
                        input_tokens=engine.total_usage.this_turn_input,
                        output_tokens=engine.total_usage.this_turn_output,
                        success=False,
                    )
                    if fail_evt is not None:
                        yield fail_evt
                    engine._transient_stream_retry_count += 1
                    delay = _transient_retry_backoff_seconds(
                        rc, engine._transient_stream_retry_count, exc
                    )
                    yield TurnEvent(
                        type=EventType.STATE_CHANGED,
                        run_id=engine.config.run_id,
                        payload={
                            "from": engine.state.value,
                            "to": engine.state.value,
                            "reason": "transient_llm_error_retry",
                            "error_class": kind,
                            "attempt": engine._transient_stream_retry_count,
                            "backoff_seconds": delay,
                            "primary_error": str(exc),
                        },
                    )
                    # Persist the partial before the backoff so a crash during
                    # the pause does not lose what the user already saw.
                    if delay > 0:
                        await asyncio.sleep(delay)
                    current_context = await _rebuild_context_for_recovery(engine)
                    continue
                # (c) fallback unavailable/engaged AND retries exhausted. A
                # transient error on a harness-forced continuation must NOT
                # bury an answer the user already received.
                if _preserve_completed_answer_on_stream_error(engine):
                    async for evt in _complete_run_on_preserved_answer(
                        engine,
                        reason="stream_error_completed_answer_preserved",
                    ):
                        yield evt
                    return
                # Persist the partial before the terminal FAILED transition so
                # a reload reflects what streamed live.
                async for evt in _emit_llm_terminal(engine, exc, kind=kind):
                    yield evt
                terminal_yielded = True
                break
            except LLMProviderError as exc:
                _persist_partial_attempt_to_history(engine, stream_result)
                # Step down the provider chain. This branch is the adapters'
                # catch-all — a 503 and a policy refusal arrive here as the same
                # Python type — so only the classification attached to the error
                # decides whether a different endpoint could serve it. An
                # unclassified one never advances.
                advanced_to = await _advance_provider_chain(
                    engine, exc, kind="llm_provider_error"
                )
                if advanced_to:
                    yield TurnEvent(
                        type=EventType.STATE_CHANGED,
                        run_id=engine.config.run_id,
                        payload={
                            "from": engine.state.value,
                            "to": engine.state.value,
                            "reason": "model_fallback_triggered",
                            "fallback_model_id": advanced_to,
                            "primary_error": str(exc),
                        },
                    )
                    # The failed attempt's deltas already streamed to SSE
                    # consumers; persist the partial assistant turn to history
                    # so the next provider's stream (and a reload) carry the
                    # same content the live view showed. Without this, the user
                    # sees a divergent second answer in the same turn window
                    # while reload shows only the second.
                    continue
                # No fallback available or already used. Wind the run down
                # rather than dropping it: the partial the user already saw is
                # in history, the model has whatever evidence it gathered, and
                # one narrowed turn is enough to turn that into an answer. Same
                # entry point as every other bound. The original error is
                # stashed so a wind-down that produces nothing still reports the
                # provider failure rather than completing silently.
                _provider_windup = _enter_soft_stop(
                    engine, cause=_soft_stop.CAUSE_PROVIDER_ERROR
                )
                if _provider_windup:
                    stored_stream_error = (exc, "llm_provider_error")
                    max_messages = _soft_stop_turn_budget(
                        engine, assistant_message_idx
                    )
                    for _evt in _provider_windup:
                        yield _evt
                    await engine._persist_snapshot()
                    current_context = await _rebuild_context_for_recovery(engine)
                    backstop_armed = True
                    break
                # Backstop disabled / already used — terminal.
                # A transient provider error on a harness-forced finalize-nudge
                # continuation must NOT override an otherwise-successful run: if
                # a complete assistant answer already streamed in a prior turn,
                # complete on that delivered answer rather than propagating the
                # continuation turn's provider error as the run's terminal status.
                if _preserve_completed_answer_on_stream_error(engine):
                    async for evt in _complete_run_on_preserved_answer(
                        engine,
                        reason="stream_error_completed_answer_preserved",
                    ):
                        yield evt
                    return
                # Persist the partial before the terminal FAILED transition;
                # otherwise a reload after the terminal event shows no record
                # of what the user saw live.
                async for evt in _emit_llm_terminal(engine, exc, kind="llm_provider_error"):
                    yield evt
                terminal_yielded = True
                break
            except Exception as exc:
                _persist_partial_attempt_to_history(engine, stream_result)
                # The generic catch-all must NOT arm the backstop. It catches
                # real parser/runtime crashes, not just provider-class
                # failures; arming here would let a later best-effort terminal
                # call complete "successfully" and SWALLOW the crash. An
                # unclassified exception stays terminal so the original error
                # is always surfaced.
                #
                # The traceback is logged HERE as well as in the terminal
                # emitter. The two records answer different questions: this one
                # names the turn the crash landed on, and it is written even if
                # driving the terminal then raises in turn.
                _logger.warning(
                    "DIAG query.stream_crashed run=%s tenant=%s turn=%s "
                    "assistant_message=%d exception=%s message=%s",
                    engine.config.run_id,
                    engine.config.tenant_id,
                    engine.turn_id(),
                    assistant_message_idx,
                    type(exc).__name__,
                    exc,
                    exc_info=exc,
                )
                async for evt in _emit_llm_terminal(
                    engine, exc, kind=INTERNAL_ERROR_KIND
                ):
                    yield evt
                terminal_yielded = True
                break

            # Mid-tool-call truncation recovery: the
            # model started emitting a tool_use block, ran out of output
            # tokens before closing the args JSON, and the SSE parser
            # synthesised a stop with ``truncated_by_output_cap=True``.
            # Dispatching that partial call would silently corrupt the
            # tool target (e.g. a Write call would write a truncated
            # file and the model would never know to continue). Instead:
            #
            # 1. Append the partial assistant turn to history so the
            #    next stream can see what was already emitted (text +
            #    tool_use blocks with the truncated args JSON).
            # 2. Append a synthetic user-role resume nudge naming the
            #    truncated tool(s) so the model knows it must re-issue
            #    the call with COMPLETE arguments.
            # 3. Re-open the LLM stream and let the model finish.
            #
            # Budgeted by ``rc.max_output_recovery_rounds`` (shared with
            # the text-only length-truncation branch — each truncation type
            # debits the same per-message counter).
            truncated_tool_calls = [
                tc
                for tc in stream_result.tool_calls
                if tc.truncated_by_output_cap
            ]
            # The recovery trigger is the PROVIDER-AGNOSTIC
            # ``truncated_by_output_cap`` signal under ANY ``finish_reason``
            # (NOT only ``length``). A content-less Write cut at the cap can
            # arrive under ``finish_reason="tool_use"``; the old
            # ``finish_reason == "length"`` gate missed it, causing the broken
            # call to fall through to dispatch → Pydantic ``Field required`` →
            # spiral. A truncated mutation call is NEVER dispatched: it is
            # routed into the bounded chunk-recovery protocol instead.
            if truncated_tool_calls:
                rc = engine.config.rc
                _persist_partial_attempt_to_history(engine, stream_result)
                if engine._max_output_recovery_count < rc.max_output_recovery_rounds:
                    engine._max_output_recovery_count += 1
                    # partition the truncated chunkable
                    # writes whose partial ``content`` body the parser RECOVERED
                    # (salvageable) from the rest. Only when the convergence
                    # driver is enabled: a disabled driver keeps the pre-FEAT
                    # discard-redo (bit-identical). The salvaged calls' partial
                    # bytes are written to disk via a CLEAN synthetic write so
                    # the genuine truncation case lands bytes + the truncation-
                    # gated driver can engage (the stand-validated path). The
                    # NON-salvaged calls (content absent / ``__raw__`` / non-
                    # content tools) keep the chunk-recovery protocol and are
                    # NEVER dispatched ("never dispatch corrupt / content-less"
                    # invariant preserved). A call qualifies for salvage ONLY
                    # when ALL hold: the driver is enabled; the tool is a
                    # BUILT-IN chunkable write (Write/AppendFile) — salvage
                    # dispatches a built-in write, so a flagged TENANT tool
                    # keeps its own recovery instead of being silently
                    # rewritten; a real target path resolves (a pathless
                    # content-present call must stay in ``unsalvaged`` and get
                    # the recovery prompt, never be dropped); and a non-empty
                    # partial ``content`` was recovered.
                    salvage_jobs: list[tuple[ToolCall, str]] = []
                    if _longfile.is_enabled(engine):
                        for _tc in truncated_tool_calls:
                            if _tc.name not in CHUNKABLE_CONTENT_MUTATION_ALLOWLIST:
                                continue
                            if _truncated_call_state_path(_tc) is None:
                                continue
                            salvaged_partial = _salvage_truncated_content(_tc)
                            if salvaged_partial:
                                salvage_jobs.append((_tc, salvaged_partial))
                    salvaged_ids = {tc.id for tc, _ in salvage_jobs}

                    # Telemetry: surface every successful trigger for dashboard
                    # observability. Counter lives in the host-side
                    # parser-recovery telemetry; the core loop signals the same
                    # The original provider attempt was persisted above with its
                    # original call ids and the partial-attempt marker. Synthetic
                    # salvage calls remain separate runtime-authored scaffolding.

                    # Dispatch the COMPLETE (non-truncated) sibling tool calls
                    # in this turn so we do not lose work the model
                    # legitimately finished alongside the truncated one. The
                    # output cap ends the stream, so every complete call
                    # PRECEDES the truncated tail call → dispatching them here
                    # preserves causal order. Without this, complete calls'
                    # ``tool_use`` ids (already appended to the assistant turn
                    # via ``partial_blocks`` above) stay unpaired, and
                    # ``_repair_outbound_tool_pairing`` forward-fills opaque
                    # synthetic ``is_error`` placeholders on every subsequent
                    # request — completed work is dropped and the model sees
                    # synthetic errors, burning extra recovery rounds.
                    # ``truncated_tool_calls`` already excludes the terminal raw
                    # envelope; salvaged calls are a subset of it, so excluding
                    # the truncated set excludes them too.
                    truncated_ids = {tc.id for tc in truncated_tool_calls}
                    sibling_pending = False
                    sibling_terminal_completed = False
                    for tool_call in stream_result.tool_calls:
                        if tool_call.id in truncated_ids:
                            continue
                        # Re-check the cancel checkpoint before EACH sibling
                        # dispatch: this is a distinct dispatch entry point
                        # that runs after several await /
                        # yield boundaries where a cancel can land. Route to the
                        # CANCELLED terminal WITHOUT dispatching; the shared
                        # teardown synthesises ``is_error`` results for any
                        # already-appended-but-undispatched ``tool_use`` (the
                        # truncated calls + the remaining siblings) so the
                        # snapshot stays pairing-valid for a cross-pod resume.
                        if engine.stop_requested:
                            async for evt in _emit_dispatch_cancel_teardown(engine):
                                yield evt
                            return
                        async for evt in _dispatch_tool(engine, tool_call):
                            if evt.type is EventType.TOOL_CALL_PENDING:
                                engine.mark_pending_approval(tool_call.id)
                                engine.transition_to(LoopState.AWAITING)
                                await engine._persist_snapshot()
                                sibling_pending = True
                                yield evt
                                break
                            yield evt
                        if sibling_pending:
                            break
                        if _history_tool_result_is_terminal(engine, tool_call.id):
                            sibling_terminal_completed = True
                            break
                    if sibling_pending:
                        return
                    if sibling_terminal_completed:
                        # pair any unpaired tool_use (the unsalvaged
                        # truncated tail blocks were appended above without a
                        # paired result, and any undispatched siblings) before
                        # driving the COMPLETED terminal. Same
                        # pairing rationale as the main terminal-completion
                        # site; without this call the persisted snapshot
                        # carries orphan tool_use blocks whose only consumer
                        # repair is the opaque wire-boundary
                        # ``_repair_outbound_tool_pairing`` forward-fill.
                        _synthesize_missing_tool_results(
                            engine.history,
                            error_content=engine.config.rc.tool_result_interrupted_placeholder,
                        )
                        # A sibling completed the run via the terminal tool.
                        # Seal any truncation-gated complete-enough-but-unsealed
                        # file (no-op when not eligible) then close the run —
                        # the truncated tail call is moot once the model has
                        # voluntarily finished. Mirrors the terminal seal at
                        # the sibling truncated-recovery path.
                        async for _seal_evt in _maybe_seal_longfile_at_voluntary_finish(
                            engine
                        ):
                            yield _seal_evt
                        async for _completion_evt in _emit_voluntary_completion(engine):
                            yield _completion_evt
                        engine.transition_to(LoopState.COMPLETED)
                        return

                    # Land each salvageable partial on disk (clean synthetic
                    # write; sets the sticky truncation latch + active-path
                    # handoff BEFORE dispatch; bytes + latch persist
                    # atomically). The salvage dispatches ``preapproved=True``
                    # (runtime-internal recovery of the model's OWN content), so
                    # the dispatcher suppresses ``TOOL_CALL_PENDING`` and the
                    # approval gate never fires — same policy as the guaranteed-
                    # terminal synthetic dispatch. The pending guard below is
                    # defensive (mirrors the normal serial-dispatch path) and is
                    # never reached while salvage stays preapproved.
                    for _stc, _salvaged in salvage_jobs:
                        salvage_pending = False
                        # PRE-DISPATCH cancel
                        # checkpoint (mirrors the non-truncated dispatch path at
                        # query.py:1869 and the sibling truncated-recovery
                        # dispatch above at query.py:1199). The recovery turn
                        # crosses several await / yield boundaries (the
                        # recovery persist, the state-change event, the
                        # ``_salvage_truncated_write_to_disk`` generator) where
                        # a cancel can land; without this checkpoint the
                        # synthetic salvage write still dispatches after
                        # ``engine.stop_requested`` flips, mutating the
                        # workspace AFTER cancellation. Route to the
                        # CANCELLED terminal via the shared teardown (the
                        # unrun salvage leaves no matching tool_result, so
                        # ``_repair_outbound_tool_pairing`` forward-fills the
                        # synthetic is_error on the NEXT outbound request if
                        # needed; the previous continue / recovery messages
                        # are already in durable history).
                        if engine.stop_requested:
                            async for evt in _emit_dispatch_cancel_teardown(engine):
                                yield evt
                            return
                        async for _salvage_evt in _salvage_truncated_write_to_disk(
                            engine, _stc, _salvaged
                        ):
                            if _salvage_evt.type is EventType.TOOL_CALL_PENDING:
                                engine.mark_pending_approval(
                                    str(_salvage_evt.payload.get("tool_call_id", ""))
                                )
                                engine.transition_to(LoopState.AWAITING)
                                await engine._persist_snapshot()
                                salvage_pending = True
                            yield _salvage_evt
                        if salvage_pending:
                            return

                    if salvage_jobs:
                        # The original truncated tool calls are never dispatched.
                        # Pair them durably with the canonical interrupted result;
                        # this keeps the original provider attempt reload-safe while
                        # the clean synthetic salvage pair records the mutation that
                        # actually reached disk.
                        _synthesize_missing_tool_results(
                            engine.history,
                            error_content=engine.config.rc.tool_result_interrupted_placeholder,
                        )

                    # The NON-salvaged truncated calls (content-absent /
                    # ``__raw__`` / non-content): chunk-recovery protocol.
                    # ``_build_truncation_chunk_recovery_text`` names the path +
                    # the per-call ``write_chunk_token_budget`` and, on a REPEAT
                    # content-absent truncation of a path, LOWERS the header
                    # budget (``_lowered_header_budget``) so the retry writes a
                    # SMALLER first chunk that fits under the cap, lands bytes,
                    # and lets the driver engage next round (the "cut before any
                    # body" shape). Skipped when everything was salvaged.
                    unsalvaged = [
                        tc for tc in truncated_tool_calls if tc.id not in salvaged_ids
                    ]
                    resume_text = ""
                    if unsalvaged:
                        resume_text = _build_truncation_chunk_recovery_text(
                            engine, unsalvaged
                        )
                        # A content-absent chunkable write cut at the cap left
                        # no bytes to salvage but IS a large-file-in-progress:
                        # latch its REAL path (state-path, never the
                        # ``"the target file"`` placeholder) so once the
                        # smaller-chunk retry lands bytes the driver engages.
                        # The INCOMPLETE-flavoured recovery message is preserved.
                        for _tc in unsalvaged:
                            if _is_content_mutation_truncation(engine, _tc):
                                _longfile.note_truncated_mutation(
                                    engine, _truncated_call_state_path(_tc)
                                )
                    # A-— a NON-salvage recovery turn added NO bytes and
                    # ``continue``s below WITHOUT reaching the turn-end seam, so
                    # advance the stall clock here. When a salvage DID land bytes,
                    # ``observe_tool_result`` already reset the clock to 0, so the
                    # advance is a correct no-op for that turn.
                    _longfile.register_completed_turn(engine)
                    if resume_text:
                        engine.history.append(
                            Message(
                                role=MessageRole.user,
                                content_blocks=[TextBlock(text=resume_text)],
                                metadata={
                                    SYNTHETIC_RECOVERY_METADATA_KEY: (
                                        SYNTHETIC_RECOVERY_TRUNCATION_CONTINUE
                                    )
                                },
                            )
                        )
                    recovery_payload: dict[str, Any] = {
                        "from": engine.state.value,
                        "to": engine.state.value,
                        "reason": "tool_call_truncation_recovery",
                        "round": engine._max_output_recovery_count,
                        "tools": [tc.name for tc in truncated_tool_calls],
                        "paths": _truncated_call_paths(truncated_tool_calls),
                    }
                    # Only add ``salvaged_paths`` key when at least one salvage
                    # actually fired. A disabled driver never populates
                    # ``salvage_jobs``, so its event payload stays BYTE-IDENTICAL
                    # to the no-salvage case (no stray ``salvaged_paths: []``).
                    if salvage_jobs:
                        recovery_payload["salvaged_paths"] = _truncated_call_paths(
                            [tc for tc, _ in salvage_jobs]
                        )
                    yield TurnEvent(
                        type=EventType.STATE_CHANGED,
                        run_id=engine.config.run_id,
                        payload=recovery_payload,
                    )
                    current_context = await _rebuild_context_for_recovery(engine)
                    continue
                # Budget exhausted: the model packed an oversized terminal call
                # (a large list argument, say) and ran out of output budget on
                # every recovery round. Wind down so it is asked for a compact
                # answer on a surface that has nothing else on it.
                _truncation_windup = _enter_soft_stop(
                    engine, cause=_soft_stop.CAUSE_OUTPUT_TOKEN_BUDGET
                )
                if _truncation_windup:
                    max_messages = _soft_stop_turn_budget(
                        engine, assistant_message_idx
                    )
                    # Pin the wind-down turns to the EXHAUSTED recovery counter:
                    # the per-message ``reset_recovery_state`` would otherwise
                    # hand them a fresh ``max_output_recovery_rounds`` budget and
                    # let them re-enter the recovery loop. With the counter left
                    # spent, a re-truncation during the wind-down re-reaches this
                    # branch, where the wind-down is already armed and the
                    # terminal below takes over with the original
                    # ``output_length_exhausted`` kind.
                    engine._terminal_backstop_turn_active = True
                    for _evt in _truncation_windup:
                        yield _evt
                    await engine._persist_snapshot()
                    current_context = await _rebuild_context_for_recovery(engine)
                    backstop_armed = True
                    break
                # Backstop disabled / already used — terminal LLM error.
                async for evt in _emit_llm_terminal(
                    engine,
                    MaxOutputTokensExhausted(
                        "max_output_recovery_rounds exhausted (mid-tool-call truncation)"
                    ),
                    kind="output_length_exhausted",
                ):
                    yield evt
                terminal_yielded = True
                break

            # Max-output-tokens recovery: model truncated mid-turn
            # without yielding tool_calls. Synthesise a resume prompt
            # and re-stream up to ``rc.max_output_recovery_rounds`` times.
            #
            # ``finish_reason is None`` is folded into this branch.
            # ``result.finish_reason`` is set ONLY by a
            # ``ProviderDeltaKind.finish`` delta (``_drive_one_stream``); the
            # empirically-observed OpenRouter SSE tail-loss shape ends the
            # upstream iterator cleanly (``data: [DONE]`` / EOF) with NO finish
            # delta, leaving it ``None``. Treating that as a normal completion
            # let a mid-sentence partial with no tool calls fall through to the
            # no-tool ``end_turn`` branch below and the run COMPLETED with the
            # truncated prefix persisted as the final answer. A finish-less,
            # tool-call-less stream is an incomplete turn: drive the SAME
            # bounded resume recovery as a ``length`` truncation (re-stream so
            # the model finishes; exhaustion goes terminal, never a silent
            # truncated completion).
            #
            # ``not engine.stop_requested`` keeps a cancel that broke the inner
            # stream (also leaves ``finish_reason`` None — the per-delta stop
            # check in ``_drive_one_stream``) on the dedicated CANCELLED path
            # below; an interrupted turn must not be "recovered".
            if (
                stream_result.finish_reason in ("length", None)
                and not stream_result.tool_calls
                and not engine.stop_requested
            ):
                rc = engine.config.rc
                _persist_partial_attempt_to_history(engine, stream_result)
                if engine._max_output_recovery_count < rc.max_output_recovery_rounds:
                    engine._max_output_recovery_count += 1
                    # Append a resume nudge after the durable partial attempt.
                    engine.history.append(
                        Message(
                            role=MessageRole.user,
                            content_blocks=[
                                TextBlock(
                                    text=("Resume directly from where you left off, without preamble or repetition.")
                                )
                            ],
                            metadata={
                                SYNTHETIC_RECOVERY_METADATA_KEY: (
                                    SYNTHETIC_RECOVERY_MAX_OUTPUT_CONTINUE
                                )
                            },
                        )
                    )
                    yield TurnEvent(
                        type=EventType.STATE_CHANGED,
                        run_id=engine.config.run_id,
                        payload={
                            "from": engine.state.value,
                            "to": engine.state.value,
                            "reason": "max_output_token_recovery",
                            "round": engine._max_output_recovery_count,
                        },
                    )
                    current_context = await _rebuild_context_for_recovery(engine)
                    continue
                # Budget exhausted (text-only max-output exhaustion). Same
                # wind-down, same counter pin as the mid-tool-call branch above.
                _max_output_windup = _enter_soft_stop(
                    engine, cause=_soft_stop.CAUSE_OUTPUT_TOKEN_BUDGET
                )
                if _max_output_windup:
                    max_messages = _soft_stop_turn_budget(
                        engine, assistant_message_idx
                    )
                    engine._terminal_backstop_turn_active = True
                    for _evt in _max_output_windup:
                        yield _evt
                    await engine._persist_snapshot()
                    current_context = await _rebuild_context_for_recovery(engine)
                    backstop_armed = True
                    break
                # Backstop disabled / already used — terminal LLM error.
                async for evt in _emit_llm_terminal(
                    engine,
                    MaxOutputTokensExhausted("max_output_recovery_rounds exhausted"),
                    kind="output_length_exhausted",
                ):
                    yield evt
                terminal_yielded = True
                break

            # Normal completion — collect the outcome and exit the
            # inner recovery loop.
            pending_tool_calls = stream_result.tool_calls
            history_tool_calls = list(stream_result.tool_calls)
            text_buffer = stream_result.text_buffer
            reasoning_buffer = stream_result.reasoning_buffer
            guard_evt = _apply_stream_loop_guard(engine, stream_result)
            if guard_evt is not None:
                text_buffer = stream_result.text_buffer
                reasoning_buffer = stream_result.reasoning_buffer
                yield guard_evt
                if engine._loop_guard_nudge_count > engine.config.rc.loop_guard_nudge_max:
                    pending_tool_calls = []
            executable, blocked_events = _block_identical_tools(
                engine, pending_tool_calls
            )
            for blocked_evt in blocked_events:
                yield blocked_evt
            pending_tool_calls = executable
            stream_result.tool_calls = history_tool_calls
            break

        if terminal_yielded:
            return

        # An exhaustion exit inside the inner stream loop armed the forced
        # terminal backstop. Restart the OUTER loop so the next assistant
        # turn streams with the injected nudge + the terminal-only latch
        # active. The ``terminal_nudge_used`` latch already prevents
        # this from looping more than once.
        if backstop_armed:
            continue

        # A real assistant stream completed — the transient-error streak is
        # broken, so refresh the in-place retry budget for any later,
        # independent 429 / timeout blip in this run.
        engine._transient_stream_retry_count = 0

        # ── Continue-prompt fallback ──
        # When the assistant turn ends with:
        #   * empty visible text (``not text_buffer``)
        #   * no pending tool_calls (``not pending_tool_calls``)
        #   * AND populated reasoning_content (``reasoning_buffer``)
        # this is the "thinking-tokens trap" — the model burned its
        # output budget on chain-of-thought and emitted nothing
        # consumable. Recover by appending an assistant turn carrying
        # ONLY the reasoning_content + a synthetic user "continue"
        # nudge, then re-stream. Bounded by
        # ``rc.max_consecutive_empty_responses`` (default 3); beyond
        # → terminal FAILED with kind=``thinking_eats_all_tokens``.
        #
        # Ported from the reference implementation of thinking-tokens recovery.
        rc = engine.config.rc
        if not text_buffer and not pending_tool_calls and reasoning_buffer and rc.max_consecutive_empty_responses > 0:
            engine._consecutive_empty_responses += 1
            _persist_partial_attempt_to_history(engine, stream_result)
            if engine._consecutive_empty_responses <= rc.max_consecutive_empty_responses:
                # Persist the empty assistant turn carrying ONLY the
                # reasoning_content so the next API call can re-inject
                # it (DeepSeek / Kimi require this). No visible
                # content_blocks — keeps the wire-format invariant
                # (system/user/tool = at most one block).
                # Inject synthetic continue-prompt user turn.
                engine.history.append(
                    Message(
                        role=MessageRole.user,
                        content_blocks=[TextBlock(text=rc.continue_prompt_text)],
                        metadata={
                            SYNTHETIC_RECOVERY_METADATA_KEY: (
                                SYNTHETIC_RECOVERY_THINKING_CONTINUE
                            )
                        },
                    )
                )
                yield TurnEvent(
                    type=EventType.STATE_CHANGED,
                    run_id=engine.config.run_id,
                    payload={
                        "from": engine.state.value,
                        "to": engine.state.value,
                        "reason": "continue_prompt_injected",
                        "round": engine._consecutive_empty_responses,
                        "reasoning_content_chars": len(reasoning_buffer),
                    },
                )
                current_context = await _rebuild_context_for_recovery(engine)
                # Loop iterates — opens the next assistant LLM stream.
                continue
            # Budget exhausted on the thinking-tokens trap: the model burned its
            # output budget on chain-of-thought and emitted nothing consumable,
            # round after round. Wind down so it is asked for the best answer its
            # evidence supports rather than producing none at all.
            # ``_consecutive_empty_responses`` is not reset by
            # ``reset_recovery_state``, so it stays exhausted: another
            # empty-reasoning response during the wind-down re-reaches this
            # branch, where the wind-down is already armed and the terminal below
            # takes over with the original ``thinking_eats_all_tokens`` kind.
            _thinking_windup = _enter_soft_stop(
                engine, cause=_soft_stop.CAUSE_PROVIDER_ERROR
            )
            if _thinking_windup:
                max_messages = _soft_stop_turn_budget(engine, assistant_message_idx)
                for _evt in _thinking_windup:
                    yield _evt
                await engine._persist_snapshot()
                current_context = await _rebuild_context_for_recovery(engine)
                continue
            # Backstop disabled / already used — terminal LLM error.
            async for evt in _emit_llm_terminal(
                engine,
                LLMProviderError(
                    "consecutive empty responses with reasoning_content exceeded rc.max_consecutive_empty_responses"
                ),
                kind="thinking_eats_all_tokens",
            ):
                yield evt
            return

        # Recovery succeeded (or never engaged) — reset the counter so a
        # later turn that triggers the trap gets a fresh budget.
        engine._consecutive_empty_responses = 0

        # ── Post-tool empty-response nudge ──
        # A model can return a FULLY-empty assistant turn (no text, no tool
        # calls, AND no reasoning) right after executing tools — distinct
        # from the thinking-tokens trap above (empty WITH reasoning, already
        # handled + ``continue``d). Some providers (mimo-v2-pro / GLM-class)
        # do this when they "expect" the tool result to be the final word.
        # Without recovery the loop would fall through to the no-tool
        # end-turn and COMPLETE with whatever (possibly nothing) is durable.
        # Recover by injecting an API-VALID synthetic pair —
        # assistant('(empty)') + user(nudge) — so the wire sequence stays
        # tool->assistant->user (never tool->user) — then re-stream ONCE.
        # Bounded by ``max_consecutive_empty_responses``. Default-off RC →
        # bit-identical (the branch is skipped entirely).
        rc = engine.config.rc
        if (
            rc.resilience_post_tool_empty_nudge_enabled
            and not text_buffer
            and not pending_tool_calls
            and not reasoning_buffer
            and tool_results_ready_at is not None
            and rc.max_consecutive_empty_responses > 0
        ):
            engine._post_tool_empty_nudge_count += 1
            if engine._post_tool_empty_nudge_count <= rc.max_consecutive_empty_responses:
                # API-valid synthetic pair: an empty-text assistant turn
                # (so the sequence is tool->assistant->user, never
                # tool->user) followed by the corrective user nudge.
                # Flag the synthetic assistant turn as recovery scaffolding so
                # ``_latest_durable_answer_text`` never mistakes the marker
                # (default ``(empty)``) for a real model answer that the
                # guaranteed-terminal backstop could then submit.
                engine.history.append(
                    Message(
                        role=MessageRole.assistant,
                        content_blocks=[TextBlock(text=rc.post_tool_empty_nudge_assistant_text)],
                        metadata={
                            SYNTHETIC_RECOVERY_METADATA_KEY: (
                                SYNTHETIC_RECOVERY_POST_TOOL_EMPTY_NUDGE
                            )
                        },
                    )
                )
                engine.history.append(
                    Message(
                        role=MessageRole.user,
                        content_blocks=[TextBlock(text=rc.post_tool_empty_nudge_user_text)],
                        metadata={
                            SYNTHETIC_RECOVERY_METADATA_KEY: (
                                SYNTHETIC_RECOVERY_POST_TOOL_EMPTY_NUDGE
                            )
                        },
                    )
                )
                yield _emit_state_change(
                    engine,
                    engine.state,
                    engine.state,
                    reason="post_tool_empty_nudge",
                )
                current_context = await _rebuild_context_for_recovery(engine)
                continue
            # Budget exhausted — fall through to the normal no-tool end-turn
            # path (which itself may fire the terminal nudge / guaranteed
            # terminal). The latch is not reset so a tenant cannot loop here.

        # A non-empty (or non-post-tool) turn clears the post-tool empty-nudge
        # counter so a one-off empty early in the run does not permanently
        # consume the budget (mirrors the thinking-trap reset).
        if text_buffer or pending_tool_calls or reasoning_buffer:
            engine._post_tool_empty_nudge_count = 0

        # ── Append assistant message to history ─────────────────────
        assistant_blocks: list[ContentBlock] = []
        if text_buffer:
            assistant_blocks.extend(
                _split_answer_text_blocks(
                    text_buffer, stream_result.narration_prefix_chars
                )
            )
        for tc in history_tool_calls or pending_tool_calls:
            assistant_blocks.append(
                ToolUseBlock(
                    tool_call_id=tc.id,
                    name=tc.name,
                    arguments_json=json.dumps(tc.arguments, ensure_ascii=False),
                )
            )
        if assistant_blocks:
            assistant_metadata = (
                {PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY: True}
                if stream_result.finish_reason == "stop"
                and any(tc.args_partial_truncated for tc in pending_tool_calls)
                else {}
            )
            engine.history.append(
                Message(
                    role=MessageRole.assistant,
                    content_blocks=assistant_blocks,
                    reasoning_content=reasoning_buffer or None,
                    metadata=assistant_metadata,
                )
            )

        # ── Truncated-tool-call recovery (args_partial_truncated) ──
        # The model emitted a tool_use_start + partial args JSON, then
        # ``finish_reason="stop"`` arrived before the args closed. The
        # SSE parser's brace balancer salvaged a parseable dict by
        # synthesising closers and set
        # :attr:`ProviderDelta.args_partial_truncated`, which the loop
        # propagated to :class:`ToolCall.args_partial_truncated` at
        # delta-receipt time. The ``truncated_by_output_cap`` branch only
        # fires on ``finish_reason="length"``; this branch handles the
        # local-model variant where the model truncates with ``stop``.
        # Without this check the loop dispatches a call with empty/malformed
        # args, the tool fails on validation, and the agent never learns it
        # has to chunk large outputs.
        #
        # Recovery: synthesize an ``is_error=True`` tool_result for each
        # truncated call (skipping the real dispatch), append the result
        # to history so the next LLM call sees the recovery instruction,
        # and continue the outer loop so the agent gets a fresh stream.
        #
        # * Mixed-batch dispatch preserves ``TOOL_CALL_PENDING`` approval
        #   semantics via the canonical guard.
        # * Per-message budget bounded by
        #   ``rc.tool_call_max_truncation_recoveries_per_message``;
        #   exhaustion surfaces a terminal LLMProviderError.
        # * M-3 — recovery message templated via RC fields
        #   (``tool_call_truncation_recovery_message_en/_ru``);
        #   chunk-byte ceiling reuses
        #   ``rc.tool_call_max_input_chunk_bytes``; both halves emitted
        #   bilingually, per the multilingual rule.
        # * M-5 — ``MESSAGE_STOP(tool_use)`` yielded AFTER all
        #   dispatches so non-truncated TOOL_RESULT events stay inside
        #   the current assistant-message window for SSE consumers.
        truncated_calls = [
            tc
            for tc in pending_tool_calls
            if tc.args_partial_truncated and not tc.truncated_by_output_cap
        ]
        if truncated_calls and stream_result.finish_reason == "stop":
            rc = engine.config.rc
            # Budget guard: a model stuck in a ``{`` + stop loop would
            # otherwise consume the entire ``max_turns_per_run``.
            if (
                engine._tool_call_truncated_recovery_count
                >= rc.tool_call_max_truncation_recoveries_per_message
            ):
                async for evt in _emit_llm_terminal(
                    engine,
                    LLMProviderError(
                        "tool_call_max_truncation_recoveries_per_message "
                        "exhausted (model kept emitting partial tool args "
                        "+ stop)"
                    ),
                    kind="tool_call_truncated_exhausted",
                ):
                    yield evt
                return
            engine._tool_call_truncated_recovery_count += 1
            chunk_bytes_hint = rc.tool_call_max_input_chunk_bytes
            # Derived placeholders for the more-directive recovery message. ``chunk_bytes_lines`` is a
            # rough lines-per-chunk proxy (~50 chars/line for source code
            # and markdown — works well for both languages). The lower
            # bound of 1 keeps the message intelligible for extremely
            # small ``chunk_bytes`` overrides (< 50). ``chunk_count_estimate``
            # gives the agent a concrete ceiling so it can plan multiple
            # turns; the ``max(2, ...)`` floor guarantees the model is
            # always told to expect at least two chunks (otherwise a
            # large ``chunk_bytes`` override would render the estimate
            # as "1 chunk", which contradicts the "do not retry" rule).
            chunk_bytes_lines = max(1, chunk_bytes_hint // 50)
            chunk_count_estimate = max(2, (10240 // chunk_bytes_hint) + 1)
            for tc in truncated_calls:
                partial_length = len(
                    json.dumps(tc.arguments, ensure_ascii=False)
                )
                _logger.warning(
                    "DIAG tool_dispatch.tool_call_truncated tool_name=%s "
                    "partial_length=%d finish_reason=stop round=%d",
                    tc.name,
                    partial_length,
                    engine._tool_call_truncated_recovery_count,
                    extra={
                        "tool_name": tc.name,
                        "partial_length": partial_length,
                        "finish_reason": "stop",
                        "round": engine._tool_call_truncated_recovery_count,
                    },
                )
                # M-3 — RC-templated bilingual recovery message. Both
                # halves emitted together (EN first, RU second), per
                # the multilingual rule. Per-half
                # placeholders:
                # ``{tool_name}`` (name of the truncated call),
                # ``{partial_length}`` (bytes of args JSON the model
                # emitted), ``{chunk_bytes}`` (sourced from
                # ``rc.tool_call_max_input_chunk_bytes`` so operators
                # can tune the per-chunk char target),
                # ``{chunk_bytes_lines}`` (line-count proxy),
                # ``{chunk_count_estimate}`` (concrete chunk-count
                # ceiling for a 10 KB target). The more-directive
                # template + lines/estimate hints fix the issue where
                # models re-emit the same oversized Write on every
                # recovery round.
                recovery_message_en = rc.tool_call_truncation_recovery_message_en.format(
                    tool_name=tc.name,
                    partial_length=partial_length,
                    chunk_bytes=chunk_bytes_hint,
                    chunk_bytes_lines=chunk_bytes_lines,
                    chunk_count_estimate=chunk_count_estimate,
                )
                recovery_message_ru = rc.tool_call_truncation_recovery_message_ru.format(
                    tool_name=tc.name,
                    partial_length=partial_length,
                    chunk_bytes=chunk_bytes_hint,
                    chunk_bytes_lines=chunk_bytes_lines,
                    chunk_count_estimate=chunk_count_estimate,
                )
                recovery_message = (
                    f"{recovery_message_en}\n\n{recovery_message_ru}"
                )
                # Surface as TOOL_RESULT envelope so SSE consumers (the
                # eval rig + dashboard) see the failure with a real
                # tool_call_id binding.
                yield TurnEvent(
                    type=EventType.TOOL_RESULT,
                    run_id=engine.config.run_id,
                    payload={
                        "tool_call_id": tc.id,
                        "success": False,
                        "error": {
                            "kind": "tool_call_truncated",
                            "message": recovery_message,
                        },
                        "content_blocks": [
                            {"type": "text", "text": recovery_message}
                        ],
                    },
                )
                # Persist the synthetic tool_result so the next LLM
                # turn's history includes the recovery instruction.
                engine.history.append(
                    Message(
                        role=MessageRole.tool,
                        content_blocks=[
                            ToolResultBlock(
                                tool_call_id=tc.id,
                                content=recovery_message,
                                is_error=True,
                            )
                        ],
                    )
                )
                # Symmetry with ``_dispatch_tool``'s post-dispatch
                # cleanup (line ~1589) — once the synthetic tool_result
                # has been emitted the dispatcher has no further need
                # for this id → name mapping.
                engine.forget_tool_name(tc.id)
            await engine._persist_snapshot()
            # Dispatch any non-truncated tool calls in this turn so we do
            # not lose work the model legitimately completed alongside the
            # truncated one. Mirror the canonical
            # approval-pending guard from the normal dispatch path
            # below so ``TOOL_CALL_PENDING`` / AWAITING semantics
            # survive mixed batches. Truncated calls are already
            # pinned to a synthetic error tool_result above.
            approval_pending = False
            terminal_tool_completed = False
            for tool_call in pending_tool_calls:
                if tool_call in truncated_calls:
                    continue
                # Re-check before EACH non-truncated dispatch in the
                # truncated-tool recovery path. This dispatch loop is a SECOND
                # dispatch entry point (distinct from the main
                # loop below) and runs AFTER several await/yield boundaries
                # where a cancel can land: the per-truncated-call TOOL_RESULT
                # yields above, the ``await engine._persist_snapshot()`` right
                # before this loop, and — between two non-truncated calls — a
                # prior tool's own dispatch ``await``. Without this checkpoint a
                # cancel that landed in any of those gaps would still dispatch a
                # NEW (non-truncated) tool here, a side effect AFTER
                # cancellation. Route to the CANCELLED terminal via the shared
                # :func:`_emit_dispatch_cancel_teardown` helper WITHOUT
                # dispatching: the truncated calls already hold synthetic
                # ``is_error`` results (idempotently skipped) and every
                # undispatched non-truncated ``tool_use`` (already appended to
                # history above) is synthesised into an ``is_error`` result so
                # the snapshot stays pairing-valid for a resume on another pod.
                if engine.stop_requested:
                    async for evt in _emit_dispatch_cancel_teardown(engine):
                        yield evt
                    return
                async for evt in _dispatch_tool(engine, tool_call):
                    if evt.type is EventType.TOOL_CALL_PENDING:
                        engine.mark_pending_approval(tool_call.id)
                        engine.transition_to(LoopState.AWAITING)
                        await engine._persist_snapshot()
                        approval_pending = True
                        yield evt
                        break
                    yield evt
                if approval_pending:
                    break
                if _history_tool_result_is_terminal(engine, tool_call.id):
                    terminal_tool_completed = True
                    break
            if approval_pending:
                return
            if terminal_tool_completed:
                # pair any unpaired tool_use (any undispatched
                # non-truncated siblings that trailed the terminal one in the
                # dispatch loop) before driving the COMPLETED terminal. Same
                # pairing rationale as the main dispatch-loop site;
                # truncated calls already have synthetic results appended
                # above, so the synthesis helper is a no-op for them.
                _synthesize_missing_tool_results(
                    engine.history,
                    error_content=engine.config.rc.tool_result_interrupted_placeholder,
                )
                # voluntary-finish terminal seal. The
                # model VOLUNTARILY completed via the terminal tool on the
                # truncated-tool recovery path; seal any truncation-gated
                # complete-enough-but-unsealed file with a SYNTHETIC FinalizeFile
                # before completing. No-op (zero-collateral) when not eligible.
                async for _seal_evt in _maybe_seal_longfile_at_voluntary_finish(
                    engine
                ):
                    yield _seal_evt
                async for _completion_evt in _emit_voluntary_completion(engine):
                    yield _completion_evt
                engine.transition_to(LoopState.COMPLETED)
                return
            # M-5 — ``MESSAGE_STOP(tool_use)`` between assistant
            # messages. Yielded AFTER all dispatches (truncated
            # synthetic + non-truncated real) complete so the SSE
            # consumer's "assistant message window" closes only after
            # every TOOL_RESULT for this turn is on the wire.
            # ``tokens_used`` / ``cache_hit_rate`` reflect partial
            # usage for the truncated stream attempt only (single
            # iteration) — full-run aggregates surface elsewhere.
            previous_tool_results_ready_at = time.perf_counter()
            yield TurnEvent(
                type=EventType.MESSAGE_STOP,
                run_id=engine.config.run_id,
                payload={
                    "turn_id": engine.turn_id(),
                    "stop_reason": "tool_use",
                    "tokens_used": _tokens_used_payload(engine),
                    "cache_hit_rate": engine.total_usage.this_turn_cache_hit_rate(),
                },
            )
            # Rebuild context and re-stream so the agent reads the
            # error tool_result and can retry with chunked writes.
            current_context = await _rebuild_context_for_recovery(engine)
            continue

        # The model produced a clean (non-truncated) assistant turn. Reset
        # the consecutive-truncations counter so a one-off truncation early
        # in the run does not permanently consume a slot. Mirrors the
        # ``_consecutive_empty_responses = 0`` reset pattern.
        engine._tool_call_truncated_recovery_count = 0

        # ── No tool calls — end_turn (terminal) ─────────────────────
        if not pending_tool_calls:
            # an interrupt that landed while this assistant stream
            # was opening breaks the inner stream immediately (the per-delta
            # ``if engine.stop_requested: break`` at the bottom of
            # :func:`_drive_one_stream`) and yields an empty result (no text,
            # no tool_calls). Without this guard the empty turn falls through
            # to the success-class ``end_turn`` → COMPLETED below, scoring an
            # aborted turn as a clean answer (indistinguishable downstream).
            # Mirror the reference's FIRST post-stream abort check
            # (``aborted_streaming``/``aborted_tools``): route to the
            # CANCELLED terminal with ``stop_reason=cancelled`` instead. This
            # runs BEFORE the terminal-tool nudge so a cancelled run is never
            # nudged into one more turn. The outer ``query()`` finally-guard
            # then sees ``engine.is_terminal`` and emits nothing further.
            if engine.stop_requested:
                from_state = engine.state
                engine.transition_to(LoopState.CANCELLED)
                yield _emit_state_change(
                    engine,
                    from_state,
                    LoopState.CANCELLED,
                    reason="stop_requested",
                )
                yield TurnEvent(
                    type=EventType.MESSAGE_STOP,
                    run_id=engine.config.run_id,
                    payload={
                        "turn_id": engine.turn_id(),
                        "stop_reason": StopReason.cancelled.value,
                    },
                )
                return

            # ── large-file convergence (prose / no-tool turn) ──
            # The model ended the turn with prose and NO tool call while a large
            # file is in flight — the dominant "write one header, then idle" /
            # done-with-content-but-unsealed shape. Advance the stall clock and,
            # on a detected stall/plateau/done, FORCE the next tool (AppendFile
            # to drive more content / FinalizeFile to seal — empty-finalize
            # guarded, bounded) and re-drive ONE more turn instead of completing
            # an incomplete file. Runs BEFORE the terminal-tool nudge + the
            # end_turn completion so a stalled large-file run is converged first.
            # No-op when disabled / no stall — the run then completes normally.
            _longfile_forced = False
            async for _conv_evt in _maybe_drive_longfile_convergence(engine):
                if isinstance(_conv_evt, bool):
                    _longfile_forced = _conv_evt
                else:
                    yield _conv_evt
            if _longfile_forced:
                max_messages = max(max_messages, assistant_message_idx + 1)
                current_context = await _rebuild_context_for_recovery(engine)
                continue

            # The terminal-tool nudge ALWAYS fires here (write-first recovery
            # + typed Finalize both depend on it; a prose-only "Done, I
            # created the file" with 0 tools MUST still be nudged into the
            # actual Write + Finalize). The post-answer META prose the nudge
            # can manufacture is instead suppressed AT THE STREAM (text-only,
            # not the tool calls) by :func:`_suppress_terminal_only_meta_text`
            # — see the ``_drive_one_stream`` text-delta path. So the nudge
            # turn still writes the file + runs Finalize, but its redundant
            # meta narration never reaches live SSE nor durable history.
            if _terminal_tool_nudge_required(engine) and not terminal_nudge_used:
                terminal_nudge_used = True
                max_messages = max(max_messages, assistant_message_idx + 1)
                _append_terminal_tool_nudge(engine)
                current_context = await _rebuild_context_for_recovery(engine)
                yield _emit_state_change(
                    engine,
                    engine.state,
                    engine.state,
                    reason="terminal_tool_nudge",
                )
                continue

            # The model produced a no-tool end-turn. If a backstop was armed
            # from a typed stream error and the run still has no terminal
            # answer (e.g. the best-effort forced turn emitted text / called
            # a non-terminal tool and then ended), surface the stored original
            # error rather than silently
            # completing with no answer.
            # The stored error is the run's outcome ONLY if the wind-down it
            # started produced nothing. A wind-down that got the model to write
            # its answer did the job it exists for, and re-raising the upstream
            # failure over that answer would throw away the recovery and report
            # a run that answered as a run that failed.
            if (
                stored_stream_error is not None
                and not _history_has_terminal_tool_result(engine)
                and not run_has_final_answer(engine)
            ):
                _exc, _kind = stored_stream_error
                async for evt in _emit_llm_terminal(engine, _exc, kind=_kind):
                    yield evt
                return

            # ── Substantive-answer floor on the plain-stop path ──
            # The model stopped without calling any tool, so the prose gate at
            # the dispatch seam never sees this run — and a run that delegated,
            # produced files and then said under a hundred characters about
            # them completes here, silently, as a success. Apply the SAME floor
            # at this completion: when the work produced nothing the user can
            # actually read, grant a bounded repair turn and re-drive. An answer
            # that is merely too SHORT shares the gate's durable latch, so that
            # test fires at most once per run across BOTH paths; an answer that
            # is only a POINTER to a file the reader cannot open draws on its
            # own attempt budget instead, because one repair turn was measured
            # to detect that failure without fixing it. Both are bounded, so
            # neither can loop, and the same RC kill switch leaves this path
            # untouched when the gate is off.
            #
            # It never competes with the empty-completion guard further down:
            # that guard owns the turn with NO visible answer at all, and this
            # predicate requires one, so the two are exact complements.
            if _plain_stop_answer_floor_applies(engine):
                repair_text = engine.config.rc.finalize_prose_gate_repair_text
                # An empty repair text would inject an empty user turn —
                # degrade to a no-op (complete as before) and leave the latch
                # unspent and the attempt uncharged, mirroring "the gate did not
                # fire" on the terminal path.
                if repair_text:
                    # Read the pointer evidence BEFORE anything is appended:
                    # the measurement is taken over the answer window, and the
                    # repair turn is part of history the moment it lands.
                    pointer = _pointer_answer_evidence(engine)
                    # Which of the two tests fired decides which bound pays for
                    # the turn: the short-answer floor spends its single shot,
                    # the pointer refusal one attempt of its own budget. Keeping
                    # them apart is what lets the pointer test ask more than
                    # once without also handing the floor a second veto it was
                    # never measured to need.
                    _attempt = 0
                    if pointer is None:
                        engine._finalize_prose_gate_used = True
                    else:
                        _attempt = _charge_pointer_answer_repair(engine)
                    max_messages = max(max_messages, assistant_message_idx + 1)
                    _append_answer_floor_repair_turn(engine)
                    # Persist IMMEDIATELY after the latch + injection so a
                    # crash / cross-pod resume in the gap cannot lose the latch
                    # (and re-fire the repair) or the correction.
                    await engine._persist_snapshot()
                    current_context = await _rebuild_context_for_recovery(engine)
                    # Two ways to get here, and they need opposite reading. The
                    # floor line says the answer was too short and repeats the
                    # threshold it was measured against; the pointer line says
                    # the answer was long enough and still delivered nothing,
                    # and carries the two sizes that make that case. When both
                    # hold, the pointer line is the one that explains the run.
                    if pointer is None:
                        _logger.warning(
                            "DIAG query.finalize_prose_gate.plain_stop_repair "
                            "run=%s tenant=%s turn=%s floor=%d",
                            engine.config.run_id,
                            engine.config.tenant_id,
                            engine.turn_id(),
                            engine.config.rc.finalize_prose_gate_min_chars,
                        )
                    else:
                        _pointer_path, _answer_chars, _written_chars = pointer
                        _rc = engine.config.rc
                        _logger.warning(
                            "DIAG query.finalize_prose_gate.pointer_answer_repair "
                            "run=%s tenant=%s turn=%s attempt=%d/%d "
                            "answer_chars=%d written_chars=%d max_fraction=%.3f "
                            "path=%s",
                            engine.config.run_id,
                            engine.config.tenant_id,
                            engine.turn_id(),
                            _attempt,
                            _rc.finalize_prose_gate_pointer_max_repair_attempts,
                            _answer_chars,
                            _written_chars,
                            _rc.finalize_prose_gate_pointer_max_answer_fraction,
                            _pointer_path,
                        )
                    yield _emit_state_change(
                        engine,
                        engine.state,
                        engine.state,
                        reason="finalize_prose_gate_plain_stop_repair",
                    )
                    continue

            # Reached only when the floor did NOT take this completion. If the
            # pointer refusal spent its whole budget and the answer is STILL a
            # filing notice, this is the moment the run gives up and hands the
            # user that notice — the one outcome the mechanism exists to make
            # visible, and invisible everywhere else (a run that finishes on a
            # repaired answer looks identical from here on).
            _release_pointer_answer_repair(engine)

            # ── voluntary-finish terminal seal ──
            # The model is completing the run VOLUNTARILY with a prose
            # ``end_turn``. The stall-driver above (L~1861) correctly stayed
            # silent for the 004-shape (the model self-continued with steady
            # AppendFile progress so no stall registered), but a truncation-gated
            # file may be complete-enough yet UNSEALED. Dispatch a SYNTHETIC
            # FinalizeFile to seal it BEFORE completing — no LLM call, no extra
            # turn. No-op (zero-collateral) when not eligible (disabled / not
            # truncation-gated / below-floor / already finalized / no budget /
            # FinalizeFile not on the surface). Runs BEFORE the empty-completion
            # guard so a longfile run whose seal produces a terminal result is
            # not misread as an unanswered empty turn.
            async for _seal_evt in _maybe_seal_longfile_at_voluntary_finish(engine):
                yield _seal_evt

            # ── Empty-completion guard ──
            # The model ended the turn with ``finish_reason='stop'`` but emitted
            # NO visible text, NO tool calls and NO reasoning, and the run has
            # neither a visible assistant answer nor a terminal tool result yet.
            # Sealing this as COMPLETED loses the turn silently — the empty
            # ``assistant_blocks`` appended nothing to history, so a reload shows
            # a completed run with no answer. Grant a bounded re-drive so the
            # model gets another chance to answer; once that budget is spent,
            # terminate FAILED with a self-evident reason rather than reporting
            # the empty turn as a clean answer. Default-on RC ⟹ this is the safe
            # default. A turn that already delivered an answer (or carried
            # text / reasoning), or a run already sealed by a terminal tool
            # result (incl. the longfile seal above), never reaches here.
            rc = engine.config.rc
            if (
                rc.empty_completion_guard_enabled
                and not text_buffer
                and not reasoning_buffer
                and not _history_has_terminal_tool_result(engine)
                and not run_has_final_answer(engine)
            ):
                if (
                    engine._empty_completion_redrive_count
                    < rc.empty_completion_guard_max_redrives
                ):
                    engine._empty_completion_redrive_count += 1
                    max_messages = max(max_messages, assistant_message_idx + 1)
                    _append_empty_completion_redrive_nudge(engine)
                    current_context = await _rebuild_context_for_recovery(engine)
                    yield _emit_state_change(
                        engine,
                        engine.state,
                        engine.state,
                        reason="empty_completion_redrive",
                    )
                    continue
                # Re-drive budget exhausted and still no answer — fail loudly
                # instead of sealing a silent empty COMPLETED.
                async for evt in _emit_empty_completion_terminal(engine):
                    yield evt
                return

            async for _completion_evt in _emit_voluntary_completion(engine):
                yield _completion_evt
            settled = _maybe_run_settled_event(engine)
            if settled is not None:
                yield settled
            await _reload_live_control(engine)
            follow_evt = _inject_follow_up_into_history(engine)
            if follow_evt is not None:
                yield follow_evt
                await _persist_live_control(engine)
                engine._run_settled_emitted = False
                max_messages = max(max_messages, assistant_message_idx + 1)
                current_context = await _rebuild_context_for_recovery(engine)
                continue
            engine.transition_to(LoopState.COMPLETED)
            return

        # ── (tool-dispatch path) — cancel before dispatch ──────
        # An interrupt that landed while THIS assistant stream was streaming
        # its tool_use(s) breaks the inner stream's per-delta stop-check
        # (``_drive_one_stream``) AFTER the ``tool_use_stop`` delta already
        # accumulated the call(s) into ``pending_tool_calls`` — so the no-tool
        # branch above does NOT fire, and without this guard the loop would
        # proceed to DISPATCH those tools, performing side effects AFTER the
        # run was cancelled. Mirror the no-tool branch's abort check here for
        # the with-tools path: route to the CANCELLED terminal
        # (``stop_reason=cancelled``) WITHOUT dispatching via the shared
        # :func:`_emit_dispatch_cancel_teardown` helper (synthesises the
        # missing ``is_error`` tool_results so the snapshot stays pairing-valid
        # for a resume on another pod). This is the FIRST of several
        # ``stop_requested`` checkpoints — the cancel guarantee is repeated
        # AFTER the awaited hook predicate and before EACH individual tool
        # dispatch, so a cancel that lands in any
        # await gap of the dispatch prelude/loop cannot dispatch a NEW tool.
        if engine.stop_requested:
            async for evt in _emit_dispatch_cancel_teardown(engine):
                yield evt
            return

        # ── Dispatch each tool call ─────────────────────────────────
        # Parallelise concurrent-safe read-only tools so a
        # single timed-out PCM read does not block sibling reads in the
        # same assistant turn. Sequential dispatch was responsible for
        # turning one 30s PCM stall into a 60s turn gap when the model
        # asked for two reads at once. Parallel-safe tools satisfy
        # ``tool.is_concurrent_safe AND not tool.is_destructive AND not
        # any-PreToolUse-hook-might-match`` (see
        # :func:`_is_parallel_safe_tool` + :func:`_pre_tool_use_match_predicate`);
        # destructive / approval-gated / hook-gated tools stay serial so
        # their causal order with sibling reads is preserved as the LLM
        # emitted them and the serial path's web-mode approval downgrade
        # (``query.py:_dispatch_tool`` web-mode branch) is honoured.
        #
        # History invariant: ``ToolResultBlock`` entries MUST land in
        # ``engine.history`` in the LLM-requested tool-call order, so
        # the parallel branch defers the history mutation via
        # :func:`_drain_dispatch_tool_deferred` + iterates the batch
        # results in the original order via
        # :func:`_apply_deferred_tool_history`. Event emission also
        # follows the original order: gather completion order is
        # discarded.
        approval_pending = False
        terminal_tool_completed = False
        # Set True when a bounded pre-terminal self-verify turn was injected
        # at a would-be-terminal site. It breaks the dispatch loop WITHOUT
        # finalising; flow then falls through to the
        # ``message_stop(tool_use)`` + context-rebuild path so the outer loop
        # re-drives one corrective turn. Default False = no self-verify.
        self_verify_injected = False

        # Pre-compute, once per turn, a predicate that tells us whether a
        # given tool name could be matched by any enabled ``PreToolUse`` hook
        # for the current tenant. If so, the tool MUST stay on the serial
        # dispatch path so the web-mode approval downgrade + first-pending
        # stop invariants live exclusively in :func:`_dispatch_tool`. The
        # predicate is awaited ONCE per turn (one ``IHookManager.list``
        # round-trip) before the batching loop walks the pending tool calls.
        hook_match_predicate = await _pre_tool_use_match_predicate(engine)

        # Re-check AFTER the awaited hook predicate.
        # ``_pre_tool_use_match_predicate`` is an ``await`` (one
        # ``IHookManager.list`` round-trip): a cancel that lands DURING that
        # await is invisible to the pre-loop guard above, yet it must still
        # abort BEFORE any dispatch. Without this checkpoint a cancel in that
        # await gap would fall through and dispatch the pending tools (side
        # effect after cancellation). Route to CANCELLED here instead.
        if engine.stop_requested:
            async for evt in _emit_dispatch_cancel_teardown(engine):
                yield evt
            return

        # Pre-compute eligibility for every call so we do not call into
        # the registry twice per call (this also keeps the partition
        # stable when the registry mutates mid-turn — defensive).
        #
        # When ``rc.parallel_read_tools_enabled`` is False the fan-out is
        # disabled entirely: every call is marked non-eligible so it
        # dispatches through the serial single-element path. Behaviour is
        # otherwise identical (rollback switch). Default True enables it.
        if engine.config.rc.parallel_read_tools_enabled:
            parallel_eligible = [
                _is_parallel_safe_tool(engine, tc, hook_match_predicate)
                for tc in pending_tool_calls
            ]
        else:
            parallel_eligible = [False] * len(pending_tool_calls)

        # Delegation fan-out eligibility — a SEPARATE partition from the read
        # fan-out above. Disabled entirely (every call marked ineligible → the
        # serial path) when the master gate is off OR the effective concurrency
        # cap resolves to ``< 2`` (a cap of 1 ⇒ sequential ⇒ the exact serial
        # path). A read-parallel-eligible call is never delegation-eligible
        # (delegation tools are NOT ``is_concurrent_safe``), so the two
        # partitions are disjoint by construction.
        subagent_concurrency_cap = max(1, engine.config.rc.max_concurrent_subagents)
        if subagent_concurrency_cap >= 2:
            delegation_eligible = [
                _is_delegation_parallel_safe(engine, tc, hook_match_predicate)
                for tc in pending_tool_calls
            ]
        else:
            delegation_eligible = [False] * len(pending_tool_calls)

        # Record that this run hands work to subagents, once, for the whole run.
        # Set from the RAW structural predicate rather than from
        # ``delegation_eligible`` above: that list answers "may these calls fan
        # out concurrently", which a concurrency cap of 1 or a matching hook
        # turns False without making the call any less a delegation. Set HERE
        # rather than at the dispatch seam because the parallel gather and the
        # serial path reach that seam through different code and only this line
        # sees both.
        if not engine._run_delegated and any(
            _tool_is_delegation(engine, tc) for tc in pending_tool_calls
        ):
            engine._run_delegated = True

        # Walk the pending list, batching adjacent parallel-safe runs
        # together so we preserve the original interleaving with any
        # serial tools (a [read, write, read] turn becomes
        # [parallel(read)] → [serial(write)] → [parallel(read)] which
        # respects the model's intended causal order between the write
        # and the second read).
        idx = 0
        n = len(pending_tool_calls)
        # Bound each gather batch at ``parallel_read_tools_max_fanout`` so a
        # turn that emits many parallel-safe reads chunks into ≤N-wide
        # sub-batches (each independently snapshot→gather→restore→replayed,
        # preserving LLM-requested order) instead of fanning out unbounded
        # against a backend that degrades under load.
        #
        # ``0`` is the value-preserving sentinel: UNLIMITED fan-out (one
        # unbounded gather per adjacent parallel-eligible run). Only chunk
        # when the configured cap is ``> 0``. ``max_fanout = n`` for the
        # unlimited case keeps the ``(idx - batch_start) < max_fanout``
        # window from ever closing a
        # batch early, so a whole adjacent parallel run fans out at once.
        configured_fanout = engine.config.rc.parallel_read_tools_max_fanout
        max_fanout = configured_fanout if configured_fanout > 0 else n
        while idx < n:
            # Re-check before EACH dispatch step. A cancel can land DURING a
            # prior tool's dispatch ``await`` (the
            # tool's own ``run()``, a PreToolUse hook, a snapshot persist) or
            # in an awaited parallel-batch ``gather``. The pre-loop guard +
            # the post-hook-predicate guard only cover the window BEFORE the
            # loop starts; this checkpoint guarantees the loop never dispatches
            # a NEW serial tool or a NEW parallel batch once a stop has been
            # observed mid-loop. Tools already dispatched this turn keep their
            # real results; the remaining undispatched ``tool_use`` blocks
            # (still in history, unpaired) are synthesised into ``is_error``
            # results by the shared teardown so the snapshot stays
            # pairing-valid, then the run routes to CANCELLED.
            if engine.stop_requested:
                async for evt in _emit_dispatch_cancel_teardown(engine):
                    yield evt
                return
            if parallel_eligible[idx]:
                batch_start = idx
                while (
                    idx < n
                    and parallel_eligible[idx]
                    and (idx - batch_start) < max_fanout
                ):
                    idx += 1
                batch = pending_tool_calls[batch_start:idx]
                if len(batch) == 1:
                    # Single-element batch — fall through to the serial
                    # dispatcher so behaviour is identical to the
                    # pre-Wave-10 path (no asyncio.gather overhead,
                    # event ordering is trivially preserved).
                    async for evt in _dispatch_tool(engine, batch[0]):
                        if evt.type is EventType.TOOL_CALL_PENDING:
                            engine.mark_pending_approval(batch[0].id)
                            engine.transition_to(LoopState.AWAITING)
                            await engine._persist_snapshot()
                            approval_pending = True
                            yield evt
                            break
                        yield evt
                    if approval_pending:
                        break
                    if _history_tool_result_is_terminal(engine, batch[0].id):
                        # One bounded self-verify turn before finalising. If a
                        # corrective turn is injected, do NOT finalise: break
                        # the dispatch loop so the outer loop runs one more
                        # bounded turn with the correction in history. The
                        # helper persists the snapshot itself on injection so
                        # the correction + latch survive a crash/resume.
                        if await _maybe_inject_pre_terminal_self_verify(engine):
                            # The corrective turn must be granted a fresh
                            # ``max_messages`` slot, or the next outer
                            # iteration immediately exceeds the budget and the
                            # run exits FAILED(max_turns) despite the
                            # already-submitted terminal answer.
                            max_messages = max(max_messages, assistant_message_idx + 1)
                            self_verify_injected = True
                            break
                        terminal_tool_completed = True
                        break
                    continue

                # ≥2 parallel-safe tool calls — fan out under
                # ``asyncio.gather`` so each PCM/HTTP RPC waits in
                # parallel rather than serialising the timeouts.
                #
                # Snapshot the per-run helper-bag streak + satisfaction state
                # BEFORE gather so we can restore it afterwards and replay
                # the state-transitions in the LLM-requested order. Without
                # the snapshot the streak state would track gather completion
                # order, not transcript order — see
                # :func:`_snapshot_dispatcher_helper_state` /
                # :func:`_replay_dispatcher_helper_state` for the contract.
                helpers_snapshot = _snapshot_dispatcher_helper_state(engine)
                results = await asyncio.gather(
                    *(_drain_dispatch_tool_deferred(engine, tc) for tc in batch),
                    return_exceptions=False,
                )
                # Restore the helper-bag streak/satisfaction state to
                # pre-gather. The parallel mutations are discarded; the
                # replay below applies the deterministic transcript-order
                # state transitions.
                _restore_dispatcher_helper_state(engine, helpers_snapshot)
                _logger.warning(
                    "DIAG query.parallel_batch.entered run=%s tenant=%s "
                    "turn=%s batch_size=%d tools=%s",
                    engine.config.run_id,
                    engine.config.tenant_id,
                    engine.turn_id(),
                    len(batch),
                    ",".join(tc.name for tc in batch),
                )
                # Emit events + apply deferred history mutations in the
                # ORIGINAL tool-call order regardless of gather
                # completion order — preserves the LLM-facing invariant.
                first_approval_seen = False
                for tool_call, (events, outcome) in zip(batch, results, strict=True):
                    if first_approval_seen:
                        # Once a tool in the batch has surfaced approval in
                        # LLM order, any further outcomes are discarded so
                        # the serial path's "stop on first pending" invariant
                        # is honoured. The discarded tools are
                        # concurrent-safe + non-destructive by construction
                        # so dropping their results has no external side
                        # effect (no history append, no event yield, no
                        # helper-bag replay).
                        engine.forget_tool_name(tool_call.id)
                        continue
                    if outcome is None:
                        # Defensive: dispatcher should always yield a
                        # final outcome. Mirror the warning emitted by
                        # the serial path so silent drops are visible.
                        _logger.warning(
                            "tool dispatcher returned no outcome for "
                            "call_id=%s (parallel batch)",
                            tool_call.id,
                        )
                        engine.forget_tool_name(tool_call.id)
                        continue
                    if outcome.approval_required:
                        # The hook match predicate should have steered any
                        # hook-gated tool onto the serial path, so we only
                        # reach this branch when a hook was added mid-turn
                        # (race) or a hook unexpectedly emitted approval for
                        # a tool whose matchers do not mention it. Mark ONLY
                        # the first approval in LLM order as pending (parity
                        # with the serial path's stop-on-first behaviour) and
                        # discard the rest of the batch.
                        _logger.warning(
                            "DIAG query.parallel_batch.approval_first_pending "
                            "run=%s tenant=%s turn=%s tool=%s call_id=%s",
                            engine.config.run_id,
                            engine.config.tenant_id,
                            engine.turn_id(),
                            tool_call.name,
                            tool_call.id,
                        )
                        approval_pending = True
                        first_approval_seen = True
                        pending_event_seen = False
                        for evt in events:
                            if evt.type is EventType.TOOL_CALL_PENDING and not pending_event_seen:
                                engine.mark_pending_approval(tool_call.id)
                                engine.transition_to(LoopState.AWAITING)
                                await engine._persist_snapshot()
                                pending_event_seen = True
                            yield evt
                        if not pending_event_seen:
                            engine.mark_pending_approval(tool_call.id)
                            engine.transition_to(LoopState.AWAITING)
                            await engine._persist_snapshot()
                        # Do not append a tool_result for an
                        # approval-pending call (parity with the serial
                        # path which short-circuits before history
                        # mutation).
                        continue
                    # a terminal-only-blocked tool produced a
                    # SYNTHETIC outcome (success=False, is_error=True) WITHOUT
                    # any dispatcher invocation (see
                    # :func:`_drain_dispatch_tool_deferred`). The SERIAL
                    # ``_dispatch_tool`` terminal-only short-circuit appends the
                    # blocked tool_result and returns WITHOUT touching the
                    # durable consecutive-error streak. Running the normal
                    # replay below would take the error path and call
                    # ``_apply_consecutive_error_cap``, INCREMENTING the streak
                    # in parallel mode but not serial mode (and the rewrite
                    # would also overwrite the ``terminal_only`` error kind
                    # with ``execution``). Detect the blocked synthetic by
                    # re-checking the SAME predicate that produced it (no
                    # terminal result has been appended for these non-terminal
                    # blocked reads, so the predicate is still True), then
                    # emit the ORIGINAL blocked events + append history with the
                    # ORIGINAL outcome — bit-identical to the serial path, no
                    # streak mutation.
                    if _terminal_only_blocks(engine, tool_call):
                        for evt in events:
                            yield evt
                        # This synthetic terminal-only veto must NOT feed the
                        # Repeated-tool-error breaker (parity with the serial path,
                        # which returns before breaker tracking). Otherwise a
                        # finalize-gate veto could trip the breaker mid-gate.
                        _apply_deferred_tool_history(
                            engine,
                            tool_call,
                            outcome,
                            track_circuit_breaker=False,
                        )
                        continue
                    # Replay the streak + satisfaction state transitions
                    # against the REAL helper bag in LLM-requested order so
                    # both the visible transcript and next-turn caps follow
                    # transcript-correct counts regardless of gather
                    # completion order.
                    _logger.warning(
                        "DIAG query.parallel_batch.helper_replay run=%s "
                        "tenant=%s turn=%s tool=%s call_id=%s success=%s",
                        engine.config.run_id,
                        engine.config.tenant_id,
                        engine.turn_id(),
                        tool_call.name,
                        tool_call.id,
                        outcome.success,
                    )
                    adjusted_outcome = _replay_dispatcher_helper_state(
                        engine, tool_call, outcome
                    )
                    for evt in _rewrite_deferred_tool_result_events(
                        events, adjusted_outcome
                    ):
                        yield evt
                    _apply_deferred_tool_history(engine, tool_call, adjusted_outcome)
                    if _dispatch_outcome_is_terminal(
                        adjusted_outcome,
                        engine=engine,
                        tool_name=tool_call.name,
                    ):
                        # One bounded self-verify turn before finalising.
                        # ``first_approval_seen`` is reused as the "stop
                        # draining the rest of this batch" signal; set it
                        # whether or not we finalise. When a corrective turn is
                        # injected we set ``self_verify_injected`` instead of
                        # ``terminal_tool_completed`` so the post-batch check
                        # breaks the dispatch loop WITHOUT finalising, and the
                        # outer loop re-drives the corrective turn. The helper
                        # persists the snapshot itself on injection.
                        if await _maybe_inject_pre_terminal_self_verify(engine):
                            # The corrective turn must be granted a fresh
                            # ``max_messages`` slot, or the next outer
                            # iteration immediately exceeds the budget and the
                            # run exits FAILED(max_turns) despite the
                            # already-submitted terminal answer.
                            max_messages = max(max_messages, assistant_message_idx + 1)
                            self_verify_injected = True
                            first_approval_seen = True
                        else:
                            terminal_tool_completed = True
                            first_approval_seen = True
                # One snapshot per batch instead of one per tool —
                # parity with the serial path's per-tool persist but
                # batched to amortise the PG/Redis round-trip.
                await engine._persist_snapshot()
                if approval_pending or terminal_tool_completed or self_verify_injected:
                    break
                continue

            if delegation_eligible[idx]:
                # Group the MAXIMAL run of adjacent delegation-eligible calls.
                delegation_start = idx
                delegation_end = idx
                while delegation_end < n and delegation_eligible[delegation_end]:
                    delegation_end += 1
                if delegation_end - delegation_start >= 2:
                    idx = delegation_end
                    batch = pending_tool_calls[delegation_start:delegation_end]
                    # ≥2 adjacent delegation calls — fan them out under a bounded
                    # semaphore so up to ``max_concurrent_subagents`` child runs
                    # execute concurrently and any excess serialise in waves. Each
                    # child goes through the SAME deferred dispatcher the read
                    # fan-out uses (history append deferred so results land in
                    # LLM-requested order); the leader still blocks until the whole
                    # group finishes (blocking join). Delegation tools are NOT
                    # ``is_concurrent_safe`` so this is a SEPARATE path from the
                    # read fan-out above — but the leader's dispatcher mutates the
                    # shared per-run helper bag (consecutive-error streak,
                    # cumulative tool-call soft cap, satisfied-precondition set) per
                    # child, so we snapshot that transcript-order state BEFORE the
                    # gather, restore it AFTER, and replay the transitions in
                    # LLM-requested order — the same correctness contract the read
                    # path relies on.
                    #
                    # TWO NESTED BOUNDS apply here. This per-turn semaphore bounds
                    # the WIDTH of THIS leader turn's group only — a fresh one is
                    # minted per turn. On its own that composes MULTIPLICATIVELY
                    # across depth (each nested group carries its own independent
                    # semaphore, so a depth-2 tree of width W runs up to W*W
                    # children at once). The SECOND bound — ``budget``, a
                    # tree-wide SubagentTreeBudget shared by reference down the
                    # whole run tree — caps the ADDITIVE sum of concurrently
                    # executing children across every nested group, so depth no
                    # longer multiplies (see rc.max_concurrent_subagents_per_tree).
                    #
                    # The tree bound is deadlock-free by construction, which the
                    # naive "one shared semaphore held around each child's whole
                    # run" is NOT: that naive scheme wedges because a parent pins
                    # its permit for the child's ENTIRE blocking join while the
                    # child needs permits for its OWN grandchildren, so at the cap
                    # the held permits starve the nested acquires. The scheme here
                    # is release-while-awaiting-children (leaf-counting): a tree
                    # slot is acquired per child at the DISPATCH site
                    # (``_dispatch_subagent_under_semaphore``), and a run that is
                    # blocked awaiting its OWN children RELEASES its slot for the
                    # duration of that wait (below) and reacquires it after. So
                    # every slot holder is a run doing local work — none is blocked
                    # on descendants — and no holder needs a further slot to finish
                    # its current slice; the budget can never form an acquisition
                    # cycle. ``budget`` is resolved (minted at the first parallel
                    # fan-out from the RC, then found in the helper bag) so the
                    # whole parallel-dispatched subtree shares the SAME object;
                    # ``tree_permit`` is THIS run's own slot (None for a run that
                    # was never dispatched under the budget — e.g. the root, or a
                    # run reached only by serial delegation) — present only for a
                    # child that was itself dispatched under the budget.
                    semaphore = asyncio.Semaphore(subagent_concurrency_cap)
                    budget = _resolve_subagent_tree_budget(engine)
                    tree_permit = _resolve_subagent_tree_permit(engine)
                    helpers_snapshot = _snapshot_dispatcher_helper_state(engine)
                    # Stable id for THIS fan-out group, shared by every child and
                    # distinct across groups/turns (tool_call ids are unique per
                    # run). Lets the parent ledger scope same-path batch-order
                    # resolution per group so a later turn's group is never frozen
                    # out by an earlier one (see AttemptLedger.declare).
                    dispatch_group = batch[0].id
                    # Release this run's tree slot BEFORE blocking on its children
                    # and reacquire AFTER (finally, so a raising gather still
                    # reacquires) — the crux of the deadlock-free scheme. No-op for
                    # the root leader (holds no permit) and under the unlimited
                    # sentinel.
                    if tree_permit is not None:
                        await tree_permit.release_while_waiting()
                    try:
                        gathered = await asyncio.gather(
                            *(
                                _dispatch_subagent_under_semaphore(
                                    engine,
                                    tc,
                                    semaphore,
                                    budget,
                                    dispatch_order=batch_pos,
                                    dispatch_group=dispatch_group,
                                )
                                for batch_pos, tc in enumerate(batch)
                            ),
                            return_exceptions=True,
                        )
                    finally:
                        if tree_permit is not None:
                            await tree_permit.reacquire()
                    _restore_dispatcher_helper_state(engine, helpers_snapshot)
                    _logger.warning(
                        "DIAG query.parallel_subagents.entered run=%s tenant=%s "
                        "turn=%s batch_size=%d cap=%d tools=%s",
                        engine.config.run_id,
                        engine.config.tenant_id,
                        engine.turn_id(),
                        len(batch),
                        subagent_concurrency_cap,
                        ",".join(tc.name for tc in batch),
                    )
                    # Normalise gather results in LLM order. A child that RAISED
                    # (defensive — the dispatcher normally returns structured
                    # error outcomes for unknown subagent_type / hook block /
                    # timeout) is converted into its own error outcome so siblings
                    # still complete. A ``BaseException`` that is NOT an
                    # ``Exception`` (``CancelledError`` / ``SystemExit`` /
                    # ``KeyboardInterrupt``) is re-raised so cancellation is never
                    # swallowed by ``return_exceptions=True``.
                    normalised_results: list[tuple[list[TurnEvent], DispatchOutcome | None]] = []
                    for tool_call, raw in zip(batch, gathered, strict=True):
                        if isinstance(raw, BaseException):
                            if not isinstance(raw, Exception):
                                raise raw
                            _logger.warning(
                                "DIAG query.parallel_subagents.child_raised run=%s "
                                "tenant=%s turn=%s tool=%s call_id=%s error=%s",
                                engine.config.run_id,
                                engine.config.tenant_id,
                                engine.turn_id(),
                                tool_call.name,
                                tool_call.id,
                                type(raw).__name__,
                            )
                            normalised_results.append(
                                _synthesize_delegation_error_result(
                                    engine, tool_call, raw
                                )
                            )
                        else:
                            normalised_results.append(raw)

                    # Emit events + apply deferred history in the ORIGINAL
                    # tool-call order regardless of gather completion order.
                    first_approval_seen = False
                    for tool_call, (events, outcome) in zip(
                        batch, normalised_results, strict=True
                    ):
                        if first_approval_seen:
                            engine.forget_tool_name(tool_call.id)
                            continue
                        if outcome is None:
                            _logger.warning(
                                "tool dispatcher returned no outcome for "
                                "call_id=%s (parallel subagent batch)",
                                tool_call.id,
                            )
                            engine.forget_tool_name(tool_call.id)
                            continue
                        if outcome.approval_required:
                            # Defensive: the eligibility predicate steers any
                            # hook-gated delegation call onto the serial path, so
                            # this only fires on a mid-turn hook race. Mark ONLY
                            # the first approval in LLM order as pending (serial
                            # parity) and discard the rest of the batch.
                            approval_pending = True
                            first_approval_seen = True
                            pending_event_seen = False
                            for evt in events:
                                if (
                                    evt.type is EventType.TOOL_CALL_PENDING
                                    and not pending_event_seen
                                ):
                                    engine.mark_pending_approval(tool_call.id)
                                    engine.transition_to(LoopState.AWAITING)
                                    await engine._persist_snapshot()
                                    pending_event_seen = True
                                yield evt
                            if not pending_event_seen:
                                engine.mark_pending_approval(tool_call.id)
                                engine.transition_to(LoopState.AWAITING)
                                await engine._persist_snapshot()
                            continue
                        if _terminal_only_blocks(engine, tool_call):
                            # Synthetic terminal-only veto (no dispatcher
                            # invocation) — emit the blocked events + append
                            # history WITHOUT feeding the circuit breaker, exactly
                            # as the serial path returns before breaker tracking.
                            for evt in events:
                                yield evt
                            _apply_deferred_tool_history(
                                engine,
                                tool_call,
                                outcome,
                                track_circuit_breaker=False,
                            )
                            continue
                        # Replay the streak + satisfaction + soft-cap transitions
                        # against the REAL helper bag in LLM-requested order so
                        # the transcript and next-turn caps follow transcript-order
                        # counts regardless of gather completion order.
                        adjusted_outcome = _replay_dispatcher_helper_state(
                            engine, tool_call, outcome
                        )
                        for evt in _rewrite_deferred_tool_result_events(
                            events, adjusted_outcome
                        ):
                            yield evt
                        _apply_deferred_tool_history(
                            engine, tool_call, adjusted_outcome
                        )
                        if _dispatch_outcome_is_terminal(
                            adjusted_outcome,
                            engine=engine,
                            tool_name=tool_call.name,
                        ):
                            # One bounded self-verify turn before finalising —
                            # parity with the read/serial paths.
                            if await _maybe_inject_pre_terminal_self_verify(engine):
                                max_messages = max(
                                    max_messages, assistant_message_idx + 1
                                )
                                self_verify_injected = True
                                first_approval_seen = True
                            else:
                                terminal_tool_completed = True
                                first_approval_seen = True
                    # One snapshot per batch (parity with the read fan-out).
                    await engine._persist_snapshot()
                    if (
                        approval_pending
                        or terminal_tool_completed
                        or self_verify_injected
                    ):
                        break
                    continue
                # Group of exactly one delegation call → fall through to the exact
                # serial path below (no gather, byte-identical single-call
                # behaviour). ``idx`` is unchanged, so the serial block dispatches
                # this call and advances.

            # Serial path — single non-parallel-safe tool call. A run holding a
            # tree-budget slot that dispatches a single delegation call blocks on
            # the child's whole nested run; the tree slot is released around that
            # join INSIDE :func:`_dispatch_tool` (the single choke point every
            # serial-style delegation await funnels through), so this call site
            # needs no permit handling of its own.
            tool_call = pending_tool_calls[idx]
            idx += 1
            async for evt in _dispatch_tool(engine, tool_call):
                if evt.type is EventType.TOOL_CALL_PENDING:
                    engine.mark_pending_approval(tool_call.id)
                    engine.transition_to(LoopState.AWAITING)
                    await engine._persist_snapshot()
                    approval_pending = True
                    yield evt
                    break
                yield evt
            if approval_pending:
                break
            #  — if ``_dispatch_tool`` vetoed this
            # terminal via the prose-gate (appended its non-terminal error
            # result + the corrective user turn), STOP draining the rest of this
            # assistant batch: do NOT dispatch later sibling tool calls AFTER
            # the injected user-repair turn (that would interleave a user
            # message between sibling tool results in the durable snapshot).
            # Mirror ``self_verify_injected`` — break WITHOUT finalising; the
            # outer loop re-drives one corrective turn.
            if _prose_gate_just_injected(engine):
                max_messages = max(max_messages, assistant_message_idx + 1)
                self_verify_injected = True
                break
            if _history_tool_result_is_terminal(engine, tool_call.id):
                # One bounded self-verify turn before finalising (serial
                # path). The helper persists the snapshot itself on injection.
                if await _maybe_inject_pre_terminal_self_verify(engine):
                    # The corrective turn must be granted a fresh
                    # ``max_messages`` slot, or the next outer iteration
                    # immediately exceeds the budget and the run exits
                    # FAILED(max_turns) despite the already-submitted terminal
                    # answer.
                    max_messages = max(max_messages, assistant_message_idx + 1)
                    self_verify_injected = True
                    break
                terminal_tool_completed = True
                break
        if approval_pending:
            return

        if terminal_tool_completed:
            # Pair any already-appended tool_use that never received a result
            # before driving the COMPLETED terminal. The dispatch loop walked
            # ``pending_tool_calls`` in order and broke on the terminal tool
            # result; any sibling tool_use blocks ALREADY in
            # ``engine.history`` (appended to the assistant turn above) were
            # never dispatched and have no paired ``ToolResultBlock``. The
            # unconditional outbound wire repair (``_repair_outbound_tool_pairing``)
            # would still forward-fill an opaque synthetic so a re-stream
            # does not 400, but the durable ``engine.history`` mirror carries
            # the orphan — a session-transcript consumer (chat reducer,
            # dashboard) renders a tool call with no result. Call the same
            # pairing helper the cancel/LLM-error/compaction teardowns use
            # so every exit path leaves a pairing-valid history.
            _synthesize_missing_tool_results(
                engine.history,
                error_content=engine.config.rc.tool_result_interrupted_placeholder,
            )
            # ── voluntary-finish terminal seal ──
            # The model VOLUNTARILY completed the run by calling the run-terminal
            # tool. A truncation-gated large file may be complete-enough but
            # UNSEALED (the model appended chunks then ended via the terminal
            # tool without FinalizeFile). Dispatch a SYNTHETIC FinalizeFile to
            # seal it BEFORE the run completes — no LLM call, no extra turn.
            # No-op (zero-collateral) when not eligible.
            async for _seal_evt in _maybe_seal_longfile_at_voluntary_finish(engine):
                yield _seal_evt
            async for _completion_evt in _emit_voluntary_completion(engine):
                yield _completion_evt
            engine.transition_to(LoopState.COMPLETED)
            return

        # ── message_stop (tool_use) — between assistant messages ────
        previous_tool_results_ready_at = time.perf_counter()
        yield TurnEvent(
            type=EventType.MESSAGE_STOP,
            run_id=engine.config.run_id,
            payload={
                "turn_id": engine.turn_id(),
                "stop_reason": "tool_use",
                "tokens_used": _tokens_used_payload(engine),
                "cache_hit_rate": engine.total_usage.this_turn_cache_hit_rate(),
            },
        )

        # ── Proactive per-iteration compaction gate ──
        # The tool result(s) are now in ``engine.history``; BEFORE rebuilding
        # the wire payload for the next assistant stream, run compaction if the
        # live history has crossed the trigger (routine) or emergency cliff
        # (proactive force). Without this gate the inner loop rebuilds context
        # from the FULL history with no compaction check (12K→39K per-
        # iteration inflation until a reactive 413). RC kill-switch
        # ``compaction_per_iteration_enabled`` (default on); idempotent via
        # the stable-key dedup + already-summary skip so re-running the gate
        # every iteration does not thrash already-compacted content.
        #
        # The result(s) just appended are the CURRENT iteration's tool batch
        # — freshly produced and not yet consumed by the model.
        # ``keep_recent_turns`` (default 4) only protects the trailing N
        # messages, so a parallel batch of >keep tool calls would leave the
        # 5th-from-last (and earlier) fresh result eligible for compaction IN
        # THIS SAME ITERATION. We pass ``protect_tail_from_index`` = the
        # index of the current assistant ``tool_use`` turn so the whole
        # in-flight batch (any size) is exempt on top of the keep window.
        # The protection is inherently kill-switched by
        # ``compaction_per_iteration_enabled``: when off, this gate never
        # fires, so no in-iteration batch compaction can occur at all.
        if engine.config.rc.compaction_per_iteration_enabled:
            _iter_emergency = (
                engine.config.rc.compaction_emergency_proactive_enabled
                and engine.needs_emergency_compaction()
            )
            if _iter_emergency or engine.needs_compaction():
                _batch_protect_index = current_tool_batch_protect_index(engine.history)
                async for evt in _run_compaction(
                    engine,
                    force=_iter_emergency,
                    reason=(
                        "proactive_per_iteration_emergency"
                        if _iter_emergency
                        else "proactive_per_iteration"
                    ),
                    protect_tail_from_index=_batch_protect_index,
                ):
                    yield evt
                # A compaction-exhausted FAILED terminal bails out of the loop
                # — mirror the turn-start gate. Pair any dangling tool_use so
                # the FAILED snapshot is wire-valid .
                if engine.state is LoopState.FAILED:
                    _synthesize_missing_tool_results(
                        engine.history,
                        error_content=engine.config.rc.tool_result_interrupted_placeholder,
                    )
                    yield TurnEvent(
                        type=EventType.MESSAGE_STOP,
                        run_id=engine.config.run_id,
                        payload={
                            "turn_id": engine.turn_id(),
                            "stop_reason": StopReason.error.value,
                        },
                    )
                    return

        # ── large-file convergence (tool-call-turn end) ──
        # The just-completed turn dispatched tools; advance the stall clock and,
        # on a detected stall/plateau, FORCE the next tool (AppendFile to drive
        # content / FinalizeFile to seal — empty-finalize guarded, bounded). The
        # injected continue message + forced tool_choice are picked up by the
        # context rebuild + next stream below. No-op when disabled / no stall.
        _longfile_forced = False
        async for _conv_evt in _maybe_drive_longfile_convergence(engine):
            if isinstance(_conv_evt, bool):
                _longfile_forced = _conv_evt
            else:
                yield _conv_evt
        if _longfile_forced:
            # Mirror the prose/no-tool seam above: a forced turn at the
            # message-budget boundary must be GRANTED an extra slot, or the
            # next outer iteration immediately exceeds ``max_messages`` and
            # the forced stream is silently killed by ``max_turns`` before
            # the model can respond (forced budget charged + history mutated,
            # but no output). No explicit ``continue`` is needed here — the
            # loop already iterates after the context rebuild below.
            max_messages = max(max_messages, assistant_message_idx + 1)

        # ── Build next context (tool_results now in history) ────────
        await _reload_live_control(engine)
        wake_ids = await _maybe_place_background_wakes(engine)
        if wake_ids:
            yield TurnEvent(
                type=EventType.BACKGROUND_WAKE,
                run_id=engine.config.run_id,
                payload={"task_ids": wake_ids},
            )
        steer_evt = _inject_steer_into_history(engine)
        if steer_evt is not None:
            yield steer_evt
            await _persist_live_control(engine)
        tool_defs = list(
            engine.tools.compute_effective_surface(
                tenant_id=engine.config.tenant_id,
                policy=engine.effective_tool_policy,
                query=engine.latest_user_message.text if engine.latest_user_message else "",
                top_k=engine.config.rc.tool_retrieval_top_k,
            )
        )
        next_history, _ = _llm_history(engine)
        # Reuse the run's once-built skill catalog (NOT rebuilt per iteration —
        # a per-turn rebuild would bust the cached system-prompt prefix).
        current_context = engine.context_manager.build_context(
            history=next_history,
            tools=tool_defs,
            system_prompt_sections=engine.config.system_prompt_sections,
            skill_index_block=await _ensure_run_skill_catalog(engine),
            skills_loaded=engine._skill_loaded_bundles,
        )
        # The rebuilt context (above) carries the injected continue message; the
        # forced tool_choice rides on a transient engine attr consumed by the
        # next stream. The ``max_messages`` bump above is the only other effect.
        # Loop iterates — opens the next assistant LLM stream.


class _StreamAttemptResult:
    """Mutable accumulator for one streaming attempt.

    Owned by :func:`_stream_one_assistant_message`; mutated by
    :func:`_drive_one_stream` so the caller can branch on
    ``finish_reason`` / ``tool_calls`` post-stream. Internal — never
    yielded to consumers.

    ``reasoning_buffer`` accumulates ``ProviderDeltaKind.thinking``
    content for detecting an assistant turn that emits reasoning_content
    but NO visible text AND NO tool_calls — the "thinking eats all tokens"
    failure mode — the loop injects a continue-prompt and re-streams up to
    ``rc.max_consecutive_empty_responses`` times before terminal.

    The text/reasoning content is accumulated into ``list[str]`` fragment
    lists and joined ONCE on read via the ``text_buffer`` /
    ``reasoning_buffer`` properties.
    Per-delta ``str += fragment`` is O(n^2) over the buffer length —
    a large Write/HTML generation streamed as thousands of tiny deltas
    turned the consumer into a quadratic CPU section on the single
    executor event loop, which (with unbounded concurrent runs)
    starved a neighbour run's pending provider socket read and produced
    a FALSE ``provider stream produced no data`` stall. These buffers
    are write-only until end-of-stream (every reader in
    :func:`_stream_one_assistant_message` runs AFTER the
    ``async for ... _drive_one_stream`` loop completes), so deferring
    the join is observationally identical.
    """

    __slots__ = (
        "_reasoning_fragments",
        "_text_fragments",
        "finish_reason",
        "narration_prefix_chars",
        "tool_calls",
    )

    def __init__(self) -> None:
        self.finish_reason: str | None = None
        self.tool_calls: list[ToolCall] = []
        self._text_fragments: list[str] = []
        self._reasoning_fragments: list[str] = []
        # Where the live stream cut this message's prose into a collapsed
        # leading narration and the answer behind it — an offset into
        # ``text_buffer``, 0 when nothing was split. Recorded by the stream so
        # the durable blocks are built from the SAME decision the reader
        # already saw, never from a second reading of the finished text.
        self.narration_prefix_chars: int = 0

    def append_text(self, fragment: str) -> None:
        if fragment:
            self._text_fragments.append(fragment)

    def append_reasoning(self, fragment: str) -> None:
        if fragment:
            self._reasoning_fragments.append(fragment)

    @property
    def text_buffer(self) -> str:
        return "".join(self._text_fragments)

    @property
    def reasoning_buffer(self) -> str:
        return "".join(self._reasoning_fragments)


async def _drive_one_stream(
    engine: QueryEngine,
    context: ContextBundle,
    result: _StreamAttemptResult,
    *,
    previous_tool_results_ready_at: float | None = None,
) -> AsyncIterator[TurnEvent]:
    """Drive one provider stream + emit per-delta TurnEvents.

 Mutates ``result`` in place so the caller can branch on
 ``finish_reason`` / ``tool_calls`` post-stream. Per the
 streaming body is a separate helper so the outer recovery loop in
 :func:`_stream_one_assistant_message` can wrap it in
 :class:`LLMContextWindowExceeded` / future recovery handlers.

 Prompt-cache breakpoint hints are added to :attr:`LLMRequest.extra`.
 The hints are produced by
 :func:`protocore.runtime.prompt_caching.apply_system_and_3` (pure
 function — same input → same output, no IO). The Anthropic adapter
 consumes them; OpenAI / vLLM ignore them.
 """
    rc = engine.config.rc
    max_output_tokens = max(
        1,
        int(context.budgets.max_context * rc.llm_output_max_tokens_ratio),
    )
    # The global per-message output cap is ``max_context * ratio`` (the KB
    # binding cap). Keep it so the Item-4 final-turn floor below can never
    # exceed it — the reserve claws tokens back DOWN within this cap, it is
    # NOT a global-cap raise.
    output_cap_before_band = max_output_tokens
    # AdaptiveSafetyBand. The band subtracts a
    # calibrated drift margin from the per-call output budget so the
    # prompt + max_tokens stay under the provider context window even
    # when the local token estimator misjudges Cyrillic-in-JSON-escape
    # inflation. The band is per-(provider, model); when no band is
    # wired (kill-switch off / test fixture) the helper returns 0 and
    # behaviour is identical to pre-A4.
    safety_band = _resolve_safety_band_value(engine)
    if safety_band > 0:
        max_output_tokens = max(1, max_output_tokens - safety_band)
    # Final-turn-specific output-token reserve. Floors the post-safety-band
    # budget on the terminal / forced-final turn so the model can emit
    # message + refs + outcome. Default-off ⇒ no-op; never raises the
    # global cap (bounded by ``output_cap_before_band``).
    max_output_tokens = _apply_terminal_synthesis_output_reserve(
        engine, max_output_tokens, output_cap_before_band
    )
    full_messages = _prepend_system_sections(
        context.system_prompt_sections,
        context.messages,
    )

    # UNCONDITIONAL tool_use<->tool_result pairing repair at the wire
    # boundary, immediately before LLMRequest assembly. Applied on every
    # request: forward-fills synthetic is_error tool_results for orphaned
    # tool_use, reverse-strips orphaned tool_results, and dedupes duplicate
    # ids — so the wire is robust to orphaning from ANY source (Tier-2
    # compaction dropping one side, resume-from-partial-batch, max_tokens
    # truncation), not just the cases the orphan PRODUCERS were patched for.
    # Runs BEFORE the cache-breakpoint computation so the breakpoint indices
    # address the final outbound list.
    full_messages = _repair_outbound_tool_pairing(
        full_messages,
        placeholder=rc.tool_result_pairing_repair_placeholder,
    )

    # vLLM-400 backstop — normalize any non-leading ``system`` message to
    # ``user`` at the SAME wire boundary as the pairing repair. vLLM 400s on a
    # ``system`` message past index 0 ("System message must be at the
    # beginning."); a mid-history Tier-2 compaction summary used to be
    # system-role (now fixed at source, but legacy persisted snapshots may
    # still carry one). The genuine system prefix at index 0 is untouched.
    full_messages, _converted_system = _normalize_outbound_system_messages(
        full_messages
    )
    if _converted_system and not getattr(
        engine, "_outbound_system_normalized_warned", False
    ):
        engine._outbound_system_normalized_warned = True  # type: ignore[attr-defined]
        _logger.warning(
            "normalized %d non-leading system message(s) to user role at the "
            "request boundary (vLLM-400 guard; likely a legacy compaction "
            "summary from a persisted snapshot)",
            _converted_system,
        )

    # Compute prompt-cache breakpoints once per provider call. The pure
    # function is cheap (O(n) over messages,
    # no allocation hot-spots) so we recompute every iteration of
    # ``_stream_one_assistant_message`` rather than thread cache state
    # through the engine. Hints land on
    # :attr:`LLMRequest.extra["cache_breakpoints"]` — adapters that
    # don't recognise the key ignore it (vLLM, OpenAI). Anthropic
    # translates each :class:`CacheBreakpoint` into a wire-format
    # ``cache_control`` block per its detected
    # :class:`CachePolicy`.
    cache_breakpoints = apply_system_and_3(list(full_messages))
    extra: dict[str, object] = {"cache_breakpoints": cache_breakpoints}
    # Thread the native-thinking axis onto every
    # assistant stream so the host vLLM adapter can translate the
    # keys to ``chat_template_kwargs.enable_thinking`` + top-level
    # ``reasoning_effort``. ``reasoning_effort`` is ALWAYS sent alongside
    # ``enable_thinking`` so CoT stays bounded even when thinking is on
    # (measured: ``enable_thinking`` alone truncates the answer). When
    # thinking is off the adapter keeps its default-disable path; the effort
    # value is inert there. Adapters that do not recognise the keys ignore
    # them (bit-identical for non-vLLM providers).
    extra["enable_thinking"] = engine.effective_thinking_enabled
    extra["reasoning_effort"] = engine.effective_reasoning_effort
    # ``extra["forced_tool_choice"]`` is a SINGLE slot carrying the tool NAME
    # the host vLLM/OpenAI-compatible adapter translates into a native
    # ``tool_choice={type:function, function:{name}}``; a provider/adapter that
    # does not recognise the key ignores it. Three independent mechanisms want
    # that slot, and in every case it may only ever name a tool THIS request
    # actually advertises — a forced choice for an unadvertised tool 400s the
    # whole request, and a compacted/BM25-clipped surface can drop a tool
    # any of them wanted.
    #
    # A run-level tool PRECONDITION wins the slot. The three are different
    # concerns — the precondition is a promise to the caller that this tool
    # runs before the agent answers, the convergence hint is an internal nudge
    # toward finishing a file, the read-back gate is a debt the agent took on
    # by delegating — and only the promise has a deadline. While a
    # precondition is outstanding the other two are not even READ, let alone
    # popped, so whatever they decided is still pending, and still forced, once
    # the preconditions are done.
    #
    # Convergence beats read-back for the same reason: its hint is transient
    # (it forces exactly the next stream, and a stream spent elsewhere is a
    # stream a half-written file spends unfinished), while the pending-read set
    # is durable and loses nothing by waiting a turn.
    precondition_tool = _preconditions.outstanding_tool(engine)
    if precondition_tool is not None:
        # Unlike the hint below, an unforceable precondition is charged as a
        # SPENT attempt rather than deferred: the caller was promised this
        # call, so a surface that never offers the tool has to end the run
        # rather than let it answer anyway.
        if any(
            getattr(t, "name", None) == precondition_tool for t in context.tools
        ):
            extra["forced_tool_choice"] = precondition_tool
            _preconditions.charge_attempt(engine)
        else:
            _preconditions.charge_attempt(
                engine,
                error=(
                    f"{precondition_tool} was not advertised on this turn's "
                    "tool surface, so it could not be forced"
                ),
            )
    else:
        # The convergence driver's hint is consumed (popped) here so it forces
        # exactly the next stream, and degrades to a strong prose nudge —
        # still bounded — on an adapter that ignores the key. PEEK first, only
        # pop when the surface includes the tool: ``AppendFile`` /
        # ``FinalizeFile`` are NOT in ``tool_surface_forced_pins`` by default,
        # so a BM25-clipped surface can drop the forced tool while the continue
        # message + ``commit_forced_*`` charge are already settled — an
        # unconditional pop dropped the hint and the model never saw a
        # ``forced_tool_choice``. A future stream whose surface does include
        # the tool picks the hint back up.
        forced_tool = _longfile.peek_force_next_tool(engine)
        if (
            forced_tool is not None
            and any(getattr(t, "name", None) == forced_tool for t in context.tools)
        ):
            # Surface includes the tool — consume the hint exactly once.
            _longfile.take_force_next_tool(engine)
            extra["forced_tool_choice"] = forced_tool
        else:
            # Nothing else wants the slot — a convergence hint that could not
            # be forced this turn has left it empty and kept its own state, so
            # taking it here costs that driver nothing.
            #
            # The read-back gate: while a tool result's declared files are
            # unread, force the read tool so the agent cannot answer out of the
            # one-line pointer it was handed. Same peek-then-charge discipline
            # as the hint above — the pending set is NOT touched when the read
            # tool is missing from this turn's surface, and no attempt is
            # charged for a stream the model was never offered it on.
            released_reads = _pending_reads.release_exhausted(engine)
            if released_reads:
                # The bound is spent: the files could not be opened. Say so on
                # the run's event stream and hand the surface back, rather than
                # forcing a read that will never land.
                yield _emit_state_change(
                    engine,
                    engine.state,
                    engine.state,
                    reason="pending_reads_released",
                )
            readback_tool = _pending_reads.peek_forced_tool(engine)
            if readback_tool is not None:
                # STRICT terminal-only finalisation permits exactly one
                # dispatch, the run's terminal tool; a forced read under it
                # surfaces a ``terminal_only`` error and burns the turn. The
                # pending set survives, so a run that leaves the latch still
                # owes its reads (mirrors the convergence driver's own bail).
                #
                # Both bail-outs are RECORDED rather than merely taken: an
                # uncharged, unforced turn is otherwise indistinguishable in a
                # log from one where nothing ever declared a read-back.
                if _terminal_only_enforced(engine):
                    _pending_reads.note_not_forced(
                        engine, reason="terminal_only_enforced"
                    )
                elif not any(
                    getattr(t, "name", None) == readback_tool for t in context.tools
                ):
                    _pending_reads.note_not_forced(
                        engine, reason="tool_not_on_surface"
                    )
                else:
                    _pending_reads.charge_forced_attempt(engine)
                    extra["forced_tool_choice"] = readback_tool
    yield TurnEvent(
        type=EventType.TOOL_SURFACE_ADVERTISED,
        run_id=engine.config.run_id,
        payload=_tool_surface_advertised_payload(engine, context),
    )
    request = LLMRequest(
        model=engine.effective_model_name,
        messages=full_messages,
        tools=list(context.tools),
        max_tokens=max_output_tokens,
        extra=extra,
        observability=LLMObservabilityContext(
            tenant_id=engine.config.tenant_id,
            run_id=engine.config.run_id,
            parent_run_id=engine.config.parent_run_id,
            session_id=engine.config.session_id,
            agent_id=engine.config.subagent_id,
            call_purpose="run",
            call_category=_provider_call_category(engine),
        ),
    )

    block_idx = engine.next_block_idx()
    # Track the KIND of the currently-open content block, not a bare
    # boolean. ``thinking`` and ``text`` deltas MUST land in SEPARATE typed
    # blocks: the chat reducer renders each block by its
    # ``content_block_start`` kind and silently drops deltas whose type
    # does not match the open block, so a thinking-then-answer stream under
    # one kind=thinking block loses the visible answer live.
    # ``None`` means no block is open.
    open_block_kind: ProviderDeltaKind | None = None
    # Track whether the most recently allocated ``block_idx`` belongs to a
    # TOOL block (set by ``tool_use_start``, cleared by ``tool_use_stop``).
    # The OpenAI wire permits interleaved ``delta.content`` while a tool
    # call is buffered open, so a text/thinking reopen can land with
    # ``open_block_kind is None`` but ``block_idx`` pointing at the
    # tool block. The reopen path uses this flag to advance
    # ``block_idx`` BEFORE emitting ``content_block_start``, guaranteeing
    # every emitted content block carries a unique ``block_idx`` per
    # turn (no two ``content_block_start`` for the same idx — a wire
    # block-model violation; the chat reducer otherwise replaces the
    # ``tool_use`` placeholder with the text block).
    last_block_was_tool: bool = False
    # Track whether a tool call that CONTINUES the run has already started
    # inside this assistant message — the whole signal behind the visibility
    # marking on every content block below. Deliberately NOT ``last_block_was_tool``:
    # that flag is block-index bookkeeping and must keep counting the terminal
    # gate call (which allocates an idx like any other), whereas this one must
    # not, because the terminal call sits in the same message as the answer.
    # Monotonic within the message — a run that has dispatched a tool cannot
    # un-dispatch it — which is what lets ``content_block_stop`` settle a value
    # that ``content_block_start`` had to send conservatively.
    tool_interleaved: bool = False
    if previous_tool_results_ready_at is not None:
        gap_ms = (time.perf_counter() - previous_tool_results_ready_at) * 1000.0
        _logger.warning(
            "DIAG query.internal_turn_gap run=%s tenant=%s turn=%s "
            "gap_ms=%.1f context_messages=%d context_tools=%d history_messages=%d",
            engine.config.run_id,
            engine.config.tenant_id,
            engine.turn_id(),
            gap_ms,
            len(full_messages),
            len(context.tools),
            len(engine.history),
        )
    upstream = engine.llm.stream_with_tools(request)

    # Decide ONCE, up-front, whether this turn's visible assistant TEXT is the
    # redundant post-nudge META narration that must be suppressed from live SSE
    # + durable history. The decision is stable for the whole turn
    # (``_terminal_only_active`` + the prior-answer/background-tool facts do
    # not change mid-stream), so no per-delta cost and NO buffering: a
    # suppressed ``text`` delta is dropped before it opens a block or appends
    # to ``text_buffer``. THINKING + ALL tool calls (Write/AppendFile/Finalize)
    # are NEVER affected — only the user-facing text leak. See
    # :func:`_suppress_terminal_only_meta_text`.
    suppress_meta_text = _suppress_terminal_only_meta_text(engine)

    # ── Leading-narration split, for a run that delegated ──
    # ``head_buffer`` holds the start of this message's FIRST text block off the
    # wire until it is known whether that block opens with process narration.
    # The hold is bounded (scan ceiling + answer floor characters) and its
    # failure mode is deliberately mild: ``result.append_text`` still runs on
    # every delta as it arrives, so the durable text is complete whatever
    # happens to the buffer, and every path that closes the block flushes it.
    # ``None`` means not buffering — the split is off, or this block's head is
    # already settled, or the message's text started in an earlier block.
    narration_split_active = _answer_narration_split_active(engine)
    head_buffer: str | None = None
    head_scan_available = narration_split_active

    def _emit_pending_head(*, complete: bool) -> Iterator[TurnEvent]:
        """Release the buffered head, splitting it when the verdict is in.

        With ``complete=True`` a verdict is always available, so this is the
        call every block-close path makes and no buffered text can be stranded.
        Mid-stream it returns nothing while the run of narration can still
        grow. Leaves ``open_block_kind`` alone: on a split it closes the
        narration block and opens the answer block itself, so the caller's own
        ``content_block_stop`` still closes exactly one open text block.
        """

        nonlocal head_buffer, block_idx
        if head_buffer is None:
            return
        split = _settled_narration_split(engine, head_buffer, complete=complete)
        if split is None:
            return
        narration, answer = split
        head_buffer = None
        if narration:
            yield from _text_delta_events(engine, narration, block_idx=block_idx)
            yield TurnEvent(
                type=EventType.CONTENT_BLOCK_STOP,
                run_id=engine.config.run_id,
                payload={
                    "turn_id": engine.turn_id(),
                    "block_idx": block_idx,
                    "visibility": _content_block_visibility(
                        ProviderDeltaKind.text,
                        tool_interleaved=tool_interleaved,
                        narration_prefix=True,
                    ).value,
                },
            )
            block_idx = engine.next_block_idx()
            yield TurnEvent(
                type=EventType.CONTENT_BLOCK_START,
                run_id=engine.config.run_id,
                payload={
                    "turn_id": engine.turn_id(),
                    "block_idx": block_idx,
                    "kind": ProviderDeltaKind.text.value,
                    "visibility": _content_block_visibility(
                        ProviderDeltaKind.text, tool_interleaved=tool_interleaved
                    ).value,
                },
            )
            # The durable blocks are built from this number, so history and the
            # wire carry one decision rather than two readings of the same text.
            result.narration_prefix_chars = len(narration)
        if answer:
            yield from _text_delta_events(engine, answer, block_idx=block_idx)

    async for delta in _iter_with_idle_watchdog(
        _as_provider_deltas(upstream),
        idle_timeout=rc.llm_stream_idle_timeout_seconds,
        stall_threshold=rc.llm_stream_stall_threshold_seconds,
        reasoning_idle_timeout=rc.llm_stream_reasoning_idle_timeout_seconds,
    ):
        if engine.stop_requested:
            break

        if delta.kind is ProviderDeltaKind.finish:
            if open_block_kind is not None:
                # Release anything the narration split is still holding first —
                # its verdict is final now, and the block cannot close over
                # text the reader never received.
                for evt in _emit_pending_head(complete=True):
                    yield evt
                # Every ``content_block_stop`` carries the SETTLED visibility of
                # the block it closes, which is the value a reader should keep.
                # Recomputing here rather than echoing what the matching
                # ``content_block_start`` sent is safe in one direction only,
                # and that is the direction we need: ``tool_interleaved`` never
                # goes back to False inside a message, so the settled value can
                # tighten to ``collapsed`` but can never relax an already-sent
                # ``collapsed`` back to ``public``.
                yield TurnEvent(
                    type=EventType.CONTENT_BLOCK_STOP,
                    run_id=engine.config.run_id,
                    payload={
                        "turn_id": engine.turn_id(),
                        "block_idx": block_idx,
                        "visibility": _content_block_visibility(
                            open_block_kind, tool_interleaved=tool_interleaved
                        ).value,
                    },
                )
                open_block_kind = None
            result.finish_reason = delta.finish_reason
            break

        if delta.kind is ProviderDeltaKind.usage and delta.usage:
            cache_read = int(
                delta.usage.get("cache_read_input_tokens", 0)
                or delta.usage.get("cache_read_tokens", 0)
                or delta.usage.get("cached_tokens", 0)
            )
            cache_creation = int(delta.usage.get("cache_creation_input_tokens", 0))
            input_tokens = int(delta.usage.get("input_tokens", 0))
            engine.total_usage.add(
                input_tokens=input_tokens,
                output_tokens=int(delta.usage.get("output_tokens", 0)),
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_creation,
            )
            # The same envelope, charged a SECOND time against the whole tree's
            # cumulative ledger. ``total_usage`` belongs to THIS engine and every
            # delegated child gets a fresh one, so it can never answer "how much
            # has this question cost in total" — the ledger is shared by
            # reference with every descendant and does.
            _charge_run_work_tokens(
                engine,
                input_tokens=input_tokens,
                output_tokens=int(delta.usage.get("output_tokens", 0)),
            )
            # Ground-truth prompt size the provider reported for this call.
            # Every provider adapter normalises the FULL prompt (prompt_tokens,
            # inclusive of any cache-read portion) into ``input_tokens``, so it
            # already reflects total context-window occupancy — do NOT add
            # ``cache_read`` on top (that subset is already inside input_tokens
            # and would double-count). Floors the compaction gate against the
            # char heuristic, which under-counts adversarial content 2-3x.
            if input_tokens > 0:
                engine.last_observed_prompt_tokens = input_tokens
            # Feed the optional cache observer one observation per usage
            # envelope. Core cannot import the host's metrics module (import
            # boundary); the host injects a concrete ``CacheObserverProtocol``
            # via ``QueryEngineConfig.cache_observer``. ``cache_breakpoints`` is
            # the placement-hint list assembled above by
            # :func:`apply_system_and_3` — adapters that ignore the key
            # still surface the count.
            observer = engine.config.cache_observer
            if observer is not None:
                cache_breakpoints_hint = request.extra.get(
                    "cache_breakpoints", []
                )
                observer.record_run_cache_hit_rate(
                    tenant_id=engine.config.tenant_id,
                    cache_read_tokens=cache_read,
                    prompt_tokens=input_tokens,
                    cache_breakpoint_count=len(cache_breakpoints_hint),
                )
            continue

        if delta.kind in (ProviderDeltaKind.text, ProviderDeltaKind.thinking):
            # Drop the redundant post-nudge META TEXT entirely: no
            # ``content_block_start`` / delta on the wire and no
            # ``append_text`` into ``text_buffer`` (so it never reaches
            # durable history nor ``result_preview``). Applies to TEXT only;
            # ``thinking`` still flows. Any open block is left as-is and is
            # closed by the next kind-transition / tool_use / finish handler,
            # so block-index bookkeeping stays valid. Tool calls on this same
            # turn (Write / AppendFile / Finalize) are unaffected — handled
            # by their own branches below.
            if suppress_meta_text and delta.kind is ProviderDeltaKind.text:
                continue
            if open_block_kind is not None and open_block_kind is not delta.kind:
                # Kind transition (thinking→text or text→thinking) — close
                # the open block and start a fresh one so every block stays
                # single-kind end-to-end, mirroring the per-kind buffers
                # the durable history keeps.
                for evt in _emit_pending_head(complete=True):
                    yield evt
                yield TurnEvent(
                    type=EventType.CONTENT_BLOCK_STOP,
                    run_id=engine.config.run_id,
                    payload={
                        "turn_id": engine.turn_id(),
                        "block_idx": block_idx,
                        "visibility": _content_block_visibility(
                            open_block_kind, tool_interleaved=tool_interleaved
                        ).value,
                    },
                )
                open_block_kind = None
                block_idx = engine.next_block_idx()
            if open_block_kind is None:
                # If the most recently allocated block was a tool
                # (``tool_use_start`` happened and ``tool_use_stop`` has not
                # yet arrived), the current ``block_idx`` belongs to that
                # tool block. Allocate a fresh idx for this content block so
                # the emitted ``content_block_start`` does not collide with
                # the tool block's idx on the wire. Mirrors the
                # ``tool_use_stop`` advance below for the post-stop reopen
                # case.
                if last_block_was_tool:
                    block_idx = engine.next_block_idx()
                # Say where this block belongs the moment it opens, so a live
                # reader can place it without waiting for the run to end and
                # without inspecting its text. ``public`` unless the model has
                # already committed to a tool call in this same message — see
                # :func:`_content_block_visibility` for why nothing stronger is
                # knowable yet and why the matching stop re-states it.
                yield TurnEvent(
                    type=EventType.CONTENT_BLOCK_START,
                    run_id=engine.config.run_id,
                    payload={
                        "turn_id": engine.turn_id(),
                        "block_idx": block_idx,
                        "kind": delta.kind.value,
                        "visibility": _content_block_visibility(
                            delta.kind, tool_interleaved=tool_interleaved
                        ).value,
                    },
                )
                open_block_kind = delta.kind
                # Arm the narration buffer on the block that STARTS this
                # message's prose. Only that one: the cut point is recorded as
                # an offset into ``text_buffer``, which is exactly this block's
                # offset while the buffer is still empty, and a later text
                # block in the same message shares a row with the tool call
                # between them and is already collapsed by ``tool_interleaved``.
                if head_scan_available and delta.kind is ProviderDeltaKind.text:
                    head_buffer = ""
                    head_scan_available = False
            if delta.kind is ProviderDeltaKind.text:
                # Accumulate BEFORE anything is emitted, so the durable text is
                # whole no matter what the wire-side buffer does with it.
                result.append_text(delta.content or "")
                if head_buffer is not None:
                    head_buffer += delta.content or ""
                    for evt in _emit_pending_head(complete=False):
                        yield evt
                    continue
            elif delta.kind is ProviderDeltaKind.thinking:
                # Accumulate reasoning_content: a turn that emits ONLY
                # thinking with no text/tool_calls is the "thinking-tokens
                # trap" — the recovery branch in
                # _stream_one_assistant_message injects a continue-prompt
                # and re-streams. List append + join-on-read instead of
                # O(n^2) ``str +=``.
                result.append_reasoning(delta.content or "")
            for evt in delta_to_turn_events(
                delta,
                run_id=engine.config.run_id,
                turn_id=engine.turn_id(),
                block_idx=block_idx,
            ):
                yield evt
            continue

        if delta.kind is ProviderDeltaKind.tool_use_start:
            # Flip the interleaving flag BEFORE closing the open block. This is
            # the moment the ambiguity the stream started with resolves: text
            # that had to open ``public`` because nothing yet proved otherwise
            # is now proven to be narration, and its ``content_block_stop`` is
            # the first frame able to say so. Doing this after the stop would
            # publish the stale value and leave the live view disagreeing with
            # the durable transcript for the rest of the run.
            tool_interleaved = tool_interleaved or _tool_call_continues_run(
                engine, delta.tool_name
            )
            if open_block_kind is not None:
                for evt in _emit_pending_head(complete=True):
                    yield evt
                yield TurnEvent(
                    type=EventType.CONTENT_BLOCK_STOP,
                    run_id=engine.config.run_id,
                    payload={
                        "turn_id": engine.turn_id(),
                        "block_idx": block_idx,
                        "visibility": _content_block_visibility(
                            open_block_kind, tool_interleaved=tool_interleaved
                        ).value,
                    },
                )
                open_block_kind = None
            block_idx = engine.next_block_idx()
            # Mark the freshly-allocated block as a tool block so a
            # subsequent text/thinking reopen path (open_block_kind is None,
            # last allocation was a tool) advances to a fresh idx instead of
            # reusing this one.
            last_block_was_tool = True
            if delta.tool_call_id and delta.tool_name:
                engine.remember_tool_name(delta.tool_call_id, delta.tool_name)
            for evt in delta_to_turn_events(
                delta,
                run_id=engine.config.run_id,
                turn_id=engine.turn_id(),
                block_idx=block_idx,
            ):
                yield evt
            continue

        if delta.kind is ProviderDeltaKind.tool_use_input:
            for evt in delta_to_turn_events(
                delta,
                run_id=engine.config.run_id,
                turn_id=engine.turn_id(),
                block_idx=block_idx,
            ):
                yield evt
            continue

        if delta.kind is ProviderDeltaKind.tool_use_stop:
            for evt in delta_to_turn_events(
                delta,
                run_id=engine.config.run_id,
                turn_id=engine.turn_id(),
                block_idx=block_idx,
            ):
                yield evt
            if delta.tool_call_id:
                # Propagate the parser's truncation signal onto the
                # :class:`ToolCall` so the recovery branch in
                # :func:`_stream_one_assistant_message` can distinguish
                # "the model emitted a complete tool call then stopped" from
                # "the model ran out of output tokens mid-tool-call args".
                # Default is ``False``; only set by the vLLM SSE parser on
                # its synthetic ``tool_use_stop`` emitted from the
                # ``finish_reason="length"`` branch.
                #
                # ``args_partial_truncated`` additionally surfaces every
                # wire-level truncation signature (stage-4 brace balancing
                # fired) regardless of ``finish_reason``. The detection
                # branch in :func:`_stream_one_assistant_message` keys off
                # this flag when ``finish_reason="stop"`` arrives mid-call.
                result.tool_calls.append(
                    ToolCall(
                        id=delta.tool_call_id,
                        name=engine.tool_name_for(delta.tool_call_id),
                        arguments=delta.tool_input_final or {},
                        truncated_by_output_cap=delta.truncated_by_output_cap,
                        args_partial_truncated=delta.args_partial_truncated,
                    )
                )
            # Clear the tool-block flag and allocate a fresh
            # ``block_idx`` for any subsequent content block. The common
            # "tool then finish" path pays no wire cost (``finish`` closes
            # whatever block is open), but a post-stop text/thinking delta
            # would otherwise reopen on the tool's idx — second-order
            # block-model violation.
            last_block_was_tool = False
            block_idx = engine.next_block_idx()
            continue

    # Close a still-open content block on EVERY non-``finish`` exit. The
    # ``finish``-delta branch above closes its block inline before breaking;
    # the two OTHER exits do not: (1) the ``stop_requested`` break at
    # the top of the loop and (2) a clean iterator exhaustion with NO finish
    # delta (the OpenRouter SSE tail-loss shape — ``data: [DONE]`` / EOF).
    # Without this, the open ``content_block_start`` has no matching
    # ``content_block_stop`` and the chat reducer is left with a dangling block.
    # Idempotent: a no-op when the finish branch already closed the block.
    if open_block_kind is not None:
        for evt in _emit_pending_head(complete=True):
            yield evt
        yield TurnEvent(
            type=EventType.CONTENT_BLOCK_STOP,
            run_id=engine.config.run_id,
            payload={
                "turn_id": engine.turn_id(),
                "block_idx": block_idx,
                "visibility": _content_block_visibility(
                    open_block_kind, tool_interleaved=tool_interleaved
                ).value,
            },
        )
        open_block_kind = None


async def _handle_context_window_exceeded(
    engine: QueryEngine,
    exc: LLMContextWindowExceeded,
) -> AsyncIterator[TurnEvent]:
    """Recover from a context-window overflow — force_compaction once and
    re-stream, else terminal.

    Tracked via ``engine._compaction_attempted_for_current_turn``: a second
    overflow within the same message drives terminal FAILED.
    """
    if engine._compaction_attempted_for_current_turn:
        # Already retried — go terminal LLM error.
        # Death-spiral guard via _emit_llm_terminal.
        async for evt in _emit_llm_terminal(engine, exc, kind="llm_context_window_exceeded"):
            yield evt
        return

    engine._compaction_attempted_for_current_turn = True
    from_state = engine.state
    engine.transition_to(LoopState.COMPACTING)
    yield _emit_state_change(engine, from_state, LoopState.COMPACTING, reason="reactive_413")

    from protocore.runtime.context.budgets import derive_budgets
    from protocore.runtime.context.manager import estimate_history_tokens

    rc = engine.context_manager._rc
    budgets = derive_budgets(rc)
    tokens_before_value = estimate_history_tokens(engine.history, rc)
    yield TurnEvent(
        type=EventType.COMPACTION_STARTED,
        run_id=engine.config.run_id,
        payload={
            "reason": "reactive_413",
            "tokens_before": tokens_before_value,
            "trigger_threshold": budgets.compaction_trigger_tokens,
            "history_messages": len(engine.history),
            "holds_settled": bool(engine.config.rc.run_settled_enabled),
        },
    )

    try:
        attempt = await engine.context_manager.force_compaction(
            history=engine.history,
            compaction_state=engine.compaction_state,
            tenant_id=engine.config.tenant_id,
            model_name=engine.config.model_name,
            observability=_observability_context(
                engine,
                call_purpose="structured",
                call_category="compaction",
            ),
        )
    except CompactionExhaustedError as inner_exc:
        # Death-spiral guard — set BEFORE the state transition.
        engine.skip_terminal_hooks = engine.config.rc.skip_terminal_hooks_on_llm_error
        compacting_from = engine.state
        engine.transition_to(LoopState.FAILED)
        yield _emit_state_change(
            engine,
            compacting_from,
            LoopState.FAILED,
            reason="reactive_413_compaction_exhausted",
        )
        yield TurnEvent(
            type=EventType.ERROR,
            run_id=engine.config.run_id,
            payload={
                "kind": "llm_context_window_exceeded",
                "message": str(inner_exc),
                "primary_error": str(exc),
            },
        )
        yield TurnEvent(
            type=EventType.MESSAGE_STOP,
            run_id=engine.config.run_id,
            payload={
                "turn_id": engine.turn_id(),
                "stop_reason": StopReason.error.value,
            },
        )
        return

    yield TurnEvent(
        type=EventType.COMPACTION_COMPLETED,
        run_id=engine.config.run_id,
        payload={
            "reason": "reactive_413",
            "tokens_before": attempt.tokens_before,
            "tokens_after": attempt.tokens_after,
            "tier1_freed": attempt.tier1.tokens_freed if attempt.tier1 else 0,
            "tier2_summarised": attempt.tier2.turns_summarised if attempt.tier2 else 0,
            "blob_refs_created": (list(attempt.tier1.blob_refs_created) if attempt.tier1 else []),
        },
    )
    # This path calls force_compaction directly (not via _run_compaction), so
    # clear the stale prompt-size floor here too — the pre-compaction history it
    # described no longer exists, and leaving it set would drive one spurious
    # compaction at the next turn-start before the next LLM call self-heals it.
    engine.last_observed_prompt_tokens = 0
    await engine._persist_snapshot()
    compacting_from = engine.state
    engine.transition_to(LoopState.RUNNING)
    yield _emit_state_change(
        engine,
        compacting_from,
        LoopState.RUNNING,
        reason="reactive_413_compaction_completed",
    )


async def _iter_with_idle_watchdog(
    source: AsyncIterator[ProviderDelta],
    *,
    idle_timeout: float,
    stall_threshold: float,
    reasoning_idle_timeout: float | None = None,
) -> AsyncIterator[ProviderDelta]:
    """Wrap ``source`` with a per-iteration idle/stall watchdog.

 Each ``__anext__`` is awaited with :func:`asyncio.wait_for` using
 ``idle_timeout`` seconds. On timeout the upstream is presumed hung
 and :class:`LLMStreamIdleError` is raised (the caller maps this to
 the terminal LLM error path).

 A best-effort warning is logged when a post-first-delta inter-delta gap
 crosses ``stall_threshold`` seconds (independent of the hard timeout);
 this surfaces as ``DIAG`` telemetry without aborting the run. A stream
 that recovers stays alive but is still observable in logs.

 Reasoning-aware extension: when the upstream has emitted at least one
 :class:`ProviderDeltaKind.thinking` delta and ``reasoning_idle_timeout``
 is set, every subsequent ``__anext__`` waits up to
 ``reasoning_idle_timeout`` seconds instead of ``idle_timeout``. Once a
 non-reasoning delta arrives (visible text, tool call, finish, usage)
 the budget reverts to the baseline. This catches the
 smoke-test-result-r3 stall pattern where Qwen3-235B over OpenRouter
 sat silent for ~90 s during second-turn reasoning before producing
 any visible token; the legacy 90 s hard cap aborted the stream before
 the model could finish thinking. The error message includes which
 threshold was active so operators can distinguish a true hang from a
 long reasoning gap when the timeout DOES eventually fire.
 """
    iterator = source.__aiter__()
    loop = asyncio.get_event_loop()
    last_delta_at: float | None = None
    in_reasoning_window = False

    while True:
        # Per-iteration budget — reasoning-extended only after a thinking
        # delta has been observed AND only until the next non-reasoning
        # delta lands. This keeps the baseline tight for normal
        # non-reasoning streams.
        active_timeout = (
            reasoning_idle_timeout
            if (in_reasoning_window and reasoning_idle_timeout is not None)
            else idle_timeout
        )
        try:
            delta = await asyncio.wait_for(
                iterator.__anext__(), timeout=active_timeout
            )
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            kind_hint = "reasoning" if in_reasoning_window else "baseline"
            raise LLMStreamIdleError(
                f"LLM stream idle for >{active_timeout:.1f}s "
                f"(window={kind_hint}, last_delta_seen=thinking={in_reasoning_window}) "
                "— terminating"
            ) from exc

        now = loop.time()
        gap = None if last_delta_at is None else now - last_delta_at
        if gap is not None and gap > stall_threshold:
            _logger.warning(
                "DIAG llm_stream.stall gap_s=%.1f stall_threshold_s=%.1f "
                "active_timeout_s=%.1f window=%s",
                gap,
                stall_threshold,
                active_timeout,
                "reasoning" if in_reasoning_window else "baseline",
            )
        # Update the reasoning-window flag AFTER observing the delta so
        # the NEXT ``wait_for`` honours the new state. A thinking delta
        # opens the window; visible output / tool call / finish close it.
        # Transport-only progress heartbeats are intentionally transparent:
        # they reset the idle clock but must not collapse the longer
        # reasoning-aware window while the provider is still thinking.
        if delta.kind is ProviderDeltaKind.thinking:
            if not in_reasoning_window:
                _logger.warning(
                    "DIAG llm_stream.reasoning_window_open "
                    "idle_timeout_s=%.1f reasoning_idle_timeout_s=%s",
                    idle_timeout,
                    reasoning_idle_timeout,
                )
            in_reasoning_window = True
        elif delta.kind is ProviderDeltaKind.progress:
            pass
        elif in_reasoning_window:
            _logger.warning(
                "DIAG llm_stream.reasoning_window_close delta_kind=%s",
                delta.kind.value,
            )
            in_reasoning_window = False
        last_delta_at = now
        yield delta


async def _emit_llm_terminal(
    engine: QueryEngine,
    exc: BaseException,
    *,
    kind: str,
) -> AsyncIterator[TurnEvent]:
    """Drive the run terminal on an LLM-provider class error.

 Used on provider timeout (PTL post-retry), fallback-exhausted /
 max-output-exhausted, and idle watchdog paths. Emits the
 state_changed → FAILED → error → message_stop sequence.

 **Death-spiral guard** — when the terminal cause is an LLM-provider
 class error (LLMProviderError, LLMStreamIdleError,
 LLMContextWindowExceeded after retry, MaxOutputTokensExhausted)
 Stop / SessionEnd hooks MUST be skipped to prevent broken-provider
 runs from cascading through error-only hooks. The guard is engaged
 by setting ``engine.skip_terminal_hooks = True`` BEFORE the state
 transition so any synchronous downstream consumer (engine.run
 finally-block, the host SessionEnd dispatcher, hook-manager
 invoke) sees a consistent value.

 The guard is opt-out via
 :attr:`RuntimeConstants.skip_terminal_hooks_on_llm_error` for
 diagnostic deployments where Stop hooks SHOULD see the LLM error.

 The classifier verdict (when present) is surfaced in the ERROR event
 payload as ``classified_reason``. LLM adapters (Anthropic / OpenAI)
 attach a ``ClassifiedError`` to every raised :class:`LLMError` subclass
 via the dynamic ``classified`` attribute (FailoverReason taxonomy).
 Surfacing the reason on the bus lets the host-side
 ``RecoveryDispatcher`` route the failure to the right recovery action
 (compaction, fallback, backoff, terminate) without re-classifying. Core
 does not own the dispatch policy — it only forwards the verdict
 downstream.

 **A crash of this process is not a failure of the upstream.** Every
 caller reaches this function with a ``kind`` describing an LLM-class
 failure, and the loop's catch-all reaches it with whatever escaped the
 stream — a parser bug, a ``RecursionError``, an ``AttributeError``.
 Those used to be recorded as a PROVIDER error, which makes the
 provider-failure metric count our own bugs and points every subsequent
 investigation at the wrong system. So the ``kind`` is decided HERE, from
 the exception's own type, and anything outside the
 :class:`~protocore.contracts.llm.LLMError` family is reported as
 :data:`INTERNAL_ERROR_KIND` no matter what the call site asked for. The
 classification lives at the single point every terminal passes through,
 because a call site that has to remember to classify is a call site that
 will eventually forget.

 The exception is logged WITH ITS TRACEBACK. Without it the whole record
 of a crash is one line naming a kind and a message, and the place the
 run actually died is not recoverable from anything the system kept.
 """
    # A typed LLM failure keeps the caller's ``kind`` (the taxonomy the
    # recovery dispatcher routes on); anything else is this process crashing.
    if not isinstance(exc, LLMError):
        kind = INTERNAL_ERROR_KIND
        _logger.warning(
            "DIAG query.internal_error run=%s tenant=%s turn=%s "
            "exception=%s message=%s",
            engine.config.run_id,
            engine.config.tenant_id,
            engine.turn_id(),
            type(exc).__name__,
            exc,
            exc_info=exc,
        )
    # pair any already-emitted tool_use that never received a
    # result before driving the FAILED terminal. A query error can throw
    # after the assistant tool_use turn was appended to history but before
    # its dispatch ran; the engine.run() finally persists this snapshot and a
    # a resume must not replay a dangling tool_use.
    _synthesize_missing_tool_results(
        engine.history,
        error_content=engine.config.rc.tool_result_interrupted_placeholder,
    )
    from_state = engine.state
    # Engage the death-spiral guard BEFORE the state transition. It exists to
    # keep a BROKEN-PROVIDER run from cascading through error-only Stop /
    # SessionEnd hooks; an internal crash is not that, and suppressing the hooks
    # there would hide the teardown of the one failure class a deployment most
    # wants to observe.
    engine.skip_terminal_hooks = (
        kind != INTERNAL_ERROR_KIND
        and engine.config.rc.skip_terminal_hooks_on_llm_error
    )
    engine.transition_to(LoopState.FAILED)
    yield _emit_state_change(
        engine,
        from_state,
        LoopState.FAILED,
        reason=kind,
    )
    # Surface the classifier verdict downstream.
    error_payload: dict[str, object] = {"kind": kind, "message": str(exc)}
    classified = getattr(exc, "classified", None)
    if classified is not None:
        reason = getattr(classified, "reason", None)
        if reason is not None:
            # ``FailoverReason`` is a ``StrEnum`` — its ``.value`` is the
            # canonical wire string. Tolerate plain-string for forward
            # compatibility (e.g. tests stubbing a classified verdict).
            error_payload["classified_reason"] = getattr(reason, "value", reason)
        retryable = getattr(classified, "retryable", None)
        if retryable is not None:
            error_payload["retryable"] = bool(retryable)
        should_compress = getattr(classified, "should_compress", None)
        if should_compress is not None:
            error_payload["should_compress"] = bool(should_compress)
        should_fallback = getattr(classified, "should_fallback", None)
        if should_fallback is not None:
            error_payload["should_fallback"] = bool(should_fallback)
    yield TurnEvent(
        type=EventType.ERROR,
        run_id=engine.config.run_id,
        payload=error_payload,
    )
    yield TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "stop_reason": StopReason.error.value,
        },
    )


async def _rebuild_context_for_recovery(
    engine: QueryEngine,
) -> ContextBundle:
    """Rebuild :class:`ContextBundle` after a recovery step mutated history.

    Used by the reactive-413 (freshly compacted history) and max-output
    recovery (synthetic resume-prompt user message) paths. The skill catalog
    block is the run's once-built value (NOT rebuilt here) so a recovery
    rebuild cannot bust the cached system-prompt prefix mid-run.
    """
    tool_defs = list(
        engine.tools.compute_effective_surface(
            tenant_id=engine.config.tenant_id,
            policy=engine.effective_tool_policy,
            query=engine.latest_user_message.text if engine.latest_user_message else "",
            top_k=engine.config.rc.tool_retrieval_top_k,
        )
    )
    recovery_history, _ = _llm_history(engine)
    return engine.context_manager.build_context(
        history=recovery_history,
        tools=tool_defs,
        system_prompt_sections=engine.config.system_prompt_sections,
        skill_index_block=await _ensure_run_skill_catalog(engine),
        skills_loaded=engine._skill_loaded_bundles,
    )


def _is_parallel_safe_tool(
    engine: QueryEngine,
    tool_call: ToolCall,
    hook_match_predicate: Callable[[str], bool] | None = None,
) -> bool:
    """Return ``True`` iff ``tool_call`` can run concurrently with siblings.

 Eligibility predicate for the parallel-dispatch branch in
 :func:`_stream_one_assistant_message`. A tool is parallel-safe only
 when ALL of these conditions hold:

 * The registry has a tool registered under ``tool_call.name`` (so we
 can read its static metadata; missing tools must fall through to
 the serial dispatcher which produces the canonical "unknown tool"
 error envelope).
 * ``tool.is_concurrent_safe is True`` — adapter explicitly opts in.
 * ``tool.is_destructive is False`` — destructive ops (writes,
 deletes, ``pcm_answer``) MUST serialise so the LLM sees a stable
 causal order between read and mutate.
 * No enabled ``PreToolUse`` hook for this tenant could match the
 tool name. Hooks can mark ANY
 tool ``requires_approval`` via
 :meth:`ToolPermissionGate._project_hook_result`; if such a tool
 ran under :func:`asyncio.gather` the serial path's web-mode
 approval downgrade (``query.py:_dispatch_tool``) would be
 bypassed AND a multi-call batch could leak multiple pending
 approvals from one turn. Steering any hook-matchable tool back
 onto the serial path keeps both invariants intact.

 The combination guarantees no approval-gate surface either: the
 safety policy only requires approval on destructive/sandbox tools
 (which the destructive predicate already excludes), so a tool that
 is concurrent-safe, non-destructive, AND outside every PreToolUse
 hook matcher cannot yield :class:`EventType.TOOL_CALL_PENDING`. The
 parallel orchestrator relies on this invariant — the in-batch
 approval handler is a defensive fallback for the narrow race where
 a hook is registered mid-turn between the predicate snapshot and
 the dispatch.

 Notes:

 * The predicate uses ``getattr(..., default)`` so it tolerates
 registries that pre-date the parallel-dispatch contract (tests +
 legacy adapters that never set the ClassVar default to ``False``,
 which is the conservative serial behaviour).
 * ``hook_match_predicate=None`` is the legacy single-argument contract
 used by the existing unit tests: it skips the hook check and assumes no
 hook could match. Production callers in
 :func:`_stream_one_assistant_message` ALWAYS pass the predicate.
 """
    tool = engine.tools.get(tool_call.name)
    if tool is None:
        return False
    if not bool(getattr(tool, "is_concurrent_safe", False)):
        return False
    if bool(getattr(tool, "is_destructive", False)):
        return False
    if hook_match_predicate is not None and hook_match_predicate(tool_call.name):
        return False
    return True


def _is_delegation_parallel_safe(
    engine: QueryEngine,
    tool_call: ToolCall,
    hook_match_predicate: Callable[[str], bool] | None = None,
) -> bool:
    """Return ``True`` iff ``tool_call`` is a fan-out-eligible delegation call.

    Distinct from :func:`_is_parallel_safe_tool` (which governs concurrent-safe
    READ tools). A delegation call — the subagent-dispatch tool — is deliberately
    NOT ``is_concurrent_safe`` (each child spawns a full nested run), so it is
    excluded from the read fan-out and keeps its serial-path safety wiring. This
    predicate instead identifies the delegation tool GENERICALLY via the
    ``is_parallel_delegation`` class flag (set only on the host dispatch
    tool) so core hardcodes no tool name, and permits fanning several ADJACENT
    delegation calls emitted in one assistant turn out under a bounded semaphore
    (see the delegation branch in :func:`_stream_one_assistant_message`).

    ``True`` only when ALL hold:

    * ``parallel_subagents_enabled`` — master gate (default on).
    * The registry has a tool under ``tool_call.name`` whose
      ``is_parallel_delegation`` flag is ``True``.
    * No enabled ``PreToolUse`` hook for this tenant could match the tool name
      (same predicate the read fan-out uses; a hook-gated delegation call MUST
      stay serial so its approval surface is honoured).

    The effective concurrency cap (``max_concurrent_subagents``) is applied by
    the caller: a cap resolving to ``< 2`` disables the fan-out so a single
    delegation call (or a cap of 1) runs on the exact serial path. Uses
    ``getattr(..., False)`` so a registry that pre-dates the delegation contract
    (legacy adapters / focused unit fixtures) conservatively stays serial.
    """
    if not engine.config.rc.parallel_subagents_enabled:
        return False
    if not _tool_is_delegation(engine, tool_call):
        return False
    if hook_match_predicate is not None and hook_match_predicate(tool_call.name):
        return False
    return True


def _tool_is_delegation(engine: QueryEngine, tool_call: ToolCall) -> bool:
    """Return ``True`` iff ``tool_call`` targets a delegation (subagent) tool.

    The RAW structural check — the ``is_parallel_delegation`` class flag on the
    registered tool — WITHOUT the ``parallel_subagents_enabled`` gate or the
    hook-steering check that :func:`_is_delegation_parallel_safe` layers on top.
    Those gates decide whether adjacent delegation calls FAN OUT concurrently;
    they do NOT change the fact that a delegation call blocks its caller on a full
    nested child run. A run holding a tree-budget slot must therefore release it
    around ANY such join — the concurrent gather AND a single hook-gated or
    gate-disabled serial dispatch — or it pins a slot while blocked on a
    descendant and the tree can wedge at the cap. ``getattr(..., False)`` keeps a
    registry that pre-dates the delegation contract conservatively non-delegation.
    """
    tool = engine.tools.get(tool_call.name)
    return tool is not None and bool(getattr(tool, "is_parallel_delegation", False))


# Operator subset needed by :func:`_hook_matchers_could_match_tool` —
# core does not import the host matcher module per the
# core-vs-the host import boundary (see
# ``protocore/tests/test_core_import_boundary.py``).
def _hook_matchers_could_match_tool(
    matchers: Mapping[str, Any],
    tool_name: str,
) -> bool:
    """Return True iff ``matchers`` could match a payload with this tool name.

    Mirrors the documented matcher subset a host's hook matcher applies to
    the ``tool_name`` field only. We are intentionally CONSERVATIVE — when the
    matcher
    references a field other than ``tool_name`` (e.g. ``tool_input.path``)
    we cannot know without invoking the dispatcher whether the payload
    would match, so we return ``True`` (treat as "could match"). This
    keeps the parallel-safe set strictly smaller in ambiguity, which is
    the right side to err on for the approval-gate invariant.

    Empty matchers ⇒ matches everything ⇒ returns True.

    Supported operators on ``tool_name``:
    ``$eq``, ``$ne``, ``$in``, ``$nin``, ``$exists``, ``$regex`` plus
    the scalar shorthand for ``$eq``. Unknown operators return True
    (conservative).
    """
    if not matchers:
        return True
    for key, spec in matchers.items():
        if key != "tool_name":
            # Matcher references a non-tool_name payload field — we
            # cannot evaluate it statically without running the full
            # dispatcher. Conservatively assume it could match.
            return True
        if isinstance(spec, dict):
            for op, expected in spec.items():
                if op == "$eq":
                    if tool_name != expected:
                        return False
                elif op == "$ne":
                    if tool_name == expected:
                        return False
                elif op == "$in":
                    if not isinstance(expected, list) or tool_name not in expected:
                        return False
                elif op == "$nin":
                    if isinstance(expected, list) and tool_name in expected:
                        return False
                elif op == "$exists":
                    # tool_name always exists in PreToolUse payloads.
                    if not bool(expected):
                        return False
                elif op == "$regex":
                    if not isinstance(expected, str):
                        return False
                    try:
                        pattern = re.compile(f"^(?:{expected})$")
                    except re.error:
                        return False
                    if not pattern.match(tool_name):
                        return False
                else:
                    # Unknown operator — conservative: could match.
                    return True
        else:
            # Scalar shorthand: implicit $eq.
            if tool_name != spec:
                return False
    return True


async def _pre_tool_use_match_predicate(
    engine: QueryEngine,
) -> Callable[[str], bool] | None:
    """Build a per-turn predicate: "does any PreToolUse hook match tool_name N?".

 Called ONCE per turn from :func:`_stream_one_assistant_message` BEFORE
 the dispatch batching loop walks the pending tool calls. Returns:

 * ``None`` when the engine has no hook manager (legacy test wiring
 or :class:`InMemoryHookManager` with no specs registered) — the
 caller falls back to the simple destructive-only predicate.
 * A predicate ``f(tool_name) -> bool`` that returns ``True`` when
 some enabled ``PreToolUse`` hook MIGHT match. Mid-turn hook
 registration is the only race window; the predicate's snapshot
 is intentionally taken once per turn so the dispatch-loop
 assumptions stay stable for that turn.

 Hook list failures (PG outage, ``IHookManager.list`` raising) are
 isolated like the rest of the hook subsystem: the predicate
 conservatively returns ``True`` for every tool, which restores serial
 dispatch behaviour for the turn. Better to lose parallelism than to
 break the approval contract.
 """
    hook_manager = getattr(engine, "hooks", None)
    if hook_manager is None:
        return None
    try:
        hooks = list(
            await hook_manager.list(
                engine.config.tenant_id, event=HookEvent.pre_tool_use
            )
        )
    except Exception:
        _logger.warning(
            "DIAG query.pre_tool_use_predicate.list_failed tenant=%s — "
            "falling back to serial dispatch for the turn",
            engine.config.tenant_id,
            exc_info=True,
        )
        # Conservative fallback: assume every tool could be hook-gated.
        return lambda _tool_name: True
    enabled_matchers: list[Mapping[str, Any]] = [
        getattr(h, "matchers", {}) or {} for h in hooks if getattr(h, "enabled", True)
    ]
    if not enabled_matchers:
        # No hooks registered → no tool can be hook-gated → predicate
        # returns False for every tool (parallelisation freely allowed
        # subject to the other predicate clauses).
        return lambda _tool_name: False

    def _predicate(tool_name: str) -> bool:
        for matchers in enabled_matchers:
            if _hook_matchers_could_match_tool(matchers, tool_name):
                return True
        return False

    return _predicate


async def _drain_dispatch_tool_deferred(
    engine: QueryEngine,
    tool_call: ToolCall,
    *,
    dispatch_order: int | None = None,
    dispatch_group: str | None = None,
    tree_permit: SubagentTreePermit | None = None,
) -> tuple[list[TurnEvent], DispatchOutcome | None]:
    """Run :func:`_dispatch_tool` but DEFER history append + persist.

 Used by the parallel-dispatch branch. The standard
 :func:`_dispatch_tool` appends a :class:`ToolResultBlock` to
 ``engine.history`` and persists a snapshot as its final side effect.
 Under :func:`asyncio.gather` the order of those appends is
 non-deterministic, which would break the LLM-facing invariant that
 tool results appear in the same order the model requested them.

 This helper drains the dispatcher into a buffer (returning the
 events list + the dispatcher's final :class:`DispatchOutcome`)
 WITHOUT touching ``engine.history``. The caller then iterates the
 parallel batch in the original ``ToolCall`` order and appends the
 result blocks sequentially via :func:`_apply_deferred_tool_history`.
 """
    # Mirror the serial dispatcher's terminal-only guard for the parallel
    # batch path. Build a synthetic deferred outcome (no dispatcher
    # invocation) so the caller's ``_apply_deferred_tool_history`` append
    # still emits the structured error blocked envelope.
    if _terminal_only_blocks(engine, tool_call):
        error_message = _terminal_only_error_message(engine, tool_call.name)
        synthetic_event = TurnEvent(
            type=EventType.TOOL_RESULT,
            run_id=engine.config.run_id,
            payload={
                "tool_call_id": tool_call.id,
                "success": False,
                "error": {
                    "kind": "terminal_only",
                    "message": error_message,
                },
                "content_blocks": [{"type": "text", "text": error_message}],
            },
        )
        synthetic_outcome = DispatchOutcome(
            tool_call=tool_call,
            success=False,
            content=error_message,
            is_error=True,
            error_kind=DispatchErrorKind.execution,
            metadata={},
        )
        return [synthetic_event], synthetic_outcome
    # Cumulative total-work guard, same synthetic shape as the terminal-only
    # veto above and for the same reason: no dispatcher invocation, so a tree
    # that has spent its delegation budget pays nothing more to be told so.
    refusal, refusal_reason = _run_work_delegation_refusal(engine, tool_call)
    if refusal:
        _logger.warning(
            "DIAG query.run_work_budget.delegation_refused run=%s tenant=%s "
            "tool=%s reason=%s %s",
            engine.config.run_id,
            engine.config.tenant_id,
            tool_call.name,
            refusal_reason,
            _resolve_run_work_ledger(engine).spent_summary(),
        )
        return _run_work_refusal_dispatch(
            engine, tool_call, refusal, refusal_reason
        )
    helpers = getattr(engine, "_helpers", None)
    metadata: dict[str, Any] = {}
    if helpers:
        # seed the per-run satisfied set from the durable
        # ``engine.history`` when the helper bag is fresh (cross-pod
        # re-drive). The dispatcher reads it from
        # ``ctx.metadata["protocore.helpers"]["tool_preconditions.satisfied"]``
        # so this MUST run before the dispatcher reads it on the
        # parallel branch.
        _rehydrate_satisfied_from_history(helpers, engine)
        metadata[_HELPERS_METADATA_KEY] = helpers
        # : skip runtime-internal keys so a forged operator
        # ``run_metadata`` cannot shadow ``tool_call_id`` / ``protocore.*``
        # on the parallel-dispatch path either.
        _merge_run_metadata_into(metadata, helpers)
    # Carry the child's LLM-requested batch position + its fan-out group id so
    # the host runner can declare its deliverables into the parent ledger in
    # batch order (not gather completion order), scoped per group so a later
    # group's declaration is never frozen out by an earlier one. Set AFTER the
    # run-metadata merge so a forged ``run_metadata`` cannot shadow them; absent
    # (serial/single dispatch) leaves declaration order at the pre-existing
    # last-writer-wins behaviour.
    if dispatch_order is not None:
        metadata[SUBAGENT_DISPATCH_ORDER_METADATA_KEY] = dispatch_order
    if dispatch_group is not None:
        metadata[SUBAGENT_DISPATCH_GROUP_METADATA_KEY] = dispatch_group
    # Carry THIS child's tree-budget permit handle (an in-memory object, not
    # serialized — like the cancel Event on the helper bag) so the host
    # runner can lodge it in the child's helper bag and the child engine can
    # release-while-awaiting around its own nested delegation gather. Set AFTER
    # the run-metadata merge so a forged ``run_metadata`` cannot shadow it; absent
    # (serial/single dispatch) means the child holds no tree slot.
    if tree_permit is not None:
        metadata[SUBAGENT_TREE_PERMIT_METADATA_KEY] = tree_permit
    ctx = ToolContext(
        tenant_id=engine.config.tenant_id,
        account_id=engine.config.account_id,
        run_id=engine.config.run_id,
        session_id=engine.config.session_id,
        evidence_origin=engine._engine_evidence_origin(),
        evidence_admission_deferred=True,
        metadata=metadata,
    )

    dispatcher = _ensure_tool_dispatcher(engine)
    events: list[TurnEvent] = []
    outcome: DispatchOutcome | None = None
    async for item in dispatcher.dispatch(
        tool_call=tool_call,
        ctx=ctx,
        # The effective policy carries the RC core tool-surface floor so the
        # gate permits exactly what was advertised (advertise/dispatch
        # parity; see ToolPermissionGate.check Stage-1 whitelist). Without
        # this, the parallel branch would compute ``allowed=visible|pinned|
        # forced_pinned`` from the raw policy and deny a tool that the
        # per-turn surface already advertised via the floor.
        visibility_policy=engine.effective_tool_policy,
        # The declared tool set of the agent driving THIS engine, when it
        # declared one. Empty declaration ⇒ ``None`` ⇒ the gate's allow-list
        # stage stays off, exactly as before it was wired.
        subagent_whitelist=engine.effective_subagent_tool_allowlist,
        timeout_seconds=engine.config.rc.tool_timeout_seconds,
        preapproved_tool_call_id=None,
        admit_evidence=lambda records, producer: engine.append_tool_evidence(
            records, producer=producer
        ),
    ):
        if isinstance(item, DispatchOutcome):
            outcome = item
            break
        events.append(item)
    return events, outcome


def _synthesize_delegation_error_result(
    engine: QueryEngine,
    tool_call: ToolCall,
    exc: BaseException,
) -> tuple[list[TurnEvent], DispatchOutcome]:
    """Build an error ``(events, outcome)`` pair for a delegation child that RAISED.

    The concurrent delegation branch gathers child dispatches with
    ``return_exceptions=True`` so ONE child raising an unexpected exception does
    not cancel its siblings. (The dispatcher normally converts subagent failures
    — unknown ``subagent_type``, hook block, timeout — into structured
    ``success=False`` outcomes, so a raised exception is the defensive edge.)
    This helper converts the exception into the same shape a normal dispatch
    yields — a ``TOOL_RESULT`` event plus a ``success=False`` ``DispatchOutcome``
    — so the raising child still contributes its OWN error ``ToolResultBlock`` in
    LLM-requested order and the leader loop continues with the successful
    siblings, matching the single-child error contract.
    """
    detail = str(exc).strip()
    message = f"subagent dispatch failed: {type(exc).__name__}"
    if detail:
        message = f"{message}: {detail[:500]}"
    event = TurnEvent(
        type=EventType.TOOL_RESULT,
        run_id=engine.config.run_id,
        payload={
            "tool_call_id": tool_call.id,
            "success": False,
            "error": {"kind": "execution", "message": message},
            "content_blocks": [{"type": "text", "text": message}],
        },
    )
    outcome = DispatchOutcome(
        tool_call=tool_call,
        success=False,
        content=message,
        is_error=True,
        error_kind=DispatchErrorKind.execution,
        metadata={},
    )
    return [event], outcome


def _resolve_subagent_tree_budget(engine: QueryEngine) -> SubagentTreeBudget:
    """Resolve the shared tree-wide subagent budget, minting it at first fan-out.

    The budget is ONE object per maximal parallel-dispatched subtree, shared by
    reference. When the helper bag already carries it (a run whose ancestor
    already minted it — the parent-helpers dict-copy propagates the SAME object
    the way ``cancel_event`` / ``root_run_id`` flow), return that. Otherwise mint
    from ``rc.max_concurrent_subagents_per_tree`` and store it back into the bag
    BEFORE any child is dispatched, so the very first ``dict(helpers)`` copy taken
    by the dispatch path carries it downward. The minting run is simply the first
    to reach a parallel fan-out without a budget in its bag — usually the root,
    but deeper if the root only ever delegates serially; either way, any two
    concurrently-executing runs share one budget (they branched at a common
    fan-out ancestor that minted it before dispatching them). When the engine has
    no helper bag (unit tests / degenerate callers) fall back to a local budget
    that still bounds THIS group but cannot propagate to descendants.
    """
    helpers = getattr(engine, "_helpers", None)
    if isinstance(helpers, dict):
        existing = helpers.get(HELPER_SUBAGENT_TREE_BUDGET_KEY)
        if isinstance(existing, SubagentTreeBudget):
            return existing
        budget = SubagentTreeBudget(
            engine.config.rc.max_concurrent_subagents_per_tree
        )
        helpers[HELPER_SUBAGENT_TREE_BUDGET_KEY] = budget
        return budget
    return SubagentTreeBudget(engine.config.rc.max_concurrent_subagents_per_tree)


def _resolve_run_work_ledger(engine: QueryEngine) -> RunWorkLedger:
    """Resolve the tree's CUMULATIVE work ledger, minting it if the bag has none.

    The live path finds one already there: the host composition root mints
    it for the root run when it builds the helper bag, and every descendant
    inherits that same object by reference. Minting here is the fallback for a
    caller that never composed a bag — it still bounds the caller's own subtree,
    which is the most a run with no shared bag can be held to.

    Unlike :func:`_resolve_subagent_tree_budget` this must NOT be minted lazily
    at the first parallel fan-out. A leader that emits one delegation call per
    turn never fans out, and that serial wave-after-wave shape is precisely the
    one an instantaneous concurrency cap cannot see.
    """
    return resolve_run_work_ledger(getattr(engine, "_helpers", None), engine.config.rc)


def _charge_run_work_tokens(
    engine: QueryEngine, *, input_tokens: int, output_tokens: int
) -> None:
    """Fold one LLM call's usage into the tree's cumulative token total.

    Called from every engine in the tree against the SHARED ledger. Deliberately
    resolves (and therefore mints) rather than reading best-effort: the very
    first LLM call of the root run happens BEFORE any delegation, and a ledger
    that only came into existence at the first dispatch would start the tree's
    token count part-way through.
    """
    _resolve_run_work_ledger(engine).charge_tokens(
        input_tokens=input_tokens, output_tokens=output_tokens
    )


def _run_work_delegation_refusal(
    engine: QueryEngine, tool_call: ToolCall
) -> tuple[str, str]:
    """``(message, reason)`` for a delegation the tree cannot afford, or ``("","")``.

    Empty for every non-delegation tool (the budget bounds delegation, not the
    leader's own work — a leader must always be able to finish its answer) and
    for a tree with budget left. The reason token comes back alongside the text
    so the caller can stamp it on the outcome without asking the ledger twice.

    The text is written for the model that is about to read it, and it has one
    job: make a RETRY look pointless. A leader that reads a refusal as transient
    spends its remaining turns re-issuing the same call and answers with nothing,
    which is a worse outcome than the unbounded delegation this bound replaces.
    So it states that the budget is cumulative, that it does not refill, that no
    further delegation in this run will be accepted, and what to do instead.
    """
    if not _tool_is_delegation(engine, tool_call):
        return "", ""
    ledger = _resolve_run_work_ledger(engine)
    reason = ledger.delegation_refusal_reason()
    if not reason:
        return "", ""
    return (
        f"Delegation budget exhausted ({reason}: {ledger.spent_summary()}). "
        "This budget is cumulative over the whole run and does NOT refill — no "
        "further delegation will be accepted, now or on any later turn, and "
        "retrying this call will fail identically. Any subagents already running "
        "will still return. Finalize your answer now from the results you "
        "already have; if something is missing, say what is missing rather than "
        "delegating again."
    ), reason


def _run_work_refusal_dispatch(
    engine: QueryEngine, tool_call: ToolCall, message: str, reason: str
) -> tuple[list[TurnEvent], DispatchOutcome]:
    """Build the synthetic ``(events, outcome)`` pair for a refused delegation.

    Refused BEFORE the dispatcher runs, so an exhausted tree spends nothing —
    not a tool invocation, not a child run — to discover it is exhausted. Shaped
    exactly like the terminal-only veto next door: a ``TOOL_RESULT`` event plus
    an ``is_error`` outcome, so the leader sees an ordinary failed tool result in
    LLM-requested order and its loop continues normally.
    """
    event = TurnEvent(
        type=EventType.TOOL_RESULT,
        run_id=engine.config.run_id,
        payload={
            "tool_call_id": tool_call.id,
            "success": False,
            "error": {"kind": "execution", "message": message},
            "content_blocks": [{"type": "text", "text": message}],
        },
    )
    outcome = DispatchOutcome(
        tool_call=tool_call,
        success=False,
        content=message,
        is_error=True,
        error_kind=DispatchErrorKind.execution,
        metadata={
            DISPATCH_STRUCTURED_ERROR_METADATA_KEY: {
                STRUCTURED_ERROR_FINALIZATION_RECOMMENDED_KEY: True,
                STRUCTURED_ERROR_REASON_KEY: reason,
            }
        },
    )
    return [event], outcome


def _resolve_subagent_tree_permit(engine: QueryEngine) -> SubagentTreePermit | None:
    """Return THIS run's own tree permit from the helper bag, or None.

    Present only for a run that was itself dispatched under the budget (
    the host runner lodges the child's permit in its fresh helper bag). The root
    leader — which was never dispatched as a subagent — holds none, so it returns
    None and neither releases nor reacquires a tree slot while awaiting children.
    """
    helpers = getattr(engine, "_helpers", None)
    if isinstance(helpers, dict):
        permit = helpers.get(HELPER_SUBAGENT_TREE_PERMIT_KEY)
        if isinstance(permit, SubagentTreePermit):
            return permit
    return None


async def _dispatch_subagent_under_semaphore(
    engine: QueryEngine,
    tool_call: ToolCall,
    semaphore: asyncio.Semaphore,
    budget: SubagentTreeBudget,
    *,
    dispatch_order: int | None = None,
    dispatch_group: str | None = None,
) -> tuple[list[TurnEvent], DispatchOutcome | None]:
    """Drain one deferred delegation dispatch while holding ``semaphore``.

    The concurrent delegation branch gathers these so at most
    ``max_concurrent_subagents`` children execute at once; the rest wait on the
    semaphore and run in waves. History append stays deferred (see
    :func:`_drain_dispatch_tool_deferred`) so the caller can append results in
    LLM-requested order regardless of completion order. ``dispatch_order`` is the
    child's 0-based position in the LLM-requested batch and ``dispatch_group`` the
    fan-out group's stable id, forwarded so parent-ledger declarations resolve in
    batch order (scoped per group) rather than completion order.

    Beyond the local per-group ``semaphore`` (which bounds this group's width),
    each child also draws ONE slot from the tree-wide ``budget`` (which bounds the
    additive sum across all nested groups). The tree slot is acquired here, on the
    child's behalf, only AFTER the local semaphore is held — so a wave waiting for
    group width never sits on a scarce tree slot. The child's permit handle is
    threaded onto its dispatch metadata so the child engine can
    release-while-awaiting around its OWN nested gather, and the parent releases
    the slot once in ``finally`` after the child run returns. Acquisition order
    (local width THEN tree slot) is uniform across every dispatch site, so the two
    bounds cannot themselves deadlock against each other.
    """
    async with semaphore:
        permit = await budget.acquire()
        try:
            return await _drain_dispatch_tool_deferred(
                engine,
                tool_call,
                dispatch_order=dispatch_order,
                dispatch_group=dispatch_group,
                tree_permit=permit,
            )
        finally:
            await permit.release()


# — the finalize-hint ``reason`` is provider-VISIBLE (it is appended to
# the tool-result content the model sees), so an UNTRUSTED future producer must
# never leak internal/tenant data or a huge string into the prompt through it.
# A reason token is echoed ONLY when it is a short, machine-token-shaped string;
# anything else is dropped from the model-visible line (the full reason still
# lives verbatim in ``outcome.metadata`` / logs). The signal
# ``transport_retry_budget_exhausted`` is a clean snake_case token and passes.
_FINALIZATION_REASON_MAX_LEN: Final[int] = 64
_FINALIZATION_REASON_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"\A[A-Za-z0-9][A-Za-z0-9 _.-]*\Z"
)


def _safe_finalization_reason_suffix(reason: object) -> str:
    """Return a bounded ``(reason: …)`` suffix, or ``""`` when not safe to echo.

 sanitiser: only a string that is non-empty, within
 :data:`_FINALIZATION_REASON_MAX_LEN`, and matches the machine-token allowlist
 :data:`_FINALIZATION_REASON_TOKEN_RE` (alnum + ``_ - . space``, no control
 chars / newlines / structural punctuation) is echoed into the provider-
 visible hint. Any other value (non-string, over-length, arbitrary prose,
 injected markup) yields ``""`` so it is dropped from the model-visible line.
 The unsanitised reason is still available in ``outcome.metadata`` and logs.
 """
    if not isinstance(reason, str):
        return ""
    if not reason or len(reason) > _FINALIZATION_REASON_MAX_LEN:
        return ""
    if not _FINALIZATION_REASON_TOKEN_RE.fullmatch(reason):
        return ""
    return f" (reason: {reason})"


def _tool_result_content_with_finalization_hint(outcome: DispatchOutcome) -> str:
    """Return the tool-result content, with a finalize hint when the tool gave up.

    A tool that exhausts its transport-retry budget raises with
    ``structured_error={"finalization_recommended": True, ...}``. The dispatch
    except-branch forwards that under
    :data:`DISPATCH_STRUCTURED_ERROR_METADATA_KEY` on
    :attr:`DispatchOutcome.metadata` and onto :attr:`ToolResultBlock.metadata`,
    but the OpenAI/vLLM wire serializer emits only
    ``{role, tool_call_id, content}`` — so the metadata signal NEVER reaches
    the model.

    Rather than change the serializer per-provider (and risk a fallback
    chain), surface a SANITIZED one-line hint in the tool-result ``content``
    itself, so it survives serialization for EVERY provider. The hint is a
    bounded budget-signal nudge appended ONCE after the existing error text.
    When the outcome carries no ``finalization_recommended`` structured error
    (the common case) the content is returned VERBATIM — bit-identical.

    Total — never raises; a malformed structured_error degrades to the raw
    content.
    """
    content = outcome.content
    metadata = outcome.metadata or {}
    structured_error = metadata.get(DISPATCH_STRUCTURED_ERROR_METADATA_KEY)
    if not isinstance(structured_error, dict):
        return content
    if structured_error.get(STRUCTURED_ERROR_FINALIZATION_RECOMMENDED_KEY) is not True:
        return content
    # : the reason is provider-visible — echo ONLY a bounded, token-shaped
    # value; anything untrusted/long/markup is dropped (full reason stays in
    # metadata/logs).
    reason_suffix = _safe_finalization_reason_suffix(
        structured_error.get(STRUCTURED_ERROR_REASON_KEY)
    )
    # Deliberately says nothing about WHICH budget ran out. Two producers reach
    # this line — a transport-retry give-up on a failing dependency, and a
    # delegation call refused because the run's cumulative work budget is spent —
    # and a sentence naming either one is a false statement about the other. The
    # reason token disambiguates, and the tool's own error text above it carries
    # the specifics.
    hint = (
        "[finalization-recommended] This tool gave up: a budget it depends on is "
        "exhausted" + reason_suffix + ". Do NOT keep retrying the same call — "
        "finalize your answer now on the best evidence already gathered."
    )
    # Append after the existing (sanitised) error text; keep one blank-line sep
    # when there is prior content so the hint reads as its own line.
    return f"{content}\n\n{hint}" if content else hint


def _apply_deferred_tool_history(
    engine: QueryEngine,
    tool_call: ToolCall,
    outcome: DispatchOutcome,
    *,
    track_circuit_breaker: bool = True,
) -> None:
    """Append the deferred :class:`ToolResultBlock` + forget the call id.

 Mirrors the tail of :func:`_dispatch_tool` (history
 append + ``forget_tool_name``) without the ``_persist_snapshot``
 await so the caller can batch multiple appends and persist once.

 ``track_circuit_breaker``: the breaker is skipped when
 ``False``. The parallel-batch caller passes ``False`` for a TERMINAL-ONLY
 finalize-gate veto (a SYNTHETIC ``is_error`` outcome produced WITHOUT a
 dispatcher invocation), exactly as the serial ``_dispatch_tool`` returns
 BEFORE the post-dispatch breaker for that case. Counting a finalize-gate
 veto as a hard tool failure would let the breaker inject a corrective turn
 DURING the finalize-background gate (meta-leak violation).
 """
    # Parity with the serial path: a SUCCESSFUL chunkable write marks the
    # path "chunking started".
    if not outcome.is_error:
        _record_chunk_write_success(engine, tool_call)
    # Parity with the serial path: observe byte production
    # (Write/AppendFile are serialised, so this rarely carries a mutation,
    # but the observation must be identical on both dispatch paths).
    _longfile.observe_tool_result(
        engine, tool_call, outcome.content, is_error=outcome.is_error
    )
    # Parity with the serial path: fold the result into run-level tool-
    # precondition progress. A SUCCESSFUL call advances the entry whether the
    # model was forced into it or reached for the tool on its own — the
    # contract is that the tool ran.
    _preconditions.observe_tool_result(
        engine, tool_call, outcome.content, is_error=outcome.is_error
    )
    # Parity with the serial path: fold the result into the declared-file
    # read-back gate — a result declaring files the caller must open engages
    # it, a successful read releases what it opened.
    _pending_reads.observe_tool_result(
        engine, tool_call, outcome.metadata, is_error=outcome.is_error
    )
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(
                    tool_call_id=tool_call.id,
                    content=_tool_result_content_with_finalization_hint(outcome),
                    is_error=outcome.is_error,
                    metadata=outcome.metadata or {},
                )
            ],
        )
    )
    # Parity with the serial path: track the consecutive same-tool/
    # same-error-class streak and, on a trip, append the bounded corrective
    # convergence turn (the caller persists once after the batch). Appended
    # AFTER this call's result block so ordering stays valid. SKIPPED for a
    # terminal-only finalize-gate veto (``track_circuit_breaker=False``) so the
    # gate cannot be misread as a hard tool failure.
    if track_circuit_breaker:
        circuit_breaker_corrective = _circuit_breaker_track_and_maybe_trip(
            engine, tool_call, outcome
        )
        if circuit_breaker_corrective is not None:
            engine.history.append(
                Message(
                    role=MessageRole.user,
                    content_blocks=[TextBlock(text=circuit_breaker_corrective)],
                    metadata={
                        SYNTHETIC_RECOVERY_METADATA_KEY: (
                            SYNTHETIC_RECOVERY_CIRCUIT_BREAKER
                        )
                    },
                )
            )
    engine.forget_tool_name(tool_call.id)


def _ingest_tool_evidence(
    engine: QueryEngine,
    outcome: DispatchOutcome,
) -> DispatchOutcome:
    """Append a successful outcome's typed evidence, or fail that outcome closed.

    The dispatcher never exposes evidence on the model-visible event/history
    path.  This is the sole query-runtime ingress, deliberately called only
    after serial or replayed dispatch status is final and before a snapshot.
    ``QueryEngine.append_tool_evidence`` validates a complete batch before it
    replaces its immutable ledger, so rejection cannot partially mutate it.
    """
    records = outcome.evidence_records
    if not records:
        return outcome
    if outcome.is_error or not outcome.success:
        # ``ToolResult`` rejects this shape at construction.  Keep the runtime
        # fail-closed if an alternate dispatcher ever constructs it directly.
        return replace(
            outcome,
            success=False,
            is_error=True,
            error_kind=DispatchErrorKind.execution,
            content="tool evidence is invalid on an unsuccessful dispatch",
            evidence_records=(),
        )
    producer = outcome.evidence_producer
    if producer is None:
        return replace(
            outcome,
            success=False,
            is_error=True,
            error_kind=DispatchErrorKind.execution,
            content="tool evidence has no registered producer binding",
            evidence_records=(),
        )
    try:
        engine.append_tool_evidence(records, producer=producer)
    except ValueError as exc:
        _logger.warning(
            "tool evidence rejected run=%s call_id=%s error=%s",
            engine.config.run_id,
            outcome.tool_call.id,
            type(exc).__name__,
        )
        return replace(
            outcome,
            success=False,
            is_error=True,
            error_kind=DispatchErrorKind.execution,
            content=f"tool evidence rejected: {exc}",
            evidence_records=(),
        )
    return outcome


def _dispatch_outcome_is_terminal(
    outcome: DispatchOutcome | None,
    *,
    engine: QueryEngine,
    tool_name: str,
) -> bool:
    """Return True when a successful tool outcome explicitly terminates the loop.

    Parallel-dispatch counterpart of :func:`_history_tool_result_is_terminal`,
    and applies the SAME expected-tool-name guard: when
    ``expected_terminal_tool`` is configured, a successful terminal-metadata
    outcome counts as terminal ONLY if ``tool_name`` matches the declared
    terminal tool. When ``expected_terminal_tool`` is None the behaviour is
    bit-identical to before (any successful terminal-metadata outcome
    counts) — no regression for a host backend or for the default.
    """
    if outcome is None or not outcome.success or outcome.is_error:
        return False
    metadata = outcome.metadata or {}
    if metadata.get(TERMINAL_TOOL_METADATA_KEY) is not True:
        return False
    expected = engine.config.expected_terminal_tool
    if expected is None:
        return True
    return tool_name == expected


# ---------------------------------------------------------------------------
# The run boundary. ``engine.history`` is a SESSION transcript, not a run's:
# cross-run history seeding prepends earlier runs of the same session verbatim.
# Every helper whose question is scoped to one run takes its messages from
# here. That is the rule, not an observation about the code: what enforces it
# is tests/unit/runtime/test_history_run_boundary.py, and that file states in
# its own docstring which shapes it cannot see.
# ---------------------------------------------------------------------------


def _this_run_messages(engine: QueryEngine) -> list[Message]:
    """Messages that belong to THIS run, in history order.

    ``engine.history`` also holds PRIOR-RUN turns the executor seeded into it
    (:data:`SESSION_HISTORY_SEED_METADATA_KEY`): the earlier runs of the same
    session, prepended verbatim — their prose, their tool calls and their tool
    results alike. Nothing else distinguishes them; ``Message`` carries no run
    id, so the seed tag IS the run boundary.

    A helper that asks a run-scoped question — did this run answer, did it
    write its deliverable, did it satisfy this precondition — takes its
    messages from here rather than from ``engine.history`` directly. The
    ordering is the point: a foreign turn is not in the sequence the helper
    iterates, so it is not something the helper can reach and then have to rule
    out. Testing provenance after the fact fails open on whatever the test
    forgot; drawing from a scoped sequence fails closed.

    Whole-history questions do NOT belong here. Wire-pairing repair, prompt
    assembly, token accounting and compaction are all about the transcript that
    goes to the provider, which is the whole of ``engine.history`` by
    definition. Pure / total — never raises.
    """
    return [
        message
        for message in engine.history
        if message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is not True
    ]


# Universal terminal-tool nudge.
#
# Any tenant declares its terminal tool name via
# ``QueryEngineConfig.expected_terminal_tool`` (routed from
# ``leader_config.expected_terminal_tool``). When set, the universal
# ``RuntimeConstants.terminal_tool_nudge_enabled`` knob gates the
# contract-repair nudge.


def _resolved_terminal_tool_name(engine: QueryEngine) -> str | None:
    """Return the configured terminal tool name for this engine.

    Resolution order:
      1. ``engine.config.expected_terminal_tool`` (per-tenant universal).
      2. ``None`` — unset; the nudge / terminal-only guard is disabled.
    """

    return engine.config.expected_terminal_tool or None


def _tool_name_for_call_id(
    engine: QueryEngine, tool_call_id: str
) -> str | None:
    """Walk history to find the assistant tool_use block matching ``tool_call_id``."""

    for message in reversed(engine.history):
        for block in reversed(message.content_blocks):
            if (
                isinstance(block, ToolUseBlock)
                and block.tool_call_id == tool_call_id
            ):
                return block.name
    return None


def _history_has_terminal_tool_result(engine: QueryEngine) -> bool:
    """Return True once THIS run has a successful terminal tool result.

    Two paths:
      * The classic ``TERMINAL_TOOL_METADATA_KEY``-flagged result satisfies
        every tenant, INCLUDING tenants with ``expected_terminal_tool``
        configured — a message-carrying terminal backend (e.g. ``pcm_answer``)
        sets the metadata key on terminal success.
      * When ``expected_terminal_tool`` is set (per-tenant generalisation),
        we additionally require the corresponding ``ToolUseBlock.name`` to
        match the configured tool name. This protects a tenant from mistaking
        a foreign tool's terminal-metadata flag for its own finalisation —
        the run is only "answered" through the declared terminal tool.

    Scoped to :func:`_this_run_messages`, and that scoping is load-bearing
    rather than tidy. This predicate is the "the run is already answered" arm
    of every gate that exists to stop a run ending unanswered — the terminal
    tool nudge and the run wind-down. Read over the whole session transcript it
    answers True
    for a run that has done nothing, because an EARLIER run of the session
    answered through the terminal tool and that turn was seeded into this one's
    history. Every one of those gates then declines to fire and the run ends
    with no answer at all, silently and in ``COMPLETED`` state. Stored history
    reaches this predicate the same way: rehydrated prior-run rows are tagged
    as seeds when they are loaded, so data written before this scoping existed
    is excluded here, but until it was, that data disarmed the next run.
    """

    expected = engine.config.expected_terminal_tool
    for message in reversed(_this_run_messages(engine)):
        for block in reversed(message.content_blocks):
            if not (
                isinstance(block, ToolResultBlock)
                and not block.is_error
                and block.metadata.get(TERMINAL_TOOL_METADATA_KEY) is True
            ):
                continue
            if expected is None:
                return True
            tool_name = _tool_name_for_call_id(engine, block.tool_call_id)
            if tool_name == expected:
                return True
    return False


def _terminal_tool_nudge_required(engine: QueryEngine) -> bool:
    """Return True iff the run needs the contract-repair terminal-tool nudge.

    Enable path: ``QueryEngineConfig.expected_terminal_tool`` is set AND
    ``rc.terminal_tool_nudge_enabled`` is True AND no terminal tool result
    is in history. Universal — keyed only on the per-tenant terminal-tool
    contract.
    """

    rc = engine.config.rc
    if _history_has_terminal_tool_result(engine):
        return False
    return (
        engine.config.expected_terminal_tool is not None
        and rc.terminal_tool_nudge_enabled
    )


def _suppress_terminal_only_meta_text(engine: QueryEngine) -> bool:
    """True iff the CURRENT terminal-only turn's visible assistant TEXT must be
 suppressed from live SSE + durable history.

 The terminal-tool nudge ALWAYS fires (write-first recovery + typed Finalize
 depend on it — a prose-only "Done, I created the file" with 0 tools must
 still be nudged into the actual Write + Finalize). The cost is that a weak
 model, on that post-nudge turn, self-narrates English 3rd-person ``META``
 prose ("The user asked … Let me finalize.") co-located with the ``Finalize``
 call. That redundant narration streams live and persists to durable
 ``session_messages`` + the memory fold + ``runs.result_preview``.

 Suppress ONLY that turn's TEXT — never the tool calls. ``Write`` /
 ``AppendFile`` (write-first recovery) AND ``Finalize`` pass through unchanged,
 so the file is still written and the typed-Finalize deliverables chip is
 preserved. Because the suppressed text is also kept out of ``text_buffer`` →
 out of ``engine.history``, even the UNfiltered ``_run_local_history`` →
 ``result_preview`` derivation is clean (the durable net then covers any path
 that bypasses this stream-suppression — pickup / reload).

 Suppress IFF ALL hold:
 * ``engine._terminal_only_active`` — we are in the post-nudge terminal-only
 turn. This latch is set by :func:`_append_terminal_tool_nudge` BEFORE
 that turn streams, so the decision is known up-front (NO buffering: the
 whole turn streams under one stable decision).
 * the resolved terminal tool is a BACKGROUND gate — its schema carries NO
 answer field (:func:`_terminal_tool_carries_answer_field` is False). For
 such a tool the user-facing answer can ONLY be prose, so the terminal-only
 turn's text is pure narration once an answer exists. A MESSAGE-CARRYING
 terminal (``pcm_answer`` / ``final_answer``) submits its answer via args
 and its visible text is the real answer surface — never suppressed.
 Unknown schema ⟹ exempt (fail-safe).
 * a PRIOR substantive answer already exists after the latest non-terminal
 work (:func:`_has_visible_assistant_prose_after_work`, floored at
 ``finalize_prose_gate_min_chars`` so a terse ``144`` counts). When NO
 prior answer exists — the terminal-only turn's text IS the answer (the
 genuinely-empty / first-answer-in-the-terminal-turn case) — it MUST stay
 visible, so this returns False and the net stays correct.

 Pure / side-effect free; cheap enough to evaluate once per stream attempt.
 """

    if not getattr(engine, "_terminal_only_active", False):
        return False
    terminal_tool = _resolved_terminal_tool_name(engine)
    if terminal_tool is None:
        return False
    # Only a BACKGROUND terminal (no answer-carrying field) routes its answer
    # through prose; a MESSAGE-CARRYING terminal answers via its args, so its
    # visible text is the real answer surface and must NOT be suppressed. Unknown
    # schema ⟹ exempt (keep the text) — fail-safe, multi-tenant.
    if _terminal_tool_carries_answer_field(engine, terminal_tool):
        return False
    return _has_visible_assistant_prose_after_work(
        engine, terminal_tool, engine.config.rc.finalize_prose_gate_min_chars
    )


def _apply_terminal_synthesis_output_reserve(
    engine: QueryEngine, max_output_tokens: int, output_cap: int
) -> int:
    """Final-turn-specific output-token floor.

    On the ACTUAL terminal / forced-final turn — i.e. the terminal-only nudge
    has fired (the durable ``engine._terminal_only_active`` latch, set by
    :func:`_append_terminal_tool_nudge`) OR the deadline backstop has fired
    (:func:`_terminal_deadline_reached`) — ensure the per-message output budget
    is at least ``rc.terminal_synthesis_output_reserve_tokens``, so the model
    has room to emit message + refs + outcome instead of being starved by the
    AdaptiveSafetyBand subtraction.

    Note: this previously keyed on :func:`_terminal_tool_nudge_required`,
    which is True on EVERY turn of a run that merely has
    ``expected_terminal_tool`` set + the nudge enabled + no terminal result
    yet — so the reserve floored the output budget on all turns, not only the
    final one. Keying on the ``_terminal_only_active`` latch restricts it
    to the genuine forced-final turn (after the nudge actually fired) +
    the deadline backstop.

    The floor is bounded by ``output_cap`` (the pre-safety-band global cap,
    ``max_context * llm_output_max_tokens_ratio``) so it can NEVER raise the
    global cap — it only reclaims tokens the safety band removed, and only on
    the final turn. ``reserve == 0`` (default) makes ``min(0, cap) == 0`` and
    the floor a no-op, so behaviour is bit-identical. Keys only on the
    generic terminal-tool contract; purely a budget number, no prompt
    wording.
    """

    reserve = engine.config.rc.terminal_synthesis_output_reserve_tokens
    if reserve <= 0:
        return max_output_tokens
    if not (
        getattr(engine, "_terminal_only_active", False)
        or _terminal_deadline_reached(engine)
    ):
        return max_output_tokens
    floor = min(reserve, output_cap)
    return max(max_output_tokens, floor)


def _history_has_file_write_result(engine: QueryEngine) -> bool:
    """Return True iff THIS run landed a successful file-write tool result.

    A file-write deliverable counts as produced when
    any tool named in ``rc.terminal_tool_nudge_file_write_tool_names``
    (default ``Write``/``AppendFile``) has a non-error tool_result in this
    run's messages. Used to decide whether the terminal-tool nudge should steer
    the model to write the deliverable FIRST. Core never hardcodes
    the host write-tool names — they come from the RC tuple so the check
    stays universal.

    Scoped to :func:`_this_run_messages`: the deliverable this run owes is one
    it writes here. A follow-up run in the same session inherits the earlier
    run's ``Write`` through cross-run history seeding, and over the whole
    transcript that earlier write would drop the write-first steer from the
    nudge for a run that has produced nothing — precisely the run that needs
    it most.
    """

    write_names = set(engine.config.rc.terminal_tool_nudge_file_write_tool_names)
    if not write_names:
        return False
    for message in _this_run_messages(engine):
        for block in message.content_blocks:
            if not (isinstance(block, ToolResultBlock) and not block.is_error):
                continue
            tool_name = _tool_name_for_call_id(engine, block.tool_call_id)
            if tool_name in write_names:
                return True
    return False


def _resolved_terminal_tool_nudge_text(engine: QueryEngine) -> str:
    """Resolve the message body for the terminal-tool nudge.

    Order: universal RC ``terminal_tool_nudge_text`` (when non-empty) →
    templated generic fallback keyed on the resolved tool name. Tenants
    that only flip ``terminal_tool_nudge_enabled`` without supplying
    override text still receive a usable nudge.

    When ``terminal_tool_nudge_write_first_enabled``
    is set AND no file-write deliverable is in history yet
    (:func:`_history_has_file_write_result`), the resolved body is PREFIXED
    with ``terminal_tool_nudge_write_first_text`` so a model that narrated
    "now let me write this file" and fired 0 tools is steered to the actual
    write tool first, not just the terminal tool. The prefix is bounded by the
    single-shot nudge latch (it never loops) and is a no-op for a run that has
    already written its deliverable.
    """

    rc = engine.config.rc
    tool_name = _resolved_terminal_tool_name(engine) or "the configured terminal tool"
    universal_text = rc.terminal_tool_nudge_text
    if universal_text:
        # Resolve the live terminal tool name DYNAMICALLY. The RC text MAY
        # carry a ``{terminal_tool}`` placeholder so a tenant never has to
        # hard-code the tool name; core fills it from
        # ``expected_terminal_tool`` at runtime. No placeholder ⟹ verbatim
        # (bit-identical for existing RC values).
        body = universal_text.replace("{terminal_tool}", tool_name)
    else:
        # Anti-echo wording: a weak model in a looping/confused state used
        # to PARAPHRASE the prior second-person imperative ("You have not called
        # X yet … Do not write normal final text.") straight into its visible
        # answer. This recasts the nudge as a self-evidently INTERNAL control
        # note (the ``[internal control — …]`` frame is obviously not answer
        # prose, so a verbatim echo is harmless), describes the run state in the
        # third person instead of commanding the model, and drops the
        # echo-attractive negative imperative — while keeping the SAME functional
        # trigger: the run still finishes by calling the terminal tool with its
        # best supported answer. Keeps the literal "Finish now by calling
        # {tool_name}" the forced-backstop test keys on.
        body = (
            f"[internal control — not part of the reply] The run is ending and "
            f"the {tool_name} tool has not been called yet. Finish now by calling "
            f"{tool_name} with the best supported answer already prepared; the "
            "answer text belongs in the reply itself, not in this note."
        )
    if (
        rc.terminal_tool_nudge_write_first_enabled
        and rc.terminal_tool_nudge_write_first_text
        and not _history_has_file_write_result(engine)
    ):
        return f"{rc.terminal_tool_nudge_write_first_text}\n\n{body}"
    return body


def _terminal_deadline_reached(engine: QueryEngine) -> bool:
    """True iff the run's wall-clock budget is (nearly) spent.

    The budget is ``rc.agent_max_seconds`` measured from ``QueryEngine.run()``
    entry (``engine._run_started_monotonic``); the early-finalize fires once
    the elapsed time reaches ``agent_max_seconds - agent_deadline_finalize_
    slack_seconds`` so a final terminal-tool round-trip can still complete
    before an external trial / reaper kills the run.

    Returns False when the budget is disabled (``agent_max_seconds <= 0``)
    or the clock was never stamped (``_run_started_monotonic == 0.0``). A
    negative start is valid after snapshot resume when persisted wall-clock
    elapsed time exceeds the new process's monotonic uptime.
    """

    rc = engine.config.rc
    budget = rc.agent_max_seconds
    if budget <= 0.0:
        return False
    started = getattr(engine, "_run_started_monotonic", 0.0)
    if started == 0.0:
        return False
    slack = rc.agent_deadline_finalize_slack_seconds
    # Threshold floored at 0 so a slack >= budget still finalises promptly
    # rather than going negative.
    threshold = budget - slack
    if threshold < 0.0:
        threshold = 0.0
    return (time.monotonic() - started) >= threshold


async def _maybe_inject_pre_terminal_self_verify(engine: QueryEngine) -> bool:
    """Inject ONE bounded corrective turn before finalising.

    Called at every terminal-completion site BEFORE the loop treats a
    terminal-tool result as final. When ALL hold:

      * ``rc.pre_terminal_self_verify_enabled`` is True,
      * the per-run latch ``engine._pre_terminal_self_verify_used`` is unset,
      * the bounded counter is below
        ``rc.pre_terminal_self_verify_max_extra_turns``,
      * an host-supplied ``config.pre_terminal_self_verify_trigger``
        returns a non-empty corrective instruction,

    this appends ONE corrective user-role turn, latches (so it fires at most
    once per run), bumps the counter, and returns ``True`` so the caller
    does NOT finalise — the outer loop runs one more bounded turn in which
    the model can fix the cited-but-unobserved ref or perform the
    declared-but-missing mutation.

    Returns ``False`` (finalise as usual) when the
    feature is disabled, already used, over budget, or the trigger declines.
    The trigger predicate is tenant-supplied so the self-verify turn stays
    universal — core never inspects a specific terminal tool's payload.
    """

    rc = engine.config.rc
    if not rc.pre_terminal_self_verify_enabled:
        return False
    if getattr(engine, "_pre_terminal_self_verify_used", False):
        return False
    if (
        getattr(engine, "_self_verify_extra_turns_used", 0)
        >= rc.pre_terminal_self_verify_max_extra_turns
    ):
        return False
    trigger = engine.config.pre_terminal_self_verify_trigger
    if trigger is None:
        return False
    try:
        corrective = trigger(engine)
    except Exception as exc:  # pragma: no cover - defensive; never break finalise
        _logger.warning(
            "DIAG query.pre_terminal_self_verify.trigger_failed run=%s error=%s",
            engine.config.run_id,
            exc,
        )
        return False
    if not corrective:
        return False
    engine._pre_terminal_self_verify_used = True
    engine._self_verify_extra_turns_used = (
        getattr(engine, "_self_verify_extra_turns_used", 0) + 1
    )
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=corrective)],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: (
                    SYNTHETIC_RECOVERY_PRE_TERMINAL_SELF_VERIFY
                )
            },
        )
    )
    # Persist the snapshot IMMEDIATELY after the corrective turn + latch
    # mutation, BEFORE the outer loop opens the next LLM call. Without this,
    # a crash or cross-pod resume between the injection and the next
    # persistence boundary loses BOTH the appended correction and the
    # ``_pre_terminal_self_verify_used`` latch — the resumed run would
    # re-fire the self-verify turn (latch lost) or finalise without the
    # correction (correction lost). The snapshot schema already carries the
    # latch + counter, so persisting here makes the at-most-once corrective
    # turn durable across a re-drive.
    await engine._persist_snapshot()
    _logger.warning(
        "DIAG query.pre_terminal_self_verify.injected run=%s tenant=%s turn=%s",
        engine.config.run_id,
        engine.config.tenant_id,
        engine.turn_id(),
    )
    return True


# Fallback corrective text when a regressed terminal turn must be re-vetoed
# but the host trigger returned no corrective for that turn. Uses the
# same vocabulary as the existing ``veto_error`` on the pre-dispatch veto
# path.
_TERMINAL_CANDIDATE_REVETO_FALLBACK = (
    "Your previous answer was withheld and this replacement is empty or much "
    "shorter than the answer you had already drafted. Do not shorten or drop "
    "your answer — re-send your full answer."
)


def _terminal_candidate_message(tool_call: ToolCall) -> str:
    """Return the stripped terminal-answer ``message`` body for ``tool_call``.

    Universal over the terminal-answer contract: the required free-text body
    of every terminal tool is the generic ``message`` argument (a host's own
    answer tool carries it too). Core never inspects
    a tenant-specific payload — only this generic field. Returns ``""`` when
    args are missing / not a mapping / the field is absent or non-string.
    """

    args = tool_call.arguments
    if not isinstance(args, dict):
        return ""
    message = args.get("message")
    if not isinstance(message, str):
        return ""
    return message.strip()


def _terminal_candidate_snapshot_args(tool_call: ToolCall) -> dict[str, Any]:
    """Return a JSON-serialisable copy of the terminal tool args.

    Stored on the durable per-run candidate so the preserved draft survives a
    cross-pod resume. Falls back to an empty mapping when args are not a
    mapping (a candidate is only ever recorded for a substantive ``message``
    body, so this is defensive).
    """

    args = tool_call.arguments
    if not isinstance(args, dict):
        return {}
    return dict(args)


def _terminal_candidate_hash(message: str) -> str:
    """Stable content hash of the preserved candidate body (audit only)."""

    return hashlib.sha256(message.encode("utf-8", errors="replace")).hexdigest()


def _resolve_terminal_candidate_corrective(
    engine: QueryEngine, tool_call: ToolCall, corrective: str | None
) -> str | None:
    """Candidate-answer preservation.

    Wraps the host pre-dispatch veto verdict (``corrective``) so that
    the first SUBSTANTIVE terminal-answer draft is not silently lost when a
    later repair turn regresses to an empty / too-short body.

    Gated entirely behind ``rc.terminal_candidate_preserve_enabled`` —
    default-off returns ``corrective`` unchanged, so the candidate is discarded
    exactly as today (bit-identical). When enabled:

    * **Preserve** — if the verdict is a veto (``corrective`` truthy) and the
      proposed body is substantive (``len(message) >=
      max(1, terminal_answer_min_message_chars)``) and no substantive candidate
      is already held, durably record ``{tool, args, veto_reason,
      candidate_hash, message_chars, substantive}`` on the engine. The verdict
      is returned unchanged (the veto still fires).
    * **Re-veto a regression** — if a substantive candidate is already held and
      the current body regresses (empty, or shorter than
      ``terminal_answer_min_message_chars`` when that floor is set), force the
      veto exactly once using the EXISTING corrective text (no new wording
      is synthesised; the model owns the corrected args). The
      one-shot is bounded by the durable ``_terminal_candidate_reveto_used``
      latch.
    * **Allow-through** — once the single re-veto repair credit is spent, a
      still-regressed body is allowed through (verdict forced to ``None``) so
      the run finalises on best evidence rather than looping.

    This function only mutates engine-side fields; the caller's existing
    ``await engine._persist_snapshot()`` on the veto path makes the write
    durable (horizontal / cross-pod safe). It NEVER mutates the answer body.
    """

    rc = engine.config.rc
    if not rc.terminal_candidate_preserve_enabled:
        return corrective

    message = _terminal_candidate_message(tool_call)
    min_chars = max(0, rc.terminal_answer_min_message_chars)
    substantive_floor = max(1, min_chars)
    is_substantive = len(message) >= substantive_floor

    saved = engine._terminal_candidate

    # ``isinstance`` inline (vs a separate ``saved_substantive`` bool) so mypy
    # narrows ``saved`` to ``dict`` for the ``saved.get(...)`` reads below —
    # behaviour-identical, fixes a pre-existing union-attr (None.get) flag.
    if isinstance(saved, dict) and bool(saved.get("substantive")):
        # A substantive draft was preserved earlier. A regression is an empty
        # body, or (when a floor is set) one shorter than the floor.
        regressed = (not message) or (min_chars > 0 and len(message) < min_chars)
        if not regressed:
            # The current body is itself substantive — defer to the normal
            # verdict; do not clobber the already-preserved candidate.
            return corrective
        if not engine._terminal_candidate_reveto_used:
            engine._terminal_candidate_reveto_used = True
            forced = corrective or _TERMINAL_CANDIDATE_REVETO_FALLBACK
            _logger.warning(
                "DIAG terminal_candidate.regressed run=%s tenant=%s "
                "action=reveto saved_chars=%s new_chars=%d",
                engine.config.run_id,
                engine.config.tenant_id,
                saved.get("message_chars"),
                len(message),
            )
            return forced
        # Repair credit already spent — allow the regressed body through so the
        # run finalises rather than looping.
        _logger.warning(
            "DIAG terminal_candidate.regressed run=%s tenant=%s "
            "action=allow saved_chars=%s new_chars=%d",
            engine.config.run_id,
            engine.config.tenant_id,
            saved.get("message_chars"),
            len(message),
        )
        return None

    # No substantive candidate held yet. Preserve the current body iff the
    # verdict is a veto AND the body is worth keeping.
    if corrective and is_substantive:
        engine._terminal_candidate = {
            "tool": tool_call.name,
            "args": _terminal_candidate_snapshot_args(tool_call),
            "veto_reason": "pre_dispatch_terminal_verify",
            "candidate_hash": _terminal_candidate_hash(message),
            "message_chars": len(message),
            "substantive": True,
        }
        _logger.warning(
            "DIAG terminal_candidate.preserved run=%s tenant=%s chars=%d hash=%s",
            engine.config.run_id,
            engine.config.tenant_id,
            len(message),
            engine._terminal_candidate["candidate_hash"],
        )
    return corrective


def _terminal_candidate_repair_applies(
    engine: QueryEngine, tool_call: ToolCall
) -> bool:
    """Gate for candidate-regression protection on the REPAIR turn, i.e. once
    the pre-dispatch-verify one-shot latch is already closed.

    The pre-dispatch terminal veto (:func:`_pre_dispatch_terminal_verify_applies`)
    is fire-at-most-once: it sets ``_pre_dispatch_terminal_verify_used`` on the
    FIRST veto, so on the model's corrected re-submission (the repair turn) the
    pre-dispatch gate is CLOSED and the candidate-regression check wired inside
    it never re-runs.

    This independent seam re-runs candidate-regression protection on the
    expected terminal tool EVEN WHEN the pre-dispatch latch is closed. ALL must
    hold:

      * ``rc.terminal_candidate_preserve_enabled`` is True (same kill-switch as
        the preservation seam — default-off is bit-identical),
      * a per-tenant terminal tool is declared (``config.expected_terminal_tool``)
        AND ``tool_call`` IS that tool (universal; never intercepts a
        non-terminal tool),
      * a SUBSTANTIVE candidate is already held
        (``engine._terminal_candidate["substantive"]``) — only true AFTER the
        first veto preserved one, which is also when the pre-dispatch latch is
        closed, so this branch and the pre-dispatch branch never both fire on
        one dispatch.

    The actual regress / re-veto-once / allow-through decision is delegated to
    the EXISTING :func:`_resolve_terminal_candidate_corrective` (keyed on the
    durable ``_terminal_candidate_reveto_used`` latch), so the repair seam and
    the preservation seam share one decision implementation and one latch.

    Cheap and side-effect-free so ``_dispatch_tool`` can call it on every
    dispatch; returns False when disabled or conditions not met.
    """

    rc = engine.config.rc
    if not rc.terminal_candidate_preserve_enabled:
        return False
    expected = engine.config.expected_terminal_tool
    if expected is None or tool_call.name != expected:
        return False
    saved = engine._terminal_candidate
    return isinstance(saved, dict) and bool(saved.get("substantive"))


# ---------------------------------------------------------------------------
# Universal prose-gate before a BACKGROUND terminal tool.
#
# The terminal tool (e.g. ``Finalize``) operates as a pure background gate: its
# answer field is removed and its tool_use / tool_result pair is filtered from
# the live stream + durable history, so the ONLY user-facing answer is the
# model's own visible assistant prose. A small empirical tail of runs call the
# terminal tool with NO substantive prose after their last real work; those
# would surface an empty answer. This gate vetoes such a dispatch ONCE and
# injects one bounded repair turn asking the model to write the answer as
# normal text first, then call the terminal tool. One-shot, snapshot-persisted.
#
# Ported from the host ``_has_visible_assistant_prose_after_work`` /
# ``_is_non_finalize_tool_activity`` predicate (it operated on engine history,
# which core owns) but driven off ``expected_terminal_tool`` — NOT a hardcoded
# ``Finalize`` name — so it stays universal.
# ---------------------------------------------------------------------------


def _strip_tool_name_prefix(name: str) -> str:
    """Strip a legacy ``tool:`` namespace prefix from a tool name (universal).

    Mirrors the host / chat ``tool:`` prefix stripping so a terminal tool
    advertised as ``tool:Finalize`` still matches the configured
    ``expected_terminal_tool`` name. Returns ``name`` unchanged when no prefix
    is present.
    """

    return name[5:] if name.startswith("tool:") else name


def _is_terminal_tool_name(name: object, terminal_tool: str) -> bool:
    """True iff ``name`` is the configured terminal tool (prefix-tolerant).

 Universal: matched against the per-tenant ``terminal_tool`` (the resolved
 ``expected_terminal_tool``), never a hardcoded tool name. Exact match after
 stripping any ``tool:`` prefix — never a substring/prefix match, so a tool
 like ``FinalizeFile`` is NOT mistaken for ``Finalize``.
 """

    return isinstance(name, str) and _strip_tool_name_prefix(name) == terminal_tool


def _is_non_terminal_tool_activity(block: object, terminal_tool: str) -> bool:
    """True for tool activity that is real work, NOT the terminal gate.

    Ported from the host ``_is_non_finalize_tool_activity`` but keyed on
    the configured ``terminal_tool``:

      * a ``ToolUseBlock`` is real work unless it is the terminal tool;
      * a ``ToolResultBlock`` is real work unless its named tool is the terminal
        tool, OR (when unnamed) it carries the terminal-metadata flag — an
        unnamed successful terminal result is still the gate, not user work.
    """

    if isinstance(block, ToolUseBlock):
        return not _is_terminal_tool_name(block.name, terminal_tool)
    if isinstance(block, ToolResultBlock):
        tool_name = block.metadata.get("tool_name")
        if isinstance(tool_name, str):
            return not _is_terminal_tool_name(tool_name, terminal_tool)
        # A successful terminal result without a name is still the terminal
        # gate, not user work. Named non-terminal results (the common path) are
        # handled above; unnamed non-terminal results remain visible work.
        return block.metadata.get(TERMINAL_TOOL_METADATA_KEY) is not True
    return False


def _has_visible_assistant_prose_after_work(
    engine: QueryEngine, terminal_tool: str, min_chars: int
) -> bool:
    """Whether the run already has substantive user-facing prose after work.

 Ported from the host ``_has_visible_assistant_prose_after_work`` (it
 operated on engine history, which core owns), generalised over
 ``terminal_tool`` + a substantive ``min_chars`` floor. Typed-terminal runs
 often look like::

 assistant tool(Bash) -> tool result -> assistant prose answer ->
 assistant tool(Finalize) -> tool result

 In that shape the prose IS the user-facing answer and the terminal payload
 is only the internal gate. A payload-only terminal (all visible prose
 occurred BEFORE the latest non-terminal work, or no substantive prose at
 all) returns False — that prose was progress narration, not the final
 answer — so the prose-gate fires.

 Reads :func:`_this_run_messages`, so the PRIOR-RUN turns cross-run history
 seeding prepends are not in the window at all — otherwise a prior run's
 seeded answer prose would falsely satisfy the gate and let a payload-only
 terminal finalise with an empty CURRENT-run answer. Same boundary as
 the host ``_run_local_history`` seed-strip.

 ``min_chars`` is the stripped-length floor a single assistant ``TextBlock``
 must reach to count as substantive (``finalize_prose_gate_min_chars``); a
 floor of 0 accepts any non-empty visible prose. Pure / total — never raises.
 """

    last_work_pos = -1
    latest_prose_pos = -1
    pos = 0
    # A floor of 0 means "any non-empty visible prose counts" (so we still
    # require at least 1 stripped char); a positive floor demands that many.
    substantive_floor = max(1, min_chars)
    for message in _this_run_messages(engine):
        for block in message.content_blocks:
            if _is_non_terminal_tool_activity(block, terminal_tool):
                last_work_pos = pos
            if (
                message.role is MessageRole.assistant
                and message.metadata.get(PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY)
                is not True
                and isinstance(block, TextBlock)
                and len(block.text.strip()) >= substantive_floor
            ):
                latest_prose_pos = pos
            pos += 1
    return latest_prose_pos > last_work_pos


def _preserve_completed_answer_on_stream_error(engine: QueryEngine) -> bool:
    """True iff a transient stream/provider error must NOT fail the run.

    Fires when the run has already produced a substantive user-facing
    assistant answer after its latest non-terminal work — the reply the user
    saw stream live. In that state a provider / idle error raised on a later
    harness-forced continuation turn (the terminal-tool nudge or a
    continue-prompt injection) is bolt-on: the answer is already delivered,
    so the run must complete on it rather than propagate the transient error
    as a FAILED terminal status. Gated by
    ``rc.preserve_completed_answer_on_stream_error`` (default on); the
    substantive floor reuses ``finalize_prose_gate_min_chars``. Pure / total.
    """

    rc = engine.config.rc
    if not getattr(rc, "preserve_completed_answer_on_stream_error", False):
        return False
    return _has_visible_assistant_prose_after_work(
        engine,
        _resolved_terminal_tool_name(engine) or "",
        rc.finalize_prose_gate_min_chars,
    )


async def _complete_run_on_preserved_answer(
    engine: QueryEngine, *, reason: str
) -> AsyncIterator[TurnEvent]:
    """Complete the run on an already-delivered answer after a stream error.

    Mirrors the voluntary-finish terminal (an ``end_turn`` ``message_stop``
    followed by a transition to ``COMPLETED``) so the run's terminal status
    reflects the substantive reply already in history rather than the
    transient provider / idle error raised on a harness-forced continuation
    turn. Emits a ``state_changed`` first so the reason is observable on the
    bus. The caller ``return``s immediately after draining these events.
    """

    from_state = engine.state
    yield _emit_state_change(engine, from_state, from_state, reason=reason)
    yield TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "stop_reason": "end_turn",
            "tokens_used": _tokens_used_payload(engine),
            "cache_hit_rate": engine.total_usage.this_turn_cache_hit_rate(),
        },
    )
    engine.transition_to(LoopState.COMPLETED)


def _transient_retry_backoff_seconds(
    rc: RuntimeConstants, attempt: int, exc: BaseException
) -> float:
    """Backoff (seconds) before the ``attempt``-th transient-error retry.

    ``attempt`` is 1-based. The delay is an exponential term
    ``base * 2 ** (attempt - 1)`` bounded by the configured ceiling. When the
    classifier surfaced a server-stated ``Retry-After`` on the error it takes
    precedence (some providers pace 429s), but is still clamped by the same
    ceiling so worst-case latency stays bounded. A zero base with no
    ``Retry-After`` retries immediately. Pure / total — never raises.
    """
    base = rc.llm_transient_error_retry_backoff_base_seconds
    ceiling = rc.llm_transient_error_retry_backoff_max_seconds
    delay = base * (2 ** (attempt - 1)) if base > 0.0 else 0.0
    classified = getattr(exc, "classified", None)
    retry_after = (
        getattr(classified, "retry_after_seconds", None)
        if classified is not None
        else None
    )
    if isinstance(retry_after, (int, float)) and retry_after > 0:
        delay = max(delay, float(retry_after))
    if ceiling > 0.0:
        delay = min(delay, ceiling)
    return max(0.0, delay)


def _this_run_model_turns(engine: QueryEngine) -> list[Message]:
    """Assistant turns carrying THIS run's model's own words, in history order.

    ``engine.history`` is not a record of one run. Two kinds of assistant turn
    sit in it that the model did not say here, and both are indistinguishable
    from a real answer once their text has been read out of the message:

    * PRIOR-RUN turns the executor seeded into this run's history
      (:data:`SESSION_HISTORY_SEED_METADATA_KEY`). Cross-run history seeding
      prepends earlier runs of the same session verbatim, so the newest
      assistant prose in ``history`` is routinely a fluent, complete answer to
      a question THIS run was never asked.
    * runtime-synthesised recovery scaffolding
      (:data:`SYNTHETIC_RECOVERY_METADATA_KEY` — the empty-completion /
      post-tool ``(empty)`` placeholders, the guaranteed-terminal tool-use
      turn). The runtime's own words, not the model's.

    Callers that ask "what did the model say in this run?" take their messages
    from here rather than from ``engine.history`` directly, so a turn from
    another run is not something they can reach and then have to rule out.
    The run half of the exclusion is :func:`_this_run_messages` — this narrows
    it to the model's own words rather than restating it, so there is exactly
    one place that knows where a run begins, and helpers that must also see
    ``role=tool`` results (write accounting, precondition rehydration) share
    that place instead of deriving a second answer to the same question.
    Pure / total — never raises.
    """
    return [
        message
        for message in _this_run_messages(engine)
        if message.role is MessageRole.assistant
        and not message.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
        and message.metadata.get(PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY) is not True
    ]


def run_has_final_answer(engine: QueryEngine) -> bool:
    """True iff THIS run already produced a substantive visible assistant answer.

    Scans :func:`_this_run_model_turns` for an assistant ``TextBlock`` with
    non-whitespace text — so neither a seeded prior answer nor the guard's own
    re-drive placeholder can mask a genuinely unanswered current run. Distinct
    from :func:`_has_visible_assistant_prose_after_work` (which requires the
    prose to come AFTER the latest non-terminal work and applies the
    substantive-char floor): this asks only "is there ANY real visible answer
    in this run yet", the precondition for the empty-completion guard.
    Pure / total — never raises.
    """
    for message in _this_run_model_turns(engine):
        for block in message.content_blocks:
            if isinstance(block, TextBlock) and block.text.strip():
                return True
    return False


def _append_empty_completion_redrive_nudge(engine: QueryEngine) -> None:
    """Append an API-valid synthetic pair to re-drive after a bare-empty turn.

    The empty turn appended nothing to history (no text / tool / reasoning), so
    the tail is whatever preceded it. To keep the wire sequence valid on the
    re-drive (never ``tool -> user`` nor a bare double-user turn), append an
    empty-marker assistant turn followed by a corrective user nudge, both flagged
    :data:`SYNTHETIC_RECOVERY_METADATA_KEY` so neither is mistaken for a real
    model answer by :func:`run_has_final_answer` /
    :func:`_latest_durable_answer_text`. Reuses the existing empty-response
    recovery scaffolding text so no new literal is introduced.
    """
    rc = engine.config.rc
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                TextBlock(text=rc.post_tool_empty_nudge_assistant_text)
            ],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: (
                    SYNTHETIC_RECOVERY_POST_TOOL_EMPTY_NUDGE
                )
            },
        )
    )
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=rc.continue_prompt_text)],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: (
                    SYNTHETIC_RECOVERY_POST_TOOL_EMPTY_NUDGE
                )
            },
        )
    )


async def _emit_empty_completion_terminal(
    engine: QueryEngine,
) -> AsyncIterator[TurnEvent]:
    """Drive the run FAILED on a bare-empty end_turn that produced no answer.

    Used by the empty-completion guard once its bounded re-drive budget is
    exhausted: the model kept ending the turn with ``finish_reason='stop'`` and
    no visible text / tool call / reasoning, and the run never produced an answer
    or a terminal tool result. Sealing that as COMPLETED would report an empty
    turn as a clean answer, so the run is terminated FAILED with a self-evident
    ``no_answer_empty_completion`` reason. Unlike :func:`_emit_llm_terminal` this
    is NOT an LLM-provider class error, so the Stop / SessionEnd death-spiral
    guard (``engine.skip_terminal_hooks``) is left untouched — ordinary terminal
    hooks still run. Emits ``state_changed -> error -> message_stop`` mirroring
    the other terminal sites.
    """

    # Pair any orphan tool_use before the FAILED transition (defensive: the
    # bare-empty turn appended no tool_use, but mirror the other terminals so a
    # resumed snapshot never carries a dangling call).
    _synthesize_missing_tool_results(
        engine.history,
        error_content=engine.config.rc.tool_result_interrupted_placeholder,
    )
    from_state = engine.state
    reason = "no_answer_empty_completion"
    engine.transition_to(LoopState.FAILED)
    yield _emit_state_change(engine, from_state, LoopState.FAILED, reason=reason)
    yield TurnEvent(
        type=EventType.ERROR,
        run_id=engine.config.run_id,
        payload={
            "kind": reason,
            "message": (
                "assistant ended the turn with no visible answer, no tool "
                "call and no reasoning, and the run produced no answer"
            ),
        },
    )
    yield TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "stop_reason": StopReason.error.value,
        },
    )


async def _emit_tool_precondition_terminal(
    engine: QueryEngine,
) -> AsyncIterator[TurnEvent]:
    """Drive the run FAILED when a tool precondition ran out of attempts.

    The run asked for a tool to be called before the agent answered, the forced
    turns were spent, and it still has not been called successfully. Completing
    would report an answer produced without the thing the caller made a
    condition of it, so the run is terminated FAILED with a reason naming the
    tool and the last error the tool reported. Like
    :func:`_emit_empty_completion_terminal` this is not an LLM-provider class
    error, so ``engine.skip_terminal_hooks`` is left untouched and the ordinary
    terminal hooks still run. Emits ``state_changed -> error -> message_stop``
    mirroring the other terminal sites.
    """

    # Pair any orphan tool_use before the FAILED transition so a resumed
    # snapshot never carries a dangling call — the exhausting turn may well
    # have appended a tool_use whose result never came back.
    _synthesize_missing_tool_results(
        engine.history,
        error_content=engine.config.rc.tool_result_interrupted_placeholder,
    )
    from_state = engine.state
    reason = "tool_precondition_unsatisfied"
    message = _preconditions.failure_message(engine)
    engine.transition_to(LoopState.FAILED)
    yield _emit_state_change(engine, from_state, LoopState.FAILED, reason=reason)
    yield TurnEvent(
        type=EventType.ERROR,
        run_id=engine.config.run_id,
        payload={"kind": reason, "message": message},
    )
    yield TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "stop_reason": StopReason.error.value,
        },
    )


def _prose_gate_just_injected(engine: QueryEngine) -> bool:
    """True iff the LAST history turn is the
    prose-gate corrective the dispatch path just appended.

    Lets the serial dispatch loop detect a prose-gate veto WITHOUT a transient
    flag: the veto in :func:`_dispatch_tool` appends a NON-terminal error
    tool_result PLUS a synthetic user turn tagged
    :data:`SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR` as the final message. When that
    is the tail, the loop breaks the batch (does NOT dispatch later sibling tool
    calls AFTER the user-repair turn) and re-drives. Pure / total."""

    if not engine.history:
        return False
    last = engine.history[-1]
    return (
        last.role is MessageRole.user
        and last.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
        == SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR
    )


def _terminal_tool_carries_answer_field(
    engine: QueryEngine, terminal_tool: str
) -> bool:
    """Return True iff the terminal tool's schema carries the user answer itself.

 The prose-gate must fire ONLY for a BACKGROUND terminal tool (one that
 does NOT carry the answer in its args, so the user-facing answer can only
 be the model's prose). A MESSAGE-CARRYING terminal tool (``pcm_answer`` /
 ``final_answer`` / ``Finalize.answer``) legitimately
 answers via its args and emits no prose — vetoing it would withhold a
 valid answer submission. Reuses the SAME schema signal as the synthesiser
 (:data:`_TERMINAL_TOOL_ANSWER_ARG_NAMES` = ``("message","answer","text"``).

 Fails SAFE (returns ``True`` ⟹ EXEMPT ⟹ no prose-gate) for multi-tenant
 safety whenever the schema cannot be introspected: no core registry, the
 tool is unknown to core (a host-backend tool whose contract core does not
 hold), or the parameters are unreadable. Only a tool that core CAN resolve
 AND whose declared properties contain NONE of the answer-carrying names is
 treated as a background terminal (returns ``False``). Cheap / side-effect
 free; mirrors :func:`_terminal_tool_accepts_refs`.
 """

    registry = getattr(engine, "tools", None)
    getter = getattr(registry, "get", None)
    if getter is None:
        return True  # no core registry → exempt (cannot prove background)
    try:
        tool = getter(terminal_tool)
    except Exception:  # pragma: no cover - defensive
        return True
    if tool is None:
        return True  # tool unknown to core (host backend) → exempt
    try:
        properties = tool.definition.parameters.properties
    except Exception:  # pragma: no cover - defensive
        return True
    return any(name in properties for name in _TERMINAL_TOOL_ANSWER_ARG_NAMES)


def _finalize_prose_gate_applies(
    engine: QueryEngine, tool_call: ToolCall
) -> bool:
    """gate for the universal prose-gate veto.

    Returns True iff the prose-gate should VETO this terminal dispatch and
    inject one bounded prose-repair turn. ALL must hold:

      * ``rc.finalize_prose_gate_enabled`` is True,
      * a per-tenant terminal tool is declared (``config.expected_terminal_tool``)
        AND ``tool_call`` IS that tool — the gate never intercepts a
        non-terminal tool (reads / writes / exec dispatch unaffected),
      * whichever of the two tests this dispatch answers to still has budget:
        the payload-only case is bounded by the durable one-shot latch
        ``_finalize_prose_gate_used`` (fire-at-most-once across resume), the
        pointer case by its own
        ``rc.finalize_prose_gate_pointer_max_repair_attempts``
        (:func:`_pointer_answer_repair_budget_spent`). Two bounds and not one,
        because the two tests fail differently: a model shown a payload-only
        terminal and told to write prose first either writes it or does not, and
        a second veto on the same run has nothing new to say — while the pointer
        refusal was measured against a model that answers the correction with a
        second filing notice, where one attempt detects the failure and does not
        repair it,
      * the terminal tool is a BACKGROUND gate — its resolved input schema has
        NO answer-carrying field (:func:`_terminal_tool_carries_answer_field` is
        False). A MESSAGE-CARRYING terminal tool (``pcm_answer`` /
        ``Finalize.answer``) answers via its args and is
        EXEMPT; an unknown schema is EXEMPT too (multi-tenant safe), and
      * the run has NO substantive visible assistant prose after its latest
        non-terminal work tool (:func:`_has_visible_assistant_prose_after_work`
        is False) — i.e. this is a payload-only terminal — OR that prose is
        principally a POINTER to a file the run wrote and the user cannot open
        (:func:`_pointer_answer_evidence`, inert unless the deployment declares
        the workspace hidden). The second is the same failure as the first with
        enough characters on it to clear a length floor.

    Returns False when the gate is disabled or the conditions are not met.
    Cheap and side-effect-free so ``_dispatch_tool`` can call it on every
    dispatch without cost when the gate is closed.
    """

    rc = engine.config.rc
    if not rc.finalize_prose_gate_enabled:
        return False
    terminal_tool = _resolved_terminal_tool_name(engine)
    if terminal_tool is None or tool_call.name != terminal_tool:
        return False
    # A MESSAGE-CARRYING terminal tool answers via its args (no prose
    # expected); only a BACKGROUND terminal (no answer-carrying field in its
    # schema) must produce prose. Unknown schema ⟹ exempt (multi-tenant
    # safe). This keeps the gate from withholding a valid payload-only answer
    # from pcm_answer / Finalize.answer.
    if _terminal_tool_carries_answer_field(engine, terminal_tool):
        return False
    # Substantive visible prose after the latest real work ⟹ the answer already
    # exists ⟹ no repair turn (the common, healthy shape) — UNLESS that prose is
    # principally a pointer to a file the user cannot open, which clears any
    # length floor while delivering nothing (:func:`_pointer_answer_evidence`,
    # inert unless the deployment says the workspace is hidden).
    if _has_visible_assistant_prose_after_work(
        engine, terminal_tool, rc.finalize_prose_gate_min_chars
    ):
        if _pointer_answer_repair_budget_spent(engine):
            return False
        return _pointer_answer_evidence(engine) is not None
    return not getattr(engine, "_finalize_prose_gate_used", False)


def _plain_stop_answer_floor_applies(engine: QueryEngine) -> bool:
    """Whether a run completing on a plain stop still owes the user an answer.

    :func:`_finalize_prose_gate_applies` can only ever intercept a TERMINAL-TOOL
    dispatch. A run in which the model simply stops — ``finish_reason='stop'``,
    no tool call — never reaches that seam, and on a deployment that declares no
    terminal tool that is how essentially every run ends, so the gate never
    participates at all. Measured shape: a leader delegated correctly, its
    subagents wrote five result files, and the reply the user actually received
    was 97 characters long. Nothing was broken; the gate was simply somewhere
    else.

    This is the SAME gate at the other completion path. It reuses the floor
    (``finalize_prose_gate_min_chars``), the durable latch
    (``_finalize_prose_gate_used``) and the repair text, so the SHORT-ANSWER
    half of the mechanism still fires AT MOST ONCE per run: whichever path
    reaches it first spends the single shot for both, and a run can never
    oscillate between them.

    The POINTER half does not share that shot. It is a different test —
    :func:`_pointer_answer_evidence`, an answer long enough to clear any floor
    that still delivers nothing — and it is bounded on its own by
    ``rc.finalize_prose_gate_pointer_max_repair_attempts``. The two remain
    exclusive per firing (a run either has substantive prose after its work or
    it does not, and that is the branch below), so neither can consume the
    other's budget; what changed is that spending one no longer silences the
    other for the rest of the run. That entanglement had teeth: a run repaired
    once for a thin answer could then file a notice about a 13 KB document and
    the pointer test, holding a spent latch, would never look at it.

    ALL must hold:

      * ``rc.finalize_prose_gate_enabled`` — the same kill switch, so an
        operator who turns the gate off restores the prior behaviour on BOTH
        paths and BOTH tests, not one of them;
      * the bound belonging to the branch this run lands in still has room: the
        durable one-shot latch for the short-answer branch (fire-at-most-once
        across resume), the attempt budget for the pointer branch
        (:func:`_pointer_answer_repair_budget_spent`, likewise resume-safe);
      * the run produced SOME visible assistant answer
        (:func:`run_has_final_answer`). A run that produced none
        at all belongs to the empty-completion guard, which owns that turn with
        its own RC, its own multi-re-drive budget and a loud FAILED terminal
        once it is spent — strictly more than the one repair turn offered here.
        The two predicates are exact complements on this point, so the paths
        never both fire, and an operator who switches that guard off keeps the
        sealed-empty behaviour they asked for rather than quietly inheriting
        this one;
      * the run has NO substantive visible assistant prose after its latest
        non-terminal work (:func:`_has_visible_assistant_prose_after_work`).
        With no terminal tool configured the predicate is passed ``""``, which
        matches no tool name, so every tool result counts as real work — the
        correct reading when nothing is a terminal gate. There is ONE shape in
        which prose that clears the floor still leaves the user with nothing:
        an answer that is principally a POINTER to a file this run wrote, on a
        surface where the user cannot open it. That is
        :func:`_pointer_answer_evidence`, and it is the second way this
        predicate can say yes. It is inert unless the deployment declares the
        workspace hidden, so the floor's behaviour is unchanged by default.

    On the ANSWER-FIELD EXEMPTION. The terminal path exempts a terminal tool
    whose schema carries the answer in its own args
    (:func:`_terminal_tool_carries_answer_field`): such a tool IS the answer
    channel, and vetoing it would withhold a valid submission. Transplanted
    verbatim onto this path that condition is not merely inert but inverted —
    by definition NO terminal tool was called on the turn that stopped, so it
    would key on the schema of a tool this run never used, and would switch the
    floor OFF for precisely the tenants whose message-carrying terminal tool
    went uncalled: the runs that delivered nothing through any channel.

    What the exemption actually asks is "has the answer already reached the user
    by some route other than prose?", and on this path exactly one thing makes
    that true: a terminal tool result is ALREADY in history AND that tool
    carries the answer in its args. That state is reachable — the
    guaranteed-terminal backstop submits on the model's behalf and then falls
    through to this same completion — so the exemption is kept, but conditioned
    on a submission having actually happened rather than on the shape of a tool
    that might never have been called. A BACKGROUND terminal whose result is in
    history is deliberately NOT exempt: its args carry no answer, so prose
    remains the only user-facing surface and the floor still applies.

    Cheap and side-effect free.
    """

    rc = engine.config.rc
    if not rc.finalize_prose_gate_enabled:
        return False
    if not run_has_final_answer(engine):
        return False
    terminal_tool = _resolved_terminal_tool_name(engine)
    if (
        terminal_tool is not None
        and _history_has_terminal_tool_result(engine)
        and _terminal_tool_carries_answer_field(engine, terminal_tool)
    ):
        return False
    if not _run_did_non_terminal_work(engine, terminal_tool or ""):
        return False
    if _has_visible_assistant_prose_after_work(
        engine, terminal_tool or "", rc.finalize_prose_gate_min_chars
    ):
        if _pointer_answer_repair_budget_spent(engine):
            return False
        return _pointer_answer_evidence(engine) is not None
    return not getattr(engine, "_finalize_prose_gate_used", False)


def _run_did_non_terminal_work(engine: QueryEngine, terminal_tool: str) -> bool:
    """True iff this run called at least one non-terminal tool.

    The floor is a length test, and a length test cannot by itself tell a reply
    that COLLAPSED from one that is correctly brief. What separates them is
    whether there was anything to report: a run that searched, delegated or
    wrote files and then answered in a few dozen characters has under-reported
    its own work, while a run that answered a greeting without touching a tool
    has reported everything it had. Without this condition the floor fires on
    the second case too, and the repair turn's only possible effect is to pad a
    correct short answer up to the threshold.

    Reads :func:`_this_run_messages` for the same reason
    :func:`_has_visible_assistant_prose_after_work` does: the obligation
    belongs to the work THIS run did, not to what an earlier run of the session
    left in history.
    """

    for message in _this_run_messages(engine):
        for block in message.content_blocks:
            if _is_non_terminal_tool_activity(block, terminal_tool):
                return True
    return False


#: The two spellings a write tool's target path arrives under. The canonical
#: field is ``path`` with ``file_path`` as a validation alias, and the model
#: reaches for either — the same pair :mod:`protocore.runtime.pending_reads` and
#: :mod:`protocore.runtime.longfile_convergence` resolve for their own reasons.
_WRITE_PATH_ARG_NAMES: Final[tuple[str, ...]] = ("file_path", "path")


def _parse_write_call(arguments_json: str) -> tuple[str, int] | None:
    """The ``(path, content_chars)`` a write call carries, or None.

    Reads the call's OWN arguments, which is where the content the run produced
    actually is: the tool result reports bytes and outcomes, the arguments hold
    the text. Core deliberately does not go looking on disk for it — it has no
    workspace, no filesystem contract and no business acquiring one, and the
    history it already owns answers the question.

    Returns None for anything that is not a write of textual content to a named
    path: unparseable arguments (they crossed a provider boundary), a missing or
    empty ``content``, or no resolvable path. A write whose path cannot be read
    is dropped rather than guessed at, because the pointer test's other half is
    "the answer names THIS file" and there is nothing to name.
    """

    try:
        arguments = json.loads(arguments_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(arguments, Mapping):
        return None
    content = arguments.get(CHUNKABLE_CONTENT_FIELD)
    if not isinstance(content, str) or not content:
        return None
    for key in _WRITE_PATH_ARG_NAMES:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), len(content)
    return None


def _run_written_content_by_path(engine: QueryEngine) -> dict[str, int]:
    """Characters of content THIS run wrote, accumulated per target file.

    Counts a write only once its result landed without error: a rejected write
    produced nothing, and treating it as a deliverable would let a failed run
    demand a longer answer about a file that does not exist. The tools that
    count are ``rc.terminal_tool_nudge_file_write_tool_names`` — the same tuple
    the write-first nudge already uses, so a tenant with differently named write
    tools configures them in one place and no the host tool name is hardcoded
    in core.

    Per PATH, not per call: a chunked deliverable arrives as one ``Write``
    followed by several ``AppendFile`` calls, and what the run produced is their
    sum. Two different files stay two entries — a run that writes a 12 KB report
    and a 200-byte scratch note has produced one document, not one 12.2 KB
    document, and the pointer test asks its question of each separately.

    Reads :func:`_this_run_messages`, so the prior-run turns cross-run history
    seeding prepends are excluded for the same reason they are excluded
    everywhere else here: the obligation belongs to the work THIS run did.
    Runtime-synthesised writes are NOT excluded — a salvage write puts real
    content in the file whoever emitted the call.
    """

    write_names = set(engine.config.rc.terminal_tool_nudge_file_write_tool_names)
    if not write_names:
        return {}
    attempted: dict[str, tuple[str, int]] = {}
    landed: set[str] = set()
    for message in _this_run_messages(engine):
        for block in message.content_blocks:
            if isinstance(block, ToolUseBlock):
                if _strip_tool_name_prefix(block.name) not in write_names:
                    continue
                parsed = _parse_write_call(block.arguments_json)
                if parsed is not None:
                    attempted[block.tool_call_id] = parsed
            elif isinstance(block, ToolResultBlock) and not block.is_error:
                landed.add(block.tool_call_id)
    written: dict[str, int] = {}
    for call_id, (path, content_chars) in attempted.items():
        if call_id in landed:
            written[path] = written.get(path, 0) + content_chars
    return written


def _visible_answer_after_work(engine: QueryEngine, terminal_tool: str) -> str:
    """The visible prose this run delivered as its ANSWER, joined.

    The same window :func:`_has_visible_assistant_prose_after_work` already
    calls the answer — every visible assistant text block after the run's latest
    non-terminal tool activity — read for its content rather than tested against
    a floor. Sharing the definition matters: this module has already ruled that
    prose emitted BEFORE the last real work is progress narration and not an
    answer, and a second, wider reading of "what the user got" here would let
    that narration pay for a reply that says nothing.

    Runtime-synthesised assistant turns (:data:`SYNTHETIC_RECOVERY_METADATA_KEY`
    — the ``(empty)`` placeholder, the guaranteed-terminal scaffolding) are the
    runtime's own words, not delivered content, and are excluded exactly as
    :func:`_latest_durable_answer_text` excludes them. Their tool activity is
    still work: a synthetic write moves the window forward like any other.
    """

    answer_parts: list[str] = []
    for message in _this_run_messages(engine):
        scaffolding = bool(message.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY))
        partial_attempt = (
            message.metadata.get(PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY) is True
        )
        for block in message.content_blocks:
            if _is_non_terminal_tool_activity(block, terminal_tool):
                # Work landed — everything said before it was narration about
                # work in progress, not the report on it.
                answer_parts.clear()
                continue
            if (
                message.role is MessageRole.assistant
                and not scaffolding
                and not partial_attempt
                and isinstance(block, TextBlock)
                and block.text.strip()
            ):
                answer_parts.append(block.text.strip())
    return "\n".join(answer_parts)


def _answer_names_file(answer: str, path: str) -> bool:
    """True iff ``answer`` refers the reader to ``path``.

    A plain substring test on the full path or on the bare filename, over
    POSIX-normalised, case-folded text — the model writes the path back in
    whatever form it likes (backquoted, inside a sentence, with or without the
    workspace prefix it wrote through), and every one of those forms contains
    one of the two.

    Deliberately permissive, because it is not the discriminating half of the
    pointer test: an answer that happens to contain the word ``notes`` while a
    file named ``notes`` exists still has to be a small fraction of that file's
    content before anything fires. What this rules out is the run that wrote a
    document and then answered about something else entirely, where refusing the
    answer would be refusing it for a file it never claimed to be reporting.
    """

    haystack = answer.replace("\\", "/").casefold()
    needle = path.strip().replace("\\", "/").rstrip("/")
    while needle.startswith("./"):
        needle = needle[2:]
    needle = needle.casefold()
    if not needle:
        return False
    if needle in haystack:
        return True
    basename = needle.rsplit("/", 1)[-1]
    return bool(basename) and basename in haystack


def _pointer_answer_evidence(engine: QueryEngine) -> tuple[str, int, int] | None:
    """The file this run's answer merely points at: ``(path, answered, written)``.

    The failure a length floor cannot see. The model is asked for an article,
    writes 13 KB of one into a workspace file, and reports back: *"Готово.
    Статья … сохранена в ``workspace/article_fem_elasticity.md`` (13 031 байт,
    ~960 слов). Структура статьи: 1. Введение …"*. Measured on a live stand that
    is the outcome of roughly two runs in three, and the reply is 1 200-1 900
    characters long — so it clears any floor an operator would set, the
    substantive-answer machinery never fires, and the run is scored a success.
    For a user with a file browser it IS a success. For a chat user it is an
    empty reply about a file they cannot open.

    So the test is not "how long is the answer" but "how does the answer compare
    to what the run produced": a reply that names a file this run wrote and
    carries a small fraction of what went into it is a filing notice, whatever
    its absolute length. Both quantities are in the run's own history — the
    write call's arguments hold the content, the assistant text is the answer.

    Returns None (nothing fires) unless ALL hold:

      * ``rc.workspace_visible_to_user`` is False. Where the user can open the
        workspace, a path IS an answer and this whole mechanism is inert — which
        is the default, so no deployment inherits it by upgrading;
      * both knobs are live —
        ``finalize_prose_gate_pointer_max_answer_fraction`` > 0 and
        ``finalize_prose_gate_pointer_min_written_chars`` > 0. Either at zero
        disables the pointer test without touching the length floor;
      * some single file (:func:`_run_written_content_by_path`) received at
        least ``_min_written_chars`` of content across this run's successful
        writes. A scratch file, a config, a patch — anything below that floor —
        can be pointed at freely;
      * the answer (:func:`_visible_answer_after_work`) names that file, and
      * the answer is shorter than ``_max_answer_fraction`` of what was written
        into it.

    When several files qualify the largest is reported, so the DIAG names the
    deliverable rather than whichever entry happened to be first.

    WHAT THIS LETS THROUGH, on purpose. A real summary that also says where the
    file lives passes: at the default fifth, a 13 KB article answered with 3 KB
    of its substance is an answer, and the run keeps it. A run whose answer
    carries the content and ALSO wrote it to a file passes, because the answer
    is not small. A run that wrote nothing substantial passes however terse its
    reply — that is the plain length floor's business, not this one's. And the
    user who explicitly asked for a file still gets the file: nothing here
    touches the write, only the claim that mentioning it was an answer.

    The remaining false positive is a run that delivers the content as prose
    FIRST and only afterwards writes it to a file and files a notice about it —
    the answer window then holds just the notice. It costs that run one repair
    turn, in exchange for the two-in-three failure this catches, and the repair
    text asks for a summary rather than a re-paste, so the second reply is not a
    duplicate of the first.

    Pure / total — reads history and RC, never the filesystem.
    """

    rc = engine.config.rc
    if rc.workspace_visible_to_user:
        return None
    max_fraction = rc.finalize_prose_gate_pointer_max_answer_fraction
    min_written = rc.finalize_prose_gate_pointer_min_written_chars
    if max_fraction <= 0.0 or min_written <= 0:
        return None
    written = _run_written_content_by_path(engine)
    if not written:
        return None
    answer = _visible_answer_after_work(
        engine, _resolved_terminal_tool_name(engine) or ""
    )
    if not answer:
        return None
    answer_chars = len(answer)
    evidence: tuple[str, int, int] | None = None
    for path, written_chars in written.items():
        if written_chars < min_written:
            continue
        if answer_chars >= max_fraction * written_chars:
            continue
        if not _answer_names_file(answer, path):
            continue
        if evidence is None or written_chars > evidence[2]:
            evidence = (path, answer_chars, written_chars)
    return evidence


def _pointer_answer_repair_budget_spent(engine: QueryEngine) -> bool:
    """True when this run may not spend another turn on the pointer refusal.

    The bound is on the run and never resets. Its neighbour
    :mod:`protocore.runtime.pending_reads` resets its counter on a productive
    read, because there "productive" is a fact: the declared file was opened or
    it was not. Here there is no such fact to reset on — the pointer test IS the
    definition of progress, and while it still says "pointer" the mechanism has,
    by its own reading, achieved nothing. A counter that reset on partial
    progress would let a model that grows its filing notice by fifty characters
    a turn hold the budget open indefinitely: an unbounded loop reached by a
    strictly improving sequence, which is the one outcome worse than the failure
    being repaired.

    Cheap (one comparison) and side-effect free, so the predicates that call it
    stay cheap; the warning that says the budget ran out lives in
    :func:`_release_pointer_answer_repair`, which is a separate call precisely
    so that this one can be asked freely.

    A budget of 0 makes this True from the start, which switches the pointer
    refusal off while leaving the plain length floor untouched — a third kill
    switch alongside the two knobs :func:`_pointer_answer_evidence` already
    honours.
    """

    return (
        engine._pointer_answer_repair_attempts
        >= engine.config.rc.finalize_prose_gate_pointer_max_repair_attempts
    )


def _charge_pointer_answer_repair(engine: QueryEngine) -> int:
    """Spend one of the pointer refusal's attempts; returns the new count.

    Charged when a repair turn is actually INJECTED, at either seam, and never
    otherwise — a turn the refusal wanted and could not take (an empty repair
    text degrades the whole gate to a no-op) costs nothing, exactly as an
    unforceable read costs :func:`pending_reads.charge_forced_attempt` nothing.

    ON A REPAIR THAT PARTLY WORKS — a second answer that grows but is still
    principally a pointer. It consumes an attempt, like any other. What is
    bounded here is the number of times the runtime asks, not the number of
    times it is refused, and the cost being bounded is paid on asking: a full
    answer regeneration, appended to the context the next answer is written
    from. Crediting partial progress back would also require a threshold for
    "enough progress" that nothing in the run can supply — the only measure
    available is the pointer test itself, which is still returning "pointer".

    Nothing is lost by that strictness, because partial progress is already
    banked where it counts. :func:`_visible_answer_after_work` reads the whole
    window since the last real work, so a second reply ACCUMULATES on to the
    first rather than replacing it: a model that adds real substance each turn
    climbs toward ``finalize_prose_gate_pointer_max_answer_fraction`` and the
    refusal stops firing of its own accord, budget unspent. The run that burns
    all three attempts is the run that added nothing across all three.
    """

    engine._pointer_answer_repair_attempts += 1
    return engine._pointer_answer_repair_attempts


def _release_pointer_answer_repair(engine: QueryEngine) -> None:
    """Say, once, that a run is finishing on an answer the refusal rejected.

    Called where the gate declined to fire, so it reports the outcome rather
    than predicting it: reaching here with the budget spent AND the evidence
    still standing means the last repair turn did not work either and the user
    is about to receive a filing notice. That is the fact an operator needs in
    order to decide whether the budget is too small or the mechanism is not
    worth its turns, and it is exactly the fact a line emitted at charge time
    could not carry — at that moment the last attempt has not been answered yet.

    The distinction it draws is the point. A run that spent every attempt and
    then succeeded ends silent here, because the evidence is gone; only the
    failure speaks. The latch keeps it to one line per run, so a resumed or
    multi-dispatch run cannot turn one outcome into a stream of them.

    Cheap on the paths that call it: a run whose pointer refusal never engaged
    returns on the first comparison, which is what lets this sit on the
    per-dispatch path without being felt.
    """

    if engine._pointer_answer_repair_attempts == 0:
        return
    if engine._pointer_answer_repair_released:
        return
    if not _pointer_answer_repair_budget_spent(engine):
        return
    evidence = _pointer_answer_evidence(engine)
    if evidence is None:
        return
    path, answer_chars, written_chars = evidence
    engine._pointer_answer_repair_released = True
    _logger.warning(
        "DIAG query.finalize_prose_gate.pointer_answer_budget_spent "
        "run=%s tenant=%s turn=%s attempts=%d/%d answer_chars=%d "
        "written_chars=%d path=%s",
        engine.config.run_id,
        engine.config.tenant_id,
        engine.turn_id(),
        engine._pointer_answer_repair_attempts,
        engine.config.rc.finalize_prose_gate_pointer_max_repair_attempts,
        answer_chars,
        written_chars,
        path,
    )


def _append_answer_floor_repair_turn(engine: QueryEngine) -> None:
    """Append the ONE bounded repair turn the plain-stop answer floor grants.

    The same synthetic user turn the dispatch-path veto appends, carrying the
    same :data:`SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR` marker so every consumer
    that already strips runtime scaffolding out of durable history, the memory
    fold and ``result_preview`` keeps doing so with no change.

    No error ``tool_result`` precedes it here. The terminal path has a vetoed
    dispatch to answer for; this path has nothing to veto — the model stopped of
    its own accord. The tail of history is therefore either this turn's
    assistant message or, when the stop carried no content at all, the last
    tool result, and a user turn is a valid successor to both (the dispatch-path
    veto appends exactly that after its own tool result).
    """

    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[
                TextBlock(text=engine.config.rc.finalize_prose_gate_repair_text)
            ],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: (
                    SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR
                )
            },
        )
    )


def _count_terminal_answer_cited_refs(tool_call: ToolCall) -> int:
    """Count cited refs on an un-submitted terminal call.

    PURE / total — reads the canonical ``refs`` slot off ``tool_call.arguments``
    (falling back to the legacy ``sources`` alias only when ``refs`` is
    absent/empty), counting non-empty string entries. Returns 0 for any
    unexpected shape; never raises. Used only for the heartbeat DIAG —
    no behavioural effect.
    """

    arguments = tool_call.arguments
    if not isinstance(arguments, Mapping):
        return 0
    for key in (_TERMINAL_ANSWER_REFS_KEY, _TERMINAL_ANSWER_REFS_LEGACY_ALIAS):
        raw = arguments.get(key)
        if isinstance(raw, list):
            count = sum(
                1 for item in raw if isinstance(item, str) and item.strip()
            )
            if count:
                return count
    return 0


def _observed_ref_ledger_size(engine: QueryEngine) -> int:
    """Best-effort size of the per-run observed-ref ledger.

    Reads the ledger off the OPAQUE helper bag (``engine._helpers``) that core
    forwards verbatim into ``ToolContext.metadata``; the host read tools
    populate it under :data:`_OBSERVED_REF_LEDGER_HELPER_KEY` (a ``set[str]``).
    This is a plain mapping lookup, NOT a host import, so core stays
    import-boundary-pure. Returns :data:`_HEARTBEAT_OBSERVED_UNAVAILABLE`
    (``-1``) when the bag/ledger is absent or not a sized collection — the
    heartbeat then reports "observed not cheaply visible from core" rather than
    a misleading 0. Pure / total — never raises, no behavioural effect.
    """

    helpers = getattr(engine, "_helpers", None)
    if not isinstance(helpers, Mapping):
        return _HEARTBEAT_OBSERVED_UNAVAILABLE
    bucket = helpers.get(_OBSERVED_REF_LEDGER_HELPER_KEY)
    try:
        return len(bucket)  # type: ignore[arg-type]
    except TypeError:
        return _HEARTBEAT_OBSERVED_UNAVAILABLE


def _observed_ref_ledger_refs(engine: QueryEngine) -> list[str]:
    """Return the real banked paths for the guaranteed-terminal backstop refs.

    Returns the sorted union of the per-run CONTENT-READ ledger (paths whose
    BODY the run actually read) and the path-OBSERVED ledger, read off the
    OPAQUE helper bag (``engine._helpers``) that core forwards verbatim —
    the host read/list/tree tools populate these buckets under
    :data:`_CONTENT_READ_LEDGER_HELPER_KEY` / :data:`_OBSERVED_REF_LEDGER_HELPER_KEY`
    (``set[str]``). The content-read set is preferred (those are the files the
    model truly opened); the path-observed set is appended only as a fallback so
    a run that listed but never full-read still cites something real.

    These are the REAL, harness-returned paths the run banked — NEVER fabricated.
    This is a plain mapping lookup, NOT a host import, so core stays
    import-boundary-pure (guard: ``tests/test_core_import_boundary.py``). Returns
    an empty list when the bag/ledgers are absent or not iterable string sets —
    the backstop then submits a message-only answer (degrades gracefully, never
    raises, no fabrication). Pure / total.
    """

    helpers = getattr(engine, "_helpers", None)
    if not isinstance(helpers, Mapping):
        return []

    def _string_set(key: str) -> set[str]:
        bucket = helpers.get(key)
        if not isinstance(bucket, (set, frozenset, list, tuple)):
            return set()
        return {item for item in bucket if isinstance(item, str) and item.strip()}

    content_read = _string_set(_CONTENT_READ_LEDGER_HELPER_KEY)
    path_observed = _string_set(_OBSERVED_REF_LEDGER_HELPER_KEY)
    # Content-read first (the files the model truly opened), then any path-only
    # observation as a real fallback. Deterministic order for stable history.
    return sorted(content_read | path_observed)


def _pre_dispatch_terminal_verify_applies(
    engine: QueryEngine, tool_call: ToolCall
) -> bool:
    """Gate for the PRE-DISPATCH terminal-tool veto.

    Returns True iff the pre-dispatch verify seam should consult its
    host-supplied trigger for ``tool_call``. ALL must hold:

      * ``rc.pre_dispatch_terminal_verify_enabled`` is True,
      * a per-tenant terminal tool is declared
        (``config.expected_terminal_tool``) AND ``tool_call`` IS that tool —
        the seam never intercepts a non-terminal tool, so reads/writes/exec
        dispatch unaffected,
      * the durable per-run latch ``_pre_dispatch_terminal_verify_used`` is
        unset (fire-at-most-once across resume), and
      * the shared corrective-turn budget
        (``rc.pre_terminal_self_verify_max_extra_turns`` vs
        ``_self_verify_extra_turns_used``) is not exhausted,
      * a host trigger is actually wired.

    Returns False (no interception) otherwise.
    Cheap and side-effect-free so ``_dispatch_tool`` can call it on every
    dispatch without the per-call trigger cost when the gate is closed.
    """

    rc = engine.config.rc
    if not getattr(rc, "pre_dispatch_terminal_verify_enabled", False):
        return False
    expected = engine.config.expected_terminal_tool
    if expected is None or tool_call.name != expected:
        return False
    if getattr(engine, "_pre_dispatch_terminal_verify_used", False):
        return False
    if (
        getattr(engine, "_self_verify_extra_turns_used", 0)
        >= rc.pre_terminal_self_verify_max_extra_turns
    ):
        return False
    return engine.config.pre_dispatch_terminal_verify_trigger is not None


def _resolve_pre_dispatch_terminal_veto(
    engine: QueryEngine, tool_call: ToolCall
) -> str | None:
    """Consult the pre-dispatch trigger; return the corrective message if the
    terminal dispatch must be VETOED, else ``None``.

    Assumes :func:`_pre_dispatch_terminal_verify_applies` already returned
    True. Invokes the host-supplied predicate with the engine + the
    UN-SUBMITTED ``ToolCall`` (so the predicate can inspect the answer the
    model is about to send against the observed-ref ledger). A trigger
    exception is swallowed (never break dispatch) and treated as "no veto".
    The caller owns the latch/counter mutation + history injection so this
    stays a pure read.
    """

    trigger = engine.config.pre_dispatch_terminal_verify_trigger
    if trigger is None:  # pragma: no cover - guarded by _applies
        return None
    try:
        corrective = trigger(engine, tool_call)
    except Exception as exc:  # pragma: no cover - defensive; never break dispatch
        _logger.warning(
            "DIAG query.pre_dispatch_terminal_verify.trigger_failed "
            "run=%s error=%s",
            engine.config.run_id,
            exc,
        )
        return None
    if not corrective:
        return None
    return corrective


def _append_terminal_tool_nudge(engine: QueryEngine) -> None:
    """Append the contract-repair nudge message + arm the terminal-only latch."""

    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=_resolved_terminal_tool_nudge_text(engine))],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: (
                    SYNTHETIC_RECOVERY_TERMINAL_TOOL_NUDGE
                )
            },
        )
    )
    # Flip the terminal-only guard latch once the nudge has been emitted so
    # subsequent dispatch calls reject every non-terminal tool with a
    # structured error pointing at finalisation. The latch persists for the
    # remainder of the run; it is implicitly cleared when
    # ``_history_has_terminal_tool_result`` returns True (the predicate
    # consulted at dispatch time).
    engine._terminal_only_active = True


# Answer-carrying field names a terminal tool may expose, in priority order.
# The guaranteed-terminal synthesiser maps the last-resort answer text to the
# FIRST of these that the terminal tool actually declares — so a legacy lean
# message-carrying contract (``message``) still receives the text in its own
# schema, with no per-tool hard-coding in core. A BACKGROUND terminal tool (the
# ``Finalize`` shape: its ``answer`` field removed, only
# ``declared_deliverables`` remaining) declares NONE of these names, so the
# synthesiser injects NO answer text and the prose-gate guarantees the
# answer is the model's prose. This same signal gates the prose-gate itself
# (:func:`_terminal_tool_carries_answer_field`).
_TERMINAL_TOOL_ANSWER_ARG_NAMES: tuple[str, ...] = ("message", "answer", "text")


# The file-target arg names the write-family tools use. They accept both
# ``path`` and the ``file_path`` alias, so recovery message/shape
# detection checks either.
_WRITE_PATH_KEYS: tuple[str, ...] = ("path", "file_path")


def _truncated_call_state_path(tool_call: ToolCall) -> str | None:
    """The REAL file-target of a truncated mutation call, or ``None``.

    The state-only twin of :func:`_truncated_call_path`: it NEVER returns the
    ``"the target file"`` display placeholder. Only this resolved path may be
    fed into convergence state (``note_truncated_mutation`` / active-path
    handoff) — latching the placeholder would poison ``_longfile_active_path``
    so a later real Write to the actual file is ignored as off-path.
    """
    args = tool_call.arguments
    if isinstance(args, dict):
        for key in _WRITE_PATH_KEYS:
            value = args.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _truncated_call_path(tool_call: ToolCall) -> str:
    """Best-effort extract the file-target of a truncated mutation call (display).

    The chunk-recovery message names the file path so the model resumes the
    SAME file. Even a truncated call usually preserves its
    target (``path`` / ``file_path`` is the first, small field; the output cap
    cut the large ``content`` after it — exactly the prod ``{"path": ...}``
    shape). Falls back to a generic placeholder when no target is recoverable
    (the cut landed before it, or the call surfaced as a raw envelope). This is
    the MODEL-VISIBLE display path; convergence STATE must use
    :func:`_truncated_call_state_path` (which never returns the placeholder).
    """
    return _truncated_call_state_path(tool_call) or "the target file"


def _truncated_call_paths(tool_calls: list[ToolCall]) -> list[str]:
    """Ordered, de-duplicated list of recoverable paths for telemetry."""
    seen: list[str] = []
    for tc in tool_calls:
        path = _truncated_call_path(tc)
        if path not in seen:
            seen.append(path)
    return seen


def _is_content_mutation_truncation(engine: QueryEngine, tool_call: ToolCall) -> bool:
    """True iff a truncated call is a chunkable file-content mutation.

 The Write->AppendFile->FinalizeFile chunk recovery only makes sense for
 a KNOWN chunkable content-mutation tool. Delegates to the ONE
 shared predicate
 (:func:`protocore.contracts.tool_chunking.is_chunkable_content_mutation`)
 that the host LLM client uses too, so both layers route identically:
 the call's tool must REQUIRE ``content`` (the body the cap cut) AND be
 explicitly flagged (``ToolParameterSchema.chunkable_content_mutation``) OR on
 the narrow built-in allowlist (``Write``/``AppendFile``). The call must also
 be missing ``content`` (the cut-body shape). This EXCLUDES a tool like
 ``Read`` (no ``content``) and — the fix — a dynamic/tenant tool that
 merely declares a ``content`` field without opting in: such a call gets the
 generic "re-issue with complete arguments" resume instead. Schema lookup uses
 the engine's tool registry (in-process; no boundary violation). An unknown
 tool is treated as non-content (safe default — generic resume).
 """
    args = tool_call.arguments
    if not isinstance(args, dict):
        return False
    if "content" in args:
        # ``content`` already present → not the cut-content shape.
        return False
    required, chunkable_flag = _tool_content_schema(engine, tool_call.name)
    return is_chunkable_content_mutation(
        tool_name=tool_call.name,
        required=required,
        chunkable_flag=chunkable_flag,
    )


def _tool_content_schema(
    engine: QueryEngine, tool_name: str
) -> tuple[list[str], bool | None]:
    """Best-effort ``(required, chunkable_content_mutation)`` for a tool.

    Reads the registered tool's ``definition.parameters`` (the same typed
    schema the host client indexes). Returns ``([], None)`` when the tool
    is unknown or its schema cannot be read — the shared predicate then treats
    it as non-content (generic resume).
    """
    if not tool_name:
        return ([], None)
    tool = engine.tools.get(tool_name)
    if tool is None:
        return ([], None)
    try:
        params = tool.definition.parameters
        required = list(getattr(params, "required", []) or [])
        chunkable_flag = getattr(params, "chunkable_content_mutation", None)
    except AttributeError:
        return ([], None)
    return (required, chunkable_flag if isinstance(chunkable_flag, bool) else None)


def _salvage_truncated_content(tool_call: ToolCall) -> str:
    """The recoverable partial ``content`` body of a truncated chunkable write.

    Mirrors the stand's ``case_e._salvage_partial`` intent
    against the PROD shape. In prod the host SSE parser already stage-4
    brace-balances the cut args and ``json.loads``-es them, so a truncated Write
    whose body was cut MID-string surfaces here as ``arguments['content']`` — an
    already-VALID (just early-terminated) Python ``str``. No raw-JSON trim is
    needed (that belongs at the raw layer; ``arguments`` never carries raw text —
    Returns that string, or ``''`` when ``content`` is absent / not a
    non-empty str (the "cut before any body" shape — nothing to salvage; the
    caller then steers a smaller first chunk instead of writing an empty
    file). Only a VALID partial is ever returned — no corrupt content is
    dispatched.
    """
    args = tool_call.arguments
    if not isinstance(args, dict):
        return ""
    content = args.get("content")
    if isinstance(content, str) and content:
        return content
    return ""


def _build_truncation_chunk_recovery_text(
    engine: QueryEngine, truncated_tool_calls: list[ToolCall]
) -> str:
    """Build the recovery message for one or more truncated tool calls.

 A chunkable file-content mutation (``path`` present, ``content`` cut) gets
 the structured Write->AppendFile->FinalizeFile message naming the PATH + the
 per-call ``write_chunk_token_budget`` (the 0/4 -> 4/4 chunking protocol).
 Any OTHER truncated call (e.g. a non-content tool whose args were cut) gets
 the generic ``tool_call_truncation_resume_prompt`` so it is not misrouted
 into an irrelevant file-chunk workflow. Both EN and RU
 halves are emitted for the chunk message (EN first), per the
 multilingual rule.

 the "continue with AppendFile" directive is emitted ONLY when a
 chunk has ACTUALLY been written for this path (``path in
 engine._mid_chunked_write_paths``, populated by a SUCCESSFUL Write/AppendFile
 dispatch). A repeat truncation BEFORE any chunk landed keeps the
 first-message ``Write(header)`` protocol (never tells the model to
 ``AppendFile`` a non-existent file) but LOWERS the header budget per prior
 no-success prompt for that path (so the next header attempt is smaller than
 the one that just truncated). This function does NOT add to
 ``_mid_chunked_write_paths`` — only a real successful write does (see
 :func:`_record_chunk_write_success`).
 """
    rc = engine.config.rc
    sections: list[str] = []
    for tc in truncated_tool_calls:
        if not _is_content_mutation_truncation(engine, tc):
            # Non-content truncation — generic resume, no file-chunk protocol.
            sections.append(
                rc.tool_call_truncation_resume_prompt.format(tool_name=tc.name)
            )
            continue
        path = _truncated_call_path(tc)
        already_chunking = path in engine._mid_chunked_write_paths
        if already_chunking:
            # A chunk has really been written → steer to AppendFile, full budget.
            chunk_budget = rc.write_chunk_token_budget
            mid_note_en = rc.truncation_chunk_recovery_mid_chunked_note_en.format(
                path=path
            )
            mid_note_ru = rc.truncation_chunk_recovery_mid_chunked_note_ru.format(
                path=path
            )
        else:
            # No chunk written yet → keep Write(header); LOWER the header budget
            # by one division step per prior no-success prompt for this path.
            prior_prompts = engine._truncation_recovery_prompt_counts.get(path, 0)
            chunk_budget = _lowered_header_budget(rc, prior_prompts)
            engine._truncation_recovery_prompt_counts[path] = prior_prompts + 1
            mid_note_en = ""
            mid_note_ru = ""
        en = rc.truncation_chunk_recovery_message_en.format(
            tool_name=tc.name,
            path=path,
            chunk_budget_tokens=chunk_budget,
            mid_chunked_note_en=mid_note_en,
        )
        ru = rc.truncation_chunk_recovery_message_ru.format(
            tool_name=tc.name,
            path=path,
            chunk_budget_tokens=chunk_budget,
            mid_chunked_note_ru=mid_note_ru,
        )
        sections.append(f"{en}\n\n{ru}")
    return "\n\n".join(sections)


def _lowered_header_budget(rc: RuntimeConstants, prior_prompts: int) -> int:
    """Header chunk budget for a repeat truncation before any chunk was written.

 Integer-divide ``write_chunk_token_budget`` by
 ``truncation_chunk_recovery_repeat_budget_divisor`` once per prior no-success
 recovery prompt for the path, floored at
 ``truncation_chunk_recovery_min_chunk_token_budget`` (itself clamped to the
 base budget). ``prior_prompts == 0`` → the full base budget (first
 truncation). A divisor of 1 (or a min == base) yields a constant budget.
 """
    base = rc.write_chunk_token_budget
    floor = min(rc.truncation_chunk_recovery_min_chunk_token_budget, base)
    divisor = rc.truncation_chunk_recovery_repeat_budget_divisor
    budget = base
    for _ in range(max(0, prior_prompts)):
        if divisor > 1:
            budget //= divisor
        if budget <= floor:
            return floor
    return max(budget, floor)


def _record_chunk_write_success(engine: QueryEngine, tool_call: ToolCall) -> None:
    """Mark a path as "chunking started" after a SUCCESSFUL chunkable write.

 Called from the successful-dispatch path for a Write/AppendFile that
 carries a ``content`` body and a resolvable target
 path. Adds the path to ``engine._mid_chunked_write_paths`` (so a later repeat
 truncation of that path gets the "continue with AppendFile" directive) and
 clears its no-success prompt count. A non-chunkable tool, or a write whose
 target cannot be resolved, is a no-op.
 """
    name = tool_call.name
    if name not in CHUNKABLE_CONTENT_MUTATION_ALLOWLIST:
        # Only the runtime's own chunk tools advance the "chunking started"
        # state; a per-tenant flagged tool drives recovery wording but its
        # append-resume semantics are tool-specific, so it is not tracked here.
        return
    args = tool_call.arguments
    if not isinstance(args, dict) or "content" not in args:
        return
    path = _truncated_call_path(tool_call)
    if path == "the target file":
        return
    engine._mid_chunked_write_paths.add(path)
    engine._truncation_recovery_prompt_counts.pop(path, None)


async def _salvage_truncated_write_to_disk(
    engine: QueryEngine,
    tool_call: ToolCall,
    salvaged_content: str,
) -> AsyncIterator[TurnEvent]:
    """land a truncated write's recovered partial on disk.

 The stand-validated salvage path ported to prod: the model's
 truncated Write/AppendFile carried a partial ``content`` body that the SSE
 parser recovered; this dispatches that VALID partial as a CLEAN synthetic
 write so bytes land on disk + are byte-tracked, after which the truncation-
 gated convergence driver can engage on the genuine target (long-en-004's
 stand 3/3). Without this prod discards the partial → 0 bytes land → the
 driver stays inert on the real file.

 Pairing-safe + snapshot-safe:
 * a NEW synthetic ``ToolCall`` (fresh id) is built with the resolved path
 + salvaged content; its tool name is remembered;
 * a MATCHING assistant ``ToolUseBlock`` is appended FIRST (flagged
 synthetic-recovery scaffold) so ``_dispatch_tool`` — which appends ONLY
 the tool RESULT — never produces an orphan result;
 * ``note_truncated_mutation`` runs BEFORE dispatch so the STICKY
 truncation latch + active-path handoff are set, and ``_dispatch_tool``'s
 post-dispatch ``observe_tool_result`` (which lands the bytes + clears the
 transient ``_longfile_last_mutation_truncated``) and snapshot persist
 capture the latch atomically with the bytes;
 * the salvage REUSES the model's OWN tool name (``tool_call.name``), which
 the caller's classifier already constrains to a built-in chunkable write
 (``Write`` / ``AppendFile``). The model's declared op IS the write
 semantics: a ``Write`` is a from-scratch REPLACE (so re-emitting a full
 truncated ``Write`` after a prior chunk landed overwrites, NOT
 duplicates, the prefix — the documented Write-spiral shape) and an
 ``AppendFile`` is a CONTINUE (so salvaging it as a fresh ``Write`` would
 destroy prior content created via Bash/sandbox or under a different
 active path). Inferring the op from the persisted active-file size
 (``>0 → AppendFile``) instead ignored that name and corrupted both shapes;
 * dispatched ``preapproved=True`` (runtime-internal salvage of the model's
 OWN content — never a new approval surface).

 Yields the dispatch events. The caller treats it as ONE recovery round
 (the shared ``_max_output_recovery_count`` is already debited by the branch).
 """
    state_path = _truncated_call_state_path(tool_call)
    if state_path is None:
        # Defensive: the caller's classifier already requires a resolvable path,
        # so this never fires for a salvage_job — but never dispatch without one.
        return
    # Reuse the model's OWN tool name (the caller already constrains it to the
    # built-in chunkable allowlist — ``Write`` / ``AppendFile``). The declared op
    # is the authoritative write semantics: a ``Write`` REPLACES from scratch (so
    # a re-emitted full truncated Write overwrites — never duplicates — a prefix
    # that landed in a prior round) and an ``AppendFile`` CONTINUES (so it is
    # never rewritten into a whole-file Write that would wipe content created via
    # Bash/sandbox or under a different active path). Inferring from the persisted
    # active-file size corrupted both shapes.
    salvage_name = tool_call.name
    # A per-run monotonic counter makes the synthetic id UNIQUE across
    # multiple salvages in one run. ``turn_id()`` advances per
    # assistant-message round, not per salvage, so two salvages in one round
    # share a ``turn_id`` and deriving from it alone would collide → the
    # outbound pairing repair would drop the later duplicate. Snapshot-persisted.
    engine._longfile_salvage_seq += 1
    synthetic_id = (
        f"toolu-longfile-salvage-{engine.config.run_id}-{engine._longfile_salvage_seq}"
    )
    salvage_call = ToolCall(
        id=synthetic_id,
        name=salvage_name,
        arguments={"path": state_path, "content": salvaged_content},
    )
    engine.remember_tool_name(synthetic_id, salvage_name)
    # Append the matching assistant tool_use FIRST (scaffold-flagged) so the
    # tool result ``_dispatch_tool`` appends is paired in durable history.
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id=synthetic_id,
                    name=salvage_name,
                    arguments_json=json.dumps(
                        salvage_call.arguments, ensure_ascii=False
                    ),
                )
            ],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_LONGFILE_SALVAGE
            },
        )
    )
    # Set the truncation latch + active-path handoff BEFORE dispatch so the
    # snapshot ``_dispatch_tool`` persists carries it atomically with the bytes.
    _longfile.note_truncated_mutation(engine, state_path)
    # Also pre-set the transient truncated-tail
    # flag BEFORE the dispatch (and pass ``keep_truncated_tail=True`` to
    # ``_dispatch_tool`` so the post-``observe_tool_result`` clear is
    # suppressed for this synthetic-recovery call). The in-``_dispatch_tool``
    # persist now captures the flag as set, atomically with the recovered
    # bytes — closing the persist-then-re-assert window that previously let a
    # pod kill between the two persists resume with a salvaged >= floor
    # half-file evaluating ``plausibly_complete=True`` and forcing a
    # PREMATURE FinalizeFile on the next stall.
    engine._longfile_last_mutation_truncated = True
    yield _emit_state_change(
        engine,
        engine.state,
        engine.state,
        reason="longfile_truncation_salvaged_write",
    )
    async for evt in _dispatch_tool(
        engine,
        salvage_call,
        preapproved=True,
        synthetic_recovery=True,
        synthetic_recovery_kind=SYNTHETIC_RECOVERY_LONGFILE_SALVAGE,
        keep_truncated_tail=True,
    ):
        yield evt
    # Re-assert the truncated-tail flag AFTER the dispatch. The
    # synthetic Write lands the recovered FIRST chunk and ``_dispatch_tool``'s
    # post-dispatch ``observe_tool_result`` would normally CLEAR the transient
    # ``_longfile_last_mutation_truncated`` on a successful byte-landing (it
    # reads the landing as a clean write). The ``keep_truncated_tail=True``
    # above suppresses that clear, so this final re-assert is a belt-and-
    # braces backstop (covers any future observe path that forgets the flag)
    # and keeps the persist in lock-step with the in-dispatch persist.
    # Re-asserting here keeps ``plausibly_complete`` False so the driver forces
    # AppendFile (continue), not FinalizeFile (seal). This does NOT wedge
    # finalization: a later GENUINE clean (non-truncated) AppendFile clears the
    # flag again via ``observe_tool_result`` → the finalize path is reachable.
    # Placed AFTER the dispatch loop (the LAST mutation of the flag this turn) and
    # snapshot-persisted so a cross-pod resume sees the re-set, not the cleared
    # value the in-dispatch persist captured. The sticky engage gate
    # (``_longfile_truncated_paths``, set by ``note_truncated_mutation`` above) is
    # already correct; this only fixes the append-vs-finalize decision.
    engine._longfile_last_mutation_truncated = True
    await engine._persist_snapshot()


async def _maybe_seal_longfile_at_voluntary_finish(
    engine: QueryEngine,
) -> AsyncIterator[TurnEvent]:
    """seal a truncation-gated unsealed file at a
    VOLUNTARY run completion seam.

    The max-turns terminal seal (``terminal_seal_required`` exhaustion block)
    covers only budget exhaustion. A run can ALSO complete VOLUNTARILY — the
    model calls the run-terminal tool, or finishes with a prose ``end_turn`` —
    leaving a truncation-gated file unsealed (after a 004-shape recovery the
    model appends chunks then ends the run without FinalizeFile). This helper is
    called at both voluntary seams BEFORE the run completes; when
    :func:`longfile_convergence.terminal_seal_required` is True it dispatches a
    SYNTHETIC ``FinalizeFile`` for ``engine._longfile_active_path`` so the file
    is sealed and ``_longfile_finalized`` flips to True.

    Mirrors the salvage synthetic-dispatch machinery
    (:func:`_salvage_truncated_write_to_disk`): a matching assistant
    ``ToolUseBlock`` is appended FIRST (scaffold-flagged) so ``_dispatch_tool``
    — which appends only the tool RESULT — never produces an orphan; the call is
    dispatched ``preapproved=True`` (a runtime-internal deterministic seal of a
    known path, never a new approval surface); and the post-dispatch
    ``observe_tool_result`` feeds the engine so ``_longfile_finalized`` flips.
    NO LLM call, NO extra turn.

    Zero-collateral: ``terminal_seal_required`` already gates on the RC
    kill-switch, the truncation latch (never fires for a non-truncated
    file), not-already-finalized, the HARD empty/below-floor finalize guard, and
    the forced-finalize budget — so an ordinary run is bit-identical. The
    one-shot ``_longfile_voluntary_seal_used`` latch caps it to ONCE per run, and
    ``commit_forced_finalize`` charges the seal against the finalize budget.

    FinalizeFile must be on the run's tool surface for dispatch — if the tool is
    NOT in the registry (``engine.tools.get("FinalizeFile") is None``) the seal
    is skipped SILENTLY (no crash): a tenant without the chunk-protocol tools
    simply cannot be sealed this way, and that is correct (the run completes as
    before). Yields any dispatch events; no-op (yields nothing) when not eligible.
    """
    if engine._longfile_voluntary_seal_used:
        return
    if not _longfile.terminal_seal_required(engine):
        return
    # Once the STRICT terminal-only latch is in force (deadline reached,
    # ``expected_terminal_tool`` configured) the only allowed dispatch is
    # the resolved terminal tool. ``FinalizeFile`` is NOT the
    # expected terminal tool, so attempting a seal would short-circuit
    # through ``_dispatch_tool`` → ``_terminal_only_blocks`` with a
    # ``terminal_only`` is_error result: forced budget charged
    # (``commit_forced_finalize`` below), one-shot
    # ``_longfile_voluntary_seal_used`` latch consumed, the synthetic
    # assistant tool_use is in history, AND the file is left unsealed —
    # every seal-related side effect fires except the seal itself. Skip
    # cleanly so the deadline path can drive the model to its
    # ``expected_terminal_tool`` and complete.
    if _terminal_only_enforced(engine):
        _logger.warning(
            "DIAG query.longfile_seal.skipped_terminal_only run=%s tenant=%s "
            "path=%s",
            engine.config.run_id,
            engine.config.tenant_id,
            engine._longfile_active_path,
        )
        return
    path = engine._longfile_active_path
    if not path:
        # Defensive: ``terminal_seal_required`` implies a truncation-gated active
        # path, but never dispatch FinalizeFile without a concrete target.
        return
    # FinalizeFile must be advertised on this run's tool surface to dispatch.
    # A run whose tenant has no chunk-protocol tools simply cannot be sealed —
    # skip silently (the run completes as it would have without the seal).
    if engine.tools.get("FinalizeFile") is None:
        return

    engine._longfile_voluntary_seal_used = True
    engine._longfile_salvage_seq += 1
    synthetic_id = (
        f"toolu-longfile-seal-{engine.config.run_id}-{engine._longfile_salvage_seq}"
    )
    seal_call = ToolCall(
        id=synthetic_id,
        name="FinalizeFile",
        arguments={"path": path},
    )
    engine.remember_tool_name(synthetic_id, "FinalizeFile")
    # Append the matching assistant tool_use FIRST (scaffold-flagged) so the
    # tool result ``_dispatch_tool`` appends is paired in durable history.
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id=synthetic_id,
                    name="FinalizeFile",
                    arguments_json=json.dumps(seal_call.arguments, ensure_ascii=False),
                )
            ],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: (
                    SYNTHETIC_RECOVERY_LONGFILE_TERMINAL_SEAL
                )
            },
        )
    )
    # Charge the seal against the forced-finalize budget so a model that already
    # exhausted it (e.g. via the stall-driver) does not double-seal — respected
    # alongside the one-shot latch above (``terminal_seal_required`` already
    # checked budget remaining; this keeps the accounting honest).
    _longfile.commit_forced_finalize(engine)
    yield _emit_state_change(
        engine,
        engine.state,
        engine.state,
        reason="longfile_terminal_seal_voluntary_finish",
    )
    async for evt in _dispatch_tool(
        engine,
        seal_call,
        preapproved=True,
        synthetic_recovery=True,
        synthetic_recovery_kind=SYNTHETIC_RECOVERY_LONGFILE_TERMINAL_SEAL,
    ):
        yield evt
    # ``_dispatch_tool``'s post-dispatch ``observe_tool_result`` flips
    # ``_longfile_finalized=True`` on the successful FinalizeFile result + the
    # in-dispatch snapshot persists it. Persist once more so the
    # ``_longfile_voluntary_seal_used`` latch (set before the dispatch) is
    # durable for a cross-pod resume even if the dispatch result was a no-op.
    await engine._persist_snapshot()


async def _maybe_drive_longfile_convergence(
    engine: QueryEngine,
) -> AsyncIterator[TurnEvent | bool]:
    """the end-of-turn convergence step (-).

 Called ONCE at every completed-assistant-turn boundary (both the
 tool-call-turn-end seam and the prose/no-tool-turn seam). It:

 1. advances the stall clock for the just-completed turn
 (:func:`longfile_convergence.register_completed_turn` — a turn that added
 no bytes increments ``turns_since_last_byte_adding_mutation``);
 2. asks the pure decision module whether to FORCE the next tool
 (:func:`longfile_convergence.decide_next_forced_tool`);
 3. if a forced tool is decided, injects the INCOMPLETE continue
 message (with the on-disk-tail anchor), records the forced ``tool_choice``
 for the next stream, charges the per-kind forced-round budget, persists
 the snapshot (cross-pod safe), and emits a ``state_changed`` event.

 Yields any emitted :class:`TurnEvent` then a FINAL ``bool`` sentinel: True
 iff a forced action was issued (the caller must ``continue`` the outer loop
 to open the forced stream). No-op (yields ``False``) when the driver is
 disabled, no stall/plateau is detected, or all forced-round budgets are
 spent — so the happy path and disabled-RC path are bit-identical to pre-FEAT.
 """
    # The longfile convergence driver forces the next assistant stream's
    # ``tool_choice`` to ``AppendFile``/``FinalizeFile``, neither of which is
    # the resolved ``expected_terminal_tool``. Once the STRICT
    # terminal-only latch is in force (deadline reached, ``expected_terminal_tool``
    # configured) the only allowed dispatch is that terminal tool, so the
    # forced tool_choice would surface a ``terminal_only`` is_error in the
    # NEXT iteration's ``_dispatch_tool`` call — forced budget charged
    # (``commit_forced_append`` / ``commit_forced_finalize``), INCOMPLETE
    # continue message appended, snapshot persisted, and a wasted LLM turn
    # burns a stream read. Bail BEFORE the budget/continue/persist side
    # effects so the deadline path can drive the model to its
    # ``expected_terminal_tool`` and complete. Stall clock still advances
    # below so the bookkeeping is honest.
    if _terminal_only_enforced(engine):
        _longfile.register_completed_turn(engine)
        _logger.warning(
            "DIAG query.longfile_convergence.skipped_terminal_only run=%s "
            "tenant=%s",
            engine.config.run_id,
            engine.config.tenant_id,
        )
        yield False
        return
    _longfile.register_completed_turn(engine)
    forced = _longfile.decide_next_forced_tool(engine)
    if forced is None:
        yield False
        return

    # Inject the INCOMPLETE continue message — bilingual, tail-anchored,
    # never "safe on disk". The forced AppendFile/FinalizeFile directive rides
    # on the native ``tool_choice`` set below (the message is the continuation
    # hint, the forcing is the active ingredient).
    continue_text = _longfile.build_continue_message(engine)
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=continue_text)],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_LONGFILE_CONTINUE
            },
        )
    )
    _longfile.set_force_next_tool(engine, forced)
    if forced == "AppendFile":
        _longfile.commit_forced_append(engine)
    else:
        _longfile.commit_forced_finalize(engine)
    await engine._persist_snapshot()
    yield _emit_state_change(
        engine,
        engine.state,
        engine.state,
        reason=(
            "longfile_forced_append"
            if forced == "AppendFile"
            else "longfile_forced_finalize"
        ),
    )
    yield True


def _terminal_only_enforced(engine: QueryEngine) -> bool:
    """Return True iff a non-terminal tool dispatch must now be REJECTED.

    Armed by the wind-down, and by nothing else. The narrowed surface already
    means a withdrawn tool is neither advertised nor admitted by the permission
    gate, so this guard is not what stops the call — it is what the model reads
    when it tries one anyway, off a stale schema in its own context. A bare
    "not permitted" would leave it guessing; :func:`_terminal_only_error_message`
    names the tool it should be calling instead.

    The predicate used to key on a deadline-specific latch, which meant the
    strictness followed one of five ways a run could be cut short. It follows
    all five now, because they are all the same wind-down.
    """
    return _soft_stop.tools_withdrawn(engine)


def _terminal_only_blocks(
    engine: QueryEngine, tool_call: ToolCall
) -> bool:
    """Return True iff ``tool_call`` should be rejected because terminal
    finalisation is in effect.

    Predicate is True iff ALL of:
      * STRICT terminal-only finalisation is in force for this tenant
        (:func:`_terminal_only_enforced`: the durable deadline-finalize
        latch). The best-effort error backstop turn is intentionally NOT
        blocked.
      * the terminal nudge latch ``engine._terminal_only_active``
        is set (the per-turn loop flipped it when the deadline-finalize /
        contract-repair nudge fired)
      * a terminal tool name actually resolves
        (:func:`_resolved_terminal_tool_name` is not None; honours the
        per-tenant ``expected_terminal_tool``). With no resolvable terminal
        tool there is nothing to force, so the guard never blocks.
      * no successful terminal tool result is in history yet
      * the tool being dispatched is NOT that resolved terminal tool name.

    The single allowed tool while the latch is active is exactly
    ``_resolved_terminal_tool_name(engine)``.
    """
    if not _terminal_only_enforced(engine):
        return False
    if not getattr(engine, "_terminal_only_active", False):
        return False
    if _history_has_terminal_tool_result(engine):
        return False
    terminal_tool_name = _resolved_terminal_tool_name(engine)
    if terminal_tool_name is None:
        return False
    return tool_call.name != terminal_tool_name


def _terminal_only_error_message(
    engine: QueryEngine, blocked_tool_name: str
) -> str:
    """Structured, model-visible error surfaced when the terminal-only
    guard blocks a non-terminal tool dispatch.

    The message names BOTH the blocked tool and the resolved terminal tool
    so the model gets an actionable instruction (call the terminal tool
    now) rather than a silent drop. Keyed on the generic
    ``expected_terminal_tool`` slot (NOT on any benchmark token); the guard
    only fires once a terminal tool resolves, so the name is always present.
    """
    terminal_tool_name = (
        _resolved_terminal_tool_name(engine) or "the configured terminal tool"
    )
    return (
        f"tool '{blocked_tool_name}' execution blocked: [terminal-only mode: "
        f"deadline reached — call {terminal_tool_name} now with your "
        "best-evidence answer]"
    )


def _history_tool_result_is_terminal(
    engine: QueryEngine,
    tool_call_id: str,
) -> bool:
    """Find the just-appended tool result and inspect its terminal metadata.

    When ``expected_terminal_tool`` is configured, a successful
    terminal-metadata result counts as terminal ONLY if the originating
    tool call's name matches the declared terminal tool — the same
    expected-tool-name guard as :func:`_history_has_terminal_tool_result`.
    This keeps the two helper families consistent so a NON-expected tool
    that returns terminal metadata can never end an
    ``expected_terminal_tool`` tenant's run as "answered" (which would
    discard a stored stream error / surface a false-positive completion).
    When ``expected_terminal_tool`` is None the behaviour is bit-identical
    to before (any successful terminal-metadata result counts) — no
    regression for a host backend or for the default.
    """
    for message in reversed(engine.history):
        for block in reversed(message.content_blocks):
            if (
                isinstance(block, ToolResultBlock)
                and block.tool_call_id == tool_call_id
            ):
                if (
                    block.is_error
                    or block.metadata.get(TERMINAL_TOOL_METADATA_KEY) is not True
                ):
                    return False
                expected = engine.config.expected_terminal_tool
                if expected is None:
                    return True
                return _tool_name_for_call_id(engine, tool_call_id) == expected
    return False


def _rewrite_deferred_tool_result_events(
    events: list[TurnEvent],
    outcome: DispatchOutcome,
) -> list[TurnEvent]:
    """Return buffered events with ``TOOL_RESULT`` text matching ``outcome``.

    Parallel dispatch executes tools under ``asyncio.gather`` but yields results
    in LLM order. Consecutive-error cap decisions are transcript-order state, so
    replay may rewrite an error outcome after gather. The SSE event must carry
    the same text that we append to history.
    """

    rewritten: list[TurnEvent] = []
    for evt in events:
        if (
            evt.type is not EventType.TOOL_RESULT
            or evt.payload.get("tool_call_id") != outcome.tool_call.id
        ):
            rewritten.append(evt)
            continue

        payload = dict(evt.payload)
        payload["success"] = outcome.success
        blocks = payload.get("content_blocks")
        if isinstance(blocks, list):
            new_blocks: list[Any] = []
            for block in blocks:
                if isinstance(block, dict):
                    new_block = dict(block)
                    if new_block.get("type") == "text":
                        new_block["text"] = outcome.content
                    new_blocks.append(new_block)
                else:
                    new_blocks.append(block)
            payload["content_blocks"] = new_blocks

        error = payload.get("error")
        if not outcome.success:
            new_error = dict(error) if isinstance(error, dict) else {}
            if outcome.error_kind is not None:
                new_error["kind"] = outcome.error_kind.value
            new_error["message"] = outcome.content
            payload["error"] = new_error
        else:
            payload.pop("error", None)
        if outcome.metadata:
            payload["metadata"] = dict(outcome.metadata)
        else:
            payload.pop("metadata", None)

        rewritten.append(evt.model_copy(update={"payload": payload}))
    return rewritten


# The dispatcher mutates per-run helper-bag state mid-execution
# (consecutive-error streak, SANDBOX_DOWN streak, string_type streak,
# satisfied-precondition set). Under ``asyncio.gather`` those mutations
# race on the shared dict and the final state depends on gather completion
# order, not the LLM-requested order. The snapshot/restore/replay helpers
# below let the parallel branch (a) preserve the pre-gather state,
# (b) restore it after gather, and (c) deterministically replay the state
# transitions in transcript order so the next-turn caps fire on the
# correct count.

_DISPATCHER_HELPER_KEYS_TO_SNAPSHOT: tuple[str, ...] = (
    "tool_dispatch.consecutive_error_state",
    "tool_dispatch.sandbox_down_streak",
    "tool_dispatch.sandbox_down_injection_pending",
    "tool_dispatch.string_type_streak",
    "tool_preconditions.satisfied",
)
"""Per-run helper-bag keys whose semantics are "transcript-order".

Mirrors the constants in :mod:`protocore.runtime.tool_dispatch` /
:mod:`protocore.runtime.tool_preconditions`. Kept as a separate tuple
so a regression in the dispatcher keys is caught by the parallel-batch
state replay tests rather than silently producing wrong streaks. New
helper-bag keys with transcript-order semantics MUST be added here.
"""


def _snapshot_dispatcher_helper_state(
    engine: QueryEngine,
) -> dict[str, Any]:
    """Deep-copy the dispatcher's transcript-order helper-bag keys.

    Returns a dict ``{key: value}`` capturing only the keys present in
    the bag prior to gather. Missing keys are NOT recorded so the
    matching :func:`_restore_dispatcher_helper_state` can ``pop`` keys
    introduced by the parallel mutations. ``getattr(engine, '_helpers',
    None)`` falls through to an empty snapshot when the bag is not
    wired (legacy tests).
    """
    helpers = getattr(engine, "_helpers", None)
    if not isinstance(helpers, dict):
        return {}
    snapshot: dict[str, Any] = {}
    for key in _DISPATCHER_HELPER_KEYS_TO_SNAPSHOT:
        if key in helpers:
            # Deep-copy because the values are mutable dicts/sets/lists
            # — a shallow copy would alias to the same nested object
            # and parallel mutations would leak into the snapshot.
            snapshot[key] = _deep_copy_helper_value(
                helpers[key], max_depth=engine.config.rc.max_data_nesting_depth
            )
    return snapshot


def _restore_dispatcher_helper_state(
    engine: QueryEngine,
    snapshot: Mapping[str, Any],
) -> None:
    """Restore the helper bag to the snapshot, discarding parallel mutations.

    Keys absent from the snapshot are ``pop``'d from the bag so any new
    keys introduced by the parallel mutations are wiped — the replay
    in :func:`_replay_dispatcher_helper_state` re-introduces them in
    LLM-requested order.
    """
    helpers = getattr(engine, "_helpers", None)
    if not isinstance(helpers, dict):
        return
    for key in _DISPATCHER_HELPER_KEYS_TO_SNAPSHOT:
        if key in snapshot:
            helpers[key] = _deep_copy_helper_value(
                snapshot[key], max_depth=engine.config.rc.max_data_nesting_depth
            )
        else:
            helpers.pop(key, None)


def _replay_dispatcher_helper_state(
    engine: QueryEngine,
    tool_call: ToolCall,
    outcome: DispatchOutcome,
) -> DispatchOutcome:
    """Apply transcript-order state transitions for one (tool_call, outcome).

    Called by the parallel-dispatch orchestrator in LLM-requested order
    after :func:`_restore_dispatcher_helper_state` has put the bag back
    into its pre-gather state. Delegates to the same classmethods on
    :class:`ToolDispatcher` that the serial dispatch path runs so the
    cap / satisfaction semantics stay defined in ONE place
    (:mod:`protocore.runtime.tool_dispatch`).

    Skips silently when:
    * The helper bag is not wired (legacy tests).
    * ``outcome`` is ``None`` (defensive — dispatcher must yield one).
    * ``outcome.approval_required`` (the caller already handled this
      before invoking us — replay would be wrong because the gate
      short-circuited the per-tool execution).

    Returns the transcript-correct outcome. When gathered dispatches hit
    consecutive-error caps in completion order, replay can produce a different
    surfaced error for this tool in LLM order. The caller must use the returned
    outcome for both SSE event text and history append.
    """
    if outcome is None or outcome.approval_required:
        return outcome

    # Gathered calls deliberately defer evidence admission so the immutable
    # ledger follows LLM order rather than completion order.  This is the
    # replay's first state transition: an admission failure becomes a normal
    # dispatch failure before any real error-streak, dependency, history, SSE,
    # or snapshot side effect can observe success.
    outcome = _ingest_tool_evidence(engine, outcome)

    helpers = getattr(engine, "_helpers", None)
    if not isinstance(helpers, dict):
        return outcome

    # Lazy import to keep the module-level import graph clean and
    # avoid any chance of a circular import via
    # ``tool_dispatch.py`` re-exporting.
    from protocore.runtime.tool_dispatch import DispatchErrorKind, ToolDispatcher

    helpers_metadata = _build_replay_metadata(engine)
    ctx = ToolContext(
        tenant_id=engine.config.tenant_id,
        account_id=engine.config.account_id,
        run_id=engine.config.run_id,
        session_id=engine.config.session_id,
        evidence_origin=engine._engine_evidence_origin(),
        metadata=helpers_metadata,
    )

    # Cumulative tool-call soft cap — count THIS executed tool call in
    # transcript order. The concurrent gather did NOT count it (the counter is a
    # snapshotted transcript-order key restored to its pre-gather value above),
    # so the increment happens HERE, once per replayed call, in LLM-requested
    # order — mirroring the serial path's per-call count. Advisory only; the
    # warning (if any) is appended to the surfaced outcome via ``_finalize``.
    # Ledger this call in LLM-requested order. The gather that executed it
    # completed in whatever order the tools finished; the replay is where
    # transcript order is re-established, so it is where the ordinal is taken.
    engine.record_tool_call(tool_call.name, ok=bool(outcome.success))

    if outcome.success:
        ToolDispatcher._reset_consecutive_error_streak(ctx)
        tool = engine.tools.get(tool_call.name)
        if tool is not None:
            ToolDispatcher._record_precondition_satisfaction(
                tool=tool,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                ctx=ctx,
            )
        return outcome

    # Error path — replay the consecutive-error cap from the original
    # pre-rewrite `(kind, message)` tuple so restored helper state matches the
    # transcript-order serial path even when the gathered dispatch already hit
    # the cap locally.
    metadata = outcome.metadata or {}
    raw_kind = metadata.get(DISPATCH_REPLAY_ERROR_KIND_METADATA_KEY)
    raw_message = metadata.get(DISPATCH_REPLAY_ERROR_MESSAGE_METADATA_KEY)
    kind: DispatchErrorKind | None = None
    if isinstance(raw_kind, str):
        try:
            kind = DispatchErrorKind(raw_kind)
        except ValueError:
            kind = None
    if kind is None:
        kind = outcome.error_kind or DispatchErrorKind.execution
    message = raw_message if isinstance(raw_message, str) else outcome.content
    final_kind, final_message = ToolDispatcher._apply_consecutive_error_cap(
        ctx,
        tool_call.name,
        kind,
        message,
        emit_diagnostics=False,
    )
    if bool(metadata.get(DISPATCH_POST_TOOL_OUTPUT_MODIFIED_METADATA_KEY)):
        # The dispatcher has already run PostToolUse against the surfaced
        # dispatch output. Replay owns transcript-order helper state, but it
        # must not undo hook-level redaction or other output modifications.
        return outcome
    if final_kind == outcome.error_kind and final_message == outcome.content:
        return outcome
    return replace(
        outcome,
        content=final_message,
        is_error=True,
        error_kind=final_kind,
    )


# the per-run ``run_metadata`` envelope is OPERATOR-supplied via the
# public ``POST /v1/runs.metadata`` API. It must never be allowed to shadow a
# RUNTIME-INTERNAL ``ToolContext.metadata`` key: those keys (the helper-bag
# namespace, the authoritative ``tool_call_id`` consumed by tool-result
# correlation / subagent-parent edges / answer-RPC binding, and any
# ``protocore.*`` control key such as synthetic-recovery / suppress-grounding)
# carry runtime trust. The merge below copies only NON-internal envelope keys.
_HELPERS_METADATA_KEY: Final[str] = "protocore.helpers"
_TOOL_CALL_ID_METADATA_KEY: Final[str] = "tool_call_id"
_RUNTIME_INTERNAL_METADATA_PREFIX: Final[str] = "protocore."


def _is_runtime_internal_metadata_key(key: str) -> bool:
    """Return ``True`` for a key that operator-supplied ``run_metadata`` must
 NOT be able to set/shadow on ``ToolContext.metadata`` .

 Covers the authoritative ``tool_call_id`` and every ``protocore.*``
 runtime-internal control key (which includes ``protocore.helpers``).
 """
    return key == _TOOL_CALL_ID_METADATA_KEY or key.startswith(
        _RUNTIME_INTERNAL_METADATA_PREFIX
    )


def _merge_run_metadata_into(
    metadata: dict[str, Any], helpers: object
) -> None:
    """Merge the per-run ``run_metadata`` envelope onto ``metadata`` in place,
 skipping runtime-internal keys so an operator-forgeable envelope cannot
 shadow trusted runtime state .

 ``helpers`` is the opaque engine helper bag; only a plain ``dict`` carries a
 ``run_metadata`` sub-mapping. Non-dict bags (TypedDict-like Mappings) are a
 no-op, matching the prior guarded behaviour.
 """
    if not isinstance(helpers, dict):
        return
    run_metadata = helpers.get("run_metadata")
    if not isinstance(run_metadata, dict):
        return
    for key, value in run_metadata.items():
        if _is_runtime_internal_metadata_key(key):
            continue
        metadata[key] = value


class HelperStateTooDeep(RuntimeError):
    """Raised when a helper-bag value nests deeper than the copier will walk.

    The bag holds tool-supplied state — a tool decides what it writes there and
    the model decides what the tool is called with — so its depth is not a
    property this module controls. Naming the condition keeps it out of the
    ``RecursionError`` class, which reaches the run loop with no attribution.
    """


def _deep_copy_helper_value(
    value: Any,
    *,
    max_depth: int = MAX_DATA_NESTING_DEPTH,
    _depth: int = 0,
) -> Any:
    """Helper-bag value deep copy that handles dicts / sets / lists / scalars.

    The dispatcher stores small dicts (streak state), sets (satisfied
    preconditions — though :func:`store_satisfied_set` normalises to a
    sorted list on persist), and scalars. A full :func:`copy.deepcopy`
    would suffice but pulls in the entire stdlib module; the explicit
    shapes here are cheap + keep the snapshot/restore path importing
    no new modules.

    Depth-bounded. This runs on the parallel-dispatch path, over values a tool
    wrote into the bag from arguments the model supplied; unbounded recursion
    there turns a nested payload into a ``RecursionError`` raised mid-dispatch,
    which the loop then reports as a provider failure. Past ``max_depth`` the
    copy raises :class:`HelperStateTooDeep` instead.
    """
    if _depth > max_depth:
        raise HelperStateTooDeep(
            f"helper-bag value nests deeper than {max_depth} levels — "
            "refusing to copy it for the parallel-dispatch snapshot"
        )
    if isinstance(value, dict):
        return {
            k: _deep_copy_helper_value(v, max_depth=max_depth, _depth=_depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _deep_copy_helper_value(v, max_depth=max_depth, _depth=_depth + 1)
            for v in value
        ]
    if isinstance(value, set):
        return set(value)
    if isinstance(value, tuple):
        return tuple(
            _deep_copy_helper_value(v, max_depth=max_depth, _depth=_depth + 1)
            for v in value
        )
    return value


def _build_replay_metadata(engine: QueryEngine) -> dict[str, Any]:
    """Build a ``ToolContext.metadata`` dict that wraps the engine's helper bag.

    Mirrors the metadata construction in
    :func:`_drain_dispatch_tool_deferred` (and :func:`_dispatch_tool`)
    so the replay's ``ToolContext`` exposes the same ``protocore.helpers``
    namespace + per-run metadata envelope the dispatcher would have seen.
    """
    helpers = getattr(engine, "_helpers", None)
    metadata: dict[str, Any] = {}
    if helpers:
        # replay the cross-pod re-drive seed so a
        # precondition check on the replay path sees the same set the
        # live recording would have produced.
        _rehydrate_satisfied_from_history(helpers, engine)
        metadata[_HELPERS_METADATA_KEY] = helpers
        # : skip runtime-internal keys on the replay path too so the
        # replay's ``ToolContext`` matches the dispatcher's sanitised one.
        _merge_run_metadata_into(metadata, helpers)
    return metadata


def _rehydrate_satisfied_from_history(
    helpers: dict[str, Any] | None,
    engine: QueryEngine,
) -> None:
    """Seed the helper-bag satisfied set from ``engine.history`` when empty.

 the helper bag is built per-pod by
 ``service_runtime.build_helper_bag`` and is NOT carried in the
 engine snapshot. On a cross-pod re-drive a fresh pod sees an
 empty satisfied set even when ``engine.history`` already contains
 a long transcript of ``AppendFile(foo)``/``Write(...)``/etc.
 Without rehydration a follow-up ``FinalizeFile(foo)`` call would
 be blocked with ``[PRECONDITION NOT MET: AppendFile:foo]`` even
 though the prereq is right there in the durable transcript.

 Reads :data:`engine.history` and writes the rebuilt set into the
 helper bag's :data:`SATISFIED_PRECONDITIONS_KEY` only when the key
 is absent or empty — a populated in-bag set always wins (in-process
 dispatches have already recorded the live satisfaction entries).

 The engine reference is the only piece of cross-pod-durable state
 for a resumed run (the helper bag itself is rebuilt by the new
 pod), so the history is the canonical source of truth for the
 satisfied set on the re-drive path.

 Rebuilt from :func:`_this_run_messages`, not from the whole
 transcript. A tool precondition is a statement about what THIS run
 has already done, and cross-run history seeding puts an earlier
 run's ``AppendFile(report.md)`` into this run's history — over the
 whole transcript that earlier call would authorise this run's
 ``FinalizeFile(report.md)`` without this run having appended
 anything, which is the same class of error as a failed call
 authorising a dependent one.
 """
    if helpers is None:
        return
    existing = helpers.get("tool_preconditions.satisfied")
    if isinstance(existing, (list, tuple, set)) and len(existing) > 0:
        return
    # Late import to avoid a circular import: tool_preconditions does
    # not import from this module, but the engine → query → preconditions
    # direction is cleaner at call-site.
    from protocore.runtime.tool_preconditions import (
        SATISFIED_PRECONDITIONS_KEY,
        record_satisfaction,
    )

    run_messages = _this_run_messages(engine)
    if not run_messages:
        return
    # A tool-use block merely records the model's request.  Rehydrating it as
    # satisfaction would let a failed tool (including evidence rejection)
    # authorize a dependent call after the helper bag is rebuilt.  Pair the
    # request with its durable non-error result instead.
    pending_calls: dict[str, tuple[str, dict[str, Any]]] = {}
    rebuilt: set[str] = set()
    for message in run_messages:
        for block in message.content_blocks:
            if isinstance(block, ToolUseBlock):
                arguments: dict[str, Any] = {}
                try:
                    decoded = json.loads(block.arguments_json)
                except (TypeError, ValueError):
                    decoded = None
                if isinstance(decoded, dict):
                    arguments = decoded
                pending_calls[block.tool_call_id] = (block.name, arguments)
            elif isinstance(block, ToolResultBlock) and not block.is_error:
                call = pending_calls.pop(block.tool_call_id, None)
                if call is not None:
                    tool_name, arguments = call
                    record_satisfaction(
                        tool_name=tool_name,
                        arguments=arguments,
                        satisfied=rebuilt,
                    )
    if rebuilt:
        helpers[SATISFIED_PRECONDITIONS_KEY] = sorted(rebuilt)


# ---------------------------------------------------------------------------
# Repeated-tool-error circuit breaker
# ---------------------------------------------------------------------------


def _resolve_max_consecutive_tool_errors(engine: QueryEngine) -> int:
    """Read ``max_consecutive_tool_errors`` from the RC snapshot.

    Defensive fallback mirrors :meth:`ToolDispatcher._resolve_consecutive_error_cap`:
    a corrupted/absent value degrades to the Pydantic default so the breaker can
    never trip on the very first error. The RC has a ``ge=2`` validator, but a
    sub-2 value here would mean "trip on first error" — clamp it out.
    """
    raw = getattr(engine.config.rc, "max_consecutive_tool_errors", 3)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 3
    return value if value >= 2 else 3


def _circuit_breaker_corrective_text(tool_name: str) -> str:
    """Bilingual corrective convergence turn for a circuit-broken tool.

    Frames the disablement as a fact ("the tool is unavailable for the rest of
    this run") and instructs the model to ANSWER from the conversation instead
    of retrying — forcing convergence. EN+RU (multilingual mandatory).
    """

    return (
        f"The '{tool_name}' tool repeatedly failed with the same error and has "
        "been disabled for the rest of this run — do NOT call it again. Answer "
        "the user's request now using what you already know from this "
        "conversation; if you cannot, say so plainly and finish. | "
        f"Инструмент '{tool_name}' многократно завершался одной и той же "
        "ошибкой и отключён до конца этого запуска — больше не вызывайте его. "
        "Ответьте на запрос пользователя сейчас, используя то, что уже известно "
        "из этого диалога; если это невозможно, прямо сообщите об этом и "
        "завершите работу."
    )


def _circuit_breaker_track_and_maybe_trip(
    engine: QueryEngine,
    tool_call: ToolCall,
    outcome: DispatchOutcome,
) -> str | None:
    """Track the consecutive same-tool/same-error-class streak and trip the
    hard circuit breaker once it crosses ``max_consecutive_tool_errors``.

    Called after every NON-approval dispatch outcome. On a SUCCESS the streak is
    cleared. On an ERROR the ``(tool_name, error_kind)`` streak increments; when
    it reaches the cap the tool is added to ``engine._circuit_broken_tools``
    (removed from the surface AND denied at dispatch via
    ``effective_tool_policy.blocked``) and — at most once per tool — a corrective
    convergence message is returned for the caller to inject as a bounded
    synthetic user turn.

    Returns the corrective text to inject, or ``None`` (no trip this dispatch,
    or the tool was already broken+notified). The in-flight streak lives on the
    engine (``_circuit_breaker_streak``) so it is snapshot-persisted across a
    cross-pod resume.
    """

    tool_name = tool_call.name

    # SUCCESS — reset the streak unconditionally. A success of ANY tool
    # breaks the "consecutive" chain: ``Read(err) → List(ok) → Read(err) →
    # Read(err)`` must NOT trip at cap 3, because the Read failures were not
    # consecutive. (Matches the dispatcher's own ``_reset_consecutive_error_
    # streak`` on a successful tool result.)
    if not outcome.is_error:
        engine._circuit_breaker_streak = None
        return None

    # Cap-INELIGIBLE soft error — a tool result that opted out of
    # consecutive-error capping (``consecutive_error_cap_eligible=False``): an
    # ordinary Bash nonzero exit used as DATA (``grep -q`` no-match, ``test``
    # false, ``diff`` difference). These are NOT "repeat the same failing call"
    # signals, so — exactly like the dispatcher's soft-error path — they must
    # neither ADVANCE nor RESET the hard-breaker streak. Reuse the SAME strict-
    # bool eligibility flag the dispatcher honours (absent ⇒ eligible).
    raw_eligible = (outcome.metadata or {}).get(
        TOOL_RESULT_CONSECUTIVE_CAP_ELIGIBLE_METADATA_KEY, True
    )
    cap_eligible = raw_eligible if isinstance(raw_eligible, bool) else True
    if not cap_eligible:
        return None

    # Already broken — keep it blocked (the surface/gate already deny it) and do
    # NOT re-inject the corrective turn (the latch lives on the engine so it
    # survives cross-pod resume).
    if tool_name in engine._circuit_broken_tools:
        return None

    error_class = outcome.error_kind.value if outcome.error_kind is not None else "error"

    state_raw = engine._circuit_breaker_streak
    last_tool: str | None = None
    last_class: str | None = None
    count = 0
    if isinstance(state_raw, dict):
        if isinstance(state_raw.get("tool_name"), str):
            last_tool = state_raw["tool_name"]
        if isinstance(state_raw.get("error_class"), str):
            last_class = state_raw["error_class"]
        raw_count = state_raw.get("count", 0)
        if isinstance(raw_count, int) and raw_count >= 0:
            count = raw_count

    if last_tool == tool_name and last_class == error_class:
        count += 1
    else:
        count = 1

    engine._circuit_breaker_streak = {
        "tool_name": tool_name,
        "error_class": error_class,
        "count": count,
    }

    cap = _resolve_max_consecutive_tool_errors(engine)
    if count < cap:
        return None

    # TRIP — hard-stop the tool for the rest of the run and inject ONE corrective
    # convergence turn.
    engine._circuit_broken_tools.add(tool_name)
    _logger.warning(
        "DIAG query.circuit_breaker.tripped run=%s tenant=%s tool=%s "
        "error_class=%s count=%d cap=%d",
        engine.config.run_id,
        engine.config.tenant_id,
        tool_name,
        error_class,
        count,
        cap,
    )
    if tool_name in engine._circuit_breaker_notified_tools:
        return None
    engine._circuit_breaker_notified_tools.add(tool_name)
    return _circuit_breaker_corrective_text(tool_name)


async def _dispatch_tool(
    engine: QueryEngine,
    tool_call: ToolCall,
    *,
    preapproved: bool = False,
    synthetic_recovery: bool = False,
    synthetic_recovery_kind: str = SYNTHETIC_RECOVERY_GUARANTEED_TERMINAL,
    keep_truncated_tail: bool = False,
) -> AsyncIterator[TurnEvent]:
    """Execute one tool_call with hooks. Yields events. Appends tool_result.

 md`. Delegates the 7-step
 dispatch lifecycle to :class:`ToolDispatcher`; performs engine-side
 state mutations (history append, snapshot, tool-name cleanup) here
 so they stay observable from the engine's perspective.

 Event ordering invariant (matches existing tests):

 (LLM-stream) tool_use_start → tool_use_input_delta → tool_use_stop
 (dispatcher) hook_fired(pre) →
 (tool_call_pending | hook_fired(post) + tool_result)

 ``sandbox_starting`` is emitted by the **sandbox adapter** (not core)
 on cold start only.
 """
    _pin_keep_flag(engine, tool_call)
    # Terminal-only finalisation guard. Once the terminal-answer nudge has
    # fired and no terminal tool result is in history yet, every
    # non-terminal dispatch is short-circuited with a structured error
    # pointing at finalisation. The model still chooses message / outcome /
    # refs — the runtime never synthesises the answer. See
    # :func:`_terminal_only_blocks` for the full predicate.
    if _terminal_only_blocks(engine, tool_call):
        error_message = _terminal_only_error_message(engine, tool_call.name)
        yield TurnEvent(
            type=EventType.TOOL_RESULT,
            run_id=engine.config.run_id,
            payload={
                "tool_call_id": tool_call.id,
                "success": False,
                "error": {
                    "kind": "terminal_only",
                    "message": error_message,
                },
                "content_blocks": [{"type": "text", "text": error_message}],
            },
        )
        engine.history.append(
            Message(
                role=MessageRole.tool,
                content_blocks=[
                    ToolResultBlock(
                        tool_call_id=tool_call.id,
                        content=error_message,
                        is_error=True,
                    )
                ],
            )
        )
        engine.forget_tool_name(tool_call.id)
        await engine._persist_snapshot()
        return

    # Cumulative total-work guard — the serial twin of the one on the deferred
    # parallel path. Both dispatch routes must refuse, or a leader whose fan-out
    # was disabled (a single delegation call, a hook-gated one, or
    # ``parallel_subagents_enabled`` off) would keep delegating past the budget
    # on exactly the serial shape the cumulative bound exists to catch.
    run_work_refusal, run_work_reason = _run_work_delegation_refusal(
        engine, tool_call
    )
    if run_work_refusal:
        _logger.warning(
            "DIAG query.run_work_budget.delegation_refused run=%s tenant=%s "
            "tool=%s reason=%s %s",
            engine.config.run_id,
            engine.config.tenant_id,
            tool_call.name,
            run_work_reason,
            _resolve_run_work_ledger(engine).spent_summary(),
        )
        refusal_events, refusal_outcome = _run_work_refusal_dispatch(
            engine, tool_call, run_work_refusal, run_work_reason
        )
        for evt in refusal_events:
            yield evt
        engine.history.append(
            Message(
                role=MessageRole.tool,
                content_blocks=[
                    ToolResultBlock(
                        tool_call_id=tool_call.id,
                        content=_tool_result_content_with_finalization_hint(
                            refusal_outcome
                        ),
                        is_error=True,
                    )
                ],
            )
        )
        engine.forget_tool_name(tool_call.id)
        await engine._persist_snapshot()
        return

    #  — universal PROSE-GATE before a BACKGROUND terminal
    # tool. The terminal tool is a pure background gate: its answer field is
    # removed and its tool_use / tool_result pair is filtered from the stream +
    # durable history, so the ONLY user-facing answer is the model's own
    # visible assistant prose. When this run is about to latch the terminal
    # tool's result but produced NO substantive visible prose after its latest
    # real-work tool (:func:`_finalize_prose_gate_applies`), we VETO the
    # dispatch ONCE — exactly like the pre-dispatch / candidate-repair seams:
    # append a NON-terminal error tool_result (so
    # ``_history_tool_result_is_terminal`` returns False and the loop does NOT
    # finalise) plus ONE bounded corrective user turn asking the model to write
    # the final answer as normal text and THEN call the terminal tool, charge
    # the bound this veto answers to (durable across resume), persist, and
    # return so the outer loop re-drives one corrective turn. Unlike the
    # pre-dispatch verify seam this needs NO the host trigger (the
    # visible-prose predicate is a pure check on the engine history core owns)
    # and does NOT debit the self-verify budget — it carries its own.
    # TWO bounds, not one: a payload-only terminal spends the gate's single
    # durable latch, while a terminal whose only prose is a filing notice
    # spends one attempt of the pointer refusal's own budget, which is larger
    # because the first correction was measured to be ignored. The repair text
    # is the OPPOSITE of the terminal-tool nudge ('write the answer as normal
    # text first'), so the two never contradict. Default-on
    # (``finalize_prose_gate_enabled``); ``_applies`` returns False for the
    # healthy prose-then-terminal shape and once the relevant bound is spent, so
    # an uncorrected terminal eventually finalises rather than looping.
    if _finalize_prose_gate_applies(engine, tool_call):
        repair_text = engine.config.rc.finalize_prose_gate_repair_text
        # An empty repair text would inject an empty user turn — degrade to a
        # no-op (let the terminal dispatch through) instead. Nothing is latched
        # or charged in that case, mirroring "the gate did not fire".
        if repair_text:
            # Read the pointer evidence BEFORE the veto's own tool_result and
            # repair turn land: that error result reads as real work, which
            # closes the answer window the measurement is taken over.
            pointer = _pointer_answer_evidence(engine)
            # Same split as the plain-stop seam: the payload-only veto spends
            # the gate's single shot, the pointer veto one attempt of the
            # pointer refusal's own budget. A run may therefore be vetoed for a
            # filing notice after it was already vetoed for having no prose at
            # all — two different failures, each answered on its own terms.
            _attempt = 0
            if pointer is None:
                engine._finalize_prose_gate_used = True
            else:
                _attempt = _charge_pointer_answer_repair(engine)
            veto_error = (
                f"tool '{tool_call.name}' submission withheld: write your final "
                "answer to the user as a normal assistant message FIRST, then "
                f"call {tool_call.name} to end the run. It was NOT submitted."
            )
            yield TurnEvent(
                type=EventType.TOOL_RESULT,
                run_id=engine.config.run_id,
                payload={
                    "tool_call_id": tool_call.id,
                    "success": False,
                    "error": {
                        "kind": "finalize_prose_gate",
                        "message": veto_error,
                    },
                    "content_blocks": [{"type": "text", "text": veto_error}],
                },
            )
            # Non-terminal error tool_result for the vetoed call (keeps the
            # assistant/tool message pairing valid; ``_history_tool_result_is_
            # terminal`` returns False so the loop does NOT finalise) ...
            engine.history.append(
                Message(
                    role=MessageRole.tool,
                    content_blocks=[
                        ToolResultBlock(
                            tool_call_id=tool_call.id,
                            content=veto_error,
                            is_error=True,
                        )
                    ],
                )
            )
            # ... followed by the bounded prose-repair user turn (write the
            # answer as normal text first, THEN call the terminal tool).
            engine.history.append(
                Message(
                    role=MessageRole.user,
                    content_blocks=[TextBlock(text=repair_text)],
                    metadata={
                        SYNTHETIC_RECOVERY_METADATA_KEY: (
                            SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR
                        )
                    },
                )
            )
            engine.forget_tool_name(tool_call.id)
            # Persist the snapshot IMMEDIATELY after the corrective turn +
            # latch mutation so a crash / cross-pod resume between the
            # injection and the next persistence boundary cannot lose the
            # latch (and re-veto) or the correction.
            await engine._persist_snapshot()
            # A prose-less terminal and a terminal whose prose was only a
            # pointer are the same veto and different failures; the second
            # carries the sizes that identify it.
            if pointer is None:
                _logger.warning(
                    "DIAG query.finalize_prose_gate.vetoed run=%s tenant=%s "
                    "turn=%s tool=%s",
                    engine.config.run_id,
                    engine.config.tenant_id,
                    engine.turn_id(),
                    tool_call.name,
                )
            else:
                _pointer_path, _answer_chars, _written_chars = pointer
                _rc = engine.config.rc
                _logger.warning(
                    "DIAG query.finalize_prose_gate.pointer_answer_vetoed "
                    "run=%s tenant=%s turn=%s tool=%s attempt=%d/%d "
                    "answer_chars=%d written_chars=%d max_fraction=%.3f path=%s",
                    engine.config.run_id,
                    engine.config.tenant_id,
                    engine.turn_id(),
                    tool_call.name,
                    _attempt,
                    _rc.finalize_prose_gate_pointer_max_repair_attempts,
                    _answer_chars,
                    _written_chars,
                    _rc.finalize_prose_gate_pointer_max_answer_fraction,
                    _pointer_path,
                )
            return

    # Reached when the prose gate let this dispatch through. If the pointer
    # refusal is out of attempts and the answer it objected to is still standing,
    # the terminal tool is about to seal the run on it — the same giving-up the
    # plain-stop completion reports, at the seam where a terminal tool exists to
    # reach it first. Costs one integer comparison on every other dispatch.
    _release_pointer_answer_repair(engine)

    # PRE-DISPATCH terminal-tool verify seam. For a terminal tool whose
    # external side effect (an answer-submission RPC) fires inside its own
    # ``run()``, the post-dispatch self-verify turn is POST-SUBMIT and cannot
    # repair the answer. This gate consults the host-supplied predicate
    # BEFORE the dispatcher runs the tool, and only for the configured
    # ``expected_terminal_tool``. If the predicate
    # vetoes (returns a corrective message), the terminal tool is NEVER
    # dispatched (no RPC fires): we append a non-terminal error tool_result
    # (so ``_history_tool_result_is_terminal`` returns False and the loop
    # does NOT finalise — this is also why the truncated-tool-call
    # terminal-completion site cannot bypass the check, R8A MEDIUM#1) plus
    # ONE bounded corrective user turn, latch (durable, fire-at-most-once),
    # debit the shared self-verify budget, persist, and return so the outer
    # loop re-drives one corrective turn. Default-off: ``_applies`` returns
    # False for every tenant that has not opted in, so behaviour is
    # bit-identical to the gate-disabled path.
    # Candidate-regression protection on the REPAIR turn. The pre-dispatch
    # terminal veto below is fire-at-most-once (it latches
    # ``_pre_dispatch_terminal_verify_used`` on the first veto), so the
    # model's corrected re-submission never re-enters that gate. Without this
    # independent seam the preserved-candidate regression check therefore
    # never runs on the repair turn and a regressed body dispatches unguarded.
    # This branch re-runs that check (via the SAME
    # ``_resolve_terminal_candidate_corrective`` decision + the SAME
    # ``_terminal_candidate_reveto_used`` one-shot latch) keyed only on a
    # held substantive candidate, independent of the pre-dispatch latch. It
    # only ever fires AFTER the first veto preserved a candidate (which is
    # also when the pre-dispatch latch is already closed), so it and the
    # pre-dispatch branch never both fire on one dispatch. Default-off (RC)
    # makes ``_applies`` return False. The core never synthesises the answer
    # body — it only re-vetoes once or allows through + finalises.
    if _terminal_candidate_repair_applies(engine, tool_call):
        repair_corrective = _resolve_terminal_candidate_corrective(
            engine, tool_call, None
        )
        if repair_corrective:
            veto_error = (
                f"tool '{tool_call.name}' submission withheld: pre-submission "
                "verification found a problem with this answer. It was NOT "
                "submitted. Review the correction below, fix your answer, then "
                f"call {tool_call.name} again."
            )
            yield TurnEvent(
                type=EventType.TOOL_RESULT,
                run_id=engine.config.run_id,
                payload={
                    "tool_call_id": tool_call.id,
                    "success": False,
                    "error": {
                        "kind": "terminal_candidate_repair_reveto",
                        "message": veto_error,
                    },
                    "content_blocks": [{"type": "text", "text": veto_error}],
                },
            )
            # Non-terminal error tool_result for the vetoed call (keeps the
            # assistant/tool message pairing valid; ``_history_tool_result_is_
            # terminal`` returns False so the loop does NOT finalise) ...
            engine.history.append(
                Message(
                    role=MessageRole.tool,
                    content_blocks=[
                        ToolResultBlock(
                            tool_call_id=tool_call.id,
                            content=veto_error,
                            is_error=True,
                        )
                    ],
                )
            )
            # ... followed by the bounded corrective user turn so the model
            # knows WHAT to fix before re-submitting.
            engine.history.append(
                Message(
                    role=MessageRole.user,
                    content_blocks=[TextBlock(text=repair_corrective)],
                    metadata={
                        SYNTHETIC_RECOVERY_METADATA_KEY: (
                            SYNTHETIC_RECOVERY_TERMINAL_REPAIR
                        )
                    },
                )
            )
            engine.forget_tool_name(tool_call.id)
            await engine._persist_snapshot()
            _logger.warning(
                "DIAG query.terminal_candidate_repair.revetoed run=%s "
                "tenant=%s turn=%s tool=%s",
                engine.config.run_id,
                engine.config.tenant_id,
                engine.turn_id(),
                tool_call.name,
            )
            return
        # No veto: either the repair body is itself substantive (not a
        # regression) or the one-shot repair credit was already spent on a
        # prior turn (which persisted the latch then). Neither case mutates new
        # engine state here, so the body dispatches normally below and the run
        # finalises on best evidence.

    if _pre_dispatch_terminal_verify_applies(engine, tool_call):
        corrective = _resolve_pre_dispatch_terminal_veto(engine, tool_call)
        # Preserve the first substantive draft across the veto and re-veto a
        # regressed (empty/1-char) repair turn once. Default-off (RC) returns
        # ``corrective`` unchanged, so the branch below is bit-identical when
        # disabled. Any engine-side candidate/latch mutation is made durable
        # by the ``await engine._persist_snapshot()`` already on this veto
        # path.
        corrective = _resolve_terminal_candidate_corrective(
            engine, tool_call, corrective
        )
        # UNCONDITIONAL heartbeat on every gate APPLICATION, regardless of
        # veto outcome. The gate's prior DIAG lived only inside
        # ``if corrective:`` below, so the no-veto path was silent. This
        # line makes "did the gate run, and what did it decide?" observable
        # from the executor log alone. LOG-ONLY: no mutation; only reached
        # when ``_pre_dispatch_terminal_verify_applies`` already returned
        # True. ``verdict`` = ``veto`` when a corrective was produced
        # (terminal submission withheld below) else ``no_veto``. ``cited``
        # is exact; ``observed`` is best-effort off the opaque helper bag
        # (``-1`` when not cheaply visible from core).
        _logger.warning(
            "DIAG query.pre_dispatch_terminal_verify.applied run=%s "
            "verdict=%s cited=%d observed=%d",
            engine.config.run_id,
            "veto" if corrective else "no_veto",
            _count_terminal_answer_cited_refs(tool_call),
            _observed_ref_ledger_size(engine),
        )
        if corrective:
            engine._pre_dispatch_terminal_verify_used = True
            engine._self_verify_extra_turns_used = (
                getattr(engine, "_self_verify_extra_turns_used", 0) + 1
            )
            veto_error = (
                f"tool '{tool_call.name}' submission withheld: pre-submission "
                "verification found a problem with this answer. It was NOT "
                "submitted. Review the correction below, fix your answer, then "
                f"call {tool_call.name} again."
            )
            yield TurnEvent(
                type=EventType.TOOL_RESULT,
                run_id=engine.config.run_id,
                payload={
                    "tool_call_id": tool_call.id,
                    "success": False,
                    "error": {
                        "kind": "pre_dispatch_terminal_verify",
                        "message": veto_error,
                    },
                    "content_blocks": [{"type": "text", "text": veto_error}],
                },
            )
            # Non-terminal error tool_result for the vetoed call (keeps the
            # assistant/tool message pairing valid) ...
            engine.history.append(
                Message(
                    role=MessageRole.tool,
                    content_blocks=[
                        ToolResultBlock(
                            tool_call_id=tool_call.id,
                            content=veto_error,
                            is_error=True,
                        )
                    ],
                )
            )
            # ... followed by the bounded corrective user turn so the model
            # knows WHAT to fix before re-submitting.
            engine.history.append(
                Message(
                    role=MessageRole.user,
                    content_blocks=[TextBlock(text=corrective)],
                    metadata={
                        SYNTHETIC_RECOVERY_METADATA_KEY: (
                            SYNTHETIC_RECOVERY_PRE_DISPATCH_TERMINAL_VERIFY
                        )
                    },
                )
            )
            engine.forget_tool_name(tool_call.id)
            await engine._persist_snapshot()
            _logger.warning(
                "DIAG query.pre_dispatch_terminal_verify.vetoed run=%s "
                "tenant=%s turn=%s tool=%s",
                engine.config.run_id,
                engine.config.tenant_id,
                engine.turn_id(),
                tool_call.name,
            )
            return

    dispatcher = _ensure_tool_dispatcher(engine)
    # Thread the engine's helper bag into ToolContext.metadata under the
    # canonical ``protocore.helpers`` namespace. Without this, every tool
    # that reads adapters from the bag (workspace, todo_storage, registry,
    # sandbox etc.) hits ``ToolInvocationError: X not wired into ToolContext``.
    # The bag is attached to the engine by the executor pod via
    # ``setattr(engine, "_helpers", helpers)`` in service_runtime.build_helper_bag.
    # Core never constructs or mutates the bag — it just forwards the
    # opaque mapping.
    helpers = getattr(engine, "_helpers", None)
    metadata: dict[str, Any] = {}
    if helpers:
        # seed the per-run satisfied set from the durable
        # ``engine.history`` when the helper bag is fresh (cross-pod
        # re-drive). The dispatcher's
        # :meth:`_check_tool_preconditions` reads it from
        # ``ctx.metadata["protocore.helpers"]["tool_preconditions.satisfied"]``
        # so this MUST run before the dispatcher reads it.
        _rehydrate_satisfied_from_history(helpers, engine)
        metadata[_HELPERS_METADATA_KEY] = helpers
        # Merge the per-run metadata
        # envelope (admitted by ``POST /v1/runs.metadata`` and persisted
        # onto the Redis run-state Hash) onto ``ToolContext.metadata`` so
        # tools (e.g. PCM remote backend) can read trial-scoped values
        # like ``pac_harness_url`` and ``pac_trial_id``. : the merge
        # skips RUNTIME-INTERNAL keys (the helper-bag namespace, the
        # authoritative ``tool_call_id``, and any ``protocore.*`` control
        # key) so a forged operator envelope cannot shadow trusted runtime
        # state. The authoritative ``tool_call_id`` is then set by the
        # dispatcher (``tool_dispatch.py`` ``metadata.setdefault``) from the
        # real ``tool_call.id``.
        _merge_run_metadata_into(metadata, helpers)
    # Flag the SYNTHETIC dispatch so a backend MAY default a required terminal
    # field (e.g. ``outcome``) ONLY for the runtime-synthesised last-resort
    # guaranteed-terminal answer, never for a model-emitted one.
    # ``synthetic_recovery_kind`` names WHICH scaffold: guaranteed-terminal
    # (default) vs the longfile salvage write. A salvage file-write must NOT
    # masquerade as a guaranteed-terminal scaffold; the metadata key is
    # unforgeable runtime state a backend may branch on. Set by core LAST
    # (after the run_metadata merge) so a forged ``run_metadata`` value cannot
    # shadow it; ALSO stripped from incoming run_metadata in
    # ``service_runtime._sanitize_run_metadata``.
    # ``False`` for every normal tool call ⟹ no key set ⟹ bit-identical.
    if synthetic_recovery:
        metadata[SYNTHETIC_RECOVERY_METADATA_KEY] = synthetic_recovery_kind
    ctx = ToolContext(
        tenant_id=engine.config.tenant_id,
        account_id=engine.config.account_id,
        run_id=engine.config.run_id,
        session_id=engine.config.session_id,
        evidence_origin=engine._engine_evidence_origin(),
        metadata=metadata,
    )

    # Buffer events so we can suppress
    # the dispatcher's ``TOOL_CALL_PENDING`` envelope when the web-mode
    # approval kill-switch (``RuntimeConstants.approval_gate_web_enabled``)
    # is off. The dispatcher yields events BEFORE the final
    # :class:`DispatchOutcome`, so we cannot inspect the verdict without
    # holding them back. The buffer is bounded by the dispatcher contract
    # (at most a single ``HOOK_FIRED(pre_tool_use)`` + a single
    # ``TOOL_CALL_PENDING`` ahead of the approval outcome) and is released
    # in-order on the happy path.
    # SINGLE CHOKE POINT for the tree-budget release-around-child-join. A run
    # holding a tree slot that dispatches a DELEGATION tool blocks on the child's
    # ENTIRE nested run inside ``dispatcher.dispatch`` below — a "permit holder
    # blocked on a descendant", which wedges the tree at the cap unless the slot
    # is released across the join. EVERY serial-style delegation await funnels
    # through :func:`_dispatch_tool` (the single-call serial path AND both
    # truncation-recovery sibling loops), so releasing HERE covers them all at
    # one site. (The >=2 parallel branch releases around its own gather and does
    # NOT pass through here.) Gate: a delegation tool AND this run actually
    # holding a slot; otherwise ``None`` ⇒ no-op for reads / non-permit runs.
    dispatch_tree_permit = (
        _resolve_subagent_tree_permit(engine)
        if _tool_is_delegation(engine, tool_call)
        else None
    )
    outcome: DispatchOutcome | None = None
    buffered: list[TurnEvent] = []
    # Release BEFORE the child join. The dispatch loop below only CONSUMES
    # (buffers) events — it never ``yield``s during the child join, so a caller
    # cannot break in mid-join; the only non-normal exits are task cancellation
    # or an exception propagating out of ``dispatcher.dispatch``. On that
    # teardown control never reaches the reacquire after the loop, so the slot
    # is deliberately LEFT
    # released and the parent-final idempotent ``release()`` reconciles it —
    # teardown never blocks on a reacquire that might have no free slot. On
    # NORMAL completion/break the reacquire restores the slot before the
    # local-work post-processing (soft caps, history append) runs.
    if dispatch_tree_permit is not None:
        await dispatch_tree_permit.release_while_waiting()
    intent = None
    if engine.config.rc.intent_settlement_enabled:
        from protocore.runtime.intent import (
            commit_intent,
            settle_intent,
            should_skip_never_replay,
        )

        existing = next(
            (item for item in engine.open_intents if item.tool_call_id == tool_call.id),
            None,
        )
        if should_skip_never_replay(existing):
            yield TurnEvent(
                type=EventType.TOOL_RESULT,
                run_id=engine.config.run_id,
                payload={
                    "tool_call_id": tool_call.id,
                    "content": "interrupted",
                    "is_error": True,
                },
            )
            return
        intent = existing or commit_intent(
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            rc=engine.config.rc,
        )
        if intent is not None and intent not in engine.open_intents:
            engine.open_intents.append(intent)
            yield TurnEvent(
                type=EventType.INTENT_COMMITTED,
                run_id=engine.config.run_id,
                payload=intent.to_dict(),
            )
        if engine.config.rc.typed_hooks_enabled and engine.typed_hook_registry is not None:
            from protocore.runtime.correctness_bind import fire_typed_hook

            hook_out, hook_evt = fire_typed_hook(
                engine,
                "before_tool",
                {"tool_name": tool_call.name, "arguments": tool_call.arguments},
            )
            if hook_evt is not None:
                yield hook_evt
            if hook_out.decision == "deny":
                return
            if hook_out.decision == "require_approval":
                engine.mark_pending_approval(tool_call.id)
                await engine._persist_snapshot()
                yield TurnEvent(
                    type=EventType.TOOL_CALL_PENDING,
                    run_id=engine.config.run_id,
                    payload={
                        "tool_call_id": tool_call.id,
                        "requires_approval": True,
                        "approval_token": hook_out.approval_token,
                    },
                )
                return
    async for item in dispatcher.dispatch(
        tool_call=tool_call,
        ctx=ctx,
        # The effective policy carries the RC core tool-surface floor so the
        # gate permits exactly what was advertised (advertise/dispatch
        # parity; see ToolPermissionGate.check Stage-1 whitelist).
        visibility_policy=engine.effective_tool_policy,
        # The declared tool set of the agent driving THIS engine, when it
        # declared one. Empty declaration ⇒ ``None`` ⇒ the gate's allow-list
        # stage stays off, exactly as before it was wired.
        subagent_whitelist=engine.effective_subagent_tool_allowlist,
        timeout_seconds=engine.config.rc.tool_timeout_seconds,
        preapproved_tool_call_id=tool_call.id if preapproved else None,
        admit_evidence=lambda records, producer: engine.append_tool_evidence(
            records, producer=producer
        ),
    ):
        if isinstance(item, DispatchOutcome):
            outcome = item
            break
        buffered.append(item)
    if intent is not None and outcome is not None and not outcome.approval_required:
        from protocore.runtime.intent import settle_intent

        settle_intent(intent, result=str(outcome.content or "")[:200])
        from protocore.runtime.correctness_bind import (
            commit_usage,
            fire_typed_hook,
            persist_correctness,
        )

        persist_correctness(engine)
        _after_tool, after_tool_evt = fire_typed_hook(
            engine,
            "after_tool",
            {"tool_name": tool_call.name, "ok": not bool(getattr(outcome, "is_error", False))},
        )
        if after_tool_evt is not None:
            yield after_tool_evt
        tool_usage = commit_usage(
            engine,
            kind="tool",
            input_tokens=0,
            output_tokens=0,
            success=not bool(getattr(outcome, "is_error", False)),
            operation_id=intent.operation_id,
        )
        if tool_usage is not None:
            yield tool_usage
    if dispatch_tree_permit is not None:
        await dispatch_tree_permit.reacquire()

    if outcome is None:
        # Defensive: dispatcher always yields a DispatchOutcome.
        _logger.warning(
            "tool dispatcher returned no outcome for call_id=%s",
            tool_call.id,
        )
        for evt in buffered:
            yield evt
        engine.forget_tool_name(tool_call.id)
        return

    if (
        outcome.approval_required
        and not engine.config.rc.approval_gate_web_enabled
    ):
        # Approval-gate kill-switch.
        # User constraint (verbatim): "система approval не должна быть
        # включена для web режима, она существует для дальнейшего создания
        # cli инструмента (в web все комманды итак выполняются в
        # изолированном sandbox)". Web-mode sandbox isolation +
        # ``dangerous_commands.py`` deny patterns ARE the safety boundary
        # so we transparently re-dispatch with ``preapproved=True`` and
        # suppress the ``TOOL_CALL_PENDING`` envelope. Forward any other
        # buffered events (e.g. ``HOOK_FIRED(pre_tool_use)``) so telemetry
        # still reflects the hook fire.
        _logger.warning(
            "approval.downgrade run=%s tool=%s reason='web_mode_default_off'",
            engine.config.run_id,
            tool_call.name,
        )
        for evt in buffered:
            if evt.type is EventType.TOOL_CALL_PENDING:
                continue
            yield evt
        # Re-run the dispatcher; the gate honours
        # ``skip_pre_tool_approval`` for this call id so the second pass
        # treats approval as already satisfied and the tool executes. This is the
        # pass that actually runs a web-mode-default-off delegation child, so it
        # gets the SAME release-around-join treatment as the primary loop above
        # (the primary pass reacquired after yielding the approval outcome).
        outcome = None
        buffered = []
        if dispatch_tree_permit is not None:
            await dispatch_tree_permit.release_while_waiting()
        async for item in dispatcher.dispatch(
            tool_call=tool_call,
            ctx=ctx,
            # Effective policy on the approval re-dispatch path too (parity).
            visibility_policy=engine.effective_tool_policy,
            # …and the same declared-tool allow-list, so an approval downgrade
            # cannot be a way past the declaration.
            subagent_whitelist=engine.effective_subagent_tool_allowlist,
            timeout_seconds=engine.config.rc.tool_timeout_seconds,
            preapproved_tool_call_id=tool_call.id,
            admit_evidence=lambda records, producer: engine.append_tool_evidence(
                records, producer=producer
            ),
        ):
            if isinstance(item, DispatchOutcome):
                outcome = item
                break
            buffered.append(item)
        if dispatch_tree_permit is not None:
            await dispatch_tree_permit.reacquire()
        if outcome is None:
            _logger.warning(
                "tool dispatcher returned no outcome on re-dispatch call_id=%s",
                tool_call.id,
            )
            for evt in buffered:
                yield evt
            engine.forget_tool_name(tool_call.id)
            return

    if outcome.approval_required:
        # Engine state transition + persistence handled by the caller
        # (`_stream_one_assistant_message`) so AWAITING semantics are
        # owned by the outer loop. We just stop here.
        for evt in buffered:
            yield evt
        return

    _activate_rules_from_tool(engine, tool_call)
    newly = list(getattr(engine, "_pending_rules_activated", []) or [])
    if newly:
        engine._pending_rules_activated = []
        yield TurnEvent(
            type=EventType.RULES_ACTIVATED,
            run_id=engine.config.run_id,
            payload={"paths": newly},
        )

    # Preserve the ORIGINAL, un-annotated tool-result body for the longfile
    # byte-parser before any soft-cap annotation runs. The soft-cap warning is
    # appended to ``outcome.content`` as free text, which makes the JSON the
    # model reads invalid; ``_longfile.observe_tool_result`` →
    # ``_parse_byte_result`` ``json.loads``-es this body, so it MUST see the
    # raw JSON or a real Write/AppendFile becomes invisible (frozen tracked
    # size, false stall → spurious forced appends mid-production). This mirrors
    # the host invariant that ``AppendFileOutput.next_step`` rides INSIDE
    # the JSON so the byte-parser stays valid.
    byte_result_content = outcome.content

    # Two advisory soft caps, both warn-only (they NEVER alter dispatch
    # success/error): the per-tool subagent cap (from the Agent tool's
    # ``tool_call_limits``) and the cumulative all-tools cap (the leader's own
    # tool calls, or a subagent's own — chosen by ``parent_run_id``). Collect
    # whichever fired and append each to the result the model reads.
    soft_cap_warnings: list[dict[str, Any]] = []
    per_tool_warning = await _record_tool_call_soft_cap_warning(
        ctx=ctx,
        tool_name=tool_call.name,
    )
    if per_tool_warning is not None:
        soft_cap_warnings.append(per_tool_warning)

    # Ledger the dispatch itself, in transcript order, alongside the counter
    # that shares this seam. Recorded from the DISPATCH rather than read back
    # out of history because compaction rewrites the turn and drops the names.
    engine.record_tool_call(tool_call.name, ok=not outcome.is_error)

    if soft_cap_warnings:
        annotated_content = outcome.content
        annotated_metadata: dict[str, Any] = dict(outcome.metadata or {})
        for _warning in soft_cap_warnings:
            annotated_content = _append_soft_cap_warning_to_content(
                annotated_content,
                _warning,
            )
            annotated_metadata = _merge_soft_cap_warning_metadata(
                annotated_metadata,
                _warning,
            )
        outcome = replace(
            outcome,
            content=annotated_content,
            metadata=annotated_metadata,
        )
        _primary_warning = soft_cap_warnings[-1]
        buffered = [
            _annotate_tool_result_event(
                evt,
                warning=_primary_warning,
                content=annotated_content,
                metadata=annotated_metadata,
            )
            for evt in buffered
        ]

    # Query owns the only ledger mutation point.  It admits the dispatcher-bound
    # evidence before flushing SSE, history, or any success-only state change.
    buffered = _rewrite_deferred_tool_result_events(buffered, outcome)

    # Flush any events that were buffered ahead of the final outcome.
    for evt in buffered:
        yield evt

    # A SUCCESSFUL chunkable write (Write/AppendFile) marks the path
    # "chunking started" so a later repeat truncation of that path gets
    # the "continue with AppendFile" directive (and not before).
    if not outcome.is_error:
        _record_chunk_write_success(engine, tool_call)

    #  — observe the byte production reported by THIS tool
    # result (Write/AppendFile size from the result payload) so the stall
    # detector tracks turns-since-last-byte-adding-mutation + the running file
    # size. No-op when the driver is disabled or the tool added no bytes.
    # Feed the ORIGINAL JSON body (``byte_result_content``), NOT
    # ``outcome.content`` — the latter may carry the soft-cap warning appended
    # as free text, which makes ``json.loads`` fail and silently hides the
    # write from the convergence driver.
    # Thread ``keep_truncated_tail`` so a
    # synthetic-recovery salvage dispatch (the longfile
    # salvage) preserves the truncated-tail flag. The post-``observe_tool_result``
    # persist that captures the dispatch result carries the flag as set by the
    # caller BEFORE the dispatch, so a pod kill between the in-dispatch persist
    # and a hypothetical post-dispatch re-assert cannot land on a half-file
    # with the flag cleared.
    _longfile.observe_tool_result(
        engine,
        tool_call,
        byte_result_content,
        is_error=outcome.is_error,
        keep_truncated=keep_truncated_tail,
    )

    # Fold the result into run-level tool-precondition progress. Fed
    # ``outcome.content`` (not ``byte_result_content``) because the failure
    # reason quotes the error the MODEL saw, warnings and all. A SUCCESSFUL
    # call advances the entry whether it was forced or voluntary.
    _preconditions.observe_tool_result(
        engine, tool_call, outcome.content, is_error=outcome.is_error
    )

    # Fold the result into the declared-file read-back gate. Fed
    # ``outcome.metadata`` — the soft-cap annotation above MERGES into that
    # dict rather than replacing it, so a tool's own declaration survives a
    # warned turn. A result that declares files the caller must open engages
    # the gate; a successful read releases the paths it opened.
    _pending_reads.observe_tool_result(
        engine, tool_call, outcome.metadata, is_error=outcome.is_error
    )

    # Append tool_result block to history + persist snapshot.
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(
                    tool_call_id=tool_call.id,
                    content=_tool_result_content_with_finalization_hint(outcome),
                    is_error=outcome.is_error,
                    metadata=outcome.metadata or {},
                )
            ],
        )
    )
    # Repeated-tool-error circuit breaker. Track the consecutive
    # same-tool/same-error-class streak; once a tool that can never succeed
    # (e.g. a ``/project`` tool on a non-project session) crosses
    # ``max_consecutive_tool_errors``, it is hard-disabled for the rest of the
    # run (via ``effective_tool_policy.blocked``) and ONE bounded corrective
    # convergence turn is injected so the model answers/finalises instead of
    # storming. The synthetic-recovery marker keeps the corrective turn out of
    # the durable transcript (runtime scaffolding) while the model still sees it
    # on the next turn. No-op for every healthy run / single tool error.
    circuit_breaker_corrective = _circuit_breaker_track_and_maybe_trip(
        engine, tool_call, outcome
    )
    if circuit_breaker_corrective is not None:
        engine.history.append(
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text=circuit_breaker_corrective)],
                metadata={
                    SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_CIRCUIT_BREAKER
                },
            )
        )
    engine.forget_tool_name(tool_call.id)
    await engine._persist_snapshot()


async def resume_approved_tool(
    engine: QueryEngine,
    tool_call: ToolCall,
) -> AsyncIterator[TurnEvent]:
    """Execute one previously-pending, now-approved tool call.

    The live approval route records approval outside core, while the engine
    has already returned from ``run()`` in ``AWAITING`` state. This explicit
    resume path verifies that the requested call exactly matches a durable
    pending ``ToolUseBlock``, skips only the already-approved pre-tool gate for
    that call id, then uses the normal dispatcher execution/post-hook/result
    path so the real ``ToolResultBlock`` lands in history before finalization.

    Replays are idempotent: if a matching ``ToolResultBlock`` already exists,
    the tool is not invoked again and no duplicate result is appended.
    """
    if _history_has_tool_result(engine, tool_call.id):
        engine.clear_pending_approval(tool_call.id)
        return

    _assert_history_has_matching_pending_tool_use(engine, tool_call)

    if engine.state is LoopState.AWAITING:
        engine.transition_to(LoopState.RUNNING)
        await engine._persist_snapshot()

    async for evt in _dispatch_tool(engine, tool_call, preapproved=True):
        yield evt

    engine.clear_pending_approval(tool_call.id)
    yield TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "stop_reason": "tool_use",
            "tokens_used": _tokens_used_payload(engine),
            "cache_hit_rate": engine.total_usage.this_turn_cache_hit_rate(),
        },
    )
    if not engine.is_terminal:
        engine.transition_to(LoopState.COMPLETED)
        await engine._persist_snapshot()


def _history_has_tool_result(engine: QueryEngine, tool_call_id: str) -> bool:
    for message in engine.history:
        for block in message.content_blocks:
            if isinstance(block, ToolResultBlock) and block.tool_call_id == tool_call_id:
                return True
    return False


def _assert_history_has_matching_pending_tool_use(
    engine: QueryEngine,
    tool_call: ToolCall,
) -> None:
    pending_tool_call_id = engine.pending_approval_tool_call_id()
    if pending_tool_call_id != tool_call.id:
        raise ValueError(
            f"approved tool call is not the pending approval: expected {pending_tool_call_id!r}, got {tool_call.id!r}"
        )
    expected_arguments_json = json.dumps(tool_call.arguments, ensure_ascii=False)
    for message in engine.history:
        for block in message.content_blocks:
            if not isinstance(block, ToolUseBlock):
                continue
            if block.tool_call_id != tool_call.id:
                continue
            if block.name != tool_call.name:
                raise ValueError(f"approved tool call does not match pending tool name: {tool_call.id}")
            if block.arguments_json != expected_arguments_json:
                raise ValueError(f"approved tool call does not match pending tool input: {tool_call.id}")
            return
    raise ValueError(f"approved tool call is not pending in history: {tool_call.id}")


def _tokens_used_payload(engine: QueryEngine) -> dict[str, int]:
    """Build the ``tokens_used`` block surfaced in ``message_stop``.

    Per the prompt-caching playbook recommendation #1, includes the
    cache split (cache_read / cache_creation) so downstream metrics
    (Prometheus / dashboard) can compute hit rate without re-reading
    the run state.
    """
    usage = engine.total_usage
    return {
        "input": usage.this_turn_input,
        "output": usage.this_turn_output,
        "total": usage.this_turn_total(),
        "cache_read": usage.this_turn_cache_read,
        "cache_creation": usage.this_turn_cache_creation,
    }


def _prepend_system_sections(
    sections: tuple[str, ...],
    messages: tuple[Message, ...],
) -> list[Message]:
    """Concatenate sections into a single system :class:`Message` prefix.

    Returns ``list(messages)`` unchanged when ``sections`` is empty so the
    test invariant "no system block added unless there is content" holds.
    Sections are joined with a blank line to preserve readability across
    the skill-index, loaded-skill, and operator-supplied blocks.
    """
    if not sections:
        return list(messages)
    body = "\n\n".join(s for s in sections if s)
    if not body:
        return list(messages)
    system_msg = Message(
        role=MessageRole.system,
        content_blocks=[TextBlock(text=body)],
    )
    return [system_msg, *messages]


# ----------------------------------------------------------------------
# tool_use <-> tool_result pairing repair
# ----------------------------------------------------------------------
#
# Enforces pairing UNCONDITIONALLY at the API boundary AND synthesises
# missing tool_results on abnormal turn exit. Anthropic / OpenAI / vLLM
# all reject a request whose assistant ``tool_use`` has no matching
# ``tool_result`` (or a ``tool_result`` with no ``tool_use``, or duplicate
# ids) with HTTP 400.
#
# Protocore's wire model: a ``tool_use`` is a :class:`ToolUseBlock` on an
# ``assistant`` message; a ``tool_result`` is a :class:`ToolResultBlock`
# on a ``tool``-role message (one block per message). The pairing key is
# ``tool_call_id`` on both sides.


def _repair_outbound_tool_pairing(
    messages: list[Message],
    *,
    placeholder: str,
) -> list[Message]:
    """Return a pairing-valid, ADJACENCY-correct copy of ``messages`` .

    The wire-boundary backstop, run UNCONDITIONALLY on the outbound message
    list right before :class:`LLMRequest` assembly — independent of whether
    compaction ran this turn. It repairs orphaning from ANY source (Tier-2
    compaction dropping one side, resume-from-partial-batch, max_tokens
    truncation, a teardown that appended a result out of position). Four
    repairs:

    1. **Forward-fill** — for an assistant ``ToolUseBlock`` whose
       ``tool_call_id`` has no matching ``ToolResultBlock`` anywhere in the
       list, emit a synthetic ``is_error=True`` tool-role
       :class:`ToolResultBlock` (content ``placeholder``) immediately after
       the orphan's assistant message.
    2. **Reposition** — every real ``ToolResultBlock`` is emitted DIRECTLY
       after the assistant message that carries its ``tool_use`` (in
       tool_use order). The Anthropic wire requires the result to be the
       immediately-following turn; a result that drifted out of position
       (e.g. a teardown that appended it after an intervening user/recovery
       message) would otherwise still 400. The OpenAI wire is id-keyed and
       tolerates either, so repositioning is safe for both.
    3. **Reverse-strip** — drop any ``ToolResultBlock`` whose
       ``tool_call_id`` has no matching ``ToolUseBlock`` (and any standalone
       tool message left empty after its block was repositioned).
    4. **Dedupe** — keep only the first occurrence of each ``tool_use`` id
       and each ``tool_result`` id (the CC-1212 duplicate-id deadlock).

    Pure function: ``messages`` is not mutated; a new list of (possibly
    new) :class:`Message` objects is returned. ``Message`` /
    :class:`ToolResultBlock` are frozen, so unchanged messages are reused
    by reference.
    """
    # Pass 1 — index every tool_use id (first occurrence) and the FIRST
    # real ToolResultBlock seen per id (anywhere in the list), so pass 2 can
    # reposition the real result directly after its tool_use regardless of
    # where it currently sits. Later duplicate results are dropped.
    tool_use_ids: set[str] = set()
    result_block_by_id: dict[str, ToolResultBlock] = {}
    for message in messages:
        for block in message.content_blocks:
            if isinstance(block, ToolUseBlock):
                tool_use_ids.add(block.tool_call_id)
            elif isinstance(block, ToolResultBlock):
                if block.tool_call_id not in result_block_by_id:
                    result_block_by_id[block.tool_call_id] = block

    result: list[Message] = []
    seen_tool_use_ids: set[str] = set()
    # tool_use ids whose result we have already emitted (repositioned real or
    # synthetic) so a standalone tool message / duplicate never re-emits it.
    emitted_result_ids: set[str] = set()

    for message in messages:
        if message.role is MessageRole.assistant:
            # Dedupe duplicate tool_use blocks within / across assistant
            # turns; preserve the FIRST occurrence.
            new_blocks: list[ContentBlock] = []
            kept_tool_use_ids: list[str] = []
            changed = False
            for block in message.content_blocks:
                if isinstance(block, ToolUseBlock):
                    if block.tool_call_id in seen_tool_use_ids:
                        changed = True
                        continue
                    seen_tool_use_ids.add(block.tool_call_id)
                    kept_tool_use_ids.append(block.tool_call_id)
                new_blocks.append(block)
            if changed:
                # An assistant turn that was ALL duplicate tool_use would now
                # be empty — drop it (its surviving sibling already carries
                # the id). Otherwise rebuild with the surviving blocks.
                if not new_blocks:
                    continue
                message = message.model_copy(update={"content_blocks": new_blocks})
            result.append(message)
            # Emit each kept tool_use's result immediately after this turn,
            # in tool_use order: the repositioned real result if one exists
            # anywhere, else a synthetic forward-fill.
            for call_id in kept_tool_use_ids:
                if call_id in emitted_result_ids:
                    continue
                real = result_block_by_id.get(call_id)
                block_to_emit = (
                    real
                    if real is not None
                    else ToolResultBlock(
                        tool_call_id=call_id,
                        content=placeholder,
                        is_error=True,
                    )
                )
                result.append(
                    Message(
                        role=MessageRole.tool,
                        content_blocks=[block_to_emit],
                    )
                )
                emitted_result_ids.add(call_id)
            continue

        # Non-assistant message — strip every ToolResultBlock (real results
        # are re-emitted in position above; orphaned/duplicate ones are
        # dropped). Tool messages carry exactly one block today, but iterate
        # defensively in case that changes; non-result blocks are preserved.
        stripped_blocks: list[ContentBlock] = []
        changed = False
        for block in message.content_blocks:
            if isinstance(block, ToolResultBlock):
                changed = True
                continue
            stripped_blocks.append(block)
        if not changed:
            result.append(message)
            continue
        if not stripped_blocks:
            # Whole message was tool_result(s) — already re-emitted / dropped.
            continue
        result.append(message.model_copy(update={"content_blocks": stripped_blocks}))

    return result


def _normalize_outbound_system_messages(
    messages: list[Message],
) -> tuple[list[Message], int]:
    """Convert every non-leading ``system`` message to ``user`` role (vLLM-400).

    vLLM (and several OpenAI-compatible servers) reject any request whose
    message array carries a ``system`` message at an index OTHER than 0 with
    HTTP 400 ``"System message must be at the beginning."``. The genuine system
    prefix produced by :func:`_prepend_system_sections` always sits at index 0
    and is left untouched; any system message AFTER it — historically a Tier-2
    compaction summary (``context/compaction.run_tier2_summarisation``), now
    fixed at source to be USER-role, but legacy persisted snapshots may still
    rehydrate one mid-array — is converted to a ``user``-role copy with the SAME
    content blocks + metadata preserved (so the
    ``COMPACTION_SUMMARY_METADATA_KEY`` flag and ``<compacted-turn>`` wrapper
    survive and downstream summary recognition keeps working).

    Defense-in-depth backstop, run UNCONDITIONALLY at the request-assembly
    boundary right after :func:`_repair_outbound_tool_pairing`. Pure function:
    ``messages`` is not mutated; unchanged
    messages are reused by reference (``Message`` is frozen). Returns the new
    list + the count of converted messages so the caller can log once per run.
    """
    converted = 0
    out: list[Message] = []
    for idx, message in enumerate(messages):
        if idx != 0 and message.role is MessageRole.system:
            # system/user share the "at most one content block" validator, so a
            # role flip is always valid here.
            out.append(message.model_copy(update={"role": MessageRole.user}))
            converted += 1
        else:
            out.append(message)
    return out, converted


def _synthesize_missing_tool_results(
    history: list[Message],
    *,
    error_content: str,
) -> int:
    """Insert synthetic ``is_error`` tool_results for orphaned tool_use .

    Mutates ``history`` in place so a cancel / LLM-error teardown leaves a
    pairing-valid AND ordered persisted snapshot: every assistant
    ``ToolUseBlock`` whose ``tool_call_id`` has no matching
    ``ToolResultBlock`` gets a tool-role :class:`ToolResultBlock`
    (content ``error_content``, ``is_error=True``) inserted IMMEDIATELY after
    the assistant message that carries the orphan — not appended at the tail,
    which (if an intervening user/recovery message already follows the
    orphan) would persist an out-of-order pair the Anthropic wire rejects.
    Ensures an interrupted/crashed run rehydrated on another pod does not
    replay a
    dangling tool_use into a provider 400.

    Idempotent: a tool_use already paired (real OR a prior synthetic result
    anywhere in history) is skipped, so calling this on every teardown path
    never double-inserts. Returns the number of synthetic results inserted
    (0 when nothing to do); ``history`` is left untouched (same objects) when
    the return is 0.
    """
    resolved_ids: set[str] = set()
    for message in history:
        for block in message.content_blocks:
            if isinstance(block, ToolResultBlock):
                resolved_ids.add(block.tool_call_id)

    # Walk in order; immediately after each assistant message, insert a
    # synthetic result for every orphaned tool_use id it introduces (in
    # first-occurrence order). ``resolved_ids`` accumulates so a duplicate
    # orphan id across turns is paired exactly once.
    rebuilt: list[Message] = []
    inserted = 0
    for message in history:
        rebuilt.append(message)
        if message.role is not MessageRole.assistant:
            continue
        for block in message.content_blocks:
            if (
                isinstance(block, ToolUseBlock)
                and block.tool_call_id not in resolved_ids
            ):
                resolved_ids.add(block.tool_call_id)
                rebuilt.append(
                    Message(
                        role=MessageRole.tool,
                        content_blocks=[
                            ToolResultBlock(
                                tool_call_id=block.tool_call_id,
                                content=error_content,
                                is_error=True,
                            )
                        ],
                    )
                )
                inserted += 1

    if inserted:
        history[:] = rebuilt
    return inserted


# ----------------------------------------------------------------------
# Skill catalog (built once per run) + <command-name> triggers (per turn)
# ----------------------------------------------------------------------

# Trigger pattern in user-authored text: ``<command-name>NAME</command-name>``
# (case-sensitive name match against the skill name).
_COMMAND_NAME_PATTERN = re.compile(
    r"<command-name>([^<\s][^<]*?)</command-name>",
    re.IGNORECASE,
)


async def _ensure_run_skill_catalog(engine: QueryEngine) -> str:
    """Return the run's skill catalog block, building it at most ONCE per run.

    The catalog is a ``<system-reminder>`` listing the account's ENABLED
    skills (plus any project pins) as ``name: description`` lines,
    alphabetical. Over-budget → deterministic names-only degrade, decided
    once here. Empty string when no skills resolve or no store is wired.

    The result is cached on ``engine._skill_catalog_block`` (a per-run
    sentinel, ``None`` until built) so the ``store.list`` + render +
    token-count cost is paid only on the first turn of a run; later turns
    reuse the byte-identical block — which both preserves the cached
    system-prompt prefix and avoids a redundant DB/LLM round-trip every
    turn. If the enabled-skill set changes mid-run the run keeps turn-1's
    catalog (acceptable per the once-per-run intent — no invalidation).

    Failures on any step are isolated with a WARNING log; the run continues
    without the catalog rather than failing.
    """
    if (
        engine._skill_catalog_block is not None
        and not engine.config.rc.skills_hot_reload_enabled
    ):
        return engine._skill_catalog_block

    store = engine.skills
    if store is None:
        engine._skill_catalog_block = ""
        return ""

    rc = engine.config.rc

    try:
        entries = list(await store.list(engine.config.account_id))
    except Exception:
        _logger.warning(
            "DIAG skill_catalog.list_failed run_id=%s",
            engine.config.run_id,
            exc_info=True,
        )
        entries = []

    # Force-include project pins even if the enabled-list dropped them.
    entries = await _merge_pinned_skills(engine, store, entries)

    budget_tokens = derive_skill_index_budget_tokens(
        model_context_window=rc.model_context_window,
        skill_index_budget_ratio=rc.skill_index_budget_ratio,
    )

    # Token counter — use the engine's LLM provider if available.
    async def _count(text: str) -> int:
        try:
            return int(engine.llm.count_tokens(text))
        except Exception:
            # Conservative heuristic: 4 chars/token (Latin-prose baseline).
            return max(1, len(text) // 4)

    try:
        block = await render_skills_catalog(
            entries,
            token_counter=_count,
            budget_tokens=budget_tokens,
        )
    except Exception:
        _logger.warning(
            "DIAG skill_catalog.render_failed run_id=%s",
            engine.config.run_id,
            exc_info=True,
        )
        block = ""

    engine._skill_catalog_block = block
    return block


async def _merge_pinned_skills(
    engine: QueryEngine,
    store: ISkillStore,
    entries: list[SkillIndexEntry],
) -> list[SkillIndexEntry]:
    """Force-include project-pinned skills into the catalog entry list.

    A project's pinned skills are surfaced-by-default: the enabled-skill list
    is the catalog baseline, and each pin that is missing from it is fetched
    and appended (one :meth:`ISkillStore.list_enabled_subset` round-trip for
    the missing names).

    Pin = surfaced, NOT a visibility restriction: existing entries are
    returned unchanged. A missing/unknown pin (deleted skill) is silently
    skipped — surfacing is best-effort and never fails the run. Empty pin set
    (the common, non-project path) returns ``entries`` untouched.

    The MISSING-pin fetch uses :meth:`ISkillStore.list_enabled_subset` (not
    the whitelist-oriented ``list_subset``, which ignores the ``enabled``
    flag): a skill the operator disabled in the account-wide bank stays OFF
    the leader's catalog even when a project still holds a stale pin for its
    name (disable gates beat pins). ``store.list`` already returns only
    enabled skills, so an already-present entry is enabled by construction.
    """

    pinned_names = engine.config.pinned_skill_names
    if not pinned_names:
        return entries

    present = {entry.name for entry in entries}
    missing = [name for name in pinned_names if name not in present]
    if not missing:
        return entries

    try:
        fetched = await store.list_enabled_subset(engine.config.account_id, missing)
    except Exception:
        _logger.warning(
            "DIAG skill_catalog.pinned_subset_failed run_id=%s",
            engine.config.run_id,
            exc_info=True,
        )
        return entries

    pinned_set = set(pinned_names)
    seen_missing: set[str] = set()
    forced: list[SkillIndexEntry] = []
    for entry in fetched:
        if entry.name not in pinned_set or entry.name in seen_missing:
            continue
        seen_missing.add(entry.name)
        forced.append(entry)
    return entries + forced


async def _load_triggered_skill_bodies(
    engine: QueryEngine,
    store: ISkillStore,
    user_text: str,
) -> list[SkillBundle]:
    """Match ``<command-name>NAME</command-name>`` references → load bodies.

    Rebuilt EVERY turn from the current turn's user message (NOT cached on
    the engine like the run-stable catalog block) so a trigger in a later
    turn still force-loads its skill body. Honors the same
    ``max_skills_per_run`` cap as the manifest-driven loaded skills; missing
    or disabled skills are silently skipped.

    Resolution order:
      1. ``store.load(account_id, name)`` — works for stores that accept a
         name or UUID directly.
      2. Fallback to ``list_subset`` + UUID lookup — covers stores where
         ``load`` requires a UUID (e.g. InMemorySkillStore).
    """
    if not user_text:
        return []
    matches = _COMMAND_NAME_PATTERN.findall(user_text)
    if not matches:
        return []

    # Dedup while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for name in matches:
        normalised = name.strip()
        if not normalised or normalised in seen:
            continue
        seen.add(normalised)
        unique.append(normalised)

    rc = engine.config.rc
    out: list[SkillBundle] = []
    for skill_name in unique[: rc.max_skills_per_run]:
        bundle = await _resolve_skill_bundle(engine, store, skill_name)
        if bundle is not None:
            out.append(bundle)
    return out


async def _resolve_skill_bundle(
    engine: QueryEngine,
    store: ISkillStore,
    skill_name: str,
) -> SkillBundle | None:
    """Resolve a ``<command-name>`` reference to a :class:`SkillBundle`.

    Tries ``store.load`` first (works for stores that accept a name or
    UUID) then falls back to ``list_subset`` + UUID-based ``load``. Failures
    are logged and yield ``None`` (the loop drops the trigger silently).

    The skill bank is account-wide (keyed on ``skills.account_id``), so every
    lookup keys on ``config.account_id`` — NOT ``tenant_id`` (the scope id,
    which differs from the account on non-default deployments). This is the
    skill-chaining path (e.g. web → frontend-design), so a
    scope-keyed lookup here would silently drop every chained skill body.
    """
    account_id = engine.config.account_id

    # Path A: direct load. Most stores accept the bare name or a UUID.
    try:
        return await store.load(account_id, skill_name)
    except SkillNotFoundError:
        pass
    except KeyError:
        # InMemorySkillStore raises KeyError on dict miss.
        pass
    except Exception:
        _logger.warning(
            "DIAG skill_index.trigger_load_failed name=%s account_id=%s",
            skill_name,
            account_id,
            exc_info=True,
        )
        return None

    # Path B: resolve via list_subset → fetch by id.
    try:
        entries = await store.list_subset(account_id, [skill_name])
    except Exception:
        _logger.warning(
            "DIAG skill_index.trigger_subset_failed name=%s account_id=%s",
            skill_name,
            account_id,
            exc_info=True,
        )
        return None

    if not entries:
        _logger.warning(
            "DIAG skill_index.trigger_not_found name=%s account_id=%s",
            skill_name,
            account_id,
        )
        return None

    try:
        return await store.load(account_id, entries[0].id)
    except Exception:
        _logger.warning(
            "DIAG skill_index.trigger_load_by_id_failed name=%s account_id=%s",
            skill_name,
            account_id,
            exc_info=True,
        )
        return None


def _ensure_tool_dispatcher(engine: QueryEngine) -> ToolDispatcher:
    """Lazily build a :class:`ToolDispatcher` for ``engine`` if absent.

 Engines instantiated before the dispatcher API existed (or in tests
 that bypass engine factories) won't carry a dispatcher reference.
 We construct a default one bound to the engine's registry +
 hook manager using the default :class:`ToolPermissionGate` chain.

 / / A2: when the host helper bag carries a
 ``tool_error_counter`` (``protocore.contracts.run.IRunToolErrorCounter``
 implementation), pass it through so every dispatch error path
 increments ``runs.tool_errors_count`` for the active run. The bag is
 attached to the engine by the executor pod via
 ``setattr(engine, "_helpers", helpers)``; tests that skip the helper
 bag get a counter-less dispatcher (no telemetry, behaviour
 unchanged).
 """
    existing = getattr(engine, "_tool_dispatcher", None)
    if isinstance(existing, ToolDispatcher):
        return existing
    helpers = getattr(engine, "_helpers", None)
    tool_error_counter = None
    if isinstance(helpers, Mapping):
        tool_error_counter = helpers.get("tool_error_counter")
    dispatcher = ToolDispatcher(
        registry=engine.tools,
        permission_gate=ToolPermissionGate(),
        hook_manager=engine.hooks,
        tool_error_counter=tool_error_counter,
    )
    engine._tool_dispatcher = dispatcher  # type: ignore[attr-defined]
    return dispatcher


async def _safe_hook_invoke(
    engine: QueryEngine,
    event: HookEvent,
    payload: dict[str, object],
) -> HookResult:
    """Invoke hooks; isolate failures (logged WARNING, treated as allow)."""
    from protocore.contracts.hooks import HookResult as _HookResult

    try:
        return await engine.hooks.invoke(event, payload, engine.config.tenant_id)
    except Exception:
        _logger.warning(
            "hook invoke raised for event=%s; isolating",
            event.value,
            exc_info=True,
        )
        return _HookResult(action=HookActionKind.ALLOW, reason="hook dispatch failed")


async def _as_provider_deltas(
    upstream: AsyncIterator[ProviderDelta | LLMStreamEvent],
) -> AsyncIterator[ProviderDelta]:
    """Adapt either a :class:`ProviderDelta` or :class:`LLMStreamEvent` stream.

 The vLLM adapter (the host) emits :class:`ProviderDelta` natively.
 The in-memory mock emits :class:`LLMStreamEvent` — translate
 via :func:`stream_events_to_provider_deltas`.
 """
    # Peek the first item to decide. `upstream` may be either an async
    # generator (synchronous call returning AsyncIterator) or an awaitable
    # returning an AsyncIterator. Normalise.
    if inspect.iscoroutine(upstream):
        upstream = await upstream

    first = None
    async for item in upstream:
        first = item
        break

    if first is None:
        return

    if isinstance(first, ProviderDelta):
        yield first
        async for item in upstream:
            if isinstance(item, ProviderDelta):
                yield item
        return

    if isinstance(first, LLMStreamEvent):
        first_evt: LLMStreamEvent = first

        # Chain the first event back into a synthetic stream.
        async def _chain() -> AsyncIterator[LLMStreamEvent]:
            yield first_evt
            async for tail_item in upstream:
                if isinstance(tail_item, LLMStreamEvent):
                    yield tail_item

        async for delta in stream_events_to_provider_deltas(_chain()):
            yield delta
        return

    # Unknown stream item — log + bail.
    _logger.warning("unknown LLM stream item type: %s", type(first).__name__)


# ---------------------------------------------------------------------------
# AdaptiveSafetyBand helpers
# ---------------------------------------------------------------------------


def _resolve_safety_band_value(engine: QueryEngine) -> int:
    """Read the current AdaptiveSafetyBand value from the helper bag.

    Returns 0 when:
      - ``RuntimeConstants.adaptive_safety_band_enabled`` is False (kill-switch).
      - No band is wired in the helper bag (test fixture / leader engine
        without the host wiring).
      - The band lookup raises (defensive — telemetry plane must never
        block the LLM call).
    """
    rc = engine.config.rc
    if not getattr(rc, "adaptive_safety_band_enabled", False):
        return 0
    helpers: Mapping[str, Any] | None = getattr(engine, "_helpers", None)
    if not isinstance(helpers, Mapping):
        return 0
    band = helpers.get("adaptive_safety_band")
    if band is None:
        return 0
    try:
        current = int(band.current())
    except Exception:
        _logger.warning(
            "adaptive_safety_band.current() raised — assuming band=0",
            exc_info=True,
        )
        return 0
    return max(0, current)


__all__ = ["query"]
