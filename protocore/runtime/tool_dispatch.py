"""``ToolDispatcher`` — the 7-step dispatch flow.

The dispatcher is the **single core entry point** for executing one
:class:`~protocore.contracts.types.ToolCall` AFTER the LLM has finished
emitting it. Concerns:

* Registry lookup — :class:`DispatchErrorKind.unknown_tool` if missing.
* Permission gate — fans out to :class:`ToolPermissionGate`
 (whitelist → safety policies → ``PreToolUse`` hook).
* Approval requirement — surfaces ``tool_call_pending`` event; emits
 no tool_result (caller transitions engine to AWAITING).
* Timeout-wrapped execute — :func:`asyncio.wait_for` honouring
 ``rc.tool_timeout_seconds`` by default, or the tool's own declared
 ``default_timeout_ms`` when it sets ``should_defer=True`` (a tool that
 runs its own long-lived async unit with an internal classified timeout —
 the ``Agent`` tool / ``SubagentRunner``). See
 :func:`_resolve_tool_timeout_seconds`.
* Event emission — emits ``hook_fired`` (pre/post) and the terminal
 ``tool_result``. **Does NOT emit** ``tool_use_start`` /
 ``tool_use_stop`` / ``tool_use_input_delta`` — those are LLM-stream
 events already produced by the assistant-message stream in
 :func:`protocore.runtime.query._stream_one_assistant_message`.
 **Does NOT emit** ``sandbox_starting`` — per
 ``,
 the sandbox adapter owns this event entirely. The adapter emits it
 only on **cold start** (with the rich payload including ``reason``,
 ``namespace``, ``pod_id``, ``sandbox_profile``); on hot-pod dispatch
 no event is emitted. The core dispatcher cannot distinguish hot from
 cold, so any emission here would be misleading double-emit noise.
* ``PostToolUse`` hook — fire-and-await; may rewrite the output.

The dispatcher is **agnostic to the tool implementation** — it depends
only on :class:`~protocore.contracts.tools.Tool.invoke`. Tests inject a
``MockTool`` via :class:`InMemoryToolRegistry`; no the host plumbing
required.

Output: every dispatch yields a stream of :class:`TurnEvent` envelopes
AND returns a final :class:`DispatchOutcome` summarising the call.
The caller (``query``) forwards events to the SSE stream and uses the
outcome to mutate engine state (append tool result to history,
transition to ``AWAITING`` on approval).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from protocore.constants import MAX_DATA_NESTING_DEPTH
from protocore.contracts.hooks import HookActionKind, IHookManager
from protocore.contracts.run import IRunToolErrorCounter
from protocore.contracts.tool_registry import (
    TOOL_VISIBILITY_POLICY_METADATA_KEY,
    IToolRegistry,
    ToolVisibilityPolicy,
)
from protocore.contracts.tools import ToolContext, ToolPolicyDenied
from protocore.contracts.types import (
    TOOL_RESULT_CONSECUTIVE_CAP_ELIGIBLE_METADATA_KEY,
    TOOL_RESULT_COUNT_AS_ERROR_METADATA_KEY,
    HookEvent,
    ToolCall,
    ToolResult,
)
from protocore.contracts.verification import EvidenceProducerBinding, EvidenceRecord
from protocore.logging_utils import get_logger
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.run_work_budget import RUN_WORK_LEDGER_HELPER_KEY
from protocore.runtime.tool_permission import (
    PermissionStage,
    ToolPermissionDecision,
    ToolPermissionGate,
    ToolPermissionOutcome,
)
from protocore.runtime.tool_preconditions import (
    check_preconditions,
    load_satisfied_set,
    record_satisfaction,
    store_satisfied_set,
)
from protocore.tools.ask_user import AskUserPauseRequested

_logger = get_logger(__name__)


def _metadata_flag(
    metadata: dict[str, Any] | None, key: str, *, default: bool
) -> bool:
    """Read a strict-boolean flag from a ToolResult metadata bag.

 Only an actual ``bool`` value overrides ``default`` — any other type
 (or a missing key) yields ``default``. Used by the soft-error path to
 honour a tool's ``count_as_tool_error`` / ``consecutive_error_cap_eligible``
 classification without trusting arbitrary truthy values that a hook
 might have injected into the bag.
 """
    if not metadata:
        return default
    value = metadata.get(key, default)
    if isinstance(value, bool):
        return value
    return default


def _helpers_of(ctx: ToolContext) -> dict[str, Any] | None:
    metadata = ctx.metadata or {}
    raw = metadata.get("protocore.helpers")
    if isinstance(raw, dict):
        return raw
    return None


def _coerce_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    if coerced < 0:
        return None
    return coerced


def _resolve_tool_timeout_seconds(tool: Any, *, flat_timeout_seconds: int) -> float:
    """Pick the dispatch wall-clock budget for one tool invocation.

    The flat ``rc.tool_timeout_seconds`` (default 90s) is the budget for an
    ordinary tool call. A tool that runs its OWN long-lived async unit with an
    internal, classified timeout — the ``Agent`` tool, whose ``SubagentRunner``
    wraps the child loop in ``asyncio.timeout(max_duration_ms)`` and owns an
    in-tool stale watchdog — declares this by setting ``should_defer=True`` plus
    a ``default_timeout_ms`` ClassVar (the host ``TypedTool``; not part of the
    core :class:`~protocore.contracts.tools.Tool` ABC, hence read defensively).
    For such a tool the dispatcher MUST give at least the tool's declared budget,
    otherwise the flat cap cancels the unit mid-run and its own classified
    timeout (``DispatchErrorKind.timeout`` / ``abort_kind``) can never fire.

    Override semantics: when ``should_defer`` is a strict ``True`` and
    ``default_timeout_ms`` is a positive int, the dispatch budget IS the tool's
    declared seconds (no min/max with the flat cap — the deferring tool owns its
    own clock). Every other tool keeps the flat budget. One field, one read.
    """
    if getattr(tool, "should_defer", False) is not True:
        return float(flat_timeout_seconds)
    declared_ms = _coerce_non_negative_int(getattr(tool, "default_timeout_ms", None))
    if not declared_ms:
        return float(flat_timeout_seconds)
    return declared_ms / 1000.0


class ToolDispatchCancelled(asyncio.CancelledError):
    """Run-level cancel interrupted an in-flight tool dispatch (#6).

    A subclass of :class:`asyncio.CancelledError` so EVERY existing
    ``except asyncio.CancelledError`` handler up the stack (the core query
    generator's cooperative unwind, the host executor ``_drive_run``
    cancelled-arm, the subagent runner's external-cancel path) treats it
    identically — but the distinct type makes the origin (a SET per-run
    ``cancel_event``, not a generic task cancellation) traceable in logs.
    """


def _run_cancel_event(ctx: ToolContext) -> asyncio.Event | None:
    """Duck-typed read of the per-run cancel ``asyncio.Event`` from the bag.

    Returns the Event ONLY when the helper bag carries a real, already-bound
    :class:`asyncio.Event` under :data:`HELPER_RUN_CANCEL_EVENT_KEY`. Any other
    shape (absent key / wrong type — older callers, unit tests) returns ``None``
    so the dispatcher keeps its pre-#6 ``await asyncio.wait_for`` path
    byte-identical. One field, one read.
    """
    helpers = _helpers_of(ctx)
    if helpers is None:
        return None
    candidate = helpers.get(HELPER_RUN_CANCEL_EVENT_KEY)
    if isinstance(candidate, asyncio.Event):
        return candidate
    return None


def _tool_cancel_drain_seconds(ctx: ToolContext) -> float:
    """Bounded drain budget for a cancelled in-flight tool task (RC-driven)."""
    helpers = _helpers_of(ctx)
    rc = helpers.get("rc") if isinstance(helpers, dict) else None
    drain = getattr(rc, "tool_cancel_drain_seconds", None)
    if isinstance(drain, bool) or not isinstance(drain, (int, float)):
        return _TOOL_CANCEL_DRAIN_FALLBACK_SECONDS
    drain_f = float(drain)
    return drain_f if drain_f > 0 else _TOOL_CANCEL_DRAIN_FALLBACK_SECONDS


async def _invoke_tool_raced_with_cancel(
    *,
    tool: Any,
    ctx: ToolContext,
    final_args: dict[str, Any],
    effective_timeout: float,
    cancel_event: asyncio.Event,
) -> ToolResult:
    """Await ``tool.invoke`` while racing the per-run cancel Event (#6).

 On the happy path this is equivalent to
 ``await asyncio.wait_for(tool.invoke(ctx, final_args), timeout=...)`` — it
 returns the tool result, and the cancel watcher is cancelled cleanly.

 When ``cancel_event`` fires FIRST, the in-flight tool task is cancelled,
 bounded-drained (so the ``Agent`` tool's subagent teardown can unwind), and
 :class:`ToolDispatchCancelled` is raised so the run finalises ``cancelled``
 promptly instead of after the whole tool (subagent) runs.

 Follows the KB anti-pattern fix (→ "asyncio
 .wait does NOT cancel its child tasks"): ``asyncio.wait`` leaves the
 not-yet-done task running, so BOTH branches explicitly cancel + drain the
 loser task; the ``finally`` guards against a cancel of THIS coroutine
 leaking either child task.
 """
    # Already cancelled BEFORE we start → never construct/await the tool
    # coroutine. A coroutine body runs its side-effects synchronously up to its
    # first ``await``, so creating the task at all would let a post-cancel
    # tool/subagent begin. Raise without touching the tool so the "no
    # post-cancel tool calls" contract holds even when the cancel landed between
    # the dispatch pre-checks and this point.
    if cancel_event.is_set():
        _logger.warning(
            "DIAG tool_dispatch.cancelled_before_invoke tool=%s run=%s — "
            "cancel already set; not starting the tool",
            getattr(tool, "name", type(tool).__name__),
            ctx.run_id,
        )
        raise ToolDispatchCancelled(
            f"tool dispatch cancelled for run {ctx.run_id!r}"
        )
    tool_task: asyncio.Task[Any] = asyncio.ensure_future(
        asyncio.wait_for(tool.invoke(ctx, final_args), timeout=effective_timeout)
    )
    cancel_task: asyncio.Task[Any] = asyncio.ensure_future(cancel_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {tool_task, cancel_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except asyncio.CancelledError:
        # THIS coroutine was cancelled (e.g. executor drain) — do not leak
        # either child task. Cancel + best-effort settle both, then re-raise.
        for task in (tool_task, cancel_task):
            if not task.done():
                task.cancel()
        for task in (tool_task, cancel_task):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        raise

    # Cancel fired before the tool finished → abort the tool task.
    if cancel_event.is_set() and tool_task not in done:
        tool_task.cancel()
        drain = _tool_cancel_drain_seconds(ctx)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(tool_task, timeout=drain)
        _logger.warning(
            "DIAG tool_dispatch.cancelled_in_flight tool=%s run=%s — "
            "run cancel raced the in-flight tool task; raising cancelled",
            getattr(tool, "name", type(tool).__name__),
            ctx.run_id,
        )
        raise ToolDispatchCancelled(
            f"tool dispatch cancelled for run {ctx.run_id!r}"
        )

    # Tool finished (or raised) first → tear down the cancel watcher and
    # surface the tool's result/exception exactly as the plain path would.
    if not cancel_task.done():
        cancel_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await cancel_task
    result: ToolResult = await tool_task
    return result


async def _record_tool_call_soft_cap_warning(
    *,
    ctx: ToolContext,
    tool_name: str,
) -> dict[str, Any] | None:
    """Increment the optional subagent soft-cap counter and return a warning.

    The counter is scoped to the engine helper bag, so the host child engines
    get one isolated count set per subagent run. A cap of 0 means unlimited.
    The warning is advisory only: the caller annotates the already-produced
    tool result and never alters dispatch success/error semantics.
    """
    helpers = _helpers_of(ctx)
    if helpers is None:
        return None
    raw_limits = helpers.get(TOOL_CALL_SOFT_CAPS_HELPER_KEY)
    if not isinstance(raw_limits, dict):
        return None
    limit = _coerce_non_negative_int(raw_limits.get(tool_name))
    if limit is None or limit == 0:
        return None

    raw_state = helpers.get(TOOL_CALL_SOFT_CAP_STATE_HELPER_KEY)
    if not isinstance(raw_state, dict):
        raw_state = {}
        helpers[TOOL_CALL_SOFT_CAP_STATE_HELPER_KEY] = raw_state

    lock = raw_state.get(_TOOL_CALL_SOFT_CAP_LOCK_KEY)
    if lock is None:
        lock = helpers.get(_TOOL_SHARED_STATE_LOCK_HELPER_KEY)
    if lock is None or not hasattr(lock, "__aenter__"):
        lock = asyncio.Lock()
    raw_state[_TOOL_CALL_SOFT_CAP_LOCK_KEY] = lock

    async with lock:
        raw_counts = raw_state.setdefault(_TOOL_CALL_SOFT_CAP_COUNTS_KEY, {})
        counts = raw_counts if isinstance(raw_counts, dict) else {}
        raw_state[_TOOL_CALL_SOFT_CAP_COUNTS_KEY] = counts
        count = _coerce_non_negative_int(counts.get(tool_name)) or 0
        count += 1
        counts[tool_name] = count

        if count < limit:
            return None

        status = "reached" if count == limit else "exceeded"
        if status == "reached":
            message = (
                f"Warning: {tool_name} call soft cap {limit} reached "
                f"(call {count}). Continue only if necessary; expected work "
                "quality may decline after this point."
            )
        else:
            message = (
                f"Warning: {tool_name} call soft cap {limit} exceeded "
                f"(call {count}). The tool still ran; expected work quality "
                "may decline."
            )
        warning = {
            "tool_name": tool_name,
            "limit": limit,
            "count": count,
            "status": status,
            "message": message,
        }
        raw_warnings = raw_state.setdefault(_TOOL_CALL_SOFT_CAP_WARNINGS_KEY, [])
        warnings = raw_warnings if isinstance(raw_warnings, list) else []
        raw_state[_TOOL_CALL_SOFT_CAP_WARNINGS_KEY] = warnings
        warnings.append(dict(warning))
        return warning


def _append_soft_cap_warning_to_content(content: str, warning: Mapping[str, Any]) -> str:
    message = str(warning.get("message") or "").strip()
    if not message:
        return content
    separator = "\n\n" if content else ""
    return f"{content}{separator}[Tool call soft-cap warning]\n{message}"


def _merge_soft_cap_warning_metadata(
    metadata: dict[str, Any] | None,
    warning: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(metadata or {})
    warning_dict = dict(warning)
    merged[TOOL_CALL_SOFT_CAP_METADATA_KEY] = warning_dict
    raw_warnings = merged.get(TOOL_CALL_SOFT_CAP_WARNINGS_METADATA_KEY)
    if isinstance(raw_warnings, list):
        warnings = [*raw_warnings, warning_dict]
    else:
        warnings = [warning_dict]
    merged[TOOL_CALL_SOFT_CAP_WARNINGS_METADATA_KEY] = warnings
    return merged


def _annotate_tool_result_event(
    event: TurnEvent,
    *,
    warning: Mapping[str, Any],
    content: str,
    metadata: dict[str, Any],
) -> TurnEvent:
    if event.type is not EventType.TOOL_RESULT:
        return event
    payload = dict(event.payload)
    blocks = payload.get("content_blocks")
    if isinstance(blocks, list) and blocks:
        next_blocks: list[Any] = []
        replaced = False
        for block in blocks:
            if (
                not replaced
                and isinstance(block, dict)
                and block.get("type") == "text"
            ):
                next_block = dict(block)
                next_block["text"] = content
                next_blocks.append(next_block)
                replaced = True
            else:
                next_blocks.append(block)
        payload["content_blocks"] = next_blocks
    else:
        payload["content_blocks"] = [{"type": "text", "text": content}]
    payload["metadata"] = metadata
    if "error" in payload and isinstance(payload["error"], dict):
        error_payload = dict(payload["error"])
        error_payload["soft_cap_warning"] = dict(warning)
        payload["error"] = error_payload
    return event.model_copy(update={"payload": payload})


# Consecutive same-tool-same-error cap.
# Research found the leader can retry an IDENTICAL failed tool call up to
# 200 times (docgen-en-005, long-en-004 Write storms). The dispatcher tracks
# a single per-run streak on the helper bag (`protocore.helpers`); when the
# (tool_name, signature) tuple repeats more times than the RC cap, the next
# error is rewritten with the `consecutive_error_cap` kind so the model sees
# a distinct stop signal instead of looping. Streak resets on (a) a different
# (tool, signature) tuple or (b) a successful tool call.
_CONSECUTIVE_ERROR_STATE_KEY: str = "tool_dispatch.consecutive_error_state"

# Fallback default when no RC is wired into the helper bag (test fixtures /
# legacy dispatch paths). Mirrors the executor's defensive ``getattr`` pattern
# for the analogous ``max_ask_user_calls_per_run``.
_DEFAULT_CONSECUTIVE_ERROR_CAP: int = 4

# Subagent tool-call soft caps. The host injects these helper-bag entries only
# for child/subagent runs; core treats them as an optional diagnostics layer
# above normal dispatch and never gates tool execution.
TOOL_CALL_SOFT_CAPS_HELPER_KEY: str = "subagent_tool_call_soft_caps"
TOOL_CALL_SOFT_CAP_STATE_HELPER_KEY: str = "subagent_tool_call_soft_cap_state"
TOOL_CALL_SOFT_CAP_METADATA_KEY: str = "tool_call_soft_cap"
TOOL_CALL_SOFT_CAP_WARNINGS_METADATA_KEY: str = "tool_call_soft_cap_warnings"

_TOOL_SHARED_STATE_LOCK_HELPER_KEY: str = "tool_shared_state_lock"
_TOOL_CALL_SOFT_CAP_COUNTS_KEY: str = "counts"
_TOOL_CALL_SOFT_CAP_WARNINGS_KEY: str = "warnings"
_TOOL_CALL_SOFT_CAP_LOCK_KEY: str = "lock"

# SANDBOX_DOWN canonical-signature streak.
# Tracks a SEPARATE counter from the generic consecutive-error cap so the
# threshold can fire earlier (default 3 vs cap default 4). When the streak
# reaches ``RuntimeConstants.sandbox_down_system_message_threshold`` the
# dispatcher posts a pending-injection signal on the helper bag for the
# host's executor loop to consume — it appends a synthetic user-role
# :class:`Message` instructing the agent to switch to inline (Write-only)
# strategy. The signal is one-shot per streak: a successful tool call OR a
# fresh non-SANDBOX_DOWN signature clears it. Observed prompts hitting
# 38-46 errored Bash calls each against rotating supervisor IPs;
# without the inline-strategy nudge the agent kept
# retrying and consumed its full token budget without making progress.
_SANDBOX_DOWN_STREAK_STATE_KEY: str = "tool_dispatch.sandbox_down_streak"
_SANDBOX_DOWN_INJECTION_PENDING_KEY: str = "tool_dispatch.sandbox_down_injection_pending"
_SANDBOX_DOWN_CANONICAL_SUFFIX: str = ":SANDBOX_DOWN"

# Fallback default when no RC is wired (test fixtures, legacy paths).
# Mirrors :data:`_DEFAULT_CONSECUTIVE_ERROR_CAP` defensive pattern.
_DEFAULT_SANDBOX_DOWN_THRESHOLD: int = 3

# Stabilization Pydantic ``string_type``
# terminal streak. Tracks consecutive validation errors with
# ``type=string_type`` on the SAME tool independently of the generic
# consecutive-error cap so this failure mode can be tuned separately.
# When the streak reaches ``RuntimeConstants.tool_dispatch_string_type_
# terminal_cap`` the dispatcher rewrites the next dispatch error to
# ``DispatchErrorKind.consecutive_error_cap`` with a stronger terminal
# guidance string (the model is told to stop retrying the same shape).
# The streak resets on a successful tool call OR a fresh non-string_type
# error signature, the same pattern as the SANDBOX_DOWN counter.
#
# A host's own write/append/bash tools are expected to coerce the common
# malformed shapes (list/dict instead of str) silently, so this guard is a
# safety net for residual
# cases (e.g. an uncoercible ``content=None`` or a future field that did
# not get a coercion validator).
_STRING_TYPE_STREAK_STATE_KEY: str = "tool_dispatch.string_type_streak"
_STRING_TYPE_CANONICAL_MARKER: str = "string_type"

# Fallback default when no RC is wired (test fixtures, legacy paths).
# Lowered from 5 to 3 so the string_type-specific TERMINAL guidance fires
# BEFORE the generic ``tool_dispatch_consecutive_error_cap`` (default 4)
# wraps the error with vague guidance.
_DEFAULT_STRING_TYPE_TERMINAL_CAP: int = 3

# Normalisation regexes — strip variable identifiers (uuid-shaped call ids,
# absolute paths, line numbers, the dispatcher's own ``after Ns`` timeout
# wording) so a real retry of the same logical error collapses to the same
# signature. Keeps the error_kind in the signature so a Write-validation error
# and a Write-execution error of the same wording are still distinct.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_HEX_TOKEN_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")
_NUMBER_RE = re.compile(r"\b\d+\b")
_TIMEOUT_DURATION_RE = re.compile(r"timed out after \d+s")

# Broaden signature normalisation so transient sandbox-unreachable /
# command-missing errors collapse to a single canonical signature regardless
# of Bash command shape. The leader varies command text on retry (different
# quoted strings, paths, regex snippets) so the (tool, signature) tuple kept
# changing and the consecutive-error cap never fired on subagent-crash cases.
#
# Canonical patterns take precedence over hashing — when a message matches
# ``_SANDBOX_DOWN_PATTERNS`` the signature is fixed to ``<tool>:SANDBOX_DOWN``
# and the cap can collapse N varied retries against a dead sandbox into a
# single streak. Similarly for ``command not found`` on Bash.
#
# When no canonical pattern matches we still hash, but first strip quoted
# strings and absolute file paths so cosmetic variation between retries
# (different file names, regex snippets, search needles) does not bump the
# signature.
#
# The literal `supervisor 5\d\d` alternative DOES NOT MATCH any real production
# wording. Every actual sandbox-down message has context between `supervisor`
# and the status code (`supervisor /exec returned 502`, `supervisor at <url>
# unreachable: <exc>`, `supervisor unreachable after respawn: <exc>`,
# `supervisor auth failed: <exc>`, `sandbox provision failed: <exc>`,
# `sandbox readiness failed after pod create: <exc>`). Those phrasings come
# from the sandbox RPC client, which raises an unreachable/auth error; the
# sandbox dispatcher wraps them into a dispatch error; and the Bash tool wraps
# that once more into the outer ``"sandbox dispatch failed: ..."`` shape that
# the dispatcher here actually sees.
#
# The legacy ``supervisor 5\d\d`` short form is RETAINED for backward
# compatibility with existing tests + as a defensive safety net
# for any future emit site that omits the descriptive context. New alternatives
# cover every production phrase grouped by emit module.
#
# Tenant sandbox capacity exhaustion is a common failure cluster: the
# capacity wording was not previously recognised by ``_SANDBOX_DOWN_PATTERNS``
# — only supervisor/provision/readiness failures were. Net effect: the
# SANDBOX_DOWN canonical signature never fired for typed admission denials,
# the inline Write-only nudge never injected, and the model burned Bash retries
# on each capacity-blocked tool call until the run hit its turn budget.
# Adding the capacity wording here causes the same inline-strategy nudge to
# fire after N capacity-blocked Bash retries (default N=3), so the model
# pivots to Write/Edit and frees up tenant quota for other concurrent runs.
# Behaviourally identical action to supervisor-down (the synthetic message
# already says "Sandbox is currently unavailable; switch to inline strategy" —
# applies verbatim to capacity).
#
# Emit sites covered:
# * the raw capacity-exhausted detail
#   ``"sandbox capacity exhausted (tenant_id=..., dimension=..., retry_after=Ns)"``.
# * the Bash tool's user-visible wrapper
#   ``"Sandbox temporarily unavailable, will retry. Reason: capacity exhausted
#   (dimension=cpu, retry_after=5s)"`` — this is the form the dispatcher sees
#   most often because Bash is the only tool that surfaces capacity to the
#   model.
_SANDBOX_DOWN_PATTERNS = re.compile(
    r"\b("
    # Direct exception name surfaced raw (e.g. uncaught raise in tests).
    r"SandboxUnreachable"
    # supervisor_rpc.py: 5xx / unexpected status / timed-out / unreachable / auth.
    r"|supervisor\s+(?:/\w+\s+)?returned\s+5\d\d"
    r"|supervisor\s+(?:/\w+\s+)?unexpected\s+\d{3}"
    r"|supervisor\s+(?:/\w+\s+)?timed\s+out"
    r"|supervisor\s+(?:at\s+\S+\s+)?unreachable"
    r"|supervisor\s+auth\s+failed"
    r"|supervisor\s+rejected"
    # Legacy short form kept for backward compatibility with existing
    # tests and as a defensive net for any future bare emit site.
    r"|supervisor\s+5\d\d"
    # dispatcher.py: SandboxDispatchError wrappers + bash.py outer wrapper.
    r"|sandbox\s+(?:provision|readiness|registration|cache\s+update|dispatch)\s+failed"
    r"|tenant\s+namespace\s+provision\s+failed"
    # No current emit site, defensive — anticipates a future "session expired"
    # surface; cheap to keep, harmless if it never fires.
    r"|sandbox\s+session\s+not\s+active"
    # Lower-level exec failure phrasings (sandbox-side).
    r"|exec\s+failed:\s*(?:connection\s+refused|pod\s+not\s+ready)"
    # typed capacity-exhausted admission denial. Both the
    # raw :class:`SandboxCapacityExhausted` detail wording and the bash.py
    # user-visible wrapper are covered. ``capacity\s+exhausted`` is the
    # canonical phrase shared by both; ``sandbox\s+(?:temporarily\s+)?
    # unavailable`` covers the bash.py wrapper prefix in isolation as a
    # defensive belt-and-braces match.
    r"|capacity\s+exhausted"
    r"|sandbox\s+(?:temporarily\s+)?unavailable"
    r")\b",
    re.IGNORECASE,
)
_BASH_CMD_MISSING_PATTERNS = re.compile(
    r"command not found",
    re.IGNORECASE,
)
# Stabilization detect Pydantic
# ``string_type`` validation errors in dispatcher error text.
# The host typed-tool adapter raises ``ToolInvocationError`` whose
# message includes the sanitized ``exc.errors(include_input=False)``
# dump, which still carries ``'type': 'string_type'`` for the offending
# field. The dispatcher uses this to maintain a separate consecutive-
# streak counter so the failure can be terminated faster than the
# generic ``tool_dispatch_consecutive_error_cap``.
#
# The SAME terminal cap must also fire on Pydantic ``type='missing'``
# (``Field required``). A content-less ``Write`` produces
# ``[{'type': 'missing', 'loc': ('content',), 'msg': 'Field required'}]``
# on EVERY retry; matching only ``string_type`` would mean the schema-specific
# TERMINAL guidance never fires and the loop spirals. Both are "the argument
# is the wrong SHAPE / absent, retrying the same shape will not succeed"
# failures, so they share the one terminal streak. (The chunk-recovery logic
# prevents a *truncated* call from ever reaching dispatch; this cap is the
# backstop for a genuinely model-omitted required field.)
_STRING_TYPE_VALIDATION_PATTERN = re.compile(
    r"'type'\s*:\s*'(?:string_type|missing)'",
)
# Alternation order is single / double / back-tick. Each alternative is
# greedy up to the next matching delimiter on the same line. We do not
# handle triple-quoted strings — they do not appear in sanitised error
# messages emitted by ``_safe_error_message``.
_QUOTED_REGEX = re.compile(r"'[^']*'|\"[^\"]*\"|`[^`]*`")
# Match absolute POSIX file paths (``/foo``, ``/foo/bar.py``). Underscores,
# dots, dashes and slashes are common in test/workspace paths and uniquely
# identify the file being acted on, so they should not contribute to the
# signature.
_FILE_PATH_REGEX = re.compile(r"(?:/[a-zA-Z0-9_.-]+)+")

# Strip rotating supervisor URLs from error text before hashing so
# multi-IP supervisor-unreachable storms collapse to a single canonical
# signature. The sandbox control plane respawns supervisors on rotating IPs
# (``10.0.0.1``, ``10.0.0.2``, ``10.0.0.3``, …) and each
# "supervisor at <url> unreachable" line embeds a fresh IP. Without this
# collapse the SANDBOX_DOWN canonical pattern still fires per-message, but
# defence in depth requires normalised text whenever the hash path is reached.
# Matches private ``10.x.x.x:port`` and other rfc1918 wordings the supervisor
# may emit.
_SUPERVISOR_URL_PATTERN = re.compile(
    r"https?://10\.\d+\.\d+\.\d+:\d+",
    re.IGNORECASE,
)


# Hallucination hint for `<*_contract>` XML blocks misemitted as tool names.
# Case-sensitive lowercase-only `^[a-z_]+_contract$` to avoid false positives
# on legitimate (e.g. CamelCase) tool names; only the model's
# "I-meant-the-XML-block" failure mode follows this snake_case shape
# (`finalization_contract`, `subagent_contract`, …).
_CONTRACT_HALLUCINATION = re.compile(r"^[a-z_]+_contract$")


def _contract_hallucination_hint(tool_name: str, *, finalize_terminal: bool = False) -> str:
    """Build the nudge message for a hallucinated ``*_contract`` tool name.

    Typed-Finalize tenants (``RuntimeConstants.agent_finalize_tool_as_terminal``)
    must NOT be steered to the legacy ``<finalization_contract>`` XML block when
    they misfire a ``finalization_contract`` "tool" — that re-introduces the very
    prose-contract leak the ``Finalize`` tool retires. For them the nudge points
    at ``Finalize`` instead. Every other ``*_contract`` shape (and the opt-out /
    default tenants) keep the inline-XML nudge byte-for-byte.
    """
    if finalize_terminal and tool_name == "finalization_contract":
        return (
            f"'{tool_name}' is not a tool. To finish, call the `Finalize` tool "
            f"with declared_deliverables and answer — do NOT write a "
            f"<finalization_contract> text block."
        )
    return (
        f"'{tool_name}' is not a tool — it appears to be an XML contract "
        f"block. Inline it directly in your assistant text as "
        f"<{tool_name}>...</{tool_name}> (especially the "
        f"<finalization_contract> block per the leader persona spec)."
    )


# ---------------------------------------------------------------------------
# Error taxonomy.
# ---------------------------------------------------------------------------


class DispatchErrorKind(StrEnum):
    """Error kinds per

 Every kind wraps into a ``tool_result(success=false)`` content block;
 the model sees the failure and recovers in the next assistant turn.

 ``consecutive_error_cap`` tracks
 dispatcher rewrites a real failure into this kind once the per-run
 consecutive-identical-error streak exceeds
 ``RuntimeConstants.tool_dispatch_consecutive_error_cap``.
 """

    validation = "validation"
    permission = "permission"
    execution = "execution"
    timeout = "timeout"
    rate_limit = "rate_limit"
    unknown_tool = "unknown_tool"
    consecutive_error_cap = "consecutive_error_cap"


@dataclass(frozen=True, slots=True)
class DispatchOutcome:
    """Final outcome of one :meth:`ToolDispatcher.dispatch` call.

 The caller uses this to (a) append a tool-result :class:`Message` to
 history and (b) decide whether to transition to ``AWAITING``.

 Attributes
 ----------
 tool_call:
 The original :class:`ToolCall` (after any hook mutation).
 success:
 ``True`` only if the tool executed and returned a non-error
 :class:`ToolResult`.
 content:
 Text content for the resulting :class:`ToolResultBlock`. On
 error this is the sanitised error message.
 is_error:
 Mirror of ``not success`` — surfaced as
 :attr:`ToolResultBlock.is_error`.
 error_kind:
 Non-None on failure paths only. Drives audit + telemetry.
 approval_required:
 ``True`` if the gate returned ``require_approval`` — the loop
 MUST transition to ``AWAITING`` and not append a tool result.
 approval_token:
 Opaque token to round-trip to the user.
 ask_user_required:
 ``True`` if the tool raised :class:`AskUserPauseRequested`
 — the host's dispatch shim
 transitions the loop to ``AWAITING`` via the ``ask_user`` resume
 path, emits the matching SSE envelope, and writes the pending
 interrupt to Redis. No ``tool_result`` is appended; the answer
 arrives on resume.
 ask_user_payload:
 Validated :class:`~protocore.tools.ask_user.AskUserInput`
 carried out of band (typed dict shape) so the host handler
 does not re-validate. Stored as a plain dict (``model_dump``) to
 keep :class:`DispatchOutcome` import-light and JSON-friendly for
 cross-pod transport.
 duration_ms:
 Wall-time of the execute step (zero on pre-execute denial).
 metadata:
 Optional bag for telemetry (provider-meta, stage, …).
 evidence_records:
 Typed tool-authored observations retained only by the runtime. They are
 intentionally absent from model-visible result blocks and event payloads.
 """

    tool_call: ToolCall
    success: bool
    content: str
    is_error: bool
    error_kind: DispatchErrorKind | None = None
    approval_required: bool = False
    approval_token: str | None = None
    ask_user_required: bool = False
    ask_user_payload: dict[str, Any] | None = None
    duration_ms: int = 0
    metadata: dict[str, Any] | None = None
    evidence_records: tuple[EvidenceRecord, ...] = ()
    evidence_producer: EvidenceProducerBinding | None = None


DISPATCH_REPLAY_ERROR_KIND_METADATA_KEY: str = "tool_dispatch.replay_error_kind"
DISPATCH_REPLAY_ERROR_MESSAGE_METADATA_KEY: str = "tool_dispatch.replay_error_message"
DISPATCH_POST_TOOL_OUTPUT_MODIFIED_METADATA_KEY: str = (
    "tool_dispatch.post_tool_output_modified"
)

# The metadata slot carrying a tool's machine-readable give-up signal, and
# the sub-key the loop reads to decide whether to surface a finalize hint to
# the model. A tool attaches ``structured_error`` to its raised exception
# (e.g. a retry-budget exhaustion: ``{"finalization_recommended": True,
# "reason": ...}``); the dispatch except-branch forwards it verbatim under
# :data:`DISPATCH_STRUCTURED_ERROR_METADATA_KEY` on the DispatchOutcome.
DISPATCH_STRUCTURED_ERROR_METADATA_KEY: str = "structured_error"
STRUCTURED_ERROR_FINALIZATION_RECOMMENDED_KEY: str = "finalization_recommended"
STRUCTURED_ERROR_REASON_KEY: str = "reason"

#: The helper-bag key under which the executor places a per-run cancel
#: ``asyncio.Event`` (``ctx.metadata["protocore.helpers"]["cancel_event"]``). The
#: the host side SETs it on a user cancel; the dispatcher RACES the in-flight
#: tool task against ``cancel_event.wait()`` so a leader blocked inside the
#: synchronous ``Agent`` tool unblocks promptly instead of after the whole
#: subagent runs. MUST stay in lockstep with
#: ``executor_main.HELPER_RUN_CANCEL_EVENT`` /
#: ``subagent_runner.HELPER_RUN_CANCEL_EVENT`` (both ``"cancel_event"``). Read
#: best-effort / duck-typed: an absent or non-Event value ⇒ pre-#6 behaviour
#: (byte-identical), so older callers / tests are unaffected.
HELPER_RUN_CANCEL_EVENT_KEY: Final[str] = "cancel_event"

#: Helper-bag key under which the first-fan-out run stores the shared
#: :class:`~protocore.runtime.subagent_budget.SubagentTreeBudget` — one object per
#: maximal parallel-dispatched subtree, threaded by reference to every descendant
#: so that subtree draws parallel-subagent slots from a single semaphore. It is
#: minted lazily in the concurrent-delegation branch from
#: ``rc.max_concurrent_subagents_per_tree`` by the first run to fan out with no
#: budget in its bag (usually the root; deeper if the root only delegates
#: serially) and stored here; descendants inherit the SAME object via the
#: parent-helpers dict-copy (identical propagation to ``cancel_event`` /
#: ``root_run_id``). Absent ⇒ no tree-wide bound (older callers / tests).
HELPER_SUBAGENT_TREE_BUDGET_KEY: Final[str] = "subagent_tree_budget"

#: Helper-bag key under which a child engine finds ITS OWN
#: :class:`~protocore.runtime.subagent_budget.SubagentTreePermit`. Unlike the
#: shared budget, this is per-child: the parent acquires a tree slot at the
#: dispatch site, stamps the handle on the child's dispatch metadata
#: (:data:`~protocore.contracts.tools.SUBAGENT_TREE_PERMIT_METADATA_KEY`), and the
#: the host runner lodges it here in the child's freshly-built helper bag. The
#: child's delegation branch reads it to release-while-awaiting around its own
#: nested gather. Absent ⇒ this run holds no tree slot to release (root leader /
#: serial dispatch).
HELPER_SUBAGENT_TREE_PERMIT_KEY: Final[str] = "subagent_tree_permit"

#: Helper-bag key under which the ROOT run's
#: :class:`~protocore.runtime.run_work_budget.RunWorkLedger` lives — the
#: CUMULATIVE total-work budget for the whole tree (child runs started, tokens
#: charged), as opposed to the instantaneous concurrency bound above. Minted for
#: the root when its bag is composed rather than lazily at the first fan-out,
#: because a leader that emits one delegation call per turn never fans out and is
#: exactly the wave-after-wave pattern the cumulative bound exists for. Inherited
#: by every descendant through the parent-helpers dict-copy, so the whole tree
#: counts into ONE ledger. Re-exported from the module that owns it rather than
#: restated, so there is one string and nothing to keep in lockstep.
HELPER_RUN_WORK_LEDGER_KEY: Final[str] = RUN_WORK_LEDGER_HELPER_KEY

#: Fallback bounded-drain budget (seconds) for a cancelled in-flight tool task
#: when the helper bag carries no ``rc`` (older callers / tests). The live path
#: uses ``RuntimeConstants.tool_cancel_drain_seconds``; this mirrors its default
#: so behaviour is identical when the RC is unreachable.
_TOOL_CANCEL_DRAIN_FALLBACK_SECONDS: Final[float] = 2.0


def _dispatch_replay_metadata(
    kind: DispatchErrorKind,
    message: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(extra or {})
    metadata[DISPATCH_REPLAY_ERROR_KIND_METADATA_KEY] = kind.value
    metadata[DISPATCH_REPLAY_ERROR_MESSAGE_METADATA_KEY] = message
    return metadata


# ---------------------------------------------------------------------------
# The dispatcher
# ---------------------------------------------------------------------------


def _nesting_exceeds(value: Any, max_depth: int) -> bool:
    """True iff ``value`` nests containers deeper than ``max_depth``.

    Iterative on an explicit stack, so measuring the depth of a pathological
    structure cannot itself be what exhausts the stack. Only the container
    types ``json.dumps`` descends into are counted.
    """
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if not isinstance(item, (dict, list, tuple)):
            continue
        if depth >= max_depth:
            return True
        children = item.values() if isinstance(item, dict) else item
        for child in children:
            stack.append((child, depth + 1))
    return False


class ToolDispatcher:
    """Orchestrates the 7-step tool-call lifecycle.

    The dispatcher is **stateless per-call** — all state flows through
    parameters. Construct once per executor pod and re-use across runs.
    """

    def __init__(
        self,
        *,
        registry: IToolRegistry,
        permission_gate: ToolPermissionGate,
        hook_manager: IHookManager | None = None,
        tool_error_counter: IRunToolErrorCounter | None = None,
    ) -> None:
        self._registry = registry
        self._gate = permission_gate
        self._hooks = hook_manager
        self._tool_error_counter = tool_error_counter

    async def _record_tool_error(self, ctx: ToolContext) -> None:
        """Best-effort increment of ``runs.tool_errors_count`` for the run.

 Tool errors are attributed to the **parent (root) run** rather than the
 per-subagent ``ctx.run_id``.

 Why: the dashboard aggregates ``tool_errors_count`` at the root-run
 level for partial-status classification. Subagent ``run_id``s are
 internal and not present in the PG ``runs`` table (no row to update —
 they are not durably persisted as their own ``runs.id`` rows), and
 any non-UUID-shaped string fed into the ``WHERE id = $1`` UUID column
 triggers ``psycopg.errors.InvalidTextRepresentation`` (~20x/run noise).
 The parent's ``root_run_id`` is already plumbed by the host's subagent
 runner and agent-dispatch adapter via ``HELPER_ROOT_RUN_ID`` and lives in
 ``ctx.metadata["protocore.helpers"]["root_run_id"]``. It is the
 parent's bare UUID — semantically the right aggregate and SQL-safe.

 Non-raising — the dispatcher must never surface a telemetry failure
 as a tool error (counter leak / cascade). Logged at WARNING so the
 diagnostic is visible in production pod logs.
 """
        counter = self._tool_error_counter
        if counter is None:
            return
        target_run_id = self._resolve_error_attribution_run_id(ctx)
        try:
            await counter.increment_tool_errors_count(target_run_id, 1)
        except Exception:
            _logger.warning(
                "tool_error_counter increment failed run_id=%s",
                target_run_id,
                exc_info=True,
            )

    @staticmethod
    def _resolve_error_attribution_run_id(ctx: ToolContext) -> str:
        """Return the run id to attribute the tool-error increment to.

        Prefers ``ctx.metadata["protocore.helpers"]["root_run_id"]`` when
        present — the parent (root) run's bare UUID, plumbed by
        the host subagent dispatch path. Falls back to ``ctx.run_id`` for
        leader runs where root_run_id == run_id and for test contexts that
        do not wire a helpers bag.
        """
        metadata = ctx.metadata
        if metadata:
            helpers = metadata.get("protocore.helpers")
            if isinstance(helpers, dict):
                root_run_id = helpers.get("root_run_id")
                if isinstance(root_run_id, str) and root_run_id:
                    return root_run_id
        return ctx.run_id

    # ------------------------------------------------------------------
    # Consecutive same-tool-same-error cap
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_consecutive_error_cap(ctx: ToolContext) -> int:
        """Read ``tool_dispatch_consecutive_error_cap`` from the RC snapshot.

        Falls back to :data:`_DEFAULT_CONSECUTIVE_ERROR_CAP` when no helper
        bag / RC snapshot is wired (legacy test fixtures, dispatch paths
        that pre-date the RC plumbing). Mirrors the defensive ``getattr``
        pattern used for ``max_ask_user_calls_per_run``.
        """
        metadata = ctx.metadata
        if not metadata:
            return _DEFAULT_CONSECUTIVE_ERROR_CAP
        helpers = metadata.get("protocore.helpers")
        if not isinstance(helpers, dict):
            return _DEFAULT_CONSECUTIVE_ERROR_CAP
        rc = helpers.get("rc")
        if rc is None:
            return _DEFAULT_CONSECUTIVE_ERROR_CAP
        raw = getattr(rc, "tool_dispatch_consecutive_error_cap", _DEFAULT_CONSECUTIVE_ERROR_CAP)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_CONSECUTIVE_ERROR_CAP
        # ``ge=2`` is enforced by the RuntimeConstants validator; defence
        # in depth here so a corrupted snapshot cannot push the cap to 1
        # (which would reject the very first error).
        return value if value >= 2 else _DEFAULT_CONSECUTIVE_ERROR_CAP

    @staticmethod
    def _resolve_sandbox_down_threshold(ctx: ToolContext) -> int:
        """Read ``sandbox_down_system_message_threshold`` from the RC snapshot.

        Falls back to :data:`_DEFAULT_SANDBOX_DOWN_THRESHOLD` when no helper
        bag / RC snapshot is wired. Mirrors :meth:`_resolve_consecutive_error_cap`.
        """
        metadata = ctx.metadata
        if not metadata:
            return _DEFAULT_SANDBOX_DOWN_THRESHOLD
        helpers = metadata.get("protocore.helpers")
        if not isinstance(helpers, dict):
            return _DEFAULT_SANDBOX_DOWN_THRESHOLD
        rc = helpers.get("rc")
        if rc is None:
            return _DEFAULT_SANDBOX_DOWN_THRESHOLD
        raw = getattr(
            rc, "sandbox_down_system_message_threshold", _DEFAULT_SANDBOX_DOWN_THRESHOLD
        )
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_SANDBOX_DOWN_THRESHOLD
        # ``gt=0`` is enforced by the RuntimeConstants validator; defence in
        # depth so a corrupted snapshot cannot push the threshold to 0
        # (which would fire on every dispatch).
        return value if value >= 1 else _DEFAULT_SANDBOX_DOWN_THRESHOLD

    @staticmethod
    def _resolve_string_type_terminal_cap(ctx: ToolContext) -> int:
        """Read ``tool_dispatch_string_type_terminal_cap`` from the RC snapshot.

 Falls back to
 :data:`_DEFAULT_STRING_TYPE_TERMINAL_CAP` when no helper bag / RC
 snapshot is wired (legacy test fixtures). Mirrors
 :meth:`_resolve_consecutive_error_cap` defensive pattern.
 """
        metadata = ctx.metadata
        if not metadata:
            return _DEFAULT_STRING_TYPE_TERMINAL_CAP
        helpers = metadata.get("protocore.helpers")
        if not isinstance(helpers, dict):
            return _DEFAULT_STRING_TYPE_TERMINAL_CAP
        rc = helpers.get("rc")
        if rc is None:
            return _DEFAULT_STRING_TYPE_TERMINAL_CAP
        raw = getattr(
            rc,
            "tool_dispatch_string_type_terminal_cap",
            _DEFAULT_STRING_TYPE_TERMINAL_CAP,
        )
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_STRING_TYPE_TERMINAL_CAP
        # ``ge=2`` is enforced by the RuntimeConstants validator; defence
        # in depth here so a corrupted snapshot cannot push the cap to 1.
        return value if value >= 2 else _DEFAULT_STRING_TYPE_TERMINAL_CAP

    @staticmethod
    def _is_string_type_error(kind: DispatchErrorKind, message: str) -> bool:
        """Return ``True`` when *message* carries a Pydantic
 ``string_type`` validation error.

 The host typed-tool adapter raises
 ``ToolInvocationError`` whose message format is
 ``tool 'X': invalid arguments: [{'type': 'string_type', ...}]``.
 Matching on the textual presence of ``'type': 'string_type'``
 avoids tight coupling to either the validation or execution
 error kind: both routes can carry the same payload depending on
 which dispatcher branch caught the error.
 """
        if not message:
            return False
        del kind  # currently informational; both validation+execution carry the text
        return bool(_STRING_TYPE_VALIDATION_PATTERN.search(message))

    @staticmethod
    def _normalize_error_text(message: str) -> str:
        """Collapse variable identifiers so retries of the same logical
 failure produce identical signatures.

 Order matters:

 1. Strip rotating supervisor URLs
 (``http://10.0.0.1:9292`` and similar rfc1918 + port shapes)
 collapse to ``<SUPERVISOR_URL>`` before any other normalisation.
 Without this, a multi-IP supervisor-unreachable storm produces fresh
 hashed signatures every iteration when the SANDBOX_DOWN canonical
 pattern does not match the surface phrasing.
 2. Strip quoted content (`'foo'`, `"bar"`, ```baz```) and absolute
 file paths.
 3. Strip uuid-shaped tokens, long hex blobs, the dispatcher's own
 ``timed out after Ns`` suffix, and bare decimal numbers.
 4. Collapse whitespace so the hash stays stable across cosmetic
 spacing differences between retries.
 """
        normalized = _SUPERVISOR_URL_PATTERN.sub("<SUPERVISOR_URL>", message)
        normalized = _QUOTED_REGEX.sub("<quoted>", normalized)
        normalized = _FILE_PATH_REGEX.sub("<path>", normalized)
        normalized = _UUID_RE.sub("<uuid>", normalized)
        normalized = _HEX_TOKEN_RE.sub("<hex>", normalized)
        normalized = _TIMEOUT_DURATION_RE.sub("timed out after <n>s", normalized)
        normalized = _NUMBER_RE.sub("<n>", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @classmethod
    def _error_signature(
        cls,
        kind: DispatchErrorKind,
        message: str,
        tool_name: str | None = None,
    ) -> str:
        """Return the per-error-streak signature.

 Canonical signatures take precedence: a sandbox-unreachable message
 collapses to ``<tool>:SANDBOX_DOWN`` and a Bash ``command not found``
 collapses to ``Bash:BASH_CMD_MISSING`` regardless of surrounding text —
 without this the consecutive-error cap never fires when the leader varies
 its Bash command shape on retry (subagent-crash regression pattern).

 Otherwise we hash ``(error_kind, normalised_message)`` where the
 normalisation already strips quoted content + file paths via
 :meth:`_normalize_error_text`. Stable across retries of the same
 logical failure but distinct across error kinds + distinct logical
 failures.

 ``tool_name`` is optional for backwards compatibility with the few
 legacy callers / tests that build a signature without a tool. When
 omitted the canonical match falls back to the ``unknown`` namespace.
 """
        # Canonical-match short-circuit traces — logged so that when the
        # consecutive-error cap fires we can verify which branch triggered.
        if _SANDBOX_DOWN_PATTERNS.search(message):
            _logger.debug(
                "DIAG tool_dispatch.canonical_error_match "
                "tool=%s kind=%s match=SANDBOX_DOWN excerpt=%r",
                tool_name or "<unknown>",
                kind.value,
                message[:100],
            )
            return f"{tool_name or 'unknown'}:SANDBOX_DOWN"
        if tool_name == "Bash" and _BASH_CMD_MISSING_PATTERNS.search(message):
            _logger.debug(
                "DIAG tool_dispatch.canonical_error_match "
                "tool=Bash kind=%s match=BASH_CMD_MISSING excerpt=%r",
                kind.value,
                message[:100],
            )
            return f"{tool_name}:BASH_CMD_MISSING"
        normalized = cls._normalize_error_text(message)
        payload = f"{kind.value}|{normalized}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _helpers_bag(ctx: ToolContext) -> dict[str, Any] | None:
        """Return the per-run helper bag if wired into ``ctx.metadata``.

        The bag is mutable (built per run by
        ``service_runtime.build_helper_bag``); the dispatcher uses it to
        park a small streak-tracking cell that is naturally per-run
        isolated.
        """
        metadata = ctx.metadata
        if not metadata:
            return None
        helpers = metadata.get("protocore.helpers")
        return helpers if isinstance(helpers, dict) else None

    @classmethod
    def _apply_consecutive_error_cap(
        cls,
        ctx: ToolContext,
        tool_name: str,
        kind: DispatchErrorKind,
        message: str,
        *,
        emit_diagnostics: bool = True,
    ) -> tuple[DispatchErrorKind, str]:
        """Track the consecutive streak and rewrite the surfaced error
 once the per-run cap is exceeded.

 Returns the (kind, message) tuple to surface to the caller — either
 the original pair (streak under cap) or a synthesised
 :class:`DispatchErrorKind.consecutive_error_cap` with guidance to
 try a different approach + the original error text appended.

 State shape: a single mutable cell on the helper bag holding the
 last (tool_name, signature) and the current consecutive count.
 Switching tool OR signature resets the count to 1.

 In addition to the generic cap, the method
 maintains a SEPARATE counter for SANDBOX_DOWN canonical signatures
 (``<tool>:SANDBOX_DOWN``) and posts a one-shot injection signal on
 the helper bag when that counter reaches
 ``RuntimeConstants.sandbox_down_system_message_threshold``. The
 signal is consumed by the host executor loop, which appends a
 synthetic user-role message instructing the agent to switch to
 inline (Write-only) strategy.
 """
        helpers = cls._helpers_bag(ctx)
        if helpers is None:
            # No bag wired — best-effort no-op so unit tests + legacy
            # dispatch paths keep working. Production paths always wire
            # the bag via ``query._dispatch_tool``.
            return kind, message

        signature = cls._error_signature(kind, message, tool_name)
        state_raw = helpers.get(_CONSECUTIVE_ERROR_STATE_KEY)
        last_tool: str | None = None
        last_sig: str | None = None
        count: int = 0
        if isinstance(state_raw, dict):
            last_tool_raw = state_raw.get("tool_name")
            last_sig_raw = state_raw.get("signature")
            count_raw = state_raw.get("count", 0)
            if isinstance(last_tool_raw, str):
                last_tool = last_tool_raw
            if isinstance(last_sig_raw, str):
                last_sig = last_sig_raw
            if isinstance(count_raw, int) and count_raw >= 0:
                count = count_raw

        if last_tool == tool_name and last_sig == signature:
            count += 1
        else:
            count = 1

        helpers[_CONSECUTIVE_ERROR_STATE_KEY] = {
            "tool_name": tool_name,
            "signature": signature,
            "count": count,
        }

        # Separately track the SANDBOX_DOWN streak
        # so the injection threshold can fire earlier (default 3) than the
        # generic cap (default 4). The two counters are kept independent so
        # operators can tune them independently per category.
        cls._track_sandbox_down_streak(
            ctx, helpers, signature, emit_diagnostics=emit_diagnostics
        )

        # separately track Pydantic
        # ``string_type`` validation errors so the terminal guard for the
        # malformed-args loop can fire independently of (and typically
        # later than) the generic cap. The streak resets on either a
        # non-string_type error or a successful tool call (handled in
        # :meth:`_reset_consecutive_error_streak`).
        string_type_terminal = cls._track_string_type_streak(
            ctx,
            helpers,
            tool_name,
            kind,
            message,
            emit_diagnostics=emit_diagnostics,
        )
        if string_type_terminal is not None:
            return string_type_terminal

        cap = cls._resolve_consecutive_error_cap(ctx)
        if count >= cap:
            if emit_diagnostics:
                _logger.warning(
                    "DIAG tool_dispatch.consecutive_error_cap "
                    "run=%s tool=%s count=%d cap=%d",
                    ctx.run_id,
                    tool_name,
                    count,
                    cap,
                )
            wrapped = (
                f"This tool+error combination repeated {count} consecutive "
                "times. Try a different tool or argument shape. "
                f"{message}"
            )
            return DispatchErrorKind.consecutive_error_cap, wrapped

        return kind, message

    @classmethod
    def _track_sandbox_down_streak(
        cls,
        ctx: ToolContext,
        helpers: dict[str, Any],
        signature: str,
        *,
        emit_diagnostics: bool = True,
    ) -> None:
        """Update the SANDBOX_DOWN canonical-signature streak counter.

 Runs every time
 :meth:`_apply_consecutive_error_cap` records a dispatch error. When
 ``signature`` ends in :data:`_SANDBOX_DOWN_CANONICAL_SUFFIX` the
 counter increments; otherwise it resets to zero (any non-SANDBOX_DOWN
 error breaks the streak so the injection signal does not fire across
 a mix of unrelated tool failures).

 Once the counter reaches the RC threshold, the helper bag gains a
 one-shot ``True`` flag at :data:`_SANDBOX_DOWN_INJECTION_PENDING_KEY`.
 The host executor loop consumes the flag by appending a
 synthetic user-role message and then clearing it via
 :meth:`_consume_sandbox_down_injection_signal`. Until the flag is
 consumed the counter does NOT re-arm the signal on every subsequent
 SANDBOX_DOWN — re-arming only happens after a successful tool call
 or a non-SANDBOX_DOWN error breaks the streak.

 Best-effort: no exceptions propagate out of this helper. The cap
 path keeps working even if the signal cannot be posted.
 """
        if not signature.endswith(_SANDBOX_DOWN_CANONICAL_SUFFIX):
            helpers.pop(_SANDBOX_DOWN_STREAK_STATE_KEY, None)
            return

        prior = helpers.get(_SANDBOX_DOWN_STREAK_STATE_KEY)
        prior_count = 0
        if isinstance(prior, dict):
            raw = prior.get("count", 0)
            if isinstance(raw, int) and raw >= 0:
                prior_count = raw

        new_count = prior_count + 1
        helpers[_SANDBOX_DOWN_STREAK_STATE_KEY] = {
            "signature": signature,
            "count": new_count,
        }

        threshold = cls._resolve_sandbox_down_threshold(ctx)
        # Only arm the signal at the exact threshold crossing so
        # the host loop sees the nudge once per streak. Subsequent
        # SANDBOX_DOWN errors keep incrementing the counter for telemetry
        # but do not re-arm — preventing a flood of synthetic messages.
        if new_count == threshold:
            if emit_diagnostics:
                _logger.warning(
                    "DIAG tool_dispatch.sandbox_down_threshold_reached "
                    "run=%s signature=%s count=%d threshold=%d",
                    ctx.run_id,
                    signature,
                    new_count,
                    threshold,
                )
            helpers[_SANDBOX_DOWN_INJECTION_PENDING_KEY] = True

    @classmethod
    def _consume_sandbox_down_injection_signal(cls, ctx: ToolContext) -> bool:
        """Pop the SANDBOX_DOWN injection-pending flag.

 The host executor loop calls this
 after every TurnEvent dispatch step. Returns ``True`` exactly once
 per streak (when the flag is armed by
 :meth:`_track_sandbox_down_streak`). Subsequent calls return
 ``False`` until the streak resets and re-arms.

 Best-effort: a missing helper bag returns ``False`` silently. The
 method never raises — failure to consume the flag must not break
 the executor loop.
 """
        helpers = cls._helpers_bag(ctx)
        if helpers is None:
            return False
        return bool(helpers.pop(_SANDBOX_DOWN_INJECTION_PENDING_KEY, False))

    @classmethod
    def _track_string_type_streak(
        cls,
        ctx: ToolContext,
        helpers: dict[str, Any],
        tool_name: str,
        kind: DispatchErrorKind,
        message: str,
        *,
        emit_diagnostics: bool = True,
    ) -> tuple[DispatchErrorKind, str] | None:
        """Update the Pydantic ``string_type`` streak counter and return
 a synthesised terminal-cap tuple when the cap is crossed.

 A model can loop many minutes retrying ``Write`` with a wrong
 argument shape (e.g. ``{content: [array]}``). The mainline coercion
 validators on ``WriteInput.content`` / ``AppendFileInput.content`` /
 ``BashInput.command`` handle common shapes silently; this terminal
 guard is the safety net for residual cases (uncoercible values,
 future fields that did not get a coercion validator).

 Returns ``None`` when the call is not a ``string_type`` error
 OR the streak is still under the cap. Returns the terminal
 ``(DispatchErrorKind.consecutive_error_cap, message)`` tuple
 when the cap is crossed so the caller can short-circuit the
 rest of :meth:`_apply_consecutive_error_cap`.

 Streak state is keyed on ``(tool_name, "string_type")`` so a
 switch to a different tool (e.g. Write -> Edit) restarts the
 counter at 1 while still surfacing the same failure mode.
 Non-``string_type`` errors break the streak (the model has
 moved to a different shape).
 """
        is_string_type = cls._is_string_type_error(kind, message)
        prior = helpers.get(_STRING_TYPE_STREAK_STATE_KEY)
        prior_tool: str | None = None
        prior_count = 0
        if isinstance(prior, dict):
            prior_tool_raw = prior.get("tool_name")
            prior_count_raw = prior.get("count", 0)
            if isinstance(prior_tool_raw, str):
                prior_tool = prior_tool_raw
            if isinstance(prior_count_raw, int) and prior_count_raw >= 0:
                prior_count = prior_count_raw

        if not is_string_type:
            # Any non-string_type error breaks the streak. We don't
            # clear it here on success — that path lives in
            # :meth:`_reset_consecutive_error_streak`.
            helpers.pop(_STRING_TYPE_STREAK_STATE_KEY, None)
            return None

        if prior_tool == tool_name:
            new_count = prior_count + 1
        else:
            new_count = 1

        helpers[_STRING_TYPE_STREAK_STATE_KEY] = {
            "tool_name": tool_name,
            "count": new_count,
        }

        cap = cls._resolve_string_type_terminal_cap(ctx)
        if new_count >= cap:
            if emit_diagnostics:
                _logger.warning(
                    "DIAG tool_dispatch.string_type_terminal_cap "
                    "run=%s tool=%s count=%d cap=%d",
                    ctx.run_id,
                    tool_name,
                    new_count,
                    cap,
                )
            wrapped = (
                f"TERMINAL: {tool_name!r} rejected {new_count} consecutive "
                "calls with the same schema error (wrong-shape or "
                "missing-required-field). The argument is the wrong shape or a "
                "required field is absent — retrying the same call will not "
                "succeed. STOP and change the call: include EVERY required "
                "field, pass `content`/`command` as a single JSON string (use "
                "embedded `\\n` for newlines), and if a file's `content` is too "
                "large to fit in one call, write it in chunks "
                "(`Write` header -> `AppendFile` chunks -> `FinalizeFile`) "
                "instead of repeating the same oversized call. "
                f"Original error: {message}"
            )
            return DispatchErrorKind.consecutive_error_cap, wrapped
        return None

    @classmethod
    def _reset_consecutive_error_streak(cls, ctx: ToolContext) -> None:
        """Clear the streak on a successful tool call.

 Called after a tool returns a non-error :class:`ToolResult` — the
 next error (even for the same tool + same signature) starts a
 fresh streak. Best-effort: a missing helper bag is a silent no-op.

 Also clears the SANDBOX_DOWN counter and
 the pending injection signal. A successful tool call (even of a
 different tool) is empirical evidence the sandbox recovered, so the
 next sandbox-down storm should restart at count=1 and the inline
 strategy can be re-armed at the threshold.

 also clears the ``string_type``
 terminal-cap counter so a subsequent malformed-args storm starts
 fresh.
 """
        helpers = cls._helpers_bag(ctx)
        if helpers is None:
            return
        helpers.pop(_CONSECUTIVE_ERROR_STATE_KEY, None)
        helpers.pop(_SANDBOX_DOWN_STREAK_STATE_KEY, None)
        helpers.pop(_SANDBOX_DOWN_INJECTION_PENDING_KEY, None)
        helpers.pop(_STRING_TYPE_STREAK_STATE_KEY, None)

    # ------------------------------------------------------------------
    # DAG tool-precondition mechanism
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_preconditions_enabled(cls, ctx: ToolContext) -> bool:
        """Read ``tool_preconditions_enabled`` from the RC snapshot.

 Falls back to ``True`` when no RC is wired (legacy test fixtures /
 dispatch paths that pre-date the RC plumbing). Mirrors
 the defensive ``getattr`` pattern used for
 ``tool_dispatch_consecutive_error_cap`` etc.
 """
        helpers = cls._helpers_bag(ctx)
        if helpers is None:
            return True
        rc = helpers.get("rc")
        if rc is None:
            return True
        return bool(getattr(rc, "tool_preconditions_enabled", True))

    @classmethod
    def _check_tool_preconditions(
        cls,
        *,
        tool: Any,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> str | None:
        """Run the precondition check for *tool* against the per-run satisfied set.

        Returns ``None`` if all preconditions are met (or none declared),
        otherwise a human-readable denial reason. Mirrors the v1
        ``dispatch.py:1179-1216`` flow ported from commit ``7dfa1ff``.

        Sources the tool's ``preconditions`` list from
        :attr:`ToolDefinition.preconditions` via ``tool.definition``. If
        the tool has no ``preconditions`` (None or empty list) the check
        returns ``None`` without consulting the helper bag.

        Disabled (returns ``None``) when
        ``RuntimeConstants.tool_preconditions_enabled`` is False.
        """
        if not cls._resolve_preconditions_enabled(ctx):
            return None
        # Read preconditions from the tool's published definition.
        # Some tool implementations may not expose ``preconditions`` on
        # their definition (legacy tools, mock tools in tests) — getattr
        # with default ``None`` covers both shapes.
        try:
            definition = tool.definition
        except Exception:
            return None
        preconditions = getattr(definition, "preconditions", None)
        if not preconditions:
            return None
        helpers = cls._helpers_bag(ctx)
        satisfied = load_satisfied_set(helpers)
        return check_preconditions(
            preconditions=list(preconditions),
            arguments=arguments,
            satisfied=satisfied,
        )

    @classmethod
    def _record_precondition_satisfaction(
        cls,
        *,
        tool: Any,
        tool_name: str,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> None:
        """Persist that *tool_name* was successfully called.

        Updates the per-run satisfied-precondition set on the helper bag
        so subsequent dispatches in the same run can see the satisfaction.
        Mirrors the v1 ``dispatch.py:2579-2607`` record path.

        Honours :attr:`ToolDefinition.path_fields` when present so tools
        with non-standard path argument names (``copy_path`` /
        ``move_path`` in v1) record the correct ``tool_name:path`` entry.

        No-op when ``tool_preconditions_enabled`` is False or the helper
        bag is missing — legacy test fixtures still dispatch successfully
        without the satisfaction set.
        """
        if not cls._resolve_preconditions_enabled(ctx):
            return
        helpers = cls._helpers_bag(ctx)
        if helpers is None:
            return
        path_fields: list[str] | None = None
        try:
            definition = tool.definition
        except Exception:
            definition = None
        if definition is not None:
            raw_fields = getattr(definition, "path_fields", None)
            if isinstance(raw_fields, list):
                path_fields = [field for field in raw_fields if isinstance(field, str)]
        satisfied = load_satisfied_set(helpers)
        record_satisfaction(
            tool_name=tool_name,
            arguments=arguments,
            satisfied=satisfied,
            path_fields=path_fields,
        )
        store_satisfied_set(helpers, satisfied)

    # ------------------------------------------------------------------
    # Public dispatch entry
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        *,
        tool_call: ToolCall,
        ctx: ToolContext,
        visibility_policy: ToolVisibilityPolicy,
        timeout_seconds: int,
        subagent_whitelist: Iterable[str] | None = None,
        preapproved_tool_call_id: str | None = None,
        admit_evidence: Callable[[tuple[EvidenceRecord, ...], EvidenceProducerBinding], None]
        | None = None,
    ) -> AsyncIterator[TurnEvent | DispatchOutcome]:
        """Drive one dispatch lifecycle.

        Yields :class:`TurnEvent` envelopes in Anthropic-style ordering
        (``hook_fired`` for PreToolUse … optional ``sandbox_starting``
        … ``hook_fired`` for PostToolUse … ``tool_result``). On approval
        path the terminal event is ``tool_call_pending`` (no
        tool_result). The final item yielded is always a
        :class:`DispatchOutcome` — callers iterate yields and switch on
        type.

        Caller responsibility: the LLM-stream ``tool_use_start`` /
        ``tool_use_input_delta`` / ``tool_use_stop`` are emitted by
        :func:`protocore.runtime.query._stream_one_assistant_message`
        BEFORE this dispatcher is invoked. We pick up after the LLM
        finishes the tool_call block.

        The dispatcher never RAISES on tool errors — all failure modes
        surface as :class:`DispatchOutcome` with ``success=False`` and a
        populated :attr:`DispatchOutcome.error_kind`.
        """
        metadata = dict(ctx.metadata)
        metadata.setdefault("tool_call_id", tool_call.id)
        # tools-initiative A2: expose the live per-run visibility policy to
        # policy-aware tools (ToolSearch) so discovery honours the SAME
        # visible/blocked contract the permission gate enforces below. A live
        # model instance, mirroring the ``protocore.helpers`` bag precedent.
        metadata.setdefault(TOOL_VISIBILITY_POLICY_METADATA_KEY, visibility_policy)
        ctx = ctx.model_copy(update={"metadata": metadata})

        # ── Step 1: registry lookup ────────────────────────────────
        tool = self._registry.get(tool_call.name)
        if tool is None:
            # Detect `<*_contract>` XML blocks the
            # model misemits as tool names and return a nudge instead of the
            # raw "unknown tool" error, prompting the model to inline the XML
            # block in its assistant text.
            if _CONTRACT_HALLUCINATION.fullmatch(tool_call.name):
                _helpers = ctx.metadata.get("protocore.helpers")
                _rc = _helpers.get("rc") if isinstance(_helpers, dict) else None
                msg = _contract_hallucination_hint(
                    tool_call.name,
                    finalize_terminal=bool(
                        getattr(_rc, "agent_finalize_tool_as_terminal", False)
                    ),
                )
            else:
                msg = f"unknown tool: {tool_call.name!r}"
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.unknown_tool, msg
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                metadata=_dispatch_replay_metadata(DispatchErrorKind.unknown_tool, msg),
            )
            return

        # The binding is runtime execution metadata: it is taken from the
        # registered tool after lookup, frozen in the context, and never read
        # from a model-visible result payload.
        ctx = ctx.model_copy(
            update={"evidence_producer_binding": tool.evidence_producer}
        )

        # ── Step 2: schema validation (input dict shape) ───────────
        # Core ABC accepts dict; tool-specific Pydantic input_model
        # validation lives in the host adapter. We
        # only enforce the byte-cap invariant here.
        #
        # The depth check comes FIRST because the serialisation below cannot be
        # relied on to report the failure: ``json.dumps`` recurses per level and
        # signals exhaustion with ``RecursionError``, which is neither
        # ``TypeError`` nor ``ValueError``, so it escapes the handler and takes
        # the run down from inside the dispatcher. Refusing the call as a
        # validation error keeps the failure where the model can see it and
        # re-issue the call with a sane payload.
        if _nesting_exceeds(tool_call.arguments, MAX_DATA_NESTING_DEPTH):
            msg = (
                "tool arguments nest deeper than "
                f"{MAX_DATA_NESTING_DEPTH} levels — re-issue the call with a "
                "flatter payload"
            )
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.validation, msg
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                metadata=_dispatch_replay_metadata(DispatchErrorKind.validation, msg),
            )
            return
        try:
            arguments_json = json.dumps(tool_call.arguments, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            msg = f"tool arguments not JSON-serialisable: {exc}"
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.validation, msg
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                metadata=_dispatch_replay_metadata(DispatchErrorKind.validation, msg),
            )
            return
        del arguments_json  # validation only

        # ── Step 3-4: permission gate ──────────────────────────────
        # Approved resume still runs the full gate. The only narrowed
        # exception is a repeated PreToolUse ``require_approval`` decision for
        # the exact call id that was already approved outside core.
        decision = await self._gate.check(
            tool=tool,
            arguments=tool_call.arguments,
            ctx=ctx,
            visibility_policy=visibility_policy,
            subagent_whitelist=subagent_whitelist,
            hook_manager=self._hooks,
            skip_pre_tool_approval=preapproved_tool_call_id == tool_call.id,
        )

        # Emit a HOOK_FIRED(pre_tool_use) event ONLY when the gate
        # actually invoked the PreToolUse hook (decision.stage == hook).
        # Earlier-stage decisions (whitelist / safety_policy / rate_limit)
        # MUST NOT advertise a hook fire — the hook stage never ran.
        # Telemetry only — never changes loop behaviour.
        if self._hooks is not None and decision.stage is PermissionStage.hook:
            yield TurnEvent(
                type=EventType.HOOK_FIRED,
                run_id=ctx.run_id,
                payload={
                    "hook_event": HookEvent.pre_tool_use.value,
                    "outcome": decision.outcome.value,
                    "stage": decision.stage.value,
                    "tool_call_id": tool_call.id,
                },
            )

        if decision.requires_approval:
            yield TurnEvent(
                type=EventType.TOOL_CALL_PENDING,
                run_id=ctx.run_id,
                payload={
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "tool_input": tool_call.arguments,
                    "requires_approval": True,
                    "approval_token": decision.approval_token,
                    "reason": decision.reason,
                },
            )
            # NO tool_use_stop yet — the call is paused; pending state
            # owns the lifecycle until user resumes.
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content="",
                is_error=False,
                approval_required=True,
                approval_token=decision.approval_token,
            )
            return

        if decision.denied:
            denial_reason = decision.reason or "blocked by policy"
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.permission, denial_reason
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                metadata=_dispatch_replay_metadata(
                    DispatchErrorKind.permission,
                    denial_reason,
                    {"stage": decision.stage.value},
                ),
            )
            return

        # Apply hook-driven mutation.
        final_args = decision.modified_input if decision.modified_input is not None else tool_call.arguments

        # ── DAG tool-precondition check ──────
        # When the tool's ``ToolDefinition.preconditions`` are unsatisfied
        # by the per-run satisfied-precondition set, return a
        # ``[PRECONDITION NOT MET: ...]`` error envelope BEFORE invoking
        # the tool. The check is gated by
        # ``RuntimeConstants.tool_preconditions_enabled`` so operators
        # can disable enforcement without redeploying the tool registry.
        precondition_reason = self._check_tool_preconditions(
            tool=tool,
            arguments=final_args,
            ctx=ctx,
        )
        if precondition_reason is not None:
            msg = f"[PRECONDITION NOT MET: {precondition_reason}]"
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.permission, msg
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                metadata=_dispatch_replay_metadata(
                    DispatchErrorKind.permission,
                    msg,
                    {"precondition_denial": precondition_reason},
                ),
            )
            return

        # NOTE: ``sandbox_starting`` is intentionally NOT emitted here.
        # The host's sandbox manager owns the event entirely — it emits only on
        # cold start (with reason=session_first_call|respawn_after_idle,
        # pod_id, namespace, sandbox_profile, run_id). The core
        # dispatcher cannot distinguish hot from cold and would emit
        # spurious events on every hot-pod dispatch.

        # ── Step 5: timeout-wrapped execute ────────────────────────
        # A tool that owns its own long-lived async unit (the ``Agent`` tool,
        # whose ``SubagentRunner`` runs an inner ``asyncio.timeout`` + stale
        # watchdog) declares a per-tool budget via ``should_defer`` +
        # ``default_timeout_ms``; honour it so the flat ``rc.tool_timeout_seconds``
        # cap does not cancel the unit before its own classified timeout fires.
        effective_timeout = _resolve_tool_timeout_seconds(
            tool, flat_timeout_seconds=timeout_seconds
        )
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        # #6 cancel propagation — when the executor put a per-run cancel
        # ``asyncio.Event`` on the helper bag, RACE the tool task against it so a
        # leader parked inside the synchronous ``Agent`` tool (whole subagent)
        # unblocks within the cancel-poll latency instead of ~2 min. Absent ⇒
        # the plain ``await asyncio.wait_for`` path (byte-identical, no-op for
        # older callers / tests). ``ToolDispatchCancelled`` (a CancelledError
        # subclass) propagates PAST the typed except-arms below (none catch
        # BaseException) up to the engine's cooperative cancel unwind + the
        # executor ``_drive_run`` cancelled-arm.
        cancel_event = _run_cancel_event(ctx)
        # Pre-dispatch guard: the gate / hook / precondition checks above ran
        # ``await``s, so a cancel may have landed since the last check. Test it
        # HERE — as close to the invoke point as possible, before the tool
        # coroutine is ever constructed — so a cancelled run never starts a new
        # tool/subagent (the raced helper repeats this check, but this covers
        # the path symmetrically and keeps the contract independent of the
        # helper's internals).
        if cancel_event is not None and cancel_event.is_set():
            _logger.warning(
                "DIAG tool_dispatch.cancelled_pre_dispatch tool=%s run=%s — "
                "cancel set before invoke; not starting the tool",
                tool_call.name,
                ctx.run_id,
            )
            raise ToolDispatchCancelled(
                f"tool dispatch cancelled for run {ctx.run_id!r}"
            )
        try:
            if cancel_event is None:
                tool_result = await asyncio.wait_for(
                    tool.invoke(ctx, final_args),
                    timeout=effective_timeout,
                )
            else:
                tool_result = await _invoke_tool_raced_with_cancel(
                    tool=tool,
                    ctx=ctx,
                    final_args=final_args,
                    effective_timeout=effective_timeout,
                    cancel_event=cancel_event,
                )
        except TimeoutError:
            duration_ms = int((loop.time() - started_at) * 1000)
            timeout_label = (
                int(effective_timeout)
                if effective_timeout.is_integer()
                else effective_timeout
            )
            msg = f"tool {tool_call.name!r} timed out after {timeout_label}s"
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.timeout, msg
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                duration_ms=duration_ms,
                metadata=_dispatch_replay_metadata(DispatchErrorKind.timeout, msg),
            )
            return
        except ToolPolicyDenied as exc:
            duration_ms = int((loop.time() - started_at) * 1000)
            reason = str(exc) or "tool policy denied"
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.permission, reason
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                duration_ms=duration_ms,
                metadata=_dispatch_replay_metadata(DispatchErrorKind.permission, reason),
            )
            return
        except AskUserPauseRequested as pause:
            # AskUser invoke raises this
            # typed signal to pause the loop. We yield a
            # ``TOOL_CALL_PENDING`` envelope (the same shape used for
            # approval gating) tagged with ``kind="ask_user"`` so
            # the host dispatch shim can fan out to the ask_user
            # resume path instead of the approval store. NO tool_result
            # is emitted — the answer arrives on resume and
            # the host side builds the final ToolResult then.
            duration_ms = int((loop.time() - started_at) * 1000)
            payload_dict = pause.payload.model_dump()
            yield TurnEvent(
                type=EventType.TOOL_CALL_PENDING,
                run_id=ctx.run_id,
                payload={
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "tool_input": tool_call.arguments,
                    "requires_approval": False,
                    "ask_user": True,
                    "kind": "ask_user",
                    "ask_user_payload": payload_dict,
                },
            )
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content="",
                is_error=False,
                ask_user_required=True,
                ask_user_payload=payload_dict,
                duration_ms=duration_ms,
            )
            return
        except Exception as exc:
            duration_ms = int((loop.time() - started_at) * 1000)
            msg = f"tool {tool_call.name!r} execution failed: {exc}"
            _logger.warning(
                "tool dispatch raised for tool=%s call_id=%s",
                tool_call.name,
                tool_call.id,
                exc_info=True,
            )
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.execution, msg
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            # Surface a machine-readable give-up signal. A tool may attach a
            # ``structured_error`` mapping to the exception it raises (e.g. a
            # retry-budget exhaustion carries
            # ``{"finalization_recommended": True, ...}``). Forward it verbatim
            # on the DispatchOutcome metadata so the loop / model can act on the
            # finalize hint instead of treating it as an opaque execution error.
            # Generic: any exception with a dict ``structured_error`` qualifies;
            # absent (the common case) leaves the metadata bit-identical.
            structured_error = getattr(exc, "structured_error", None)
            structured_error_extra: dict[str, Any] | None = (
                {DISPATCH_STRUCTURED_ERROR_METADATA_KEY: structured_error}
                if isinstance(structured_error, dict)
                else None
            )
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                duration_ms=duration_ms,
                metadata=_dispatch_replay_metadata(
                    DispatchErrorKind.execution, msg, structured_error_extra
                ),
            )
            return

        duration_ms = int((loop.time() - started_at) * 1000)
        if not isinstance(tool_result, ToolResult):
            msg = f"tool {tool_call.name!r} returned non-ToolResult shape: {type(tool_result).__name__}"
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.execution, msg
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                duration_ms=duration_ms,
                metadata=_dispatch_replay_metadata(DispatchErrorKind.execution, msg),
            )
            return

        success = not tool_result.is_error
        content = tool_result.content
        result_metadata = dict(tool_result.metadata)
        evidence_records = tool_result.evidence_records
        evidence_producer: EvidenceProducerBinding | None = None
        if success and evidence_records:
            evidence_producer = ctx.evidence_producer_binding
            if evidence_producer is None:
                success = False
                content = "tool evidence rejected: registered tool has no evidence producer binding"
                evidence_records = ()
            else:
                origin = ctx.evidence_origin
                if origin is None:
                    success = False
                    content = "tool evidence rejected: dispatch context has no evidence origin"
                    evidence_records = ()
                    evidence_producer = None
                else:
                    evidence_records = tuple(
                        record.model_copy(
                            update={
                                "origin": origin,
                                "producer_id": evidence_producer.producer_id,
                                "producer_revision": evidence_producer.producer_revision,
                            }
                        )
                        for record in evidence_records
                    )
                    if not ctx.evidence_admission_deferred:
                        if admit_evidence is None:
                            success = False
                            content = "tool evidence rejected: dispatcher has no private ledger admission channel"
                            evidence_records = ()
                            evidence_producer = None
                        else:
                            try:
                                admit_evidence(evidence_records, evidence_producer)
                            except ValueError as exc:
                                _logger.warning(
                                    "tool evidence rejected run=%s call_id=%s error=%s",
                                    ctx.run_id,
                                    tool_call.id,
                                    type(exc).__name__,
                                )
                                success = False
                                content = f"tool evidence rejected: {exc}"
                                evidence_records = ()
                                evidence_producer = None
        # A tool that returns ``ToolResult(is_error=True)`` still counts as
        # a tool error for the per-run counter (parity with the
        # dispatcher-detected error paths in :meth:`_emit_failure`). The
        # PostToolUse hook below may rewrite
        # content but cannot flip ``success`` back to ``True``, so this is
        # the canonical point to record the increment.
        if not success:
            # A tool may stamp its soft ``is_error=True`` result as NOT a
            # genuine failure for counting/capping purposes.
            # ``Bash`` does this for an ordinary nonzero process exit (a
            # ``grep -q`` no-match, a ``test``/``[`` false, a ``diff``
            # difference — exit status used as DATA, not a tool failure). Such
            # results must NOT increment ``tool_errors_count`` (which would
            # downgrade an otherwise-successful run to ``partial`` via
            # the host terminal classifier) and must NOT feed the generic
            # consecutive-error cap. The flags default to True (absent ⇒
            # count + cap) so every historical soft/hard error path is
            # unchanged. A genuine hard failure (e.g. a Bash timeout) leaves
            # both flags True and remains fully count/cap eligible.
            count_as_error = _metadata_flag(
                result_metadata,
                TOOL_RESULT_COUNT_AS_ERROR_METADATA_KEY,
                default=True,
            )
            cap_eligible = _metadata_flag(
                result_metadata,
                TOOL_RESULT_CONSECUTIVE_CAP_ELIGIBLE_METADATA_KEY,
                default=True,
            )
            if count_as_error:
                await self._record_tool_error(ctx)
            if cap_eligible:
                # Soft is_error path participates in
                # the consecutive-error cap. The streak is keyed on
                # ``(tool_name, signature(execution, content))`` which matches
                # how dispatcher-detected execution errors are keyed below.
                soft_kind, soft_msg = self._apply_consecutive_error_cap(
                    ctx, tool_call.name, DispatchErrorKind.execution, content
                )
                content = soft_msg
                soft_error_kind = soft_kind
            else:
                # Not cap-eligible: leave the streak untouched and surface the
                # original error kind/content. An ordinary nonzero exit is not
                # a "repeat the same failing call" signal, so it must neither
                # advance nor reset the streak (a real failure interleaved with
                # benign predicate exits should still accumulate).
                soft_error_kind = DispatchErrorKind.execution
        else:
            soft_error_kind = None
            # A gathered call is replayed in transcript order.  Its helper
            # mutations must wait for that replay, where deferred evidence is
            # admitted first; otherwise a completion-order success could
            # satisfy a dependency before a later rejection turns it into an
            # error.  Serial dispatch commits the normal state immediately.
            if not ctx.evidence_admission_deferred:
                # Successful tool result breaks the consecutive-error streak so a
                # subsequent failure restarts at count=1. Without this reset,
                # ``ToolA(err)…ToolA(ok)…ToolA(err)`` would carry the count
                # across the success, contrary to the "consecutive" semantics.
                self._reset_consecutive_error_streak(ctx)

                # ── DAG record_satisfaction ──────
                # On successful dispatch, persist that this (tool_name, path)
                # tuple has been satisfied so future tools whose
                # ``preconditions`` reference it can proceed.
                self._record_precondition_satisfaction(
                    tool=tool,
                    tool_name=tool_call.name,
                    arguments=final_args,
                    ctx=ctx,
                )

        # ── Step 7: PostToolUse hook (may mutate output) ───────────
        post_tool_output_modified = False
        if self._hooks is not None:
            try:
                post_hook = await self._hooks.invoke(
                    HookEvent.post_tool_use,
                    {
                        "run_id": ctx.run_id,
                        "tenant_id": ctx.tenant_id,
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.name,
                        "tool_output": content,
                        "success": success,
                        "duration_ms": duration_ms,
                    },
                    ctx.tenant_id,
                )
            except Exception:
                _logger.warning(
                    "PostToolUse hook raised for tool=%s; isolating",
                    tool_call.name,
                    exc_info=True,
                )
                post_hook = None
            if post_hook is not None:
                yield TurnEvent(
                    type=EventType.HOOK_FIRED,
                    run_id=ctx.run_id,
                    payload={
                        "hook_event": HookEvent.post_tool_use.value,
                        "outcome": ("modify" if post_hook.action == HookActionKind.MODIFY else "success"),
                        "tool_call_id": tool_call.id,
                    },
                )
                if post_hook.action == HookActionKind.MODIFY:
                    modified = post_hook.modifications.get("tool_output")
                    if isinstance(modified, str):
                        content = modified
                        post_tool_output_modified = True

        outcome_metadata: dict[str, Any] | None = result_metadata or None
        if not success:
            replay_extra = (
                {DISPATCH_POST_TOOL_OUTPUT_MODIFIED_METADATA_KEY: True}
                if post_tool_output_modified
                else None
            )
            outcome_metadata = {
                **result_metadata,
                **_dispatch_replay_metadata(
                    DispatchErrorKind.execution,
                    tool_result.content,
                    replay_extra,
                ),
            }

        # ── Step 6: ToolResult envelope event ──────────────────────
        yield TurnEvent(
            type=EventType.TOOL_RESULT,
            run_id=ctx.run_id,
            payload={
                "tool_call_id": tool_call.id,
                "success": success,
                "duration_ms": duration_ms,
                "content_blocks": [{"type": "text", "text": content}],
                **({"metadata": outcome_metadata} if outcome_metadata else {}),
            },
        )

        yield DispatchOutcome(
            tool_call=tool_call,
            success=success,
            content=content,
            is_error=not success,
            error_kind=None if success else soft_error_kind,
            duration_ms=duration_ms,
            metadata=outcome_metadata,
            evidence_records=evidence_records,
            evidence_producer=evidence_producer,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _emit_failure(
        self,
        tool_call: ToolCall,
        ctx: ToolContext,
        kind: DispatchErrorKind,
        message: str,
    ) -> AsyncIterator[TurnEvent]:
        """Emit the terminal ``tool_result(error=...)`` envelope.

 The LLM-stream ``tool_use_start`` / ``tool_use_stop`` for the
 originating call were emitted upstream — they are not
 synthesised here. Every failure path of :meth:`dispatch` calls
 this helper exactly once for symmetry.

 Increments the run's ``tool_errors_count`` (atomic, idempotent) BEFORE
 yielding the error envelope so the terminal classifier can downgrade runs
 with tool errors to ``partial`` status.
 """
        await self._record_tool_error(ctx)
        yield TurnEvent(
            type=EventType.TOOL_RESULT,
            run_id=ctx.run_id,
            payload={
                "tool_call_id": tool_call.id,
                "success": False,
                "error": {
                    "kind": kind.value,
                    "message": message,
                },
                "content_blocks": [{"type": "text", "text": message}],
            },
        )


def consume_sandbox_down_injection_signal(
    helpers: dict[str, Any] | None,
) -> bool:
    """Public consumer for the SANDBOX_DOWN injection-pending flag.

 The host executor loop calls this
 once per dispatched ``TurnEvent``. Returns ``True`` exactly once per
 streak (when :meth:`ToolDispatcher._track_sandbox_down_streak` armed
 the flag). Subsequent calls in the same streak return ``False`` until
 a successful tool call or a non-SANDBOX_DOWN error resets the counter.

 Parameters
 ----------
 helpers:
 The per-run helper bag (``ctx.metadata["protocore.helpers"]``).
 ``None`` is treated as "no signal" — the function never raises.

 Returns
 -------
 bool
 ``True`` iff the dispatcher armed the flag and this is the first
 consumer to pop it. ``False`` otherwise.
 """
    if helpers is None:
        return False
    return bool(helpers.pop(_SANDBOX_DOWN_INJECTION_PENDING_KEY, False))


__all__ = [
    "DispatchErrorKind",
    "DispatchOutcome",
    "ToolDispatcher",
    "ToolPermissionDecision",
    "ToolPermissionOutcome",
    "consume_sandbox_down_injection_signal",
]
