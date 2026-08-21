"""EventBus + EventName StrEnum — in-process typed pub/sub.

Distinct from :class:`~protocore.contracts.events.IEventStream`:

 - :class:`EventBus`: in-process pub/sub for sibling-handler signalling
 (handlers run in the same pod). Used by HookManager, ContextManager,
 and other in-process subsystems.
 - :class:`IEventStream`: cross-pod durable stream (Redis Streams) for
 SSE reconnect/replay.

~70 event names; v1's ~120 names trimmed to drop ~30 dead names tied to
multi-agent / knowledge-capture. Strict review on new event names.
"""
from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from protocore.logging_utils import get_logger

_logger = get_logger(__name__)


class EventName(StrEnum):
    """Typed event taxonomy. ~70 events.

    Categories:
        - Lifecycle: session_start, session_end, run_*, turn_*
        - LLM: llm_call_start, llm_call_end, llm_stream_*
        - Streaming (Anthropic-aligned): message_*, content_block_*, tool_use_*
        - Tools: tool_call_*, tool_dispatch_*, tool_permission_*
        - Compaction: compaction_*, snapshot_*
        - Hooks: hook_fired, hook_denied, hook_failed
        - Sandbox: sandbox_starting, sandbox_ready, sandbox_failed
        - Subagent: subagent_spawn, subagent_complete, subagent_failed
        - GC: gc_started, gc_completed
        - Audit: audit_emit
    """

    # ----- Lifecycle -----
    session_start = "session_start"
    session_end = "session_end"
    run_created = "run_created"
    run_started = "run_started"
    run_completed = "run_completed"
    run_failed = "run_failed"
    run_cancelled = "run_cancelled"
    turn_start = "turn_start"
    turn_end = "turn_end"

    # ----- LLM -----
    llm_call_start = "llm_call_start"
    llm_call_end = "llm_call_end"
    llm_call_failed = "llm_call_failed"
    llm_context_exceeded = "llm_context_exceeded"

    # ----- Streaming (Anthropic-aligned) -----
    message_start = "message_start"
    message_stop = "message_stop"
    content_block_start = "content_block_start"
    content_block_delta = "content_block_delta"
    content_block_stop = "content_block_stop"
    tool_use_start = "tool_use_start"
    tool_use_input_delta = "tool_use_input_delta"
    tool_use_stop = "tool_use_stop"

    # ----- Tools -----
    tool_call_start = "tool_call_start"
    tool_call_end = "tool_call_end"
    tool_call_failed = "tool_call_failed"
    tool_call_pending = "tool_call_pending"  # approval flow
    tool_permission_decided = "tool_permission_decided"
    tool_dispatch_start = "tool_dispatch_start"
    tool_dispatch_end = "tool_dispatch_end"
    tool_result_emitted = "tool_result_emitted"

    # ----- Tool surface -----
    tool_surface_resolved = "tool_surface_resolved"
    tool_retrieval_invoked = "tool_retrieval_invoked"
    tool_progressive_discovery = "tool_progressive_discovery"

    # ----- Compaction -----
    compaction_routine_start = "compaction_routine_start"
    compaction_routine_end = "compaction_routine_end"
    compaction_auto_start = "compaction_auto_start"
    compaction_auto_end = "compaction_auto_end"
    compaction_emergency_start = "compaction_emergency_start"
    compaction_emergency_end = "compaction_emergency_end"
    compaction_snapshot_persisted = "compaction_snapshot_persisted"
    compaction_recall_artifact = "compaction_recall_artifact"

    # ----- Hooks -----
    hook_fired = "hook_fired"
    hook_denied = "hook_denied"
    hook_failed = "hook_failed"
    hook_modified = "hook_modified"

    # ----- Sandbox -----
    sandbox_starting = "sandbox_starting"
    sandbox_ready = "sandbox_ready"
    sandbox_failed = "sandbox_failed"
    sandbox_teardown = "sandbox_teardown"

    # ----- Subagent -----
    subagent_spawn = "subagent_spawn"
    subagent_complete = "subagent_complete"
    subagent_failed = "subagent_failed"

    # ----- Skills -----
    skill_loaded = "skill_loaded"
    skill_load_failed = "skill_load_failed"
    skill_index_emitted = "skill_index_emitted"

    # ----- Workspace -----
    file_changed = "file_changed"
    workspace_policy_decision = "workspace_policy_decision"
    workspace_approval_requested = "workspace_approval_requested"
    workspace_approval_resolved = "workspace_approval_resolved"

    # ----- Collapse / safety -----
    collapse_detected = "collapse_detected"
    safety_policy_denied = "safety_policy_denied"

    # ----- Persistence -----
    blob_put = "blob_put"
    run_persisted = "run_persisted"
    session_persisted = "session_persisted"

    # ----- Token / budget -----
    # This is an event name, not a credential.
    token_budget_exceeded = "token_budget_exceeded"  # nosec B105
    iteration_budget_exceeded = "iteration_budget_exceeded"

    # ----- GC -----
    gc_started = "gc_started"
    gc_completed = "gc_completed"
    gc_failed = "gc_failed"

    # ----- Audit -----
    audit_emit = "audit_emit"

    # ----- Error -----
    error = "error"
    warning = "warning"


# Handler signature: synchronous or async, takes a payload dict.
type _HandlerSync = Callable[[dict[str, Any]], None]
type _HandlerAsync = Callable[[dict[str, Any]], Awaitable[None]]
type Handler = _HandlerSync | _HandlerAsync


class EventBus:
    """In-process typed pub/sub. Async-safe within one pod.

 Cross-pod fanout is NOT this class — that's :class:`IEventStream`.
 """

    def __init__(self) -> None:
        self._subs: dict[EventName, list[Handler]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def subscribe(self, name: EventName, handler: Handler) -> None:
        """Register a handler. Synchronous; safe pre-loop-start."""
        self._subs[name].append(handler)

    def unsubscribe(self, name: EventName, handler: Handler) -> bool:
        """Remove a handler. Return ``True`` if removed."""
        try:
            self._subs[name].remove(handler)
            return True
        except ValueError:
            return False

    async def publish(self, name: EventName, payload: dict[str, Any]) -> None:
        """Fan-out to all registered handlers; handler errors isolated."""
        async with self._lock:
            handlers = list(self._subs.get(name, []))
        for handler in handlers:
            try:
                result = handler(payload)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                _logger.warning(
                    "event handler raised for %s (handler=%s); isolating",
                    name.value,
                    getattr(handler, "__qualname__", repr(handler)),
                    exc_info=True,
                )

    def subscriber_count(self, name: EventName) -> int:
        """Return number of subscribers for an event name."""
        return len(self._subs.get(name, []))


__all__ = ["EventBus", "EventName", "Handler"]
