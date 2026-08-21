"""HookManager — pluggy dispatcher with error isolation.

In-process only; the host
ships the HTTP / LLM-as-hook executors . The manager:

 - Holds the pluggy.PluginManager instance
 - Dispatches in registration order
 - Isolates handler failures (logged WARNING; never breaks siblings)
 - Honors the 'deny' short-circuit semantics
"""
from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import pluggy

from protocore.contracts.hooks import HookActionKind, HookResult
from protocore.contracts.types import HookEvent
from protocore.hooks.specs import PROJECT_NAME, AgentHookSpecs
from protocore.logging_utils import get_logger

_logger = get_logger(__name__)


class HookManager:
    """Async-safe pluggy dispatcher for the 8 hook events.

    Register a hook implementation by passing any object with
    ``@hookimpl``-decorated coroutine methods to :meth:`register`.

    All ``invoke_*`` methods aggregate per-handler results into a single
    :class:`HookResult`; ``deny`` from any hook short-circuits.
    """

    def __init__(self) -> None:
        self._pm = pluggy.PluginManager(PROJECT_NAME)
        self._pm.add_hookspecs(AgentHookSpecs)
        self._registered: list[str] = []

    def register(self, plugin: object, *, name: str | None = None) -> str:
        """Register a hook implementation. Returns the registered name."""
        registered = self._pm.register(plugin, name=name)
        if isinstance(registered, str):
            self._registered.append(registered)
            return registered
        # pluggy returns the plugin name; fall back to repr
        rname = name or repr(plugin)
        self._registered.append(rname)
        return rname

    def unregister(self, name_or_plugin: str | object) -> bool:
        """Unregister a hook implementation.

        Returns ``True`` if a plugin was removed, ``False`` if nothing matched.
        Pluggy returns ``None`` for an unknown name and raises
        :class:`AssertionError` for an unknown plugin instance; both are
        normalized to a ``False`` return here.
        """
        try:
            removed = (
                self._pm.unregister(name=name_or_plugin)
                if isinstance(name_or_plugin, str)
                else self._pm.unregister(plugin=name_or_plugin)
            )
        except (AssertionError, KeyError, ValueError):
            return False
        if removed is None:
            return False
        # Drop from cached name list (best-effort; pluggy is authoritative).
        if isinstance(name_or_plugin, str) and name_or_plugin in self._registered:
            self._registered.remove(name_or_plugin)
        return True

    def registered(self) -> Sequence[str]:
        """Return all registered hook plugin names."""
        return tuple(self._registered)

    async def invoke(
        self,
        event: HookEvent,
        payload: dict[str, Any],
    ) -> HookResult:
        """Invoke all handlers for an event in registration order.

 Aggregates per-handler results. First ``deny`` short-circuits.
 Handler errors are isolated (logged WARNING) but SURFACED: a
 crashed handler appends an ``{"outcome": "error", ...}`` record to
 ``raw_results`` so the failure is observable rather than silently
 swallowed . The aggregate allow/deny verdict is left
 unchanged — failure-mode (allow vs deny on crash) is a host
 policy decision.
 """
        caller = getattr(self._pm.hook, event.value, None)
        if caller is None:
            return HookResult(action=HookActionKind.ALLOW, reason="no spec for event")

        try:
            raw_results = caller(**payload)
        except Exception as exc:
            # : pluggy invokes SYNCHRONOUS hookimpls inside
            # ``caller(...)``, so a sync handler that raises lands here. Surface
            # the crash (error record + WARNING) instead of silently swallowing
            # it; the aggregate stays ALLOW (failure-mode is a host
            # policy decision). Async handlers raise in the per-result loop
            # below and are surfaced there.
            _logger.warning(
                "hook dispatch raised while invoking handlers (event=%s)",
                event.value,
                exc_info=True,
            )
            return HookResult(
                action=HookActionKind.ALLOW,
                reason="dispatch failed",
                raw_results=[
                    {
                        "outcome": "error",
                        "event": event.value,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ],
            )

        # pluggy returns a list of results from sync impls; for async impls
        # we resolve each awaitable explicitly + isolate failures.
        materialized: list[dict[str, Any]] = []
        action = HookActionKind.ALLOW
        reason = ""
        modifications: dict[str, Any] = {}

        for result in raw_results:
            try:
                resolved = await result if inspect.isawaitable(result) else result
            except Exception as exc:
                # : a crashed handler must be SURFACED, not silently
                # swallowed. We isolate the failure (siblings keep running and
                # the aggregate allow/deny is unchanged — that is a
                # failure_mode policy decision owned by the host
                # adapter), but we append an ``outcome='error'`` record so the
                # crash is observable to any consumer of ``raw_results`` and
                # mirror the reference's ``non_blocking_error`` surfacing.
                _logger.warning(
                    "hook handler raised for event=%s; isolating",
                    event.value,
                    exc_info=True,
                )
                materialized.append(
                    {
                        "outcome": "error",
                        "event": event.value,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if resolved is None:
                continue
            if not isinstance(resolved, dict):
                continue
            materialized.append(resolved)
            handler_action = resolved.get("action", HookActionKind.ALLOW)
            if handler_action == HookActionKind.DENY:
                return HookResult(
                    action=HookActionKind.DENY,
                    reason=str(resolved.get("reason", "")),
                    modifications=dict(resolved.get("modifications", {})),
                    raw_results=materialized,
                )
            if handler_action == HookActionKind.MODIFY:
                action = HookActionKind.MODIFY
                modifications.update(resolved.get("modifications", {}))
                if not reason:
                    reason = str(resolved.get("reason", ""))

        return HookResult(
            action=action,
            reason=reason,
            modifications=modifications,
            raw_results=materialized,
        )


__all__ = ["HookManager"]
