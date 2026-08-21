"""Observability contracts — optional hooks the core exposes to its host.

Prompt-caching metrics belong to whatever the host already runs for
observability, so the producer lives outside the core and is wired in
through an injectable :class:`CacheObserverProtocol` carried on
:class:`protocore.runtime.query_engine.QueryEngineConfig`.

The shape is intentionally minimal — one method, four kwargs. Adding
fields to the recorder requires a contract bump; removing fields would
break host implementations silently. Keep this stable.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CacheObserverProtocol(Protocol):
    """Optional sink for per-LLM-call prompt-caching observations.

    Implementations live in the host (Prometheus histograms, for
    instance). The runtime calls this once per
    ``ProviderDeltaKind.usage`` envelope, after the engine's
    :class:`~protocore.runtime.usage.TokenUsage` has been updated.

    The protocol is :func:`typing.runtime_checkable` so tests can verify
    a concrete implementation's shape without depending on its module.
    """

    def record_run_cache_hit_rate(
        self,
        *,
        tenant_id: str,
        cache_read_tokens: int,
        prompt_tokens: int,
        cache_breakpoint_count: int,
    ) -> None:
        """Record a single LLM-call cache observation.

        Arguments
        ---------
        tenant_id:
            The tenant this LLM call belongs to. Used as the primary
            label dimension on the underlying Prometheus histograms.
        cache_read_tokens:
            Tokens served from the provider's prompt cache for this call
            (Anthropic ``cache_read_input_tokens`` / OpenAI+vLLM
            ``cached_tokens``).
        prompt_tokens:
            Total prompt tokens for this call (``input_tokens`` from the
            provider usage envelope). Combined with ``cache_read_tokens``
            this yields the hit rate the implementation may bucket.
        cache_breakpoint_count:
            Number of :class:`~protocore.contracts.llm.CacheBreakpoint`
            hints attached to the originating
            :class:`~protocore.contracts.llm.LLMRequest`. Surfaces the
            placement-strategy effect on cache success.

        Implementations MUST be cheap (lock-free counter / histogram
        observation). The runtime calls this on the hot streaming path
        and does NOT spawn a thread / task to defer the call.
        """
        ...


__all__ = ["CacheObserverProtocol"]
