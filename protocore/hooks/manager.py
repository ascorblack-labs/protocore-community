"""In-process implementation of the lifecycle contract.

One registry, one dispatcher, one set of rules — the contract itself lives in
:mod:`protocore.contracts.middleware`; this module is what runs it.

Four properties the loop can rely on:

order
    ``priority`` ascending, then registration order. Two handlers never swap
    places between runs.
timeout
    Per registration, in seconds. A handler that overruns is treated exactly
    as one that raised.
exception policy
    ``decide`` / ``transform`` / ``around`` fail **closed**: a handler that
    raises, overruns, or answers with something unreadable produces a ``deny``
    carrying its owner's name. ``observe`` / ``notify`` fail **isolated**: the
    failure is logged and recorded on the outcome, and the verdict, the
    payload, and the siblings are untouched.
cancellation
    :class:`asyncio.CancelledError` is never converted into a verdict. It
    propagates, so a cancelled run stops instead of quietly denying itself.
"""
from __future__ import annotations

import asyncio
import inspect
import itertools
from collections.abc import Mapping
from typing import Any

from protocore.contracts.middleware import (
    ANY_SCOPE,
    AroundHandler,
    AroundNext,
    LifecycleContext,
    LifecycleDecision,
    LifecycleDisposer,
    LifecycleFailure,
    LifecycleHandler,
    LifecycleOutcome,
    LifecycleRegistration,
    LifecycleScope,
    LifecycleVerdict,
    RegistrationKind,
)
from protocore.contracts.types import HookEvent
from protocore.logging_utils import get_logger

_logger = get_logger(__name__)


class HookManager:
    """The core's lifecycle registry and dispatcher.

    Register with :meth:`register`, which hands back an idempotent disposer;
    dispatch a coordinate with :meth:`dispatch`, or wrap real work in the
    ``around`` chain with :meth:`around`.
    """

    def __init__(self) -> None:
        self._by_point: dict[HookEvent, list[LifecycleRegistration]] = {}
        self._seq = itertools.count()

    # -- registration ----------------------------------------------------

    def register(
        self,
        point: HookEvent,
        kind: RegistrationKind,
        handler: LifecycleHandler | AroundHandler,
        *,
        owner: str,
        scope: LifecycleScope | None = None,
        priority: int = 100,
        timeout_s: float | None = None,
    ) -> LifecycleDisposer:
        """Place ``handler`` on ``point`` and return the disposer that lifts it.

        ``owner`` is required and shows up in every verdict, log line, and
        failure record the registration causes: an anonymous seam cannot be
        held to account for a denial.
        """
        if not owner:
            raise ValueError("registration_requires_owner")
        registration = LifecycleRegistration(
            point=point,
            kind=kind,
            owner=owner,
            handler=handler,
            scope=scope if scope is not None else ANY_SCOPE,
            priority=priority,
            timeout_s=timeout_s,
            seq=next(self._seq),
        )
        bucket = self._by_point.setdefault(point, [])
        bucket.append(registration)
        bucket.sort(key=lambda item: (item.priority, item.seq))

        def _remove() -> bool:
            live = self._by_point.get(point)
            if not live:
                return False
            for index, candidate in enumerate(live):
                if candidate.seq == registration.seq:
                    del live[index]
                    if not live:
                        self._by_point.pop(point, None)
                    return True
            return False

        return LifecycleDisposer(_remove, owner=owner, point=point)

    def dispose_owner(self, owner: str) -> int:
        """Drop every registration made by ``owner``. Returns how many went."""
        removed = 0
        for point in list(self._by_point):
            live = self._by_point[point]
            kept = [item for item in live if item.owner != owner]
            removed += len(live) - len(kept)
            if kept:
                self._by_point[point] = kept
            else:
                self._by_point.pop(point, None)
        return removed

    def registrations(
        self, point: HookEvent | None = None
    ) -> tuple[LifecycleRegistration, ...]:
        """Live registrations in dispatch order (all points, or just one)."""
        if point is not None:
            return tuple(self._by_point.get(point, ()))
        everything = [item for bucket in self._by_point.values() for item in bucket]
        everything.sort(key=lambda item: (item.point.value, item.priority, item.seq))
        return tuple(everything)

    def points(self) -> tuple[HookEvent, ...]:
        """Coordinates that currently carry at least one registration."""
        return tuple(sorted(self._by_point, key=lambda item: item.value))

    def owners(self) -> tuple[str, ...]:
        """Every owner holding a live registration, once each, sorted."""
        return tuple(sorted({item.owner for item in self.registrations()}))

    # -- dispatch --------------------------------------------------------

    async def dispatch(self, context: LifecycleContext) -> LifecycleOutcome:
        """Run the chain at ``context.point``.

        ``observe`` runs first (it may not change anything, so it sees the
        payload as the loop built it), then ``decide``, then ``transform``,
        then ``notify``. ``around`` registrations are not part of this chain —
        they need work to wrap, which only :meth:`around` supplies.
        """
        chain = [
            item
            for item in self._by_point.get(context.point, ())
            if item.kind is not RegistrationKind.around and item.scope.matches(context)
        ]
        if not chain:
            return LifecycleOutcome(payload=point_payload_unchanged(context))

        failures: list[LifecycleFailure] = []
        current = context

        for kind in (
            RegistrationKind.observe,
            RegistrationKind.decide,
            RegistrationKind.transform,
            RegistrationKind.notify,
        ):
            for registration in [item for item in chain if item.kind is kind]:
                try:
                    raw = await self._call(registration, current)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    failure = _failure(registration, exc)
                    failures.append(failure)
                    if failure.isolated:
                        _logger.warning(
                            "lifecycle %s handler failed at %s (owner=%s); isolated",
                            registration.kind.value,
                            registration.point.value,
                            registration.owner,
                            exc_info=True,
                        )
                        continue
                    _logger.warning(
                        "lifecycle %s handler failed at %s (owner=%s); denying",
                        registration.kind.value,
                        registration.point.value,
                        registration.owner,
                        exc_info=True,
                    )
                    return LifecycleOutcome(
                        verdict=LifecycleVerdict.deny,
                        reason=failure.error,
                        payload=dict(current.payload),
                        decided_by=registration.owner,
                        failures=tuple(failures),
                    )

                if registration.kind in (
                    RegistrationKind.observe,
                    RegistrationKind.notify,
                ):
                    continue

                decision = _read_decision(registration, raw)
                if decision is None:
                    failure = _unreadable(registration, raw)
                    failures.append(failure)
                    _logger.warning(
                        "lifecycle %s handler at %s (owner=%s) returned %r; denying",
                        registration.kind.value,
                        registration.point.value,
                        registration.owner,
                        type(raw).__name__,
                    )
                    return LifecycleOutcome(
                        verdict=LifecycleVerdict.deny,
                        reason=failure.error,
                        payload=dict(current.payload),
                        decided_by=registration.owner,
                        failures=tuple(failures),
                    )

                if decision.verdict is not LifecycleVerdict.allow:
                    return LifecycleOutcome(
                        verdict=decision.verdict,
                        reason=decision.reason,
                        approval_token=decision.approval_token,
                        payload=dict(
                            decision.replacement
                            if decision.replacement is not None
                            else current.payload
                        ),
                        decided_by=registration.owner,
                        failures=tuple(failures),
                    )

                if (
                    registration.kind is RegistrationKind.transform
                    and decision.replacement is not None
                ):
                    # The whole point of the kind: what the transform returned
                    # is what everything downstream — the next registration and
                    # the caller alike — actually works with.
                    current = current.with_payload(decision.replacement)

        return LifecycleOutcome(payload=dict(current.payload), failures=tuple(failures))

    async def around(
        self, context: LifecycleContext, next_: AroundNext
    ) -> LifecycleOutcome:
        """Wrap ``next_`` in the ``around`` chain registered at the coordinate.

        Handlers are entered in dispatch order, so the first-registered one is
        outermost. A handler that never awaits its ``next`` short-circuits: the
        work does not happen and its decision is the outcome.
        """
        chain = [
            item
            for item in self._by_point.get(context.point, ())
            if item.kind is RegistrationKind.around and item.scope.matches(context)
        ]
        if not chain:
            await next_(context)
            return LifecycleOutcome(payload=dict(context.payload))

        failures: list[LifecycleFailure] = []
        reached = {"inner": False}

        async def _invoke(index: int, ctx: LifecycleContext) -> LifecycleOutcome:
            if index == len(chain):
                reached["inner"] = True
                await next_(ctx)
                return LifecycleOutcome(payload=dict(ctx.payload))
            registration = chain[index]
            inner_outcome: dict[str, LifecycleOutcome] = {}

            async def _next(passed: LifecycleContext) -> Any:
                inner_outcome["value"] = await _invoke(index + 1, passed)
                return inner_outcome["value"]

            try:
                raw = await self._call_around(registration, ctx, _next)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                failure = _failure(registration, exc)
                failures.append(failure)
                _logger.warning(
                    "lifecycle around handler failed at %s (owner=%s); denying",
                    registration.point.value,
                    registration.owner,
                    exc_info=True,
                )
                return LifecycleOutcome(
                    verdict=LifecycleVerdict.deny,
                    reason=failure.error,
                    payload=dict(ctx.payload),
                    decided_by=registration.owner,
                )

            produced = inner_outcome.get("value")
            if raw is None:
                return produced if produced is not None else LifecycleOutcome(
                    verdict=LifecycleVerdict.deny,
                    reason=f"{registration.owner}: around handler skipped next without deciding",
                    payload=dict(ctx.payload),
                    decided_by=registration.owner,
                )
            decision = _read_decision(registration, raw)
            if decision is None:
                failure = _unreadable(registration, raw)
                failures.append(failure)
                return LifecycleOutcome(
                    verdict=LifecycleVerdict.deny,
                    reason=failure.error,
                    payload=dict(ctx.payload),
                    decided_by=registration.owner,
                )
            payload = (
                decision.replacement
                if decision.replacement is not None
                else (produced.payload if produced is not None else ctx.payload)
            )
            if decision.verdict is LifecycleVerdict.allow and produced is not None:
                return produced.model_copy(update={"payload": dict(payload)})
            return LifecycleOutcome(
                verdict=decision.verdict,
                reason=decision.reason,
                approval_token=decision.approval_token,
                payload=dict(payload),
                decided_by=(
                    registration.owner
                    if decision.verdict is not LifecycleVerdict.allow
                    else ""
                ),
            )

        outcome = await _invoke(0, context)
        return outcome.model_copy(
            update={"failures": tuple([*outcome.failures, *failures])}
        )

    # -- calling ---------------------------------------------------------

    async def _call(
        self, registration: LifecycleRegistration, context: LifecycleContext
    ) -> Any:
        handler = registration.handler
        result = handler(context)  # type: ignore[call-arg]
        if inspect.isawaitable(result):
            if registration.timeout_s is not None:
                return await asyncio.wait_for(result, registration.timeout_s)
            return await result
        return result

    async def _call_around(
        self,
        registration: LifecycleRegistration,
        context: LifecycleContext,
        next_: AroundNext,
    ) -> Any:
        handler = registration.handler
        result = handler(context, next_)  # type: ignore[call-arg]
        if inspect.isawaitable(result):
            if registration.timeout_s is not None:
                return await asyncio.wait_for(result, registration.timeout_s)
            return await result
        return result


def point_payload_unchanged(context: LifecycleContext) -> dict[str, Any]:
    """The empty-chain answer: the payload comes back exactly as it went in."""
    return dict(context.payload)


def _failure(
    registration: LifecycleRegistration, exc: BaseException
) -> LifecycleFailure:
    kind_of_error = (
        "timed out" if isinstance(exc, TimeoutError) else f"{type(exc).__name__}: {exc}"
    )
    return LifecycleFailure(
        owner=registration.owner,
        point=registration.point,
        kind=registration.kind,
        error=f"{registration.owner}: {kind_of_error}",
        isolated=not registration.fail_closed,
    )


def _unreadable(registration: LifecycleRegistration, raw: Any) -> LifecycleFailure:
    return LifecycleFailure(
        owner=registration.owner,
        point=registration.point,
        kind=registration.kind,
        error=(
            f"{registration.owner}: {registration.kind.value} handler returned "
            f"{type(raw).__name__}, which is not a decision"
        ),
        isolated=False,
    )


def _read_decision(
    registration: LifecycleRegistration, raw: Any
) -> LifecycleDecision | None:
    """Normalise a handler's return value, or ``None`` if it is unreadable."""
    if raw is None:
        return LifecycleDecision()
    if isinstance(raw, LifecycleDecision):
        return raw
    if isinstance(raw, LifecycleVerdict):
        return LifecycleDecision(verdict=raw)
    if isinstance(raw, str):
        try:
            return LifecycleDecision(verdict=LifecycleVerdict(raw))
        except ValueError:
            return None
    if registration.kind is RegistrationKind.transform and isinstance(raw, Mapping):
        return LifecycleDecision(replacement=dict(raw))
    return None


def refuse_lifecycle_when_disabled(enabled: bool) -> None:
    """Raise when the lifecycle seam is reached with its switch turned off.

    A caller that asks the registry to do work while the seam is disabled has
    a bug, not a no-op: silence there is how a host ends up believing its
    registrations run.
    """
    if not enabled:
        raise ValueError("lifecycle_hooks_disabled")


__all__ = ["HookManager", "refuse_lifecycle_when_disabled"]
