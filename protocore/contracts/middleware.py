"""The lifecycle contract — one extension seam for the whole run.

Everything a host wants to do *around* the loop goes through this module: it
declares the coordinates a run passes through, the five kinds of registration
that can sit on a coordinate, and the shapes a registration reads and returns.
There is no second surface. A registration is made once, is owned by a named
party, is limited to a scope, and is removed by the disposer handed back at
registration time.

Kinds
-----
``observe``
    Sees the context, changes nothing. A raising observer is logged and the
    run continues exactly as if it had returned.
``decide``
    Returns a verdict. This is the security kind: the first non-``allow``
    verdict wins and the rest of the chain is skipped. **A decide handler that
    raises, times out, or returns something unreadable denies** — a seam that
    cannot answer must not be read as consent.
``transform``
    Returns a replacement payload, which is applied: later registrations at the
    same coordinate see the replacement, and the caller receives it in
    :attr:`LifecycleOutcome.payload`. A transform that promises a rewrite and
    does not get one is worse than no transform at all, so a raising or timing
    out transform denies rather than letting the untransformed payload through.
``around``
    Wraps the work at the coordinate: it receives the context and a ``next``
    callable, may skip ``next`` entirely (short-circuit), and may post-process
    what ``next`` produced. Raising or timing out denies.
``notify``
    Fire-and-forget after the fact. Like ``observe``, a failure is logged and
    never changes the outcome; unlike ``observe``, it is not expected to be
    consulted before anything happens.

Ordering is ``priority`` ascending, then registration order — stable and
declared, so a chain of five is debuggable.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from protocore.contracts.types import HookEvent


class RegistrationKind(StrEnum):
    """What a registration is allowed to do at its coordinate."""

    observe = "observe"
    decide = "decide"
    transform = "transform"
    around = "around"
    notify = "notify"


class LifecycleVerdict(StrEnum):
    """The answer a coordinate carries back to the loop."""

    allow = "allow"
    deny = "deny"
    require_approval = "require_approval"
    fail_run = "fail_run"


#: Verdicts that stop the chain and are never produced by an ``observe`` or
#: ``notify`` registration.
BLOCKING_VERDICTS: frozenset[LifecycleVerdict] = frozenset(
    {
        LifecycleVerdict.deny,
        LifecycleVerdict.require_approval,
        LifecycleVerdict.fail_run,
    }
)

#: Kinds whose failure is a security failure and therefore denies.
FAIL_CLOSED_KINDS: frozenset[RegistrationKind] = frozenset(
    {
        RegistrationKind.decide,
        RegistrationKind.transform,
        RegistrationKind.around,
    }
)

#: Kinds whose failure is isolated: logged, never able to change the outcome.
FAIL_ISOLATED_KINDS: frozenset[RegistrationKind] = frozenset(
    {RegistrationKind.observe, RegistrationKind.notify}
)


class LifecycleScope(BaseModel):
    """Where a registration applies. An unset field means "any"."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None

    def matches(self, context: LifecycleContext) -> bool:
        """Does this scope admit ``context``?"""
        if self.tenant_id is not None and self.tenant_id != context.tenant_id:
            return False
        if self.session_id is not None and self.session_id != context.session_id:
            return False
        return not (self.run_id is not None and self.run_id != context.run_id)


#: The scope a registration gets when its owner names none.
ANY_SCOPE: LifecycleScope = LifecycleScope()


class LifecycleContext(BaseModel):
    """What a registration is handed: the coordinate plus the run's identity.

    ``payload`` is the coordinate's data. It is replaced, never mutated: a
    ``transform`` returns a new mapping and the dispatcher builds the next
    context from it.
    """

    model_config = ConfigDict(frozen=True)

    point: HookEvent
    run_id: str = ""
    session_id: str = ""
    tenant_id: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)

    def with_payload(self, payload: Mapping[str, Any]) -> LifecycleContext:
        """Same coordinate and identity, different payload."""
        return self.model_copy(update={"payload": dict(payload)})


class LifecycleDecision(BaseModel):
    """What a ``decide``, ``transform``, or ``around`` registration returns.

    A handler may also return ``None`` (no opinion), a bare
    :class:`LifecycleVerdict`, or — for ``transform`` — a plain mapping, which
    is read as a replacement payload.
    """

    model_config = ConfigDict(frozen=True)

    verdict: LifecycleVerdict = LifecycleVerdict.allow
    reason: str = ""
    approval_token: str | None = None
    replacement: dict[str, Any] | None = None
    """Payload that supersedes the one the handler was given."""


class LifecycleFailure(BaseModel):
    """A registration that could not answer, and what was done about it."""

    model_config = ConfigDict(frozen=True)

    owner: str
    point: HookEvent
    kind: RegistrationKind
    error: str
    isolated: bool
    """``True`` when the failure was logged and the outcome left untouched."""


class LifecycleOutcome(BaseModel):
    """The aggregate the loop reads after a coordinate has been dispatched."""

    model_config = ConfigDict(frozen=True)

    verdict: LifecycleVerdict = LifecycleVerdict.allow
    reason: str = ""
    approval_token: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    """The payload as it stands after every applied ``transform``."""

    decided_by: str = ""
    """Owner of the registration that produced a non-``allow`` verdict."""

    failures: tuple[LifecycleFailure, ...] = ()

    @property
    def allowed(self) -> bool:
        return self.verdict is LifecycleVerdict.allow


LifecycleHandler = Callable[[LifecycleContext], Any]
"""Sync or async; return value is read according to the registration's kind."""

AroundNext = Callable[[LifecycleContext], Awaitable[Any]]
AroundHandler = Callable[[LifecycleContext, AroundNext], Any]


class LifecycleDisposer:
    """Removes one registration. Calling it twice is not an error.

    The disposer is the only way a registration goes away, and it is
    idempotent by construction: the second call finds nothing to remove and
    reports ``False``, so a caller that disposes in both a normal path and a
    cleanup path leaves no orphan and raises nothing.
    """

    __slots__ = ("_disposed", "_remove", "owner", "point")

    def __init__(
        self,
        remove: Callable[[], bool],
        *,
        owner: str,
        point: HookEvent,
    ) -> None:
        self._remove = remove
        self._disposed = False
        self.owner = owner
        self.point = point

    @property
    def disposed(self) -> bool:
        return self._disposed

    def __call__(self) -> bool:
        """Remove the registration. Returns ``True`` only on the first call."""
        if self._disposed:
            return False
        self._disposed = True
        return self._remove()

    def __repr__(self) -> str:
        state = "disposed" if self._disposed else "live"
        return f"<LifecycleDisposer {self.owner}@{self.point.value} {state}>"


@dataclass(frozen=True, slots=True)
class LifecycleRegistration:
    """One handler placed on one coordinate by one owner."""

    point: HookEvent
    kind: RegistrationKind
    owner: str
    handler: LifecycleHandler | AroundHandler
    scope: LifecycleScope = ANY_SCOPE
    priority: int = 100
    timeout_s: float | None = None
    seq: int = field(default=0, compare=False)
    """Registration order; breaks priority ties so the chain is deterministic."""

    @property
    def fail_closed(self) -> bool:
        return self.kind in FAIL_CLOSED_KINDS


@runtime_checkable
class ILifecycleRegistry(Protocol):
    """The single seam. Implemented in core by ``protocore.hooks.HookManager``."""

    def register(
        self,
        point: HookEvent,
        kind: RegistrationKind,
        handler: LifecycleHandler | AroundHandler,
        *,
        owner: str,
        scope: LifecycleScope | None = ...,
        priority: int = ...,
        timeout_s: float | None = ...,
    ) -> LifecycleDisposer:
        """Place ``handler`` on ``point``; the return value removes it again."""
        ...

    def registrations(
        self, point: HookEvent | None = ...
    ) -> tuple[LifecycleRegistration, ...]:
        """Every live registration, in dispatch order."""
        ...

    async def dispatch(self, context: LifecycleContext) -> LifecycleOutcome:
        """Run the chain at ``context.point`` and aggregate the answer."""
        ...

    async def around(
        self, context: LifecycleContext, next_: AroundNext
    ) -> LifecycleOutcome:
        """Run the ``around`` chain at ``context.point`` wrapping ``next_``."""
        ...


__all__ = [
    "ANY_SCOPE",
    "BLOCKING_VERDICTS",
    "FAIL_CLOSED_KINDS",
    "FAIL_ISOLATED_KINDS",
    "AroundHandler",
    "AroundNext",
    "ILifecycleRegistry",
    "LifecycleContext",
    "LifecycleDecision",
    "LifecycleDisposer",
    "LifecycleFailure",
    "LifecycleHandler",
    "LifecycleOutcome",
    "LifecycleRegistration",
    "LifecycleScope",
    "LifecycleVerdict",
    "RegistrationKind",
]
