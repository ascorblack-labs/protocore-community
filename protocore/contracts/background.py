"""The session work pool, as the loop sees it.

One pool, two kinds of work. A shell command started in the background and a
child run started by delegation are the same thing from the loop's side: a unit
of work with an address, a status, a way to wait for it and a way to stop it.
They used to be two mechanisms — the command was a pool record with an id, and
the child run was a function call that blocked its caller for as long as it took
and had no address at all. Nothing could ask a child run how far along it was,
nothing could stop one, and a parent waiting on one held its turn and its slot in
the tree budget for the whole descendant run.

So the pool takes both, told apart by :attr:`TaskRecord.kind`, and a subagent
handle is simply a :class:`WorkHandle` over a record of kind ``agent``.

A background task outlives the run that started it. The command is spawned in
one run, the run ends, and whichever run is live when the command finishes is
the one that has to be told — that delivery is the whole point of the pool, and
it is what ``notify_on_finish`` promises the agent.

Which means the pool is not run state and cannot be. It is a collaborator the
host injects, and on a cold start — a fresh process picking up a session whose
tasks were spawned by a process that is gone — the host has to put the session's
still-running commands back in the new pool's hands before the loop asks it
anything. :meth:`IBackgroundTaskPool.ensure_session_attached` is where the loop
asks whether that happened, and a pool that answers ``False`` gets an explicit
event on the run rather than the empty wake list that reads exactly like a
session with nothing running.

Stateless, structural contract — a Protocol, so the host's own pool satisfies it
by shape and does not import an implementation to inherit from.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar, runtime_checkable

#: Statuses from which a background task will never report anything again. A
#: record in one of these is finished business: nothing has to watch it, and the
#: loop stops asking about it.
BACKGROUND_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"succeeded", "failed", "timed_out", "stopped", "orphaned"}
)


#: The two kinds of work one pool holds. A shell command started in the
#: background, and a child run started by delegation.
TaskKind = Literal["command", "agent"]

#: The result a :class:`WorkHandle` produces. Left open because the pool holds
#: both kinds: a command finishes with its own record, a child run finishes with
#: whatever the dispatcher builds out of it.
WorkResultT = TypeVar("WorkResultT", covariant=True)


@dataclass(frozen=True, slots=True)
class AgentRef:
    """Which agent a record of kind ``agent`` is running, and as which run.

    ``child_run_id`` is empty until the child run has an id of its own, which is
    the window between the pool minting the task id and the dispatcher starting
    the run. Both ids exist on purpose: the task id addresses the WORK, the run
    id addresses the RUN, and a caller that conflated them could not ask about a
    child that had been launched and had not yet started.
    """

    name: str
    child_run_id: str = ""


@runtime_checkable
class BackgroundTaskView(Protocol):
    """What the loop reads off a task record.

    Read-only on purpose: the loop reports what the pool holds and never writes
    back through this view, and declaring the fields as properties lets an
    implementation narrow ``status`` to its own status enumeration.

    The fields past ``status`` are the ones a finished task has to be able to
    NAME. A wake line built from an id and a status alone tells the agent that
    something ended and nothing about how, which is not enough to decide what to
    do next — see :func:`describe_finished_task`.
    """

    @property
    def id(self) -> str:
        """Pool-unique task id."""

    @property
    def status(self) -> str:
        """Current lifecycle status, e.g. ``running`` / ``succeeded``."""

    @property
    def kind(self) -> str:
        """``command`` or ``agent`` — which sort of work this record holds."""

    @property
    def label(self) -> str:
        """Short human-facing name for the work, or empty when it has none."""

    @property
    def exit_code(self) -> int | None:
        """Process exit status, or ``None`` for work that has no exit code."""

    @property
    def error(self) -> str:
        """Why the work failed, or empty when it did not."""

    @property
    def duration_seconds(self) -> float | None:
        """How long the work ran, or ``None`` while it is still running."""

    @property
    def agent(self) -> AgentRef | None:
        """The agent behind a record of kind ``agent``; ``None`` otherwise."""


@dataclass(frozen=True, slots=True)
class WorkSpec:
    """Everything the pool needs to mint a record BEFORE the work starts.

    The order matters and is the point: a record with an id exists, and is
    durable, before anything is spawned. A pool that minted the id from whatever
    the spawn returned had a window in which the work was running and unnamed,
    and work that cannot be named cannot be stopped, waited on, or reported.
    """

    session_id: str
    kind: TaskKind = "command"
    label: str = ""
    tool_call_id: str = ""
    notify_on_finish: bool = False
    expected_seconds: float | None = None
    timeout_seconds: float | None = None
    agent: AgentRef | None = None
    owner_scope: str = ""
    """Which run's obligation this work is, when that is not the session's.

    A run started by delegation shares the session — so it shares the
    workspace, and its records are listed and woken with the session's — but
    the work it starts is its own to end. Empty means the session owns it,
    which is the ordinary case and every case there was before a run could be
    delegated.
    """


@dataclass(slots=True)
class TaskRecord:
    """One unit of pool work, of either kind.

    Mutable, because a record is the thing that changes: it is minted pending,
    becomes running, and settles terminal. Satisfies :class:`BackgroundTaskView`
    by shape, so the loop reads a record through the same view it reads any
    pool's own record through.
    """

    id: str
    session_id: str
    kind: TaskKind = "command"
    owner_scope: str = ""
    status: str = "pending"
    label: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    error: str = ""
    timeout_seconds: float | None = None
    expected_seconds: float | None = None
    notify_on_finish: bool = False
    tool_call_id: str = ""
    agent: AgentRef | None = None

    @property
    def terminal(self) -> bool:
        """Whether this record will ever report anything again."""
        return self.status in BACKGROUND_TERMINAL_STATUSES

    @property
    def duration_seconds(self) -> float | None:
        """Wall time between start and finish, or ``None`` while it runs."""
        if self.started_at is None or self.finished_at is None:
            return None
        return max(0.0, self.finished_at - self.started_at)

    @classmethod
    def minted(cls, task_id: str, spec: WorkSpec) -> TaskRecord:
        """The record a pool hands back from :meth:`IWorkPool.launch`."""
        return cls(
            id=task_id,
            session_id=spec.session_id,
            kind=spec.kind,
            owner_scope=spec.owner_scope,
            label=spec.label,
            timeout_seconds=spec.timeout_seconds,
            expected_seconds=spec.expected_seconds,
            notify_on_finish=spec.notify_on_finish,
            tool_call_id=spec.tool_call_id,
            agent=spec.agent,
        )


@runtime_checkable
class WorkHandle(Protocol[WorkResultT]):
    """The three things anyone holding launched work can do with it.

    A subagent handle is this and nothing more: ``SubagentHandle`` is
    ``WorkHandle`` over a subagent result, declared where the dispatch contract
    lives. There is no second handle type, because there is no second kind of
    ownership — waiting on a child run and waiting on a background command are
    the same question asked of the same pool.
    """

    def identity(self) -> TaskRecord:
        """The record this handle is over, id and all, without waiting."""

    async def wait(self) -> WorkResultT:
        """Block until the work settles and return what it produced."""

    async def stop(self, grace_seconds: float = 0.0) -> TaskRecord:
        """Ask the work to end, allowing ``grace_seconds`` before it is forced."""


#: What a pool calls to actually begin work it has already minted a record for.
#: Takes the id the record already has, so whatever it spawns is born knowing
#: the name the rest of the system will use for it.
StartWork = Callable[[str], Awaitable[WorkHandle[object]]]

#: Called with a record that has just reached a terminal status.
TerminalObserver = Callable[[TaskRecord], None]


def describe_finished_task(task: BackgroundTaskView) -> str:
    """One line naming what a finished task did, for the agent that reads it.

    ``id status`` was the whole line, and it is the wrong line: an agent woken
    by it learns that something ended and has to spend a turn asking what
    happened. So the line carries what the decision actually turns on — which
    work it was, how it ended, and how long it took — and stays one line,
    because the wake batches every task that finished at once.

    Every field past the status is optional in the rendering, not in the
    contract: a task that failed has no exit code worth printing next to its
    error, and one that is still being described mid-flight has no duration.
    """
    parts = [f"{task.id} [{task.status}]"]
    if task.label:
        parts.append(task.label)
    if task.kind == "agent" and task.agent is not None and task.agent.name:
        parts.append(f"agent {task.agent.name}")
    if task.error:
        parts.append(f"error: {task.error}")
    elif task.exit_code is not None:
        parts.append(f"exit {task.exit_code}")
    duration = task.duration_seconds
    if duration is not None:
        parts.append(f"ran {duration:.1f}s")
    return ", ".join(parts)


@runtime_checkable
class IBackgroundTaskPool(Protocol):
    """Session-scoped background commands the run can be woken by."""

    def mark_session_attached(self, session_id: str) -> None:
        """Declare that this session's live commands are back in the pool's hands.

        The host calls this once it has done whatever re-attachment it can for
        the session — re-adopting each still-running command it can find, or
        nothing at all if it keeps no durable record of them. It is the
        AFFIRMATIVE half of :meth:`ensure_session_attached`: a pool that has
        never been told anything about a session must not claim to speak for
        it, because "I hold no records" and "nothing is running" are different
        answers and only one of them is safe to act on.
        """

    async def ensure_session_attached(self, session_id: str) -> bool:
        """Report whether this session's still-running records are bound to handles.

        Called before every wake check, so it MUST be idempotent and cheap on
        an already-attached session.

        Returns ``True`` when the pool has been told this session is its own
        AND every non-terminal record it holds for the session is bound to
        something that can report on it. ``False`` when it cannot vouch for the
        session — which is what a resumed run finds when the host has not
        re-attached it. Returning ``True`` on an unattached session is the one
        thing an implementation must not do: the loop would then read an empty
        wake list as "nothing finished" and the agent would wait forever on a
        command that already did. Holding no records is NOT evidence of
        attachment, which is why the declaration is required and an empty pool
        is not trivially attached.

        A pool that keeps its records only in memory cannot detect a session
        whose commands were spawned by a process that is gone — there is
        nothing left to notice missing. The loop carries the other half of that
        check itself, comparing the ids its own durable state recorded against
        what the pool holds.
        """

    def list(self, session_id: str) -> Sequence[BackgroundTaskView]:
        """Every record the pool holds for ``session_id``."""

    def get(self, task_id: str) -> BackgroundTaskView | None:
        """One record by id, or ``None`` when the pool does not hold it."""

    async def refresh(self, task_id: str) -> object:
        """Pull the live status of one task into its record."""

    def drain_wakes(self, session_id: str) -> Sequence[str]:
        """Ids of finished tasks that asked to wake the run, consumed once."""


@runtime_checkable
class IWorkPool(IBackgroundTaskPool, Protocol):
    """The pool that holds both kinds of work, with the handles to drive them.

    Everything :class:`IBackgroundTaskPool` promises, plus the four operations
    that make a unit of work addressable: mint it before starting it, wait on
    it, stop it, and stop everything one session owns. A child run reaches the
    loop through exactly these, which is what lets a delegation call be started
    in the background and collected later instead of pinning its caller.
    """

    async def launch(self, spec: WorkSpec, start: StartWork) -> TaskRecord:
        """Mint a record, then start the work, and answer with the record.

        The id is minted FIRST and handed to ``start``, so a caller that never
        gets to see the return value — a crash between the two, a stop that
        arrives immediately — has still left behind a named, durable record of
        work that may be running. A pool that named the work afterwards would
        leave an unnamed process instead.
        """

    def handle(self, task_id: str) -> WorkHandle[object] | None:
        """The live handle for a record, or ``None`` when nothing holds it."""

    async def stop(self, task_id: str, grace_seconds: float = 0.0) -> TaskRecord | None:
        """Stop one task, or answer ``None`` when the pool does not hold it."""

    async def stop_session(
        self, scope: str, grace_seconds: float = 0.0
    ) -> Sequence[TaskRecord]:
        """Stop every non-terminal task ``scope`` owns, and say which.

        What a run owes the work it started before it ends. A run that walks
        away from its background commands leaks them: the pool holds records no
        live run is watching, and nothing else in the tree knows to look.

        A record is owned by ``scope`` when its :attr:`TaskRecord.owner_scope`
        is ``scope``, or — for the ordinary work that belongs to nobody
        narrower — when its ``session_id`` is. Both, because the two callers ask
        different questions with the same verb: a delegated run ends and stops
        what it alone started, and a session ends and stops everything under
        it, including whatever its children left behind. Listing and waking stay
        keyed on the session, which is where a wake has to be delivered.
        """

    def subscribe(self, on_terminal: TerminalObserver) -> Callable[[], None]:
        """Call ``on_terminal`` with each record that settles; returns a canceller.

        Push, not poll. A finished child run has to reach whichever run is live
        at that moment, and a pool that could only be asked would deliver it
        whenever somebody next happened to ask.
        """


__all__ = [
    "BACKGROUND_TERMINAL_STATUSES",
    "AgentRef",
    "BackgroundTaskView",
    "IBackgroundTaskPool",
    "IWorkPool",
    "StartWork",
    "TaskKind",
    "TaskRecord",
    "TerminalObserver",
    "WorkHandle",
    "WorkSpec",
    "describe_finished_task",
]
