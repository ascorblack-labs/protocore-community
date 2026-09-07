"""A background-pool double for loop tests.

The loop reads its session's background commands through
:class:`~protocore.contracts.background.IBackgroundTaskPool` — a structural
contract, deliberately without an implementation in the core: whatever spawns
and watches processes belongs to the host. These tests are about what the loop
does with the pool's answers, so the pool they inject keeps records and nothing
else.
"""
from __future__ import annotations

import builtins
from dataclasses import dataclass, field

from protocore.contracts.background import BACKGROUND_TERMINAL_STATUSES, AgentRef


@dataclass(slots=True)
class FakeBackgroundTask:
    """One record, in the fields the loop reads plus what the double needs."""

    id: str
    session_id: str
    #: Which run has to end this work, when that is not the session. Empty for
    #: work the session owns, which is every record there was before a run
    #: could be delegated.
    owner_scope: str = ""
    status: str = "running"
    kind: str = "command"
    label: str = ""
    exit_code: int | None = None
    error: str = ""
    duration_seconds: float | None = None
    agent: AgentRef | None = None
    notify_on_finish: bool = False
    #: Whether anything is still able to report on this record. A non-terminal
    #: record with nothing behind it is what a process death leaves.
    handle_bound: bool = True


@dataclass
class FakeBackgroundPool:
    """Records for a session, with the attachment declaration the contract asks for."""

    tasks: builtins.list[FakeBackgroundTask] = field(default_factory=list)
    attached_sessions: set[str] = field(default_factory=set)
    refreshed: builtins.list[str] = field(default_factory=list)
    drained: builtins.list[str] = field(default_factory=list)
    stopped_sessions: builtins.list[tuple[str, float]] = field(default_factory=list)
    _minted: int = 0
    _woken: set[str] = field(default_factory=set)

    # -- what the test drives ------------------------------------------------

    def start(
        self,
        session_id: str,
        *,
        owner_scope: str = "",
        notify_on_finish: bool = False,
        kind: str = "command",
        label: str = "",
        agent: AgentRef | None = None,
    ) -> FakeBackgroundTask:
        self._minted += 1
        task = FakeBackgroundTask(
            id=f"bg-{self._minted}",
            session_id=session_id,
            owner_scope=owner_scope,
            notify_on_finish=notify_on_finish,
            kind=kind,
            label=label,
            agent=agent,
        )
        self.tasks.append(task)
        self.attached_sessions.add(session_id)
        return task

    def finish(
        self,
        task_id: str,
        status: str = "succeeded",
        *,
        exit_code: int | None = 0,
        error: str = "",
        duration_seconds: float | None = None,
    ) -> None:
        task = self.get(task_id)
        assert task is not None
        task.status = status
        task.exit_code = exit_code
        task.error = error
        task.duration_seconds = duration_seconds

    def lose_handle(self, task_id: str) -> None:
        task = self.get(task_id)
        assert task is not None
        task.handle_bound = False

    # -- the contract --------------------------------------------------------

    def mark_session_attached(self, session_id: str) -> None:
        self.attached_sessions.add(session_id)

    async def ensure_session_attached(self, session_id: str) -> bool:
        if session_id not in self.attached_sessions:
            return False
        return all(
            item.status in BACKGROUND_TERMINAL_STATUSES or item.handle_bound
            for item in self.list(session_id)
        )

    def list(self, session_id: str) -> builtins.list[FakeBackgroundTask]:
        return [item for item in self.tasks if item.session_id == session_id]

    def get(self, task_id: str) -> FakeBackgroundTask | None:
        for item in self.tasks:
            if item.id == task_id:
                return item
        return None

    async def refresh(self, task_id: str) -> object:
        self.refreshed.append(task_id)
        return self.get(task_id)

    def owned_by(self, scope: str) -> builtins.list[FakeBackgroundTask]:
        """Records ``scope`` has to end — its own, or its session's."""
        return [
            item
            for item in self.tasks
            if scope in (item.owner_scope, item.session_id)
        ]

    async def stop_session(
        self, scope: str, grace_seconds: float = 0.0
    ) -> builtins.list[FakeBackgroundTask]:
        self.stopped_sessions.append((scope, grace_seconds))
        stopped: builtins.list[FakeBackgroundTask] = []
        for task in self.owned_by(scope):
            if task.status in BACKGROUND_TERMINAL_STATUSES:
                continue
            task.status = "stopped"
            stopped.append(task)
        return stopped

    def drain_wakes(self, session_id: str) -> builtins.list[str]:
        self.drained.append(session_id)
        ids = [
            item.id
            for item in self.list(session_id)
            if item.notify_on_finish
            and item.status in BACKGROUND_TERMINAL_STATUSES
            and item.id not in self._woken
        ]
        self._woken.update(ids)
        return ids
