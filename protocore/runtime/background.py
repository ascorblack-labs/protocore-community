"""Session-scoped background tasks: statuses, timeouts, wake, adopt, reap.

The loop pod never holds a subprocess. A :class:`BackgroundPool` talks to a
:class:`ProcessRunner` (sandbox worker in production, a test double here).
"""
from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from uuid import uuid4

from protocore.contracts.runtime_constants import RuntimeConstants

BackgroundStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "timed_out",
    "stopped",
    "orphaned",
]
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"succeeded", "failed", "timed_out", "stopped", "orphaned"}
)
AdoptDecision = Literal[
    "continue",
    "adopt",
    "kill_cancel",
    "kill_pool_full",
    "kill_disabled",
]


@dataclass(slots=True)
class BackgroundTask:
    """One session-scoped background command."""

    id: str
    session_id: str
    tenant_id: str
    command: str
    status: BackgroundStatus
    notify_on_finish: bool = False
    expected_seconds: int | None = None
    timeout_seconds: int = 900
    output: str = ""
    worker_id: str = ""
    pid: int | None = None
    start_token: str = ""
    adopted: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "command": self.command,
            "status": self.status,
            "notify_on_finish": self.notify_on_finish,
            "expected_seconds": self.expected_seconds,
            "timeout_seconds": self.timeout_seconds,
            "output": self.output,
            "worker_id": self.worker_id,
            "pid": self.pid,
            "start_token": self.start_token,
            "adopted": self.adopted,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BackgroundTask:
        return cls(
            id=str(raw["id"]),
            session_id=str(raw.get("session_id") or ""),
            tenant_id=str(raw.get("tenant_id") or ""),
            command=str(raw.get("command") or ""),
            status=raw.get("status") or "queued",
            notify_on_finish=bool(raw.get("notify_on_finish")),
            expected_seconds=raw.get("expected_seconds"),
            timeout_seconds=int(raw.get("timeout_seconds") or 900),
            output=str(raw.get("output") or ""),
            worker_id=str(raw.get("worker_id") or ""),
            pid=raw.get("pid"),
            start_token=str(raw.get("start_token") or ""),
            adopted=bool(raw.get("adopted")),
            reason=str(raw.get("reason") or ""),
        )


def new_task_id() -> str:
    return f"bg_{uuid4().hex[:10]}"


def compute_hard_timeout_seconds(
    *,
    explicit: int | None,
    expected_seconds: int | None,
    rc: RuntimeConstants,
) -> int:
    """Resolve the hard timeout; never exceed the RC cap."""
    if explicit is not None and explicit > 0:
        chosen = explicit
    elif expected_seconds is not None and expected_seconds > 0:
        chosen = max(
            expected_seconds * rc.background_expected_timeout_multiplier,
            rc.background_expected_timeout_floor_seconds,
        )
    else:
        chosen = rc.background_default_timeout_seconds
    return min(int(chosen), rc.background_max_timeout_seconds)


def refuse_notify_when_disabled(*, enabled: bool, notify_on_finish: bool) -> None:
    if notify_on_finish and not enabled:
        raise ValueError("background_tasks_disabled")


def running_count(tasks: list[BackgroundTask]) -> int:
    return sum(1 for item in tasks if item.status in {"queued", "running"})


def pool_is_full(tasks: list[BackgroundTask], rc: RuntimeConstants) -> bool:
    return running_count(tasks) >= rc.background_max_concurrent_per_session


def decide_foreground(
    *,
    background_enabled: bool,
    adopt_enabled: bool,
    elapsed_seconds: float,
    timeout_seconds: float,
    cancelled: bool,
    pool_full: bool,
) -> AdoptDecision:
    """Decide whether a timed-out foreground command is adopted or killed."""
    if elapsed_seconds < timeout_seconds:
        return "continue"
    if cancelled:
        return "kill_cancel"
    if not background_enabled or not adopt_enabled:
        return "kill_disabled"
    if pool_full:
        return "kill_pool_full"
    return "adopt"


def adopt_notice(task_id: str, output_so_far: str, rc: RuntimeConstants) -> str:
    """Model-visible content: notice first, captured output last (truncates from the end)."""
    notice = (
        f"Command still running after {rc.bash_foreground_timeout_seconds}s. "
        f"It was NOT cancelled: it now runs as background task {task_id}. "
        "Do NOT run this command again with a larger timeout."
    )
    budget = rc.tool_result_content_max_chars
    room = max(0, budget - len(notice) - 2)
    tail = output_so_far[-room:] if room else ""
    if tail:
        return f"{notice}\n\n{tail}"
    return notice


def mark_orphaned_on_worker_death(
    tasks: list[BackgroundTask],
    dead_worker_id: str,
) -> list[BackgroundTask]:
    """In-flight tasks on a dead worker read as orphaned; meta is not rewritten in place."""
    out: list[BackgroundTask] = []
    for task in tasks:
        if task.worker_id == dead_worker_id and task.status in {"queued", "running"}:
            out.append(
                BackgroundTask(
                    **{
                        **task.to_dict(),
                        "status": "orphaned",
                        "reason": "worker_dead",
                    }
                )
            )
        else:
            out.append(task)
    return out


def reap_is_safe(
    *,
    recorded_pid: int | None,
    recorded_start_token: str,
    live_pid: int | None,
    live_start_token: str,
) -> bool:
    """Refuse to signal a reused pid that is no longer our process group."""
    if recorded_pid is None or live_pid is None:
        return False
    if recorded_pid != live_pid:
        return False
    if not recorded_start_token or recorded_start_token != live_start_token:
        return False
    return True


def collect_wake(
    finished: list[BackgroundTask],
    *,
    shutting_down: bool,
    wakes_used: int,
    rc: RuntimeConstants,
) -> list[str]:
    """Return task ids that should share one wake turn, or empty."""
    if shutting_down:
        return []
    if wakes_used >= rc.background_max_wakes_per_session:
        return []
    return [item.id for item in finished if item.notify_on_finish]


def clip_output(raw: str, rc: RuntimeConstants) -> str:
    limit = rc.background_output_buffer_bytes
    encoded = raw.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return raw
    return encoded[-limit:].decode("utf-8", errors="replace")


class ProcessHandle(Protocol):
    pid: int | None
    start_token: str
    worker_id: str

    async def kill_group(self) -> None: ...

    async def snapshot(self) -> tuple[BackgroundStatus, str]: ...


class ProcessRunner(Protocol):
    async def spawn(
        self,
        command: str,
        *,
        timeout_seconds: int,
        session_id: str,
        tenant_id: str,
    ) -> ProcessHandle: ...


class FakeHandle:
    """Test double for a sandbox process group."""

    def __init__(
        self,
        pid: int,
        start_token: str,
        worker_id: str = "worker-a",
    ) -> None:
        self.pid: int | None = pid
        self.start_token = start_token
        self.worker_id = worker_id
        self.status: BackgroundStatus = "running"
        self.output = ""
        self.killed = False

    async def kill_group(self) -> None:
        self.killed = True
        self.status = "stopped"

    async def snapshot(self) -> tuple[BackgroundStatus, str]:
        return self.status, self.output

    def finish(self, status: BackgroundStatus, output: str = "") -> None:
        self.status = status
        if output:
            self.output = output


class LocalProcessGroupHandle:
    """A real OS process group started with ``start_new_session=True``."""

    def __init__(
        self,
        proc: Any,
        *,
        start_token: str,
        worker_id: str,
        timeout_seconds: int,
    ) -> None:
        self.proc = proc
        self.pid: int | None = proc.pid
        self.start_token = start_token
        self.worker_id = worker_id
        self.timeout_seconds = timeout_seconds
        self._chunks: list[str] = []
        self._status: BackgroundStatus = "running"
        self._watch: Any = None
        # Held for the task's lifetime. A bare ``create_task`` keeps only a weak
        # reference, so an unreferenced pump can be collected mid-read and the
        # process output silently stops accumulating.
        self._pump_task: Any = None

    def start_watchers(self) -> None:
        import asyncio

        self._watch = asyncio.create_task(self._watch_proc())
        if self.proc.stdout is not None:
            self._pump_task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        # ``start_watchers`` only starts the pump when there is a stdout to
        # read, but the check is repeated here rather than asserted: an
        # ``assert`` is compiled out under ``-O``, which would turn a missing
        # stream into an ``AttributeError`` inside a detached task nobody
        # awaits.
        if self.proc.stdout is None:
            return
        while True:
            chunk = await self.proc.stdout.read(4096)
            if not chunk:
                return
            self._chunks.append(chunk.decode("utf-8", errors="replace"))

    async def _watch_proc(self) -> None:
        import asyncio

        try:
            await asyncio.wait_for(self.proc.wait(), timeout=self.timeout_seconds)
        except TimeoutError:
            await self.kill_group()
            self._status = "timed_out"
            return
        code = self.proc.returncode
        if self._status == "stopped":
            return
        self._status = "succeeded" if code == 0 else "failed"

    async def kill_group(self) -> None:
        import os
        import signal

        if self.proc.returncode is not None:
            if self._status == "running":
                self._status = "stopped"
            return
        pid = self.proc.pid
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            self.proc.kill()
        try:
            await self.proc.wait()
        except (ProcessLookupError, ChildProcessError):
            # The group kill above already reaped it; there is nothing to wait
            # for. Any other failure is a real one and belongs to the caller.
            pass
        self._status = "stopped"

    async def snapshot(self) -> tuple[BackgroundStatus, str]:
        return self._status, "".join(self._chunks)

    def extend_timeout(self, seconds: int) -> None:
        self.timeout_seconds = seconds


class LocalProcessGroupRunner:
    """Shipped runner: a real process group, not a recorded FakeHandle.

    Production prefers a sandbox-worker runner; this class is the process-group
    implementation used by tests that need a live command and as the fallback
    when no sandbox manager is wired.
    """

    worker_id = "local-pg"

    async def spawn(
        self,
        command: str,
        *,
        timeout_seconds: int,
        session_id: str,
        tenant_id: str,
    ) -> LocalProcessGroupHandle:
        import asyncio
        from uuid import uuid4

        proc = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        handle = LocalProcessGroupHandle(
            proc,
            start_token=f"pg-{proc.pid}-{uuid4().hex[:8]}",
            worker_id=self.worker_id,
            timeout_seconds=timeout_seconds,
        )
        handle.start_watchers()
        return handle


class FakeRunner:
    """Injectable runner: tests finish/kill handles without a real subprocess."""

    def __init__(self) -> None:
        self.handles: dict[str, FakeHandle] = {}
        self.spawned: list[str] = []
        self._next_pid = 1000
        self.worker_id = "worker-a"

    async def spawn(
        self,
        command: str,
        *,
        timeout_seconds: int,
        session_id: str,
        tenant_id: str,
    ) -> FakeHandle:
        self._next_pid += 1
        handle = FakeHandle(
            pid=self._next_pid,
            start_token=f"st-{self._next_pid}",
            worker_id=self.worker_id,
        )
        self.handles[command] = handle
        self.spawned.append(command)
        return handle


@dataclass
class BackgroundPool:
    """Shipped pool: persist-before-emit records + runner, never a loop-pod fork."""

    runner: ProcessRunner
    rc: RuntimeConstants
    tasks: list[BackgroundTask] = field(default_factory=list)
    handles: dict[str, ProcessHandle] = field(default_factory=dict)
    wakes_used: int = 0
    shutting_down: bool = False

    def list(self, session_id: str) -> list[BackgroundTask]:
        return [item for item in self.tasks if item.session_id == session_id]

    def get(self, task_id: str) -> BackgroundTask | None:
        for item in self.tasks:
            if item.id == task_id:
                return item
        return None

    async def start(
        self,
        *,
        command: str,
        session_id: str,
        tenant_id: str,
        notify_on_finish: bool = False,
        expected_seconds: int | None = None,
        timeout_seconds: int | None = None,
    ) -> BackgroundTask:
        refuse_notify_when_disabled(
            enabled=self.rc.background_tasks_enabled,
            notify_on_finish=notify_on_finish,
        )
        if not self.rc.background_tasks_enabled:
            raise ValueError("background_tasks_disabled")
        session_tasks = self.list(session_id)
        if pool_is_full(session_tasks, self.rc):
            raise ValueError("background_pool_full")
        hard = compute_hard_timeout_seconds(
            explicit=timeout_seconds,
            expected_seconds=expected_seconds,
            rc=self.rc,
        )
        handle = await self.runner.spawn(
            command,
            timeout_seconds=hard,
            session_id=session_id,
            tenant_id=tenant_id,
        )
        task = BackgroundTask(
            id=new_task_id(),
            session_id=session_id,
            tenant_id=tenant_id,
            command=command,
            status="running",
            notify_on_finish=notify_on_finish,
            expected_seconds=expected_seconds,
            timeout_seconds=hard,
            worker_id=handle.worker_id,
            pid=handle.pid,
            start_token=handle.start_token,
        )
        self.tasks.append(task)
        self.handles[task.id] = handle
        return task

    async def adopt(
        self,
        *,
        command: str,
        session_id: str,
        tenant_id: str,
        output_so_far: str,
        handle: ProcessHandle,
        expected_seconds: int | None = None,
    ) -> BackgroundTask:
        if pool_is_full(self.list(session_id), self.rc):
            raise ValueError("background_pool_full")
        hard = self.rc.background_max_timeout_seconds
        task = BackgroundTask(
            id=new_task_id(),
            session_id=session_id,
            tenant_id=tenant_id,
            command=command,
            status="running",
            notify_on_finish=False,
            expected_seconds=expected_seconds,
            timeout_seconds=hard,
            output=clip_output(output_so_far, self.rc),
            worker_id=handle.worker_id,
            pid=handle.pid,
            start_token=handle.start_token,
            adopted=True,
        )
        self.tasks.append(task)
        self.handles[task.id] = handle
        return task

    async def refresh(self, task_id: str) -> BackgroundTask | None:
        task = self.get(task_id)
        handle = self.handles.get(task_id)
        if task is None or handle is None:
            return task
        status, output = await handle.snapshot()
        task.status = status
        task.output = clip_output(output, self.rc)
        return task

    async def stop(self, task_id: str) -> BackgroundTask | None:
        task = self.get(task_id)
        handle = self.handles.get(task_id)
        if task is None:
            return None
        if handle is not None:
            await handle.kill_group()
            task.status = "stopped"
            task.reason = "stopped"
        return task

    async def reap(self, task_id: str) -> BackgroundTask | None:
        task = self.get(task_id)
        handle = self.handles.get(task_id)
        if task is None:
            return None
        if handle is None:
            task.status = "orphaned"
            task.reason = "missing_handle"
            return task
        if not reap_is_safe(
            recorded_pid=task.pid,
            recorded_start_token=task.start_token,
            live_pid=handle.pid,
            live_start_token=handle.start_token,
        ):
            task.reason = "reap_identity_mismatch"
            return task
        await handle.kill_group()
        task.status = "stopped"
        task.reason = "reaped"
        return task

    def worker_died(self, worker_id: str) -> None:
        self.tasks = mark_orphaned_on_worker_death(self.tasks, worker_id)

    def drain_wakes(self, session_id: str) -> builtins.list[str]:
        finished = [
            item
            for item in self.list(session_id)
            if item.status in TERMINAL_STATUSES and item.notify_on_finish
        ]
        ids = collect_wake(
            finished,
            shutting_down=self.shutting_down,
            wakes_used=self.wakes_used,
            rc=self.rc,
        )
        if ids:
            self.wakes_used += 1
            for item in finished:
                if item.id in ids:
                    item.notify_on_finish = False
        return ids


__all__ = [
    "TERMINAL_STATUSES",
    "AdoptDecision",
    "BackgroundPool",
    "BackgroundStatus",
    "BackgroundTask",
    "FakeHandle",
    "FakeRunner",
    "LocalProcessGroupHandle",
    "LocalProcessGroupRunner",
    "ProcessHandle",
    "ProcessRunner",
    "adopt_notice",
    "clip_output",
    "collect_wake",
    "compute_hard_timeout_seconds",
    "decide_foreground",
    "mark_orphaned_on_worker_death",
    "new_task_id",
    "pool_is_full",
    "reap_is_safe",
    "refuse_notify_when_disabled",
    "running_count",
]
