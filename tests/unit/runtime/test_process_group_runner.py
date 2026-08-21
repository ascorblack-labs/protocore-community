"""Live process-group runner — real echo/sleep, not FakeRunner theater."""
from __future__ import annotations

import asyncio

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.runtime.background import BackgroundPool, LocalProcessGroupRunner


def _on(**overrides: object) -> RuntimeConstants:
    values: dict[str, object] = {
        "model_context_window": 4096,
        "background_tasks_enabled": True,
        "foreground_adopt_enabled": True,
        "background_max_concurrent_per_session": 2,
        "bash_foreground_timeout_seconds": 1,
    }
    values.update(overrides)
    return RuntimeConstants(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_echo_persists_and_wait_reads_output() -> None:
    pool = BackgroundPool(runner=LocalProcessGroupRunner(), rc=_on())
    task = await pool.start(
        command="echo hello-from-pg",
        session_id="s",
        tenant_id="t",
        notify_on_finish=True,
    )
    assert task.status == "running"
    assert task.pid is not None and task.pid > 1
    assert task.start_token
    listed = pool.list("s")
    assert listed[0].id == task.id
    for _ in range(40):
        current = await pool.refresh(task.id)
        assert current is not None
        if current.status in {"succeeded", "failed", "timed_out", "stopped"}:
            break
        await asyncio.sleep(0.05)
    done = pool.get(task.id)
    assert done is not None
    assert done.status == "succeeded"
    assert "hello-from-pg" in done.output
    wakes = pool.drain_wakes("s")
    assert wakes == [task.id]
    assert pool.drain_wakes("s") == []


@pytest.mark.asyncio
async def test_sleep_stop_and_reap_identity() -> None:
    pool = BackgroundPool(runner=LocalProcessGroupRunner(), rc=_on())
    task = await pool.start(
        command="sleep 8",
        session_id="s",
        tenant_id="t",
    )
    await asyncio.sleep(0.1)
    stopped = await pool.stop(task.id)
    assert stopped is not None
    assert stopped.status == "stopped"
    reaped = await pool.reap(task.id)
    assert reaped is not None
    assert reaped.reason in {"reaped", "stopped"}


@pytest.mark.asyncio
async def test_pool_full_refuses_another_start() -> None:
    rc = _on(background_max_concurrent_per_session=1)
    pool = BackgroundPool(runner=LocalProcessGroupRunner(), rc=rc)
    first = await pool.start(command="sleep 5", session_id="s", tenant_id="t")
    with pytest.raises(ValueError, match="background_pool_full"):
        await pool.start(command="echo extra", session_id="s", tenant_id="t")
    await pool.stop(first.id)


@pytest.mark.asyncio
async def test_worker_death_marks_orphaned() -> None:
    pool = BackgroundPool(runner=LocalProcessGroupRunner(), rc=_on())
    task = await pool.start(command="sleep 5", session_id="s", tenant_id="t")
    pool.worker_died(task.worker_id)
    assert pool.get(task.id) is not None
    assert pool.get(task.id).status == "orphaned"
    await pool.stop(task.id)
