"""The token heuristic is scaled to the provider's count; an empty compaction pass backs the gate off."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from protocore.contracts.llm import LLMRequest
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, StopReason, TextBlock
from protocore.runtime.context.compaction import estimate_history_tokens, estimate_message_tokens
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import _calibrate_token_estimate
from protocore.runtime.turn_policies.compaction import PerIterationCompactionPolicy


def _msg(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def test_calibration_scales_the_estimate_and_invalidates_the_cache() -> None:
    rc = LoopConstants()
    message = _msg("x" * 4000)
    base = estimate_message_tokens(message, rc)
    doubled = estimate_message_tokens(message, rc.model_copy(update={"token_estimate_calibration": 2.0}))
    assert base > 0 and doubled == 2 * base
    assert estimate_history_tokens([message, message], rc.model_copy(update={"token_estimate_calibration": 1.5})) == 2 * round(base * 1.5)


@dataclass
class _Manager:
    rc: LoopConstants
    updated: list[LoopConstants] = field(default_factory=list)

    def update_rc(self, rc: LoopConstants) -> None:
        self.rc = rc
        self.updated.append(rc)


@dataclass
class _Config:
    rc: LoopConstants


@dataclass
class _Engine:
    config: _Config
    context_manager: _Manager


def _engine(rc: LoopConstants) -> _Engine:
    return _Engine(config=_Config(rc=rc), context_manager=_Manager(rc=rc))


def test_calibration_moves_towards_the_provider_count_and_is_damped() -> None:
    rc = LoopConstants()
    request = LLMRequest(model="m", messages=[_msg("y" * 8000)], tools=[])
    raw = estimate_history_tokens(list(request.messages), rc)
    engine = _engine(rc)
    _calibrate_token_estimate(engine, request, observed=raw * 2)  # type: ignore[arg-type]
    factor = engine.config.rc.token_estimate_calibration
    assert 1.4 < factor < 1.6, factor  # half-way to 2.0 on the first observation
    assert engine.context_manager.rc is engine.config.rc
    _calibrate_token_estimate(engine, request, observed=raw * 2)  # type: ignore[arg-type]
    assert engine.config.rc.token_estimate_calibration > factor
    # A provider that counts fewer tokens than the heuristic never pulls the factor below 1.
    engine = _engine(rc)
    _calibrate_token_estimate(engine, request, observed=raw // 2)  # type: ignore[arg-type]
    assert engine.config.rc.token_estimate_calibration == 1.0 and engine.context_manager.updated == []


def test_calibration_can_be_switched_off() -> None:
    rc = LoopConstants(token_estimate_calibration_enabled=False)
    request = LLMRequest(model="m", messages=[_msg("y" * 8000)], tools=[])
    engine = _engine(rc)
    _calibrate_token_estimate(engine, request, observed=10**6)  # type: ignore[arg-type]
    assert engine.config.rc.token_estimate_calibration == 1.0


class _GateEngine:
    def __init__(self, rc: LoopConstants) -> None:
        self.rc = rc
        self.history: list[Message] = []
        self.state = LoopState.RUNNING
        self.compaction_backoff_left = 0
        self.needs = True

    def needs_emergency_compaction(self) -> bool:
        return False

    def needs_compaction(self) -> bool:
        return self.needs


@pytest.mark.asyncio
async def test_empty_routine_pass_backs_the_gate_off() -> None:
    rc = LoopConstants(compaction_no_gain_backoff_iterations=2, compaction_min_gain_ratio=0.03)
    engine = _GateEngine(rc)
    calls: list[str] = []

    async def compact(eng: Any, *, force: bool, reason: str, protect_tail_from_index: int | None) -> AsyncIterator[TurnEvent]:
        calls.append(reason)
        yield TurnEvent(type=EventType.COMPACTION_COMPLETED, run_id="r", payload={"tokens_before": 60_000, "tokens_after": 59_900})

    policy = PerIterationCompactionPolicy(compact=compact, protect_index=lambda h: None, pair_orphans=lambda e: None, message_stop=lambda e, s: TurnEvent(type=EventType.MESSAGE_STOP, run_id="r", payload={"stop_reason": StopReason.error.value}))

    async def run() -> list[TurnEvent]:
        turn = type("Turn", (), {"engine": engine, "outcome": type("O", (), {"directive": None, "reason": None})()})()
        return [e async for e in policy.apply(turn)]  # type: ignore[arg-type]

    assert len(await run()) == 1 and calls == ["proactive_per_iteration"]
    assert engine.compaction_backoff_left == 2
    assert await run() == [] and await run() == [] and calls == ["proactive_per_iteration"]  # two skipped iterations
    assert len(await run()) == 1 and len(calls) == 2  # then it is consulted again
