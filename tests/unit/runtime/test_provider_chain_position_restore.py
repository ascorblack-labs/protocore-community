"""A run resumes on the provider rung it was demoted to, or not at all.

The provider chain cursor is one-way for the life of a run: a demotion is a
statement that the current provider is unhealthy right now. The cursor lives in
the host's chain object, so a process picking the run up starts at position 0 —
and before this the run walked straight back onto the endpoint it had already
proved dead, with a fresh advance budget. Under a degrading upstream that is a
loop that pays for a full turn each round.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace

import pytest

from protocore.contracts.llm import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, StopReason, TextBlock


class _Verdict:
    def __init__(self, reason: str) -> None:
        self.reason = reason


def _classified(exc: BaseException, reason: str) -> BaseException:
    object.__setattr__(exc, "classified", _Verdict(reason))
    return exc


class _Chain:
    """Minimal IProviderChain over a fixed list of model names."""

    def __init__(self, provider: object, names: list[str]) -> None:
        self._provider = provider
        self._names = names
        self._index = 0
        self.advance_reasons: list[str] = []

    def current(self) -> object:
        return self._provider

    def current_model_name(self) -> str:
        return self._names[self._index]

    async def advance(self, *, reason: str) -> bool:
        self.advance_reasons.append(reason)
        if self._index + 1 >= len(self._names):
            return False
        self._index += 1
        return True

    def attempted(self) -> list[tuple[str, str]]:
        return []


class _PartialThenFailLLM:
    """Streams a partial answer, fails the first call, then recovers."""

    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []
        self._n = 0

    async def stream_with_tools(
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        n = self._n
        self._n += 1
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(
            name="content_block_start", payload={"index": 0, "type": "text"}
        )
        if n == 0:
            yield LLMStreamEvent(
                name="content_block_delta", payload={"index": 0, "text": "partial"}
            )
            raise _classified(LLMProviderError("primary is down"), "server_error")
        yield LLMStreamEvent(
            name="content_block_delta", payload={"index": 0, "text": "recovered"}
        )
        yield LLMStreamEvent(name="content_block_stop", payload={"index": 0})
        yield LLMStreamEvent(
            name="message_stop", payload={"stop_reason": "end_turn"}
        )

    async def complete_structured(
        self, request: LLMRequest, schema: object
    ) -> LLMResponse:
        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text: str, model: str | None = None) -> int:
        return max(1, len(text) // 4)


def _rc() -> LoopConstants:
    return LoopConstants(model_context_window=4_096)


def test_advance_count_is_in_the_snapshot(engine_factory) -> None:
    engine = engine_factory()
    engine._provider_chain_advances = 2

    assert engine.snapshot()["provider_chain_advances"] == 2


async def test_a_demoted_run_resumes_on_the_rung_it_reached(engine_factory) -> None:
    engine = engine_factory(rc=_rc(), model_name="primary-model")
    llm = _PartialThenFailLLM()
    engine.llm = llm
    engine.provider_chain = _Chain(llm, ["primary-model", "fallback-model-x"])

    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    ):
        pass

    assert engine._provider_chain_advances == 1
    assert engine.config.model_name == "fallback-model-x"
    snapshot = engine.snapshot()

    resumed = engine_factory(rc=_rc(), model_name="primary-model")
    fresh_llm = _PartialThenFailLLM()
    resumed.llm = fresh_llm
    resumed.provider_chain = _Chain(
        fresh_llm, ["primary-model", "fallback-model-x"]
    )
    await resumed.resume_from_snapshot(snapshot)

    assert resumed.config.model_name == "fallback-model-x"
    assert resumed._provider_chain_advances == 1
    assert resumed.provider_chain.current_model_name() == "fallback-model-x"


async def test_an_undemoted_run_does_not_move(engine_factory) -> None:
    source = engine_factory(rc=_rc(), model_name="primary-model")
    llm = _PartialThenFailLLM()
    source.llm = llm
    chain = _Chain(llm, ["primary-model", "fallback-model-x"])
    source.provider_chain = chain

    resumed = engine_factory(rc=_rc(), model_name="primary-model")
    resumed.llm = llm
    resumed_chain = _Chain(llm, ["primary-model", "fallback-model-x"])
    resumed.provider_chain = resumed_chain
    await resumed.resume_from_snapshot(source.snapshot())

    assert resumed_chain.advance_reasons == []
    assert resumed.config.model_name == "primary-model"


async def test_a_rung_missing_from_the_chain_is_refused(engine_factory) -> None:
    source = engine_factory(rc=_rc(), model_name="primary-model")
    source._provider_chain_advances = 1
    snapshot = source.snapshot()
    snapshot["model_name"] = "fallback-model-x"
    snapshot["provider_chain_model_name"] = "fallback-model-x"

    llm = _PartialThenFailLLM()
    resumed = engine_factory(rc=_rc(), model_name="primary-model")
    resumed.llm = llm
    # The installation reordered its providers between the two processes.
    resumed.provider_chain = _Chain(llm, ["primary-model", "some-other-model"])

    with pytest.raises(ValueError, match="fallback-model-x"):
        await resumed.resume_from_snapshot(snapshot)
    # Refused before any state was applied.
    assert resumed.history == []


async def test_an_exhausted_chain_is_refused(engine_factory) -> None:
    source = engine_factory(rc=_rc(), model_name="primary-model")
    source._provider_chain_advances = 3
    snapshot = source.snapshot()
    snapshot["model_name"] = "fallback-model-x"
    snapshot["provider_chain_model_name"] = "fallback-model-x"

    llm = _PartialThenFailLLM()
    resumed = engine_factory(rc=_rc(), model_name="primary-model")
    resumed.llm = llm
    resumed.provider_chain = _Chain(llm, ["primary-model", "fallback-model-x"])

    with pytest.raises(ValueError, match="exhausted"):
        await resumed.resume_from_snapshot(snapshot)


async def test_a_demotion_without_a_chain_to_re_seat_it_is_refused(
    engine_factory,
) -> None:
    source = engine_factory(rc=_rc(), model_name="primary-model")
    source._provider_chain_advances = 1
    snapshot = source.snapshot()
    snapshot["model_name"] = "fallback-model-x"

    resumed = engine_factory(rc=_rc(), model_name="primary-model")

    with pytest.raises(ValueError, match="no provider chain"):
        await resumed.resume_from_snapshot(snapshot)


async def test_the_model_name_is_applied_not_compared(engine_factory) -> None:
    """A chainless engine takes the snapshot's model name as authoritative."""
    source = engine_factory(rc=_rc(), model_name="chosen-model")

    resumed = engine_factory(rc=_rc(), model_name="whatever-this-pod-started-on")
    await resumed.resume_from_snapshot(source.snapshot())

    assert resumed.config.model_name == "chosen-model"


async def test_a_snapshot_without_a_model_name_is_refused(engine_factory) -> None:
    source = engine_factory(rc=_rc())
    snapshot = source.snapshot()
    del snapshot["model_name"]

    resumed = engine_factory(rc=_rc())
    with pytest.raises(ValueError, match="model_name"):
        await resumed.resume_from_snapshot(snapshot)


async def test_a_pinned_model_name_does_not_refuse_an_undemoted_resume(
    engine_factory,
) -> None:
    """The host may pin a run to a model the chain rows never name.

    The rung the run sits on and the model name the run was configured with are
    two different facts. Asserting the chain against the configured name made
    every resume in such a scope impossible — including runs that were never
    demoted and had nothing to re-seat.
    """
    llm = _PartialThenFailLLM()
    source = engine_factory(rc=_rc(), model_name="an-operator-pinned-model")
    source.llm = llm
    source.provider_chain = _Chain(llm, ["row-model-a", "row-model-b"])
    snapshot = source.snapshot()
    assert snapshot["provider_chain_advances"] == 0

    resumed = engine_factory(rc=_rc(), model_name="whatever-this-pod-started-on")
    resumed.llm = llm
    resumed.provider_chain = _Chain(llm, ["row-model-a", "row-model-b"])

    await resumed.resume_from_snapshot(snapshot)

    assert resumed.config.model_name == "an-operator-pinned-model"
    assert resumed.provider_chain.current_model_name() == "row-model-a"
    assert resumed._provider_chain_advances == 0


async def test_resuming_a_live_engine_from_its_own_snapshot_moves_nothing(
    engine_factory,
) -> None:
    """The walk is relative, so a run already seated stays where it is.

    The approval continuation resumes the SAME live engine from a snapshot it
    just took of itself. Walking the cursor absolutely would demote such a run
    one further rung on every continuation, and refuse outright once the chain
    ran out of spares.
    """
    llm = _PartialThenFailLLM()
    engine = engine_factory(rc=_rc(), model_name="primary-model")
    engine.llm = llm
    chain = _Chain(llm, ["primary-model", "fallback-b", "fallback-c"])
    engine.provider_chain = chain
    assert await chain.advance(reason="demotion") is True
    engine._provider_chain_advances = 1
    engine.config = replace(engine.config, model_name="fallback-b")

    snapshot = engine.snapshot()
    await engine.resume_from_snapshot(snapshot)

    assert chain.current_model_name() == "fallback-b"
    assert engine._provider_chain_advances == 1

    await engine.resume_from_snapshot(snapshot)

    assert chain.current_model_name() == "fallback-b"


async def test_a_refused_resume_leaves_the_shared_chain_where_it_was(
    engine_factory,
) -> None:
    """A snapshot that will not parse must not demote the host's chain.

    The cursor is one-way and the chain object outlives the call, so a resume
    that comes up refused has to leave it standing.
    """
    llm = _PartialThenFailLLM()
    source = engine_factory(rc=_rc(), model_name="primary-model")
    source.llm = llm
    source.provider_chain = _Chain(llm, ["primary-model", "fallback-b"])
    source._provider_chain_advances = 1
    snapshot = source.snapshot()
    snapshot["provider_chain_model_name"] = "fallback-b"
    snapshot["history"] = [{"role": "not-a-real-role"}]

    resumed = engine_factory(rc=_rc(), model_name="primary-model")
    resumed.llm = llm
    chain = _Chain(llm, ["primary-model", "fallback-b"])
    resumed.provider_chain = chain

    with pytest.raises(Exception):
        await resumed.resume_from_snapshot(snapshot)

    assert chain.current_model_name() == "primary-model"
    assert chain.advance_reasons == []


async def test_a_demoted_snapshot_without_the_rung_recorded_is_refused(
    engine_factory,
) -> None:
    """A non-zero position with no rung beside it cannot be verified.

    The count alone does not identify a provider: it says how far the cursor
    walked in a chain this process may not have in the same order. Seating on
    "whatever is at index 1 here" is the silent wrong-endpoint resume the rung
    was recorded to prevent.
    """
    llm = _PartialThenFailLLM()
    source = engine_factory(rc=_rc(), model_name="primary-model")
    source._provider_chain_advances = 1
    snapshot = source.snapshot()
    snapshot.pop("provider_chain_model_name", None)

    resumed = engine_factory(rc=_rc(), model_name="primary-model")
    resumed.llm = llm
    chain = _Chain(llm, ["primary-model", "fallback-b"])
    resumed.provider_chain = chain

    with pytest.raises(ValueError, match="provider_chain_model_name"):
        await resumed.resume_from_snapshot(snapshot)

    assert chain.current_model_name() == "primary-model"


async def test_an_undemoted_snapshot_without_the_rung_recorded_still_resumes(
    engine_factory,
) -> None:
    """Position zero has no rung to verify, so the key is not required there.

    Requiring it unconditionally made every snapshot written before the rung
    was recorded unresumable in every scope that configures a chain — the runs
    that never left the primary included, which is nearly all of them.
    """
    llm = _PartialThenFailLLM()
    source = engine_factory(rc=_rc(), model_name="primary-model")
    snapshot = source.snapshot()
    snapshot.pop("provider_chain_model_name", None)
    assert snapshot["provider_chain_advances"] == 0

    resumed = engine_factory(rc=_rc(), model_name="whatever-this-pod-started-on")
    resumed.llm = llm
    chain = _Chain(llm, ["primary-model", "fallback-b"])
    resumed.provider_chain = chain

    await resumed.resume_from_snapshot(snapshot)

    assert resumed.config.model_name == "primary-model"
    assert resumed._provider_chain_advances == 0
    assert chain.advance_reasons == []


async def test_a_snapshot_behind_this_runs_own_position_is_refused(
    engine_factory,
) -> None:
    """The cursor does not rewind, so an older snapshot of a live run refuses.

    Two snapshots of the same run can be taken either side of a demotion. The
    older one describes a rung this run has already left; re-seating on it
    would promote a run back onto an endpoint it proved dead this same run,
    and the chain object cannot walk backwards to do it anyway.
    """
    llm = _PartialThenFailLLM()
    engine = engine_factory(rc=_rc(), model_name="primary-model")
    engine.llm = llm
    chain = _Chain(llm, ["primary-model", "fallback-b"])
    engine.provider_chain = chain
    stale = engine.snapshot()

    assert await chain.advance(reason="demotion") is True
    engine._provider_chain_advances = 1
    engine.config = replace(engine.config, model_name="fallback-b")

    with pytest.raises(ValueError, match="does not rewind"):
        await engine.resume_from_snapshot(stale)

    assert chain.current_model_name() == "fallback-b"
    assert engine._provider_chain_advances == 1


async def test_a_malformed_state_refuses_before_the_chain_moves(
    engine_factory,
) -> None:
    """Every value that can refuse is parsed before the one-way move.

    The history was already; the loop state, the turn count, the usage totals
    and the compaction block were not, and each of them raises on a snapshot
    that has been truncated or hand-edited. Refusing after the re-seat leaves
    the host's shared chain demoted for a run that never came up.
    """
    llm = _PartialThenFailLLM()
    source = engine_factory(rc=_rc(), model_name="primary-model")
    source.llm = llm
    source.provider_chain = _Chain(llm, ["primary-model", "fallback-b"])
    source._provider_chain_advances = 1
    snapshot = source.snapshot()
    snapshot["provider_chain_model_name"] = "fallback-b"
    snapshot["state"] = "not-a-loop-state"

    resumed = engine_factory(rc=_rc(), model_name="primary-model")
    resumed.llm = llm
    chain = _Chain(llm, ["primary-model", "fallback-b"])
    resumed.provider_chain = chain

    with pytest.raises(ValueError):
        await resumed.resume_from_snapshot(snapshot)

    assert chain.current_model_name() == "primary-model"
    assert chain.advance_reasons == []
