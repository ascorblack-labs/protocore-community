"""The tool-name lookup is resolved once per pass, not once per result block.

Naming the tool behind a call id is a whole-transcript walk. Asking it per
shed result block makes the walk quadratic in the transcript — and compaction
and result eviction run precisely on the largest transcripts, so that is the
one place the quadratic term is guaranteed to be paid. Both passes therefore
build the id -> name map ONCE and index it.

That is invisible in an output assertion: a per-block lookup and a hoisted map
return the same names. So it is pinned here by counting the walks.
"""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.prompts import bundled_prompt_provider
from protocore.runtime import result_eviction
from protocore.runtime.context import compaction
from protocore.runtime.context.budgets import derive_budgets
from protocore.runtime.context.compaction import run_tier1_truncation
from protocore.runtime.result_eviction import (
    evict_history_for_llm,
    tool_name_for_result,
    tool_names_by_call_id,
)
from protocore.tests_support.adapters import InMemoryBlobStore

_BLOCKS_PER_MESSAGE = 4
_MESSAGES = 6


def _transcript(body: str) -> list[Message]:
    """A transcript whose tool-role messages each answer several calls."""
    history: list[Message] = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")])
    ]
    for msg in range(_MESSAGES):
        ids = [f"c{msg}-{i}" for i in range(_BLOCKS_PER_MESSAGE)]
        history.append(
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    ToolUseBlock(tool_call_id=cid, name="Read", arguments_json="{}")
                    for cid in ids
                ],
            )
        )
        history.append(
            Message(
                role=MessageRole.tool,
                content_blocks=[
                    ToolResultBlock(tool_call_id=cid, content=body) for cid in ids
                ],
            )
        )
    return history


def test_the_map_answers_exactly_what_the_single_id_lookup_answers() -> None:
    history = _transcript("body")
    names = tool_names_by_call_id(history)

    assert names == {
        f"c{msg}-{i}": "Read"
        for msg in range(_MESSAGES)
        for i in range(_BLOCKS_PER_MESSAGE)
    }
    for call_id in names:
        assert names[call_id] == tool_name_for_result(history, call_id)
    # An id that was never issued is absent from the map, which is the same
    # answer the single-id lookup gives as None.
    assert "never-issued" not in names
    assert tool_name_for_result(history, "never-issued") is None


@pytest.mark.asyncio
async def test_tier1_walks_the_transcript_for_names_once_per_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    walks = 0
    real = result_eviction.tool_names_by_call_id

    def counting(history):  # type: ignore[no-untyped-def]
        nonlocal walks
        walks += 1
        return real(history)

    monkeypatch.setattr(compaction, "tool_names_by_call_id", counting)

    rc = LoopConstants(model_context_window=4_096)
    history = _transcript("X" * 6_000)
    shed_blocks = _MESSAGES * _BLOCKS_PER_MESSAGE

    result = await run_tier1_truncation(
        history=history,
        blob_store=InMemoryBlobStore(),
        tenant_id="t1",
        rc=rc,
        truncation_threshold_tokens=derive_budgets(rc).tool_result_truncation_threshold,
        keep_recent_turns=0,
    )

    # Every result block was over threshold and shed, so a per-block lookup
    # would have walked the transcript once for each of them.
    assert len(result.blob_refs_created) == shed_blocks
    assert shed_blocks > 1
    assert walks == 1


def test_eviction_walks_the_transcript_for_names_once_per_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    walks = 0
    real = result_eviction.tool_names_by_call_id

    def counting(history):  # type: ignore[no-untyped-def]
        nonlocal walks
        walks += 1
        return real(history)

    monkeypatch.setattr(result_eviction, "tool_names_by_call_id", counting)

    rc = LoopConstants(
        result_eviction_enabled=True, result_eviction_tool_names=["Read"]
    )
    history = _transcript("body")

    _, evicted = evict_history_for_llm(history, rc, bundled_prompt_provider())

    assert len(evicted) == _MESSAGES * _BLOCKS_PER_MESSAGE
    assert walks == 1
