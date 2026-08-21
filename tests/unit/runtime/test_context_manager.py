"""Tests for :mod:`protocore.runtime.context.manager`."""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import (
    ImageRefBlock,
    Message,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.context.manager import (
    ContextManager,
    detect_active_language,
    estimate_history_tokens,
)
from protocore.tests_support.adapters import InMemoryBlobStore, InMemoryLLMProvider


def test_detect_active_language_english() -> None:
    msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hello world")])
    assert detect_active_language(msg) == "en"


def test_detect_active_language_russian() -> None:
    msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="Привет мир")])
    assert detect_active_language(msg) == "ru"


def test_detect_active_language_cyrillic_in_json_escape() -> None:
    msg = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text=r'{"q": "Привет"}')],
    )
    assert detect_active_language(msg) == "ru"


def test_detect_active_language_none_message() -> None:
    assert detect_active_language(None) == "en"


def test_detect_active_language_empty_message() -> None:
    msg = Message(role=MessageRole.user, content_blocks=[])
    assert detect_active_language(msg) == "en"


def test_estimate_history_tokens_zero_for_empty() -> None:
    rc = RuntimeConstants()
    assert estimate_history_tokens([], rc) == 0


def test_estimate_history_tokens_scales_with_content() -> None:
    rc = RuntimeConstants()
    short = [Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])]
    long_ = [Message(role=MessageRole.user, content_blocks=[TextBlock(text="x" * 1000)])]
    assert estimate_history_tokens(long_, rc) > estimate_history_tokens(short, rc)


# ---------------------------------------------------------------------------
# / / exhaustive per-ContentBlock estimation
# ---------------------------------------------------------------------------


def test_estimate_counts_tool_use_arguments() -> None:
    """a large ToolUseBlock argument payload must NOT count as 0.

    ToolUseBlock has neither ``text`` nor ``content`` — only ``name`` +
    ``arguments_json``. The old attr-probe estimator counted it as 0, so a
    20 KB Write tool-call vanished from the pre-flight compaction gate.
    """
    rc = RuntimeConstants()
    big_args = '{"path": "x.txt", "content": "' + ("Z" * 20_000) + '"}'
    history = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="t1", name="write_file", arguments_json=big_args),
            ],
        ),
    ]
    estimate = estimate_history_tokens(history, rc)
    # 20 KB of args at ~4 chars/token must be thousands of tokens, never 0.
    assert estimate > 1_000


def test_estimate_counts_reasoning_content() -> None:
    """Message.reasoning_content (re-emitted thinking) counts.

    reasoning_content is a top-level Message field (not a content block) that
    thinking-capable providers re-emit on the wire; it must be included.
    """
    rc = RuntimeConstants()
    base = Message(
        role=MessageRole.assistant,
        content_blocks=[TextBlock(text="ok")],
    )
    with_reasoning = Message(
        role=MessageRole.assistant,
        content_blocks=[TextBlock(text="ok")],
        reasoning_content="R" * 4_000,
    )
    assert estimate_history_tokens([with_reasoning], rc) > estimate_history_tokens([base], rc)
    # The reasoning text alone must contribute a non-trivial amount.
    assert estimate_history_tokens([with_reasoning], rc) > 500


def test_estimate_counts_thinking_block() -> None:
    """ThinkingBlock.text is counted (it has .text, but assert)."""
    rc = RuntimeConstants()
    history = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[ThinkingBlock(text="T" * 4_000)],
        ),
    ]
    assert estimate_history_tokens(history, rc) > 500


def test_estimate_counts_image_ref_block_via_constant() -> None:
    """ImageRefBlock has neither text nor content; it must count
    a fixed RC-configurable image-token constant, never 0."""
    rc = RuntimeConstants()
    history = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[ImageRefBlock(blob_ref="blob://img1")],
        ),
    ]
    estimate = estimate_history_tokens(history, rc)
    assert estimate == rc.token_count_image_tokens
    assert estimate > 0


def test_estimate_counts_tool_result_block() -> None:
    """A ToolResultBlock (has .content) is counted (regression anchor)."""
    rc = RuntimeConstants()
    history = [
        Message(
            role=MessageRole.tool,
            content_blocks=[ToolResultBlock(tool_call_id="t1", content="R" * 4_000)],
        ),
    ]
    assert estimate_history_tokens(history, rc) > 500


@pytest.mark.asyncio
async def test_context_manager_build_context_returns_bundle() -> None:
    rc = RuntimeConstants(model_context_window=4_096)
    blobs = InMemoryBlobStore()
    llm = InMemoryLLMProvider()
    mgr = ContextManager(rc=rc, blob_store=blobs, compaction_llm=llm)

    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hello")]),
    ]
    bundle = mgr.build_context(history=history, tools=())
    assert bundle.active_language == "en"
    assert bundle.budgets.max_context == 4_096
    assert len(bundle.messages) == 1


@pytest.mark.asyncio
async def test_context_manager_detects_russian() -> None:
    rc = RuntimeConstants(model_context_window=4_096)
    blobs = InMemoryBlobStore()
    llm = InMemoryLLMProvider()
    mgr = ContextManager(rc=rc, blob_store=blobs, compaction_llm=llm)

    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="Привет, как дела?")]),
    ]
    bundle = mgr.build_context(history=history, tools=())
    assert bundle.active_language == "ru"


def test_context_manager_needs_compaction_when_history_exceeds_trigger() -> None:
    rc = RuntimeConstants(model_context_window=512)
    blobs = InMemoryBlobStore()
    llm = InMemoryLLMProvider()
    mgr = ContextManager(rc=rc, blob_store=blobs, compaction_llm=llm)

    # 8000 chars of text should exceed 0.8 * 512 = ~410 tokens estimate.
    big = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="x" * 8000)],
    )
    assert mgr.needs_compaction([big]) is True


def test_context_manager_no_compaction_for_short_history() -> None:
    rc = RuntimeConstants(model_context_window=49_152)
    blobs = InMemoryBlobStore()
    llm = InMemoryLLMProvider()
    mgr = ContextManager(rc=rc, blob_store=blobs, compaction_llm=llm)

    small = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    assert mgr.needs_compaction([small]) is False


def test_observed_prompt_tokens_floors_compaction_gate() -> None:
    """The char heuristic under-counts adversarial content, so a history that
    the estimate reads as tiny must still trip the gate once the provider has
    reported a real prompt size above the trigger. Regression for a 65536-window
    provider that received ~148K real input tokens with the estimate far below
    trigger and never compacted."""
    rc = RuntimeConstants(model_context_window=65_536)
    blobs = InMemoryBlobStore()
    llm = InMemoryLLMProvider()
    mgr = ContextManager(rc=rc, blob_store=blobs, compaction_llm=llm)

    # A short message whose char estimate is far below 0.8 * 65536 = 52428.
    tiny = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    assert estimate_history_tokens([tiny], rc) < 52_428

    # No real measurement yet → gate relies on the (low) estimate → no compaction.
    assert mgr.needs_compaction([tiny], observed_prompt_tokens=0) is False
    # Provider reported 148K real prompt tokens on the prior call (2.25x window).
    assert mgr.needs_compaction([tiny], observed_prompt_tokens=147_892) is True
    assert (
        mgr.needs_emergency_compaction([tiny], observed_prompt_tokens=147_892)
        is True
    )


def test_observed_prompt_tokens_below_trigger_does_not_force_compaction() -> None:
    """A real measurement UNDER the trigger must not spuriously trip the gate;
    the floor is a max, never an override that ignores a healthy prompt."""
    rc = RuntimeConstants(model_context_window=65_536)
    blobs = InMemoryBlobStore()
    llm = InMemoryLLMProvider()
    mgr = ContextManager(rc=rc, blob_store=blobs, compaction_llm=llm)

    tiny = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    # 10K real tokens < 0.8 * 65536 trigger and < 0.95 * 65536 emergency.
    assert mgr.needs_compaction([tiny], observed_prompt_tokens=10_000) is False
    assert (
        mgr.needs_emergency_compaction([tiny], observed_prompt_tokens=10_000)
        is False
    )


def test_estimate_still_governs_when_it_exceeds_observed() -> None:
    """When the char estimate is the larger of the two (e.g. right after a
    resume with a stale-zero observation but a genuinely large history), the
    estimate still drives the gate — the floor is max(estimate, observed)."""
    rc = RuntimeConstants(model_context_window=512)
    blobs = InMemoryBlobStore()
    llm = InMemoryLLMProvider()
    mgr = ContextManager(rc=rc, blob_store=blobs, compaction_llm=llm)

    big = Message(role=MessageRole.user, content_blocks=[TextBlock(text="x" * 8000)])
    assert mgr.needs_compaction([big], observed_prompt_tokens=0) is True


# ---------------------------------------------------------------------------
# run_compaction Tier-1-failure early return must NOT report
# tokens_after=0 (false "full clear" telemetry).
# ---------------------------------------------------------------------------


class _FailingBlobStore(InMemoryBlobStore):
    """Blob store whose ``put`` always raises — forces Tier 1 to fail."""

    async def put(self, *args: object, **kwargs: object):  # type: ignore[override]
        raise RuntimeError("blob store unavailable")


@pytest.mark.asyncio
async def test_run_compaction_tier1_failure_early_return_sets_tokens_after() -> None:
    """when Tier 1 raises and the retry budget is not yet exhausted,
    run_compaction early-returns. That attempt must carry a real
    ``tokens_after`` (the current estimate), never the default 0 — otherwise
    the caller emits COMPACTION_COMPLETED with tokens_after=0 ≪ tokens_before,
    falsely telling operators the whole context was cleared."""
    from protocore.runtime.context.compaction import CompactionState

    rc = RuntimeConstants(
        model_context_window=4_096,
        # Generous retry budget so the failure early-returns (not raises).
        compaction_failed_max_retries=5,
        # Keep only the last turn so the big tool result is eligible for Tier 1.
        compaction_keep_recent_turns=1,
    )
    mgr = ContextManager(
        rc=rc,
        blob_store=_FailingBlobStore(),
        compaction_llm=InMemoryLLMProvider(),
    )
    # A big tool result in the eligible (non-kept) region so Tier 1 actually
    # reaches blob_store.put → raises → run_compaction early-returns.
    history = [
        Message(
            role=MessageRole.tool,
            content_blocks=[ToolResultBlock(tool_call_id="t1", content="X" * 8_000)],
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    state = CompactionState()
    attempt = await mgr.run_compaction(
        history=history,
        compaction_state=state,
        tenant_id="t1",
        model_name="mock",
    )

    assert attempt.tokens_before > 0
    # The bug: tokens_after defaulted to 0. The fix sets it to the real estimate.
    assert attempt.tokens_after > 0
    assert attempt.tokens_after == attempt.tokens_before


# ---------------------------------------------------------------------------
# pin LRU + cap enforcement
# ---------------------------------------------------------------------------


def _new_manager(*, cap: int) -> ContextManager:
    rc = RuntimeConstants(pinned_tool_max_count=cap)
    return ContextManager(
        rc=rc,
        blob_store=InMemoryBlobStore(),
        compaction_llm=InMemoryLLMProvider(),
    )


def test_pin_tool_under_cap_does_not_evict() -> None:
    """Pinning below the cap is a plain append; no eviction returned."""
    mgr = _new_manager(cap=3)
    assert mgr.pin_tool("A") is None
    assert mgr.pin_tool("B") is None
    assert mgr.pinned_tool_names() == ("A", "B")


def test_pin_tool_at_cap_evicts_oldest() -> None:
    """F8 happy path — pinning a new tool when the list is at the cap
    evicts the LRU (oldest) entry and surfaces the evicted name."""
    mgr = _new_manager(cap=3)
    mgr.pin_tool("A")
    mgr.pin_tool("B")
    mgr.pin_tool("C")
    # Cap is 3; pinning a 4th evicts A.
    evicted = mgr.pin_tool("D")
    assert evicted == "A"
    assert mgr.pinned_tool_names() == ("B", "C", "D")


def test_pin_tool_repin_promotes_to_mru() -> None:
    """Re-pinning an already-pinned name moves it to the MRU end so
    the next eviction skips it."""
    mgr = _new_manager(cap=3)
    mgr.pin_tool("A")
    mgr.pin_tool("B")
    mgr.pin_tool("C")
    # Re-pin A → moves to end; A is no longer the LRU.
    assert mgr.pin_tool("A") is None
    assert mgr.pinned_tool_names() == ("B", "C", "A")
    # Now pinning D evicts B (the new LRU).
    assert mgr.pin_tool("D") == "B"
    assert mgr.pinned_tool_names() == ("C", "A", "D")


def test_pin_tool_saturation_evicts_in_strict_order() -> None:
    """F8 saturation: cycle 16 distinct pins through a cap of 15 and
    confirm the eviction order matches strict LRU."""
    mgr = _new_manager(cap=15)
    for i in range(15):
        assert mgr.pin_tool(f"T{i}") is None
    assert len(mgr.pinned_tool_names()) == 15
    # 16th pin evicts T0.
    assert mgr.pin_tool("T15") == "T0"
    assert "T0" not in mgr.pinned_tool_names()
    assert mgr.pinned_tool_names()[-1] == "T15"


def test_pin_tool_empty_name_is_noop() -> None:
    """Defensive guard: empty name is rejected silently (no LRU shift,
    no return)."""
    mgr = _new_manager(cap=3)
    assert mgr.pin_tool("") is None
    assert mgr.pinned_tool_names() == ()
