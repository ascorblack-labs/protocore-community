"""Session-memory gates are denominated in the ONE shared token estimate.

This module's gates once ran on a private, narrower estimate that costed only
text / tool_use / tool_result text plus ``reasoning_content`` and priced
thinking blocks, image refs and unknown block kinds at ZERO. They now run on
:func:`~protocore.runtime.context.compaction.estimate_message_tokens`, which
costs every block.

That is a change of SCALE, not just of implementation: the same transcript
reads higher than it used to, so ``session_memory_fold_min_tokens``,
``summary_fold_threshold_tokens`` and ``session_memory_tail_budget_fraction``
all bind on marginally less transcript. Nothing in an output assertion shows
that, so it is pinned here — if the zero-cost blocks ever go back to costing
nothing, these fail.
"""
from __future__ import annotations

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    ImageRefBlock,
    Message,
    MessageRole,
    TextBlock,
    ThinkingBlock,
)
from protocore.runtime.context.compaction import (
    estimate_history_tokens,
    estimate_message_tokens,
)
from protocore.runtime.context.session_memory import SessionMemory, fold_run

_TEXT = "the quick brown fox " * 20


def _text_only() -> Message:
    return Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=_TEXT)])


def _with_thinking_and_image() -> Message:
    return Message(
        role=MessageRole.assistant,
        content_blocks=[
            TextBlock(text=_TEXT),
            ThinkingBlock(text=_TEXT),
            ImageRefBlock(blob_ref="blob-1"),
        ],
        reasoning_content=_TEXT,
    )


def test_thinking_and_image_blocks_cost_tokens() -> None:
    rc = LoopConstants()
    text_only = estimate_message_tokens(_text_only(), rc)
    enriched = estimate_message_tokens(_with_thinking_and_image(), rc)

    assert text_only > 0
    # The image is priced at the flat constant; the thinking block and the
    # re-emitted reasoning are each priced as their own text.
    assert enriched == text_only * 3 + rc.token_count_image_tokens
    assert enriched > text_only


def test_the_folded_raw_size_is_read_on_the_shared_scale() -> None:
    """``cumulative_raw_tokens`` — what the lazy-fold gate reads — moved too."""
    rc = LoopConstants()
    base = SessionMemory()

    text_run = [_text_only()]
    rich_run = [_with_thinking_and_image()]

    text_folded = fold_run(base, text_run, "summary", rc).memory
    rich_folded = fold_run(base, rich_run, "summary", rc).memory

    assert text_folded.cumulative_raw_tokens == estimate_history_tokens(text_run, rc)
    assert rich_folded.cumulative_raw_tokens == estimate_history_tokens(rich_run, rc)
    # The discriminating comparison: the same visible text costs strictly more
    # once thinking / image / reasoning are carried alongside it.
    assert rich_folded.cumulative_raw_tokens > text_folded.cumulative_raw_tokens
