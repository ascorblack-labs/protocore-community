"""Two-tier compaction state + Tier 1 / Tier 2 implementations.

Tier 1 — tool-result truncation:
 Replace tool-result content > ``tool_result_truncation_threshold`` with
 the canonical ``PROTOCOL_COMPACTED_TOOL_RESULT_V1:SNAPSHOT`` placeholder
 (renderer = :mod:`protocore.runtime.wire_format`). Full payload stored
 in :class:`IBlobStore`. Loop oldest-first; stop when enough freed.

Tier 2 — old-turn summarisation:
 For turns older than ``compaction_keep_recent_turns``, call
 :meth:`ILLMProvider.complete_structured` with :func:`build_summary_schema`
 and replace the turn with a system message containing the summary. Strip
 injection patterns before sending to the summariser.

Cyrillic-in-JSON-escape safety preserved via
:mod:`protocore.runtime.token_counting`.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Final

from protocore.contracts.blob import IBlobStore
from protocore.contracts.llm import ILLMProvider, LLMObservabilityContext, LLMRequest
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import (
    COMPACTION_REFERENCE_METADATA_KEY,
    COMPACTION_SUMMARY_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
    CompactionSourceRef,
    ContentBlock,
    ImageRefBlock,
    Message,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.logging_utils import get_logger
from protocore.runtime.token_counting import estimate_tokens
from protocore.runtime.wire_format import (
    is_compacted_placeholder,
    render_compacted_placeholder,
)

_logger = get_logger(__name__)


# Anti-injection patterns applied ONLY to content sent to the summariser, never to user-visible text.
_INJECTION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?i)ignore (?:previous|prior|all|above) instructions?"),
    re.compile(r"(?i)you are now"),
    re.compile(r"(?i)(?:^|\n)system:\s*"),
    re.compile(r"---\s*END OF CONVERSATION\s*---"),
    re.compile(r"(?i)return (?:exactly|only) this json"),
    re.compile(r"(?i)disregard (?:the |all )?(?:above|previous)"),
)

_INJECTION_REPLACEMENT: Final[str] = "[REDACTED-INJECTION-PATTERN]"


def build_summary_schema(rc: RuntimeConstants) -> dict[str, Any]:
    """Build the summariser JSON schema with the RC-driven ``maxLength``.

    The ``summary.maxLength`` cap is sourced from
    ``rc.compaction_summary_string_max_chars`` so dashboard tuning takes
    effect without a code change (no inline magic numbers). XGrammar enforces
    this at decode time.

    One field, because a declared field is not free. Under a response schema a
    model fills every field it is given whether or not the source material
    supports one, so a field nothing reads is paid for twice: once in the output
    budget (``compaction_summary_max_output_tokens``, which this shares with the
    summary that actually survives) and again in the risk that the model invents
    plausible content to fill it. This schema carried two such arrays of up to
    twenty strings each; nothing ever extracted them.
    """
    return {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "minLength": 1,
                "maxLength": rc.compaction_summary_string_max_chars,
            },
        },
        "required": ["summary"],
        "additionalProperties": False,
    }


class CompactionExhaustedError(RuntimeError):
    """Compaction failed beyond :attr:`RuntimeConstants.compaction_failed_max_retries`."""


@dataclass(frozen=True, slots=True)
class Tier1Result:
    """Outcome of Tier 1 — tool-result truncation."""

    tokens_freed: int
    blob_refs_created: tuple[str, ...]
    messages_modified: int


@dataclass(frozen=True, slots=True)
class Tier2Result:
    """Outcome of Tier 2 — old-turn summarisation."""

    turns_summarised: int
    tokens_freed: int


@dataclass(slots=True)
class CompactionAttempt:
    """One compaction attempt's combined outcome.

    Used by :class:`QueryEngine` to emit ``compaction_started`` /
    ``compaction_completed`` event payloads with full provenance.
    """

    tier1: Tier1Result | None = None
    tier2: Tier2Result | None = None
    tokens_before: int = 0
    tokens_after: int = 0


@dataclass(slots=True)
class CompactionState:
    """Per-engine compaction state — counts retries + tracks summarised IDs."""

    retry_count: int = 0
    summarised_turn_ids: set[str] = field(default_factory=set)
    blob_refs_created: list[str] = field(default_factory=list)

    def reset_retries(self) -> None:
        self.retry_count = 0


def _strip_injection_patterns(text: str) -> str:
    """Redact known injection patterns before sending to the summariser."""
    if not text:
        return text
    redacted = text
    for pattern in _INJECTION_PATTERNS:
        redacted = pattern.sub(_INJECTION_REPLACEMENT, redacted)
    return redacted


def _block_text_for_estimation(block: ContentBlock) -> str:
    """Return the estimation/summariser text for one content block.

 Exhaustive per :data:`~protocore.contracts.types.ContentBlock` kind so no
 block is ever silently dropped. Mirrors the reference rough estimator:

 * :class:`TextBlock` → ``text``
 * :class:`ThinkingBlock` → ``text``
 * :class:`ToolUseBlock` → ``name`` + ``arguments_json`` (the model-generated
 tool-call payload the provider re-serializes on the wire — )
 * :class:`ToolResultBlock` → ``content``
 * :class:`ImageRefBlock` → a compact serialized marker (image token weight
 is added separately as a flat constant by :func:`estimate_message_tokens`;
 this only gives the summariser a textual breadcrumb)
 * any other / future kind → ``model_dump_json`` serialized form so it is
 never counted as 0 (catch-all).
 """
    if isinstance(block, TextBlock):
        return block.text
    if isinstance(block, ThinkingBlock):
        return block.text
    if isinstance(block, ToolUseBlock):
        return f"{block.name}{block.arguments_json}"
    if isinstance(block, ToolResultBlock):
        return block.content
    if isinstance(block, ImageRefBlock):
        return f"[image:{block.mime_type}:{block.blob_ref}]"
    # Defensive catch-all for any future ContentBlock kind — serialize so the
    # estimate is never silently zero.
    return block.model_dump_json()


def _message_text_for_estimation(message: Message, rc: RuntimeConstants) -> str:
    """Return concatenated text content used for token estimation + summarising.

 Exhaustive across every content block kind PLUS
 :attr:`Message.reasoning_content` (re-emitted chain-of-thought for
 thinking-capable providers — ). The ``rc`` parameter is accepted for
 signature symmetry with :func:`estimate_message_tokens` and forward
 compatibility (per-kind text shaping may become RC-tunable); it is not used
 for plain text assembly today.
 """
    _ = rc  # reserved for future RC-tunable text shaping; keeps the call sites aligned
    parts: list[str] = [_block_text_for_estimation(block) for block in message.content_blocks]
    if message.reasoning_content:
        parts.append(message.reasoning_content)
    return "\n".join(part for part in parts if part)


def estimate_message_tokens(message: Message, rc: RuntimeConstants) -> int:
    """Estimate the token weight of a single :class:`Message` exhaustively.

 The single source of truth for the cheap pre-flight estimate, shared by
 :func:`protocore.runtime.context.manager.estimate_history_tokens` and the
 Tier-2 freed-token accounting below. Every content block contributes:

 * text-bearing blocks (text / thinking / tool_use / tool_result / unknown)
 via :func:`~protocore.runtime.token_counting.estimate_tokens` on their
 extracted text (tool_use args no longer count as 0);
 * :class:`ImageRefBlock` via the flat
 :attr:`RuntimeConstants.token_count_image_tokens` constant — image blocks
 carry only a blob ref, so a size-derived estimate is impossible ;
 * :attr:`Message.reasoning_content` via ``estimate_tokens`` .
 """
    total = 0
    for block in message.content_blocks:
        if isinstance(block, ImageRefBlock):
            total += rc.token_count_image_tokens
            continue
        total += estimate_tokens(_block_text_for_estimation(block), rc)
    if message.reasoning_content:
        total += estimate_tokens(message.reasoning_content, rc)
    return total


def _content_is_already_compacted(text: str) -> bool:
    return is_compacted_placeholder(text)


def _stable_turn_key(message: Message) -> str:
    """Return a DURABLE dedup key for one turn.

    The prior key was ``str(id(message))`` — Python object identity, which
    is reborn on every ``Message.model_validate`` (snapshot/resume), so the
    persisted ``summarised_turn_ids`` set matched nothing after a resume and
    every resume re-summarised the same old turns (churn + summary-of-summary
    decay + cross-pod non-determinism).

 This key is a SHA-256 over the message role + its canonical content text +
 the per-block ``tool_call_id`` of every ``tool_use`` / ``tool_result`` block,
 so it is identical for the same logical turn across processes/pods and
 stable across snapshot round-trips. It does NOT include ``created_at`` /
 ``metadata`` (which can drift) — only the wire-relevant content the
 summariser would consume.

 Why the ``tool_call_id`` is part of the key (the "collision is
 safe" claim was WRONG for one by-construction case): two DISTINCT turns can
 carry byte-identical content text. A model that re-emits the same tool call
e.g. the documented content-missing Write-retry spiral — produces the
 SAME ``name`` + ``arguments_json`` each time (``_block_text_for_estimation``
 folds neither the id) and, after Tier-1 sheds ``reasoning_content``, even
 aged copies converge further. Without the id in the key, every later
 identical turn collides with the first one's entry in
 ``state.summarised_turn_ids`` and is silently skipped by the anchor-skip
 guard in :func:`run_tier2_summarisation` — so its whole unit (the assistant
 ``tool_use`` turn AND its tool results) is never summarised and never
 dropped, on every pass, persistently. Tier-2 then frees almost nothing
 beyond the first copy and compaction can abort the run via
 :class:`CompactionExhaustedError` where summarising the duplicates would
 have freed the space. The provider assigns a fresh ``tool_call_id`` per
 emission, and that id is a persisted, snapshot-stable, cross-pod-deterministic
 field — so folding it in disambiguates distinct spiral turns while keeping
 the SAME turn's key identical across snapshot/resume (the A4 invariant
 below). Pure-text turns carry no ``tool_call_id`` and are unaffected.

 Invariant (why ``reasoning_content`` in the key is safe despite
 Tier-1 shedding it): the digest folds in ``reasoning_content`` (line
 below), and Tier-1's reasoning-shed (:func:`run_tier1_truncation`)
 replaces an aged assistant turn with a ``reasoning_content=None`` copy —
 which would change this key. The theoretical drift (a turn summarised with
 a "with-reasoning" key in an earlier pass, later shed, then re-checked with
 a "without-reasoning" key) is NON-REACHABLE by construction: once Tier-2
 summarises a turn it REPLACES that turn in-place with a system summary
 message. The original ``reasoning_content``-bearing assistant turn no
 longer exists in history, so Tier-1 can never shed it afterwards, and on
 the next pass the replacement is caught by :func:`_is_compaction_summary`
 (early skip) BEFORE this key is ever computed for it. The exact hash order
 of Tier-1-vs-Tier-2 is therefore irrelevant for already-summarised turns.
 This is load-bearing: a future change to the Tier-2 in-place-replacement
 pattern (e.g. keeping the original turn alongside the summary) would break
 the dedup invariant silently.
 """
    digest = hashlib.sha256()
    digest.update(message.role.value.encode("utf-8"))
    digest.update(b"\x00")
    for block in message.content_blocks:
        digest.update(_block_text_for_estimation(block).encode("utf-8"))
        digest.update(b"\x00")
        # distinguish DISTINCT turns that share byte-identical content
        # (the model re-emitting the same tool call with a fresh tool_call_id).
        # The id is a persisted, snapshot-stable, cross-pod-deterministic field;
        # text-only blocks carry none and are unaffected.
        if isinstance(block, (ToolUseBlock, ToolResultBlock)):
            digest.update(block.tool_call_id.encode("utf-8"))
            digest.update(b"\x00")
    if message.reasoning_content:
        digest.update(message.reasoning_content.encode("utf-8"))
    return digest.hexdigest()


def current_tool_batch_protect_index(history: list[Message]) -> int | None:
    """Return the index of the most recent assistant ``tool_use`` turn, or None.

    The per-iteration compaction gate (:mod:`protocore.runtime.query`) fires
    AFTER all tool
    results from the current assistant turn's just-executed batch have been
    appended to ``history``. Those results are freshly produced and the model
    has NOT yet consumed them. The ``compaction_keep_recent_turns`` window
    (default 4) only protects the trailing N messages, so a parallel batch of
    more than ``keep`` tool calls leaves the 5th-from-last (and earlier) fresh
    result inside the eligible zone — Tier-1 can blob it to a placeholder and
    Tier-2 can summarise it away in the SAME iteration, before the next
    assistant stream ever sees it.

    The protection point is the lowest index of the CURRENT batch: the most
    recent assistant turn that emits a :class:`ToolUseBlock`. Everything from
    that index to the end of history (the assistant tool_use turn plus every
    tool-result message answering it, regardless of batch size) is the
    in-flight, unconsumed batch and must be exempt from compaction this
    iteration. Returns ``None`` when no assistant ``tool_use`` turn exists
    (no batch to protect — e.g. a plain text turn), in which case callers fall
    back to the unmodified ``keep_recent_turns`` window.

    Only the per-iteration gate passes this; the turn-start gate keeps the
    pre-existing keep-window-only behaviour (no in-flight batch exists when a
    fresh turn begins).
    """
    for idx in range(len(history) - 1, -1, -1):
        message = history[idx]
        if message.role is MessageRole.assistant and any(
            isinstance(block, ToolUseBlock) for block in message.content_blocks
        ):
            return idx
    return None


def _effective_eligible_upper(
    history: list[Message],
    keep: int,
    protect_tail_from_index: int | None,
) -> int:
    """Compute ``eligible_upper`` honouring keep-window + current-batch guard.

    The base is ``max(0, len(history) - keep)`` (the trailing keep-window is
    never eligible). When ``protect_tail_from_index`` is set (per-iteration
    gate), the eligible region is additionally clamped so that
    NO message at or after that index is eligible — protecting the current
    just-executed tool-result batch on top of the keep window.
    """
    eligible_upper = max(0, len(history) - keep)
    if protect_tail_from_index is not None:
        eligible_upper = min(eligible_upper, max(0, protect_tail_from_index))
    return eligible_upper


def _is_compaction_summary(message: Message) -> bool:
    """Return ``True`` if ``message`` is an already-produced Tier-2 summary.

    Recognised by the durable ``COMPACTION_SUMMARY_METADATA_KEY`` flag (set
    on every summary this module produces) OR — defensively, for summaries
    produced before the flag existed — by a content body that is a
    ``<compacted-turn ...>`` wrapper, OR — for LEGACY persisted snapshots —
    by ``role is MessageRole.system`` (the role this module used for summaries
    before the vLLM-400 fix flipped them to ``MessageRole.user``; such
    snapshots may still rehydrate into the eligible region and MUST keep being
    recognised as already-compacted). Used to skip re-summarising a summary
    (idempotency under the per-iteration gate) and so the
    unit-builder never folds a summary into a new component.
    """
    if message.metadata.get(COMPACTION_SUMMARY_METADATA_KEY) is True:
        return True
    if message.role is MessageRole.system:
        return True
    text = message.text.strip()
    return text.startswith("<compacted-turn")


def _content_preview(text: str, max_chars: int) -> str:
    """Head/tail excerpt of ``text`` capped at ``max_chars``.

    Keeps a head and tail window (most-informative ends of a tool result —
    headers + final lines) joined by an ellipsis when the content is longer
    than the cap. Returns the full text when it already fits, and ``""`` when
    the cap is 0 (preview disabled). Newlines are collapsed to spaces so the
    preview stays a single readable line inside the pipe-delimited placeholder
    (the wire renderer base64-encodes it regardless, so this is purely for
    readability once decoded).
    """
    if max_chars <= 0 or not text:
        return ""
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    head_len = max_chars // 2
    tail_len = max_chars - head_len
    return f"{flat[:head_len]}…{flat[-tail_len:]}"


def _tool_name_by_call_id(history: list[Message]) -> dict[str, str]:
    """Map each ``tool_call_id`` to its originating ``ToolUseBlock.name``.

    Used by Tier-1 to enrich a compacted tool-result
    placeholder with the name of the tool that produced it, so the model
    knows what was shed and can re-fetch it. A result whose originator is no
    longer in history (already compacted/summarised away) maps to ``""``.
    """
    names: dict[str, str] = {}
    for message in history:
        if message.role is not MessageRole.assistant:
            continue
        for block in message.content_blocks:
            if isinstance(block, ToolUseBlock) and block.tool_call_id not in names:
                names[block.tool_call_id] = block.name
    return names


async def run_tier1_truncation(
    history: list[Message],
    blob_store: IBlobStore,
    tenant_id: str,
    rc: RuntimeConstants,
    truncation_threshold_tokens: int,
    *,
    keep_recent_turns: int | None = None,
    protect_tail_from_index: int | None = None,
) -> Tier1Result:
    """Replace large tool-result blocks with blobbed wire-format placeholders.

    Mutates ``history`` in place — oldest first. Stops when nothing further
    can be freed (i.e. all remaining tool_results are below threshold or
    already compacted).

    Additional compaction passes (both RC-gated, default-on):

    * ``compaction_shed_reasoning_enabled`` — strip ``reasoning_content`` from
      assistant turns older than ``keep_recent_turns``. Re-emitted CoT is
      single-turn scaffolding the model never needs from prior turns and is
      otherwise uncompactable bloat on a small window.
    * ``compaction_bound_reference_blocks_enabled`` — compact an OVER-BUDGET
      frozen reference block (a non-tool single-text message tagged
      ``COMPACTION_REFERENCE_METADATA_KEY``, e.g. the executor's
      ``<environment_context>``/``<memory-context>`` bootstrap) to a blobbed
      placeholder, the same way large tool results are shed. The original task
      user turn is never tagged, so it is never shed here.

    The tool-result placeholder is enriched with the originating tool name +
    a short head/tail preview (RC-bounded by
    ``compaction_placeholder_preview_chars``) so a compacted result is
    recoverable rather than an information dead-end.

    Args:
        history: full message list (mutated in place).
        blob_store: target durable store for the original content.
        tenant_id: scoping for blob refs.
        rc: token-counting + RC fields.
        truncation_threshold_tokens: tool_result tokens above this get blobbed.
        keep_recent_turns: trailing turns to skip (anchor caching). ``None``
            defaults to :attr:`RuntimeConstants.compaction_keep_recent_turns`.
        protect_tail_from_index: When set, NO message at or after this index is
            eligible — protects the current iteration's just-executed
            tool-result batch (any batch size) on top of ``keep_recent_turns``.
            ``None`` keeps the keep-window-only behaviour (turn-start gate).
            See :func:`current_tool_batch_protect_index`.

    Returns:
        :class:`Tier1Result` with stats for telemetry.
    """
    if not history:
        return Tier1Result(tokens_freed=0, blob_refs_created=(), messages_modified=0)

    keep = rc.compaction_keep_recent_turns if keep_recent_turns is None else keep_recent_turns
    eligible_upper = _effective_eligible_upper(history, keep, protect_tail_from_index)

    preview_cap = rc.compaction_placeholder_preview_chars
    tool_names = _tool_name_by_call_id(history)

    tokens_freed = 0
    refs_created: list[str] = []
    modified = 0

    for idx in range(eligible_upper):
        message = history[idx]

        # ── A2(1) — shed re-emitted reasoning_content on aged assistant turns.
        # Free the tokens BEFORE the tool-result branch's continue so a
        # tool-pairing assistant turn with bloated CoT is still trimmed even
        # when its tool result is the thing being blobbed.
        if (
            rc.compaction_shed_reasoning_enabled
            and message.role is MessageRole.assistant
            and message.reasoning_content
        ):
            freed = estimate_tokens(message.reasoning_content, rc)
            if freed > 0:
                history[idx] = message.model_copy(update={"reasoning_content": None})
                message = history[idx]
                tokens_freed += freed
                modified += 1

        if not message.content_blocks:
            continue

        block = message.content_blocks[0]

        # ── A2(2) — bound an over-budget FROZEN reference block (bootstrap).
        if (
            rc.compaction_bound_reference_blocks_enabled
            and message.role is not MessageRole.tool
            and message.metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is True
            and isinstance(block, TextBlock)
            and not _content_is_already_compacted(block.text)
        ):
            ref_tokens = estimate_tokens(block.text, rc)
            if ref_tokens >= truncation_threshold_tokens:
                ref_bytes = block.text.encode("utf-8")
                ref_sha = hashlib.sha256(ref_bytes).hexdigest()
                ref_blob = await blob_store.put(
                    tenant_id=tenant_id,
                    content=ref_bytes,
                    content_type="text/plain; charset=utf-8",
                    metadata={"label": "reference_block", "tier": "tier1"},
                )
                ref_placeholder = render_compacted_placeholder(
                    CompactionSourceRef(
                        blob_ref=ref_blob.ref,
                        sha256=ref_sha,
                        original_tokens=ref_tokens,
                        label="reference_block",
                        tool_name="",
                        preview=_content_preview(block.text, preview_cap),
                    ),
                    "SNAPSHOT",
                )
                history[idx] = message.model_copy(
                    update={
                        "content_blocks": [TextBlock(text=ref_placeholder)],
                        "metadata": {**message.metadata, "compacted": True, "blob_ref": ref_blob.ref},
                    }
                )
                modified += 1
                refs_created.append(ref_blob.ref)
                tokens_freed += ref_tokens - estimate_tokens(ref_placeholder, rc)
            continue

        if message.role is not MessageRole.tool:
            continue
        # a tool-role message may carry MORE THAN ONE ToolResultBlock
        # — the ``Message`` validator caps only system/user at one block
        # (types.py comment that claims tool is capped is wrong), and
        # ``_build_summarisation_units`` explicitly models "a single tool-role
        # message may answer more than one assistant tool_use turn". Iterate
        # EVERY block, blob each over-threshold ToolResultBlock, and only
        # replace the message when at least one block was shed — the prior
        # code took ``block = message.content_blocks[0]`` and rebuilt with
        # ``content_blocks=[new_block]``, which (a) left blocks [1:] as
        # permanent uncompactable bloat and (b) DROPPED non-ToolResultBlock
        # siblings, corrupting the message.
        new_blocks: list[ContentBlock] = []
        msg_modified = False
        for block in message.content_blocks:
            if not isinstance(block, ToolResultBlock):
                new_blocks.append(block)
                continue
            if _content_is_already_compacted(block.content):
                new_blocks.append(block)
                continue
            original_tokens = estimate_tokens(block.content, rc)
            if original_tokens < truncation_threshold_tokens:
                new_blocks.append(block)
                continue

            content_bytes = block.content.encode("utf-8")
            sha256 = hashlib.sha256(content_bytes).hexdigest()
            blob_md = await blob_store.put(
                tenant_id=tenant_id,
                content=content_bytes,
                content_type="text/plain; charset=utf-8",
                metadata={
                    "tool_call_id": block.tool_call_id,
                    "label": "tool_result",
                    "tier": "tier1",
                },
            )

            placeholder = render_compacted_placeholder(
                CompactionSourceRef(
                    blob_ref=blob_md.ref,
                    sha256=sha256,
                    original_tokens=original_tokens,
                    label="tool_result",
                    tool_name=tool_names.get(block.tool_call_id, ""),
                    preview=_content_preview(block.content, preview_cap),
                ),
                "SNAPSHOT",
            )

            new_blocks.append(
                ToolResultBlock(
                    tool_call_id=block.tool_call_id,
                    content=placeholder,
                    is_error=block.is_error,
                    metadata={**block.metadata, "compacted": True, "blob_ref": blob_md.ref},
                )
            )
            msg_modified = True
            refs_created.append(blob_md.ref)
            tokens_freed += original_tokens - estimate_tokens(placeholder, rc)

        if msg_modified:
            history[idx] = message.model_copy(update={"content_blocks": new_blocks})
            modified += 1

    return Tier1Result(
        tokens_freed=max(0, tokens_freed),
        blob_refs_created=tuple(refs_created),
        messages_modified=modified,
    )


def _tool_use_ids(message: Message) -> tuple[str, ...]:
    """Return the tool_call_ids of every :class:`ToolUseBlock` on ``message``."""
    return tuple(
        block.tool_call_id
        for block in message.content_blocks
        if isinstance(block, ToolUseBlock)
    )


def _tool_result_ids(message: Message) -> tuple[str, ...]:
    """Return the tool_call_ids of every :class:`ToolResultBlock` on ``message``."""
    return tuple(
        block.tool_call_id
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    )


@dataclass(slots=True)
class _SummarisationUnit:
    """One atomic compaction unit .

    ``indices`` are the history positions that MUST be summarised/skipped
    together — a tool-pairing connected component: every assistant ``tool_use``
    turn and every tool-role ``tool_result`` message that share a
    ``tool_call_id`` (transitively, e.g. parallel calls answered by a shared
    tool message) belong to the same component. ``anchor_idx`` is the lowest
    assistant index in the component; it becomes the system summary and ALL
    other members (other assistant turns + every matching tool-result message)
    are removed, so no side of any pair is ever orphaned. A plain turn (no tool
    blocks) is a singleton component.
    """

    anchor_idx: int
    indices: tuple[int, ...]


def _first_user_turn_index(history: list[Message]) -> int | None:
    """Return the index of the FIRST user-role turn (the original task), or None.

    This turn carries the verbatim task + constraints and must survive every
    compaction once the per-iteration gate makes compaction fire often. Skips
    runtime-prepended reference blocks (``COMPACTION_REFERENCE_METADATA_KEY``)
    — those are bootstrap context, not the user's task, and Tier-1 may shed
    them.

    Also skips PRIOR-RUN user turns the executor seeded into history
    (``SESSION_HISTORY_SEED_METADATA_KEY``). Those precede the new task in
    history, so without this skip the "first user turn" would be a seeded
    prior-run turn and the protect-first-user-turn guard would shield the wrong
    message, leaving the NEW task summarisable. With the skip, the guard
    correctly protects the new task — the first user turn that is neither a
    reference block nor a seeded prior-run turn.
    """
    for idx, message in enumerate(history):
        if (
            message.role is MessageRole.user
            and message.metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is not True
            and message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is not True
            # vLLM-400 fix: Tier-2 summaries are now USER-role. A summary is
            # never the user's verbatim task, so it must not be picked as the
            # protect-first-user-turn target.
            and not _is_compaction_summary(message)
        ):
            return idx
    return None


def _session_history_seed_indices(history: list[Message]) -> frozenset[int]:
    """Return the history indices of executor-seeded prior-run turns.

    These are protected from Tier-2 summarisation so a lossy summary never
    silently collapses seeded prior-run content into an UNtagged
    ``<compacted-turn>`` system message — which would (a) defeat the host
    finalization filter that excludes seed-tagged turns from re-persistence and
    (b) re-write prior-run conversation under the new ``run_id``.

    Tier-1 still bounds these turns under budget pressure, by TWO mechanisms
    (both preserve the seed tag because ``model_copy`` keeps ``metadata``):

    * seeded ``role=tool`` results are blobbed by the Tier-1 main shed path
      (large seeded tool results → SNAPSHOT placeholders);
    * the synthetic running-summary + artifact-ledger seed blocks are
      ``role=user`` ``TextBlock`` messages that DUAL-TAG
      :data:`SESSION_HISTORY_SEED_METADATA_KEY` AND
      :data:`COMPACTION_REFERENCE_METADATA_KEY` (see
      ``session_memory._tag_seeded_reference``), so the Tier-1 A2(2)
      reference-block path blobs THEM to a recoverable placeholder when they
      exceed the truncation threshold. Without the reference dual-tag these
      large user-text blocks had NO shed path and were permanently immovable.

    Only the lossy Tier-2 collapse is withheld from ALL seed-tagged turns; the
    reference dual-tag does not weaken that (Tier-2 still skips them by seed tag).
    """
    return frozenset(
        idx
        for idx, message in enumerate(history)
        if message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True
    )


def _compaction_reference_indices(history: list[Message]) -> frozenset[int]:
    """Return the history indices of frozen reference blocks (bootstrap context).

    Reference blocks (``COMPACTION_REFERENCE_METADATA_KEY``, e.g. the
    executor's ``<environment_context>``/``<memory-context>`` bootstrap) are a
    Tier-1-ONLY shed surface: the A2(2) path blobs an over-budget block to a
    RECOVERABLE ``PROTOCOL_COMPACTED…SNAPSHOT`` placeholder (the blob ref stays
    in history). They are protected from the lossy Tier-2 collapse in BOTH
    states:

    * un-blobbed (small) — a Tier-2 summary would lossily collapse frozen
      bootstrap context that the Tier-1-only shed design deliberately keeps
      verbatim until it is over budget;
    * blobbed — the placeholder message is the ONLY in-history pointer to the
      blob. Summarising it sends the raw placeholder marker to the summariser
      (one wasted LLM call) and replaces the message, erasing the blob ref and
      breaking the A2(2) recoverable-snapshot contract.

    The tag survives the A2(2) blob-shed (`run_tier1_truncation` spreads the
    existing ``message.metadata`` into the placeholder copy), so this single
    tag check covers the placeholder state too.
    """
    return frozenset(
        idx
        for idx, message in enumerate(history)
        if message.metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is True
    )


def _build_summarisation_units(
    history: list[Message],
    eligible_upper: int,
    *,
    protected_indices: frozenset[int] = frozenset(),
) -> list[_SummarisationUnit]:
    """Partition the eligible region into atomic tool-pairing units .

    ``protected_indices`` are history positions that must never be summarised
    — any component containing one is skipped wholesale (so the original task
    user turn stays verbatim and no tool-pairing partner is half-dropped).

    Pairing is computed as CONNECTED COMPONENTS over the message graph so the
    rewrite can never half-drop a pair: a tool_call_id may be
    answered by more than one tool-role message, and a single tool-role message
    may answer more than one assistant ``tool_use`` turn (parallel calls). Two
    message indices are in the same component when they share any
    ``tool_call_id`` (assistant ``tool_use`` ↔ tool-role ``tool_result``).

    Rules:

    * A component is summarised/skipped atomically. It is eligible ONLY if EVERY
      member index is inside the eligible region (``< eligible_upper``); if any
      member is anchored in the kept-recent tail, the whole component is skipped
      — never half-summarised.
    * The anchor (summary target) is the LOWEST assistant index in the
      component; every other member (other assistant turns + ALL matching
      tool-result messages) is dropped. A component with NO assistant member
      (orphan tool results whose originator is not in history, e.g. already
      compacted away) is left untouched — we never synthesise or strip a bare
      result here.
    * Already-compacted summaries (``_is_compaction_summary``) are skipped and
      never join a component. This matches both new USER-role summaries (the
      vLLM-400 fix: a mid-history ``MessageRole.system`` 400s vLLM) and LEGACY
      ``MessageRole.system`` summaries rehydrated from persisted snapshots.
    """
    # Map every tool_call_id -> the tool-role result message indices answering
    # it (a list, NOT first-writer-wins: duplicates must all be grouped).
    result_indices_by_call: dict[str, list[int]] = {}
    # Map every tool_call_id -> the assistant message indices that emit it.
    tool_use_indices_by_call: dict[str, list[int]] = {}
    for idx in range(len(history)):
        msg = history[idx]
        if _is_compaction_summary(msg):
            continue
        if msg.role is MessageRole.tool:
            for call_id in _tool_result_ids(msg):
                result_indices_by_call.setdefault(call_id, []).append(idx)
        elif msg.role is MessageRole.assistant:
            for call_id in _tool_use_ids(msg):
                tool_use_indices_by_call.setdefault(call_id, []).append(idx)

    # Union-find over message indices linked by a shared tool_call_id.
    parent: dict[int, int] = {}

    def _find(i: int) -> int:
        parent.setdefault(i, i)
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Seed every non-summary message index as its own node.
    for idx in range(len(history)):
        if not _is_compaction_summary(history[idx]):
            _find(idx)

    # Link assistant tool_use turns with the tool-role results that answer them.
    for call_id, use_indices in tool_use_indices_by_call.items():
        members = list(use_indices) + result_indices_by_call.get(call_id, [])
        for other in members[1:]:
            _union(members[0], other)
    # A duplicated result with no in-history originator still links its own
    # result messages together so they are treated as one (orphan) component.
    for call_id, res_indices in result_indices_by_call.items():
        if call_id in tool_use_indices_by_call:
            continue
        for other in res_indices[1:]:
            _union(res_indices[0], other)

    components: dict[int, list[int]] = {}
    for idx in range(len(history)):
        if _is_compaction_summary(history[idx]):
            continue
        components.setdefault(_find(idx), []).append(idx)

    units: list[_SummarisationUnit] = []
    for member_indices in components.values():
        member_indices.sort()
        # Atomicity: the whole component must live inside the eligible region.
        if any(member >= eligible_upper for member in member_indices):
            continue
        # A component touching a protected index (the original task user turn)
        # is skipped wholesale so the verbatim task survives and no
        # tool-pairing partner is half-dropped.
        if protected_indices and any(member in protected_indices for member in member_indices):
            continue
        assistant_members = [
            i for i in member_indices if history[i].role is MessageRole.assistant
        ]
        if not assistant_members:
            # Orphan tool result(s) / pure-non-assistant component. A standalone
            # plain user turn is its own component and IS summarisable; a bare
            # tool result with no originator must be left intact.
            non_tool_members = [
                i for i in member_indices if history[i].role is not MessageRole.tool
            ]
            if not non_tool_members:
                continue
            anchor_idx = non_tool_members[0]
        else:
            anchor_idx = assistant_members[0]
        units.append(
            _SummarisationUnit(anchor_idx=anchor_idx, indices=tuple(member_indices))
        )

    # Deterministic order: summarise components by their anchor position.
    units.sort(key=lambda u: u.anchor_idx)
    return units


def _wrap_compaction_summary(anchor_key: str, summary_text: str) -> str:
    """Build the ``<compacted-turn>`` replacement body for a summarised unit.

    Single source of truth for the wrapper so the no-net-gain floor
    (:func:`_compaction_wrapper_floor_tokens`) and the actual replacement stay
    byte-for-byte in sync.
    """
    return f"<compacted-turn id='{anchor_key}'>{summary_text}</compacted-turn>"


def _compaction_wrapper_floor_tokens(anchor_key: str, rc: RuntimeConstants) -> int:
    """Estimated token weight of an EMPTY ``<compacted-turn>`` wrapper.

    A unit can only shrink under Tier-2 if its current token estimate is
    strictly above this floor — the minimum size the replacement can ever be,
    reached when the summariser returns an empty body. A unit at or below the
    floor cannot be made smaller by summarising it; replacing it would GROW
    history (measured: a 1-token turn becomes a ~39-token wrapper) while a
    ``max(0, ...)`` clamp would hide that growth from ``tokens_freed``. So such
    units are skipped before any LLM call — this both prevents the inflation
    (which can push ``run_compaction``'s ``tokens_after`` above
    ``tokens_before`` and count a no-progress retry toward
    :class:`CompactionExhaustedError`) and avoids spending a ~5-11s summariser
    call that cannot help.
    """
    return estimate_tokens(_wrap_compaction_summary(anchor_key, ""), rc)


async def run_tier2_summarisation(
    history: list[Message],
    compaction_llm: ILLMProvider,
    state: CompactionState,
    rc: RuntimeConstants,
    *,
    model_name: str,
    observability: LLMObservabilityContext | None = None,
    protect_tail_from_index: int | None = None,
    free_target_tokens: int | None = None,
) -> Tier2Result:
    """Summarise old turns via the compaction LLM.

 For each ATOMIC unit older than ``rc.compaction_keep_recent_turns`` that has
 not yet been summarised, call
 :meth:`ILLMProvider.complete_structured` with the summary schema and replace
 the unit's anchor turn in-place with a system message wrapping the summary.

 tool pairing is atomic: an assistant ``tool_use`` turn and the
 tool-role ``tool_result`` message(s) that answer it are summarised (the
 assistant turn becomes the summary, the result messages are removed) or
 skipped together. The function never replaces a tool-role result while
 leaving its originating ``ToolUseBlock`` (or vice versa), so the rewritten
 history always satisfies tool_use ↔ tool_result pairing and the next
 provider call cannot 400 on an orphan.

 Anti-injection: every turn body is stripped of
 :data:`_INJECTION_PATTERNS` before being included in the prompt.

 The dedup key is a DURABLE content hash (:func:`_stable_turn_key`), not
 ``str(id(obj))``, so a turn already summarised before a snapshot/resume is
 recognised after rehydration and is NOT re-summarised (no churn, no
 summary-of-summary decay, deterministic across pods). The produced summary
 system message is tagged with :data:`COMPACTION_SUMMARY_METADATA_KEY` and
 an already-summary anchor is
 skipped, so the per-iteration gate (A1) is idempotent.

 The original task user turn is protected from summarisation when
 ``rc.compaction_protect_first_user_turn`` is set.

 When ``protect_tail_from_index`` is set (the per-iteration gate), the
 current iteration's just-executed tool batch (assistant ``tool_use`` turn +
 its results) is exempt from summarisation on top of the keep window, so a
 >keep parallel batch's fresh results cannot be summarised away before the
 next assistant stream consumes them. See
 :func:`current_tool_batch_protect_index`.

 No-net-gain floor — a unit whose combined token estimate is at or below
 :func:`_compaction_wrapper_floor_tokens` (the empty ``<compacted-turn>``
 wrapper) cannot shrink under summarisation; replacing it would only GROW
 history. Such units are skipped before any LLM call, so Tier-2 never inflates
 a small-turn-dominated history (which would push ``tokens_after`` above
 ``tokens_before`` and miscount a no-progress retry toward
 :class:`CompactionExhaustedError`) and never spends a summariser call that
 cannot free tokens.

 Bounded per-pass cost — ``free_target_tokens`` (when set by the caller) is
 the freed-token budget for this pass: once that many tokens have been freed
 the loop STOPS issuing further summariser calls. A many-turn history therefore
 no longer triggers dozens of sequential ~5-11s LLM calls in one
 ``COMPACTING`` pass; the remaining eligible units are summarised on a later
 pass if still needed.

 Mutates ``history`` in place.
 """
    if not history:
        return Tier2Result(turns_summarised=0, tokens_freed=0)

    keep = rc.compaction_keep_recent_turns
    eligible_upper = _effective_eligible_upper(history, keep, protect_tail_from_index)
    if eligible_upper == 0:
        return Tier2Result(turns_summarised=0, tokens_freed=0)

    protected: frozenset[int] = frozenset()
    if rc.compaction_protect_first_user_turn:
        first_user = _first_user_turn_index(history)
        if first_user is not None:
            protected = frozenset({first_user})

    # Protect executor-seeded prior-run turns from the lossy Tier-2 collapse so
    # a summary never drops the SESSION_HISTORY_SEED tag (which the host
    # finalization filter relies on to avoid re-persisting prior-run
    # conversation under the new run_id). Tier-1 still bounds them: seeded tool
    # results via the main shed path, and the session summary/ledger seed blocks
    # via the reference path (they DUAL-TAG COMPACTION_REFERENCE — see
    # ``_session_history_seed_indices``).
    seed_indices = _session_history_seed_indices(history)
    if seed_indices:
        protected = protected | seed_indices

    # Reference blocks are a Tier-1-ONLY shed surface (A2(2) recoverable blob
    # path). Tier-2 must never collapse them — un-blobbed, that loses frozen
    # bootstrap context; blobbed, the placeholder is the only in-history
    # pointer to the blob and summarising it both wastes an LLM call on the
    # raw placeholder marker and erases the blob ref. See
    # ``_compaction_reference_indices``.
    reference_indices = _compaction_reference_indices(history)
    if reference_indices:
        protected = protected | reference_indices

    units = _build_summarisation_units(history, eligible_upper, protected_indices=protected)

    summarised = 0
    freed = 0
    # Replacement Message keyed by anchor index; indices to delete after the loop.
    replacements: dict[int, Message] = {}
    indices_to_drop: set[int] = set()

    for unit in units:
        # Bounded per-pass cost — stop issuing summariser calls once this pass
        # has freed its budget. Prevents one COMPACTING pass over a many-turn
        # history from firing dozens of sequential ~5-11s LLM calls.
        if free_target_tokens is not None and freed >= free_target_tokens:
            break

        anchor = history[unit.anchor_idx]
        # A4 idempotency — never re-summarise an existing compaction summary
        # (would nest <compacted-turn> wrappers and decay the summary).
        if _is_compaction_summary(anchor):
            continue
        anchor_key = _stable_turn_key(anchor)
        if anchor_key in state.summarised_turn_ids:
            continue

        # Exhaustive text across EVERY member of the unit (assistant turn +
        # its tool results), so the summary preserves the tool exchange.
        unit_messages = [history[member] for member in unit.indices]
        raw_text = "\n".join(
            _message_text_for_estimation(member, rc) for member in unit_messages
        ).strip()
        if not raw_text:
            continue
        before_tokens = sum(estimate_message_tokens(member, rc) for member in unit_messages)

        # No-net-gain floor — a unit at or below the empty-wrapper size cannot
        # shrink; replacing it would only GROW history (the inflation the
        # max(0, ...) freed clamp would mask). Skip it before spending an LLM
        # call that cannot free tokens.
        if before_tokens <= _compaction_wrapper_floor_tokens(anchor_key, rc):
            continue

        sanitised = _strip_injection_patterns(raw_text)
        prompt = (
            "<turn>\n"
            f"{sanitised}\n"
            "</turn>\n\n"
            "Summarise the above turn in 1-2 sentences. Preserve tool names, "
            "key user intent, and file paths touched. Output STRICT JSON only."
        )

        request = LLMRequest(
            model=model_name,
            messages=[Message(role=MessageRole.user, content_blocks=[TextBlock(text=prompt)])],
            tools=[],
            max_tokens=rc.compaction_summary_max_output_tokens,
            temperature=rc.compaction_summary_temperature,
            observability=observability,
        )

        try:
            response = await compaction_llm.complete_structured(
                request, build_summary_schema(rc)
            )
        except Exception as exc:
            _logger.warning(
                "tier2 summariser failed for unit anchor_idx=%s; skipping (err=%s)",
                unit.anchor_idx,
                exc,
            )
            continue

        # Reconstruct the summary text from the structured response.
        # ``complete_structured`` is invoked with :func:`build_summary_schema`
        # (json_object), and the openai-compat provider's
        # ``_structured_response_from_body`` returns the model's RAW content
        # without parsing it. ``response.message.text`` therefore carries the
        # full JSON envelope — ``{"summary": "..."}``. Detect it by the leading
        # ``{`` (the schema is a top-level object), parse it strictly, and read
        # the ``summary`` string. Any other key is ignored rather than trusted:
        # ``additionalProperties`` is false, but a provider that does not enforce
        # the grammar can still return one, and an unread key must never reach
        # the history the summary replaces. A response that is NOT a JSON object
        # (a future provider that pre-parses, or the test-only
        # InMemoryLLMProvider) is taken verbatim as the summary — that path is
        # exercised by the existing tests.
        raw = response.message.text
        if not raw:
            continue
        if raw.lstrip().startswith("{"):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                _logger.warning(
                    "tier2 summary response looked like JSON but failed to parse "
                    "for unit anchor_idx=%s; skipping (err=%s)",
                    unit.anchor_idx,
                    exc,
                )
                continue
            if not isinstance(parsed, dict) or not isinstance(parsed.get("summary"), str):
                _logger.warning(
                    "tier2 summary response missing 'summary' string field "
                    "for unit anchor_idx=%s; skipping",
                    unit.anchor_idx,
                )
                continue
            summary_text: str = parsed["summary"]
        else:
            summary_text = raw
        if not summary_text:
            continue

        wrapped = _wrap_compaction_summary(anchor_key, summary_text)
        after_tokens = estimate_tokens(wrapped, rc)
        # Net-gain guard — a verbose summary can come back at or above the
        # original even when the unit cleared the empty-wrapper floor. Committing
        # such a replacement would GROW history while the max(0, ...) freed clamp
        # masked it (and could tip run_compaction's tokens_after over
        # tokens_before into a no-progress retry). Discard it: leave the original
        # turn intact rather than inflate. The turn is NOT marked summarised, so
        # a later pass may retry it.
        if after_tokens >= before_tokens:
            continue
        # vLLM-400 fix: the summary replaces an aged turn IN THE MIDDLE of
        # history. vLLM rejects any ``system`` message past index 0
        # ("System message must be at the beginning."), so the summary turn is
        # USER-role. It stays recognisable as a summary via the durable
        # ``COMPACTION_SUMMARY_METADATA_KEY`` flag + the ``<compacted-turn>``
        # wrapper (``_is_compaction_summary``); legacy persisted system-role
        # summaries remain recognised too. (The request-assembly boundary in
        # ``query._normalize_outbound_system_messages`` is the defense-in-depth
        # backstop for those legacy snapshots.)
        replacements[unit.anchor_idx] = Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=wrapped)],
            metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
        )
        # Every non-anchor member of the unit (the matching tool results) is
        # removed so the dropped ToolUseBlock leaves no orphaned tool_result.
        indices_to_drop.update(member for member in unit.indices if member != unit.anchor_idx)

        state.summarised_turn_ids.add(anchor_key)
        summarised += 1
        # after_tokens < before_tokens is guaranteed by the net-gain guard above,
        # so the delta is always a real positive freeing (the clamp is now only a
        # defensive floor).
        freed += max(0, before_tokens - after_tokens)

    if replacements or indices_to_drop:
        rebuilt: list[Message] = []
        for idx in range(len(history)):
            if idx in indices_to_drop:
                continue
            rebuilt.append(replacements.get(idx, history[idx]))
        history[:] = rebuilt

    return Tier2Result(turns_summarised=summarised, tokens_freed=freed)


__all__ = [
    "CompactionAttempt",
    "CompactionExhaustedError",
    "CompactionState",
    "Tier1Result",
    "Tier2Result",
    "build_summary_schema",
    "run_tier1_truncation",
    "run_tier2_summarisation",
]
