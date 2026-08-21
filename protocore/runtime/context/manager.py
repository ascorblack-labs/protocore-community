"""``ContextManager`` — assembles the 8-layer context bundle + drives compaction.

Pure-ish: every call rebuilds budgets from the latest RC snapshot. No
module-level cache — horizontal scaling rule (no per-pod state for
correctness-affecting decisions).
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from protocore.contracts.blob import IBlobStore
from protocore.contracts.llm import ILLMProvider, LLMObservabilityContext
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.skills import SkillBundle
from protocore.contracts.types import Message, ToolDefinition
from protocore.logging_utils import get_logger
from protocore.runtime.context.budgets import TokenBudgets, derive_budgets
from protocore.runtime.context.compaction import (
    CompactionAttempt,
    CompactionExhaustedError,
    CompactionState,
    Tier1Result,
    Tier2Result,
    estimate_message_tokens,
    run_tier1_truncation,
    run_tier2_summarisation,
)
from protocore.runtime.token_counting import LanguageProfile, detect_profile

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ContextBundle:
    """Output of :meth:`ContextManager.build_context`.

    The bundle is what the loop forwards to
    :meth:`ILLMProvider.stream_with_tools`. ``system_prompt_sections`` is
    the assembled prefix; ``messages`` is the (possibly compacted) history.
    """

    system_prompt_sections: tuple[str, ...]
    tools: tuple[ToolDefinition, ...]
    messages: tuple[Message, ...]
    active_language: str
    budgets: TokenBudgets


def detect_active_language(latest_message: Message | None) -> str:
    """Return ``"ru"`` or ``"en"`` based on Cyrillic-ratio heuristic.

    Mirrors v1 detection rule (Cyrillic ratio > 30% → RU). Defaults to
    ``"en"`` for empty / undetectable content.
    """
    if latest_message is None:
        return "en"
    text = latest_message.text
    if not text:
        return "en"
    profile = detect_profile(text)
    if profile in (LanguageProfile.cyrillic_prose, LanguageProfile.cyrillic_in_json_escape):
        return "ru"
    return "en"


def estimate_history_tokens(
    history: Sequence[Message],
    rc: RuntimeConstants,
) -> int:
    """Sum estimated tokens across all messages in ``history``.

 Used as the pre-flight cheap counter before calling
 :meth:`ILLMProvider.count_tokens` (the authoritative endpoint).

 Delegates per-message accounting to
 :func:`protocore.runtime.context.compaction.estimate_message_tokens`, the
 single exhaustive-by-ContentBlock estimator that covers ``TextBlock`` /
 ``ThinkingBlock`` / ``ToolUseBlock`` / ``ToolResultBlock`` / ``ImageRefBlock``
 PLUS :attr:`Message.reasoning_content` and a serialized-form catch-all so no
 block (or future kind) is ever silently counted as 0 tokens
 (/ / ).
 """
    return sum(estimate_message_tokens(message, rc) for message in history)


class ContextManager:
    """Builds per-turn :class:`ContextBundle` and drives compaction.

 The manager is stateless across calls
 with respect to context assembly — it reads :class:`RuntimeConstants`
 fresh and operates on the history list provided.

 Compaction state IS persisted on the :class:`QueryEngine` (retry
 counter, summarised-turn IDs) — passed in as :class:`CompactionState`.

 pin LRU state IS persisted on the
 manager so :meth:`pin_tool` can cap the per-run pin list at
 :attr:`RuntimeConstants.pinned_tool_max_count` and evict the
 least-recently-pinned entry on overflow. Pin state lives on the
 manager because the QueryEngine builds at-most-one ContextManager
 per run, so the pin list is naturally per-run scoped without
 needing extra plumbing on :class:`CompactionState`.
 """

    def __init__(
        self,
        *,
        rc: RuntimeConstants,
        blob_store: IBlobStore,
        compaction_llm: ILLMProvider,
    ) -> None:
        self._rc = rc
        self._blob_store = blob_store
        self._compaction_llm = compaction_llm
        # LRU pin tracking. OrderedDict
        # gives us O(1) ``move_to_end`` for re-pins and FIFO eviction
        # on overflow. Stored as ``dict[str, None]`` because we only
        # care about names + their relative order; the actual tool
        # descriptors live in the :class:`ToolRegistry`.
        self._pinned_tools: OrderedDict[str, None] = OrderedDict()

    @property
    def rc(self) -> RuntimeConstants:
        return self._rc

    # ------------------------------------------------------------------
    # pin LRU
    # ------------------------------------------------------------------

    def pin_tool(self, name: str) -> str | None:
        """Pin a tool by name with LRU + cap enforcement.

 The ``ToolVisibilityPolicy.pinned`` set can grow unbounded without a cap —
 a latent KV-cache bloat risk (every pinned tool stays in the prompt prefix
 regardless of retrieval scoring). The cap is taken fresh from
 :attr:`RuntimeConstants.pinned_tool_max_count` so an operator
 tightening the value via the dashboard takes effect on the
 next pin call without a redeploy.

 Behaviour:

 * Re-pinning an already-pinned name moves it to the
 most-recently-used position (no eviction).
 * Pinning a new name when the list is already at the cap
 evicts the oldest entry and returns its name so the caller
 can update the visibility policy that consumes the list.
 * Pinning when the cap is at or below zero is a no-op
 (defensive — the RC field is ``gt=0`` so this only fires
 for adversarial test setups).
 """

        if not name:
            return None
        cap = max(0, int(self._rc.pinned_tool_max_count))
        if cap <= 0:
            return None
        if name in self._pinned_tools:
            self._pinned_tools.move_to_end(name)
            return None
        evicted: str | None = None
        if len(self._pinned_tools) >= cap:
            # popitem(last=False) returns the OLDEST insertion — that's
            # the LRU because every re-pin moves the entry to the end.
            evicted_name, _ = self._pinned_tools.popitem(last=False)
            evicted = evicted_name
        self._pinned_tools[name] = None
        return evicted

    def pinned_tool_names(self) -> tuple[str, ...]:
        """Return the current LRU-ordered pinned tool names.

 Oldest-first (insertion order with
 re-pin promoting to the end). Callers building
 :class:`ToolVisibilityPolicy` pass ``frozenset(...)`` because
 the policy field is a set; the order is preserved here so the
 caller MAY render or log the recency.
 """

        return tuple(self._pinned_tools.keys())

    def build_context(
        self,
        *,
        history: Sequence[Message],
        tools: Sequence[ToolDefinition],
        skills_loaded: Sequence[SkillBundle] = (),
        system_prompt_sections: Sequence[str] = (),
        skill_index_block: str = "",
    ) -> ContextBundle:
        """Assemble the 8-layer context bundle.

        Filtering / retrieval of tools is the caller's concern (the loop
        delegates that to :class:`IToolRegistry.compute_effective_surface`).

        ``skill_index_block`` is the pre-rendered ``<system-reminder>`` skill
        catalog produced by
        :func:`~protocore.runtime.skill_index.render_skills_catalog`. Injected
        at Layer 2 (skill catalog sits between the system prompt proper and
        Layer 3 loaded skill bodies).
        """
        budgets = derive_budgets(self._rc)
        latest = history[-1] if history else None
        language = detect_active_language(latest)

        # Render skill bodies as system-prompt prepends (Layer 3). Each body
        # is capped at ``loaded_skills_budget_tokens // max_skills_per_run``
        # to keep the prefix stable.
        prepended_skills: list[str] = []
        if skills_loaded:
            max_per_skill = max(
                1,
                budgets.loaded_skills_budget_tokens
                // max(1, self._rc.max_skills_per_run),
            )
            for bundle in list(skills_loaded)[: self._rc.max_skills_per_run]:
                body = bundle.body or ""
                # Token-cap each loaded skill body — soft truncation by char
                # count using the RC-tunable chars-per-token heuristic
                # (Latin-prose baseline).
                budget_chars = max_per_skill * self._rc.skill_body_chars_per_token
                if len(body) > budget_chars:
                    body = body[:budget_chars] + "…"
                prepended_skills.append(
                    f"<loaded-skill name=\"{bundle.manifest.name}\">\n{body}\n</loaded-skill>"
                )

        sections: list[str] = list(system_prompt_sections)
        if skill_index_block:
            sections.append(skill_index_block)
        sections.extend(prepended_skills)
        assembled_sections = tuple(sections)

        return ContextBundle(
            system_prompt_sections=assembled_sections,
            tools=tuple(tools),
            messages=tuple(history),
            active_language=language,
            budgets=budgets,
        )

    async def run_compaction(
        self,
        *,
        history: list[Message],
        compaction_state: CompactionState,
        tenant_id: str,
        model_name: str,
        observability: LLMObservabilityContext | None = None,
        protect_tail_from_index: int | None = None,
    ) -> CompactionAttempt:
        """Run Tier 1 truncation; fall through to Tier 2 if needed.

        Increments :attr:`CompactionState.retry_count` on every failed
        attempt; raises :class:`CompactionExhaustedError` when
        :attr:`RuntimeConstants.compaction_failed_max_retries` is breached.

        ``protect_tail_from_index`` (set only by the per-iteration gate)
        exempts the current just-executed tool-result batch from BOTH tiers on
        top of ``compaction_keep_recent_turns``, so a >keep parallel batch's
        fresh, unconsumed results cannot be compacted in the same iteration
        they were produced.
        """
        budgets = derive_budgets(self._rc)
        tokens_before = estimate_history_tokens(history, self._rc)

        attempt = CompactionAttempt(tokens_before=tokens_before)

        try:
            tier1 = await run_tier1_truncation(
                history=history,
                blob_store=self._blob_store,
                tenant_id=tenant_id,
                rc=self._rc,
                truncation_threshold_tokens=budgets.tool_result_truncation_threshold,
                protect_tail_from_index=protect_tail_from_index,
            )
        except Exception as exc:
            compaction_state.retry_count += 1
            if compaction_state.retry_count > self._rc.compaction_failed_max_retries:
                raise CompactionExhaustedError("tier1 truncation exhausted retries") from exc
            # a failed Tier-1 pass freed nothing, but tokens_after
            # defaults to 0. Stamp the real current estimate before returning so
            # the caller's COMPACTION_COMPLETED event does not report a phantom
            # "full clear" (tokens_after=0 ≪ tokens_before).
            attempt.tokens_after = estimate_history_tokens(history, self._rc)
            return attempt

        attempt.tier1 = tier1
        compaction_state.blob_refs_created.extend(tier1.blob_refs_created)

        # Only fire Tier 2 if Tier 1 didn't clear enough. The bar is
        # ``compaction_trigger_tokens * routine_min_clear_ratio`` per the
        # cascade definition.
        min_clear_target = int(
            budgets.compaction_trigger_tokens
            * self._rc.compaction_routine_min_clear_ratio
        )
        if tier1.tokens_freed < min_clear_target and self._compaction_llm is not None:
            try:
                tier2 = await run_tier2_summarisation(
                    history=history,
                    compaction_llm=self._compaction_llm,
                    state=compaction_state,
                    rc=self._rc,
                    model_name=model_name,
                    observability=observability,
                    protect_tail_from_index=protect_tail_from_index,
                    # Tier-2 only needs to make up the shortfall Tier-1 left
                    # against the min-clear target; bound its per-pass LLM-call
                    # count to that budget rather than summarising every eligible
                    # turn serially.
                    free_target_tokens=min_clear_target - tier1.tokens_freed,
                )
            except Exception as exc:
                compaction_state.retry_count += 1
                if compaction_state.retry_count > self._rc.compaction_failed_max_retries:
                    raise CompactionExhaustedError("tier2 summarisation exhausted retries") from exc
                tier2 = Tier2Result(turns_summarised=0, tokens_freed=0)
            attempt.tier2 = tier2

        tokens_after = estimate_history_tokens(history, self._rc)
        attempt.tokens_after = tokens_after

        # Success → reset retry counter for next compaction
        if attempt.tokens_after < tokens_before:
            compaction_state.reset_retries()
        else:
            compaction_state.retry_count += 1
            if compaction_state.retry_count > self._rc.compaction_failed_max_retries:
                raise CompactionExhaustedError(
                    "compaction made no progress over retry budget"
                )

        return attempt

    async def force_compaction(
        self,
        *,
        history: list[Message],
        compaction_state: CompactionState,
        tenant_id: str,
        model_name: str,
        observability: LLMObservabilityContext | None = None,
        protect_tail_from_index: int | None = None,
    ) -> CompactionAttempt:
        """Run BOTH Tier 1 + Tier 2 unconditionally for reactive-413 recovery.

 Unlike :meth:`run_compaction` (which gates Tier 2 behind a
 "Tier 1 didn't free enough" check), this method always runs both
 tiers — the upstream provider has already signalled that the
 request exceeds the window, so we must free as much as possible
 before re-streaming.

 Raises :class:`CompactionExhaustedError` per the same retry budget
 as :meth:`run_compaction`.

 ``protect_tail_from_index`` (set only by the per-iteration
 emergency-cliff gate) exempts the current just-executed tool-result
 batch from BOTH tiers on top of ``compaction_keep_recent_turns``.
 The reactive-413 caller passes ``None`` (the provider already rejected
 the request, so the whole history is fair game and the most-recent
 batch was never wire-accepted).
 """
        budgets = derive_budgets(self._rc)
        tokens_before = estimate_history_tokens(history, self._rc)

        attempt = CompactionAttempt(tokens_before=tokens_before)

        # Tier 1 always runs.
        try:
            tier1 = await run_tier1_truncation(
                history=history,
                blob_store=self._blob_store,
                tenant_id=tenant_id,
                rc=self._rc,
                truncation_threshold_tokens=budgets.tool_result_truncation_threshold,
                protect_tail_from_index=protect_tail_from_index,
            )
        except Exception as exc:
            compaction_state.retry_count += 1
            if compaction_state.retry_count > self._rc.compaction_failed_max_retries:
                raise CompactionExhaustedError(
                    "force_compaction tier1 truncation exhausted retries"
                ) from exc
            tier1 = Tier1Result(
                tokens_freed=0,
                blob_refs_created=(),
                messages_modified=0,
            )

        attempt.tier1 = tier1
        compaction_state.blob_refs_created.extend(tier1.blob_refs_created)

        # Tier 2 ALWAYS runs in force mode (provider already signalled PTL).
        if self._compaction_llm is not None:
            # Free aggressively but bounded — enough to bring the post-Tier-1
            # history back under the trigger threshold so the request can
            # re-stream, without summarising every eligible turn serially.
            tokens_after_tier1 = estimate_history_tokens(history, self._rc)
            free_target = tokens_after_tier1 - budgets.compaction_trigger_tokens
            try:
                tier2 = await run_tier2_summarisation(
                    history=history,
                    compaction_llm=self._compaction_llm,
                    state=compaction_state,
                    rc=self._rc,
                    model_name=model_name,
                    observability=observability,
                    protect_tail_from_index=protect_tail_from_index,
                    free_target_tokens=free_target if free_target > 0 else None,
                )
            except Exception as exc:
                compaction_state.retry_count += 1
                if compaction_state.retry_count > self._rc.compaction_failed_max_retries:
                    raise CompactionExhaustedError(
                        "force_compaction tier2 summarisation exhausted retries"
                    ) from exc
                tier2 = Tier2Result(turns_summarised=0, tokens_freed=0)
            attempt.tier2 = tier2

        tokens_after = estimate_history_tokens(history, self._rc)
        attempt.tokens_after = tokens_after

        # Force-compaction success rule: ANY progress (tokens freed OR
        # any Tier 1 messages_modified OR any Tier 2 summarisation) resets
        # the retry counter. A no-progress force pass increments the
        # counter the same as in :meth:`run_compaction`.
        progress_made = (
            tokens_after < tokens_before
            or (attempt.tier1 is not None and attempt.tier1.messages_modified > 0)
            or (attempt.tier2 is not None and attempt.tier2.turns_summarised > 0)
        )
        if progress_made:
            compaction_state.reset_retries()
        else:
            compaction_state.retry_count += 1
            if compaction_state.retry_count > self._rc.compaction_failed_max_retries:
                raise CompactionExhaustedError(
                    "force_compaction made no progress over retry budget"
                )

        return attempt

    def current_prompt_tokens(
        self,
        history: Sequence[Message],
        *,
        observed_prompt_tokens: int = 0,
    ) -> int:
        """Best estimate of the current prompt size in tokens.

        The char-based :func:`estimate_history_tokens` heuristic is only a
        cold-start proxy: it systematically under-counts adversarial content
        (digit-dense tables, multilingual prose) by 2-3x relative to the real
        provider tokenizer, so a history that already occupies >2x the context
        window can still read below the compaction trigger. When the provider
        has reported a real prompt token count for a prior LLM call
        (``observed_prompt_tokens``), that ground-truth measurement is the
        floor — history only grows between calls, so the last real prompt size
        is a valid lower bound on the current one. The heuristic still governs
        cold start (turn 1, no usage yet) and post-compaction (observed is
        reset to 0 so the freshly-shrunk history is re-measured cheaply).
        """
        estimated = estimate_history_tokens(history, self._rc)
        return max(estimated, max(0, observed_prompt_tokens))

    def needs_compaction(
        self,
        history: Sequence[Message],
        *,
        observed_prompt_tokens: int = 0,
    ) -> bool:
        """Return ``True`` if the current prompt exceeds the trigger threshold."""
        budgets = derive_budgets(self._rc)
        current = self.current_prompt_tokens(
            history, observed_prompt_tokens=observed_prompt_tokens
        )
        return current > budgets.compaction_trigger_tokens

    def needs_emergency_compaction(
        self,
        history: Sequence[Message],
        *,
        observed_prompt_tokens: int = 0,
    ) -> bool:
        """Return ``True`` if the current prompt exceeds the emergency cliff.

        Activates :attr:`RuntimeConstants.compaction_emergency_ratio`. When
        this is True the runtime should run :meth:`force_compaction` (both
        tiers, unconditional) proactively rather than waiting for the provider
        to raise a context-window-exceeded error. ``compaction_emergency_tokens``
        is strictly above ``compaction_trigger_tokens`` (the RC validator
        enforces ``compaction_trigger_ratio < compaction_emergency_ratio``).
        """
        budgets = derive_budgets(self._rc)
        current = self.current_prompt_tokens(
            history, observed_prompt_tokens=observed_prompt_tokens
        )
        return current > budgets.compaction_emergency_tokens


__all__ = [
    "ContextBundle",
    "ContextManager",
    "detect_active_language",
    "estimate_history_tokens",
]
