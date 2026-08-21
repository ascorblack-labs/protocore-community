"""``ContextManager`` + token-budget derivation + two-tier compaction.

"""
from __future__ import annotations

from protocore.runtime.context.budgets import TokenBudgets, derive_budgets
from protocore.runtime.context.compaction import (
    CompactionAttempt,
    CompactionExhaustedError,
    CompactionState,
    Tier1Result,
    Tier2Result,
)
from protocore.runtime.context.manager import ContextBundle, ContextManager
from protocore.runtime.context.session_memory import (
    SUMMARY_SYSTEM,
    ArtifactLedger,
    FoldResult,
    SessionMemory,
    build_seed,
    build_summary_user_message,
    estimate_messages_tokens,
    extract_artifacts,
    fold_run,
    render_ledger,
    running_summary_needed,
    summary_fold_threshold_tokens,
)

__all__ = [
    "SUMMARY_SYSTEM",
    "ArtifactLedger",
    "CompactionAttempt",
    "CompactionExhaustedError",
    "CompactionState",
    "ContextBundle",
    "ContextManager",
    "FoldResult",
    "SessionMemory",
    "Tier1Result",
    "Tier2Result",
    "TokenBudgets",
    "build_seed",
    "build_summary_user_message",
    "derive_budgets",
    "estimate_messages_tokens",
    "extract_artifacts",
    "fold_run",
    "render_ledger",
    "running_summary_needed",
    "summary_fold_threshold_tokens",
]
