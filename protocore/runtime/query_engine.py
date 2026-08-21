"""``QueryEngine`` — per-conversation engine instance.

1.
One :class:`QueryEngine` per active run; owns the **mutable
per-conversation state** (history, state machine, compaction state, token
usage). The actual turn execution happens in :func:`protocore.runtime.query.query`
(an async iterator over this engine; it lowers turn-start state when called,
so it is a coroutine returning the iterator rather than a generator itself).

Persistence: snapshot ↔ resume via
:meth:`QueryEngine.snapshot`/:meth:`QueryEngine.resume_from_snapshot`. Per
:subplan:.5 the snapshot fires after every
``tool_result`` append, every compaction completion, and every
``message_stop``.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from protocore.contracts.blob import IBlobStore
from protocore.contracts.events import IEventStream
from protocore.contracts.hooks import IHookManager
from protocore.contracts.llm import ILLMProvider, IProviderChain
from protocore.contracts.observability import CacheObserverProtocol
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.skills import ISkillStore, SkillBundle
from protocore.contracts.tool_registry import IToolRegistry, ToolVisibilityPolicy
from protocore.contracts.types import Message, MessageRole, ToolCall, ToolPrecondition
from protocore.contracts.verification import (
    CandidateBundle,
    CandidateReleasedProjection,
    EvidenceLedger,
    EvidenceProducerBinding,
    EvidenceRecord,
    RunTreeOrigin,
    VerificationDelivery,
    VerificationLifecycle,
    VerificationState,
)
from protocore.runtime.candidate_delivery import CandidateDeliveryGate
from protocore.runtime.context.compaction import CompactionState
from protocore.runtime.context.manager import ContextManager
from protocore.runtime.deadline_clock import (
    consumed_deadline_seconds,
    restore_deadline_clock,
)
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import (
    InvalidStateTransitionError,
    LoopState,
    assert_transition,
    is_terminal,
)
from protocore.runtime.tool_dispatch import clear_run_scoped_helpers
from protocore.runtime.usage import TokenUsage

if TYPE_CHECKING:
    pass


# The two orthogonal loop axes. Kept as the
# single source of truth for the ``QueryEngineConfig.__post_init__`` validators
# so the accepted vocabularies are declared once (mirrors the Field ``pattern``
# carried by the Pydantic-side RuntimeConstants defaults).
RUN_MODES: tuple[str, ...] = ("direct", "deep")
REASONING_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")

# These envelopes are structurally children of an assistant reader turn.  In
# gated mode they cannot precede the withheld ``message_start``; retaining the
# typed events until the matching stop lets a genuinely tool-only turn retain
# its ordinary stream shape without guessing from any payload.
_READER_TURN_INTERIOR_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_SURFACE_ADVERTISED,
        EventType.TOOL_USE_START,
        EventType.TOOL_USE_INPUT_DELTA,
        EventType.TOOL_USE_STOP,
        EventType.TOOL_RESULT,
        EventType.TOOL_CALL_PENDING,
    }
)


@dataclass(frozen=True)
class QueryEngineConfig:
    """Immutable per-conversation config bound at engine construction.

 ``model_name`` is a REQUIRED field — it MUST come from the
 ``llm_provider_config`` PG row the executor resolves at admission
 time. There is no fallback to a baked-in model name (no inline magic
 numbers / no hardcoded defaults).
 """

    run_id: str
    tenant_id: str
    """Scope id (``tenants.id``). Keys hooks, secrets, RC, workspace, sessions."""
    session_id: str
    model_name: str
    account_id: str = ""
    """Owning account id (``tenants.account_id``) for the run's scope.

    The skill bank is ACCOUNT-WIDE and flat (keyed on ``skills.account_id``), so
    the run's skill catalog build + project-pin merge (``query._ensure_run_skill_catalog``
    / ``_merge_pinned_skills``) and the per-turn ``ToolContext`` the engine builds
    MUST key skill-store reads on this id — NOT ``tenant_id``, which is the
    (possibly different) scope id. The host executor resolves
    scope→account once per run via ``_resolve_account_id_for_scope`` and threads
    it here. Empty only when the scope's account could not be resolved; a skill
    lookup then resolves nothing rather than silently keying on the wrong id.
    """
    subagent_id: str | None = None
    parent_run_id: str | None = None
    root_run_id: str = ""
    """Durable root of this run tree.

    Root engines default to their own ``run_id``.  Child engines must be
    constructed with their leader's root id; this is persisted in every
    snapshot so evidence binding never depends on process-local ancestry.
    """
    run_depth: int = 0
    """How far below :attr:`root_run_id` this run sits.

    Supplied by whoever dispatched the run, which already counts this to bound
    recursion, rather than derived here.  An engine sees one hop of its tree,
    so any local derivation would be a second account of the same distance,
    free to disagree with the one the dispatcher enforced — and this value is
    stamped on evidence that is read long after both are gone.
    """
    verification_delivery: VerificationDelivery | None = None
    """Reader visibility selected by an already-admitted verification profile.

    ``None`` preserves the historical public stream.  A caller may select
    ``gated`` only after it has durably bound a verification profile; core does
    not infer that choice from generated output or a tenant-specific policy.
    """
    system_prompt_sections: tuple[str, ...] = field(default_factory=tuple)
    tool_visibility_policy: ToolVisibilityPolicy = field(default_factory=ToolVisibilityPolicy)
    pinned_skill_names: frozenset[str] = field(default_factory=frozenset)
    """project-pinned skill names.

    When the run's session belongs to a Project (and ``projects_enabled`` is on
    for the scope), the project's pinned skills are force-included in the
    once-per-run skill catalog even if disabled-by-default would otherwise have
    dropped them (pin = surfaced, NOT a visibility restriction). Routed from
    ``agent_project_skills`` by the host executor. Empty (the default)
    means the catalog is exactly the account's enabled skills.
    """
    rc: RuntimeConstants = field(default_factory=RuntimeConstants)
    execution_profile: str = "default"
    """Published tool profile: ``default`` | ``plan``. Orthogonal to run_mode."""
    run_mode: str = "direct"
    """Loop strategy axis: ``direct`` | ``deep``.

    ``direct`` reproduces today's auto-tool loop. ``deep`` runs the
    stand-validated SGR step first (a forced ``plan`` tool with native CoT
    bounded by ``reasoning_effort``) → emits one ``REASONING_STEP`` event →
    then drives the shared action loop. Validated in ``__post_init__``
    against ``^(direct|deep)$``. Routed from ``run.mode`` by the executor;
    default per tenant from ``RuntimeConstants.agent_loop_default_mode``.
    """
    thinking_enabled: bool = False
    """Native chain-of-thought toggle (the second, orthogonal axis).

    When True the runtime sets ``LLMRequest.extra['enable_thinking']`` so
    the host vLLM adapter flips ``chat_template_kwargs.enable_thinking``.
    Always paired with ``reasoning_effort`` so CoT stays bounded — measured,
    ``enable_thinking`` alone truncates the answer because the thinking
    consumes the whole output budget. ``deep`` mode implies thinking on; the executor
    enforces ``deep ⇒ thinking`` server-side.
    """
    reasoning_effort: str = "low"
    """Native CoT effort throttle: ``minimal`` | ``low`` | ``medium`` | ``high`` | ``xhigh``.

    Threaded into ``LLMRequest.extra['reasoning_effort']`` on every assistant
    stream so the adapter can map the level onto the provider (vLLM chat
    template / OpenRouter ``reasoning.effort``). Default ``low``. Validated
    in ``__post_init__`` against ``REASONING_EFFORTS``.
    """
    expected_terminal_tool: str | None = None
    """Per-tenant terminal-tool name.

    When set, the query loop consults ``rc.terminal_tool_nudge_enabled``
    and emits a single contract-repair nudge if the run is about to
    finish without that tool's successful terminal result in history.
    Routed from ``leader_config.expected_terminal_tool`` by the executor.
    ``None`` disables the terminal-tool nudge / terminal-only guard.
    """
    cache_observer: CacheObserverProtocol | None = None
    """Optional sink for per-LLM-call prompt-caching observations.

 When set, the runtime invokes
 ``cache_observer.record_run_cache_hit_rate(...)`` once per
 ``ProviderDeltaKind.usage`` envelope (see
 ``protocore.runtime.query._stream_one_assistant_message``). Core
 cannot import its host — implementations live there.
 """
    pre_terminal_self_verify_trigger: (
        Callable[[QueryEngine], str | None] | None
    ) = None
    """Host-supplied pre-terminal self-verify predicate.

    Called by the query loop at the moment a terminal-tool result would be
    committed (and only when ``rc.pre_terminal_self_verify_enabled`` is True
    and the per-run latch has not yet fired). It receives the engine (so it
    can inspect ``history`` and any caller-provided helper-bag state) and
    returns:

      * a non-empty corrective-instruction string → the loop injects ONE
        bounded corrective user turn instead of finalising, so the model can
        fix a cited-but-unobserved ref or perform a declared-but-missing
        mutation before re-finalising;
      * ``None`` / empty → no correction needed; finalisation proceeds.

    Core never hardcodes domain-specific verification logic — the trigger is
    tenant-supplied so the self-verify turn stays universal. ``None`` (the
    default) means no self-verify turn ever fires, reproducing prior behaviour.
    """
    pre_dispatch_terminal_verify_trigger: (
        Callable[[QueryEngine, ToolCall], str | None] | None
    ) = None
    """Host-supplied PRE-DISPATCH terminal-tool verify predicate.

    Called by :func:`protocore.runtime.query._dispatch_tool` BEFORE the
    dispatcher runs the tool, and ONLY when:

      * ``rc.pre_dispatch_terminal_verify_enabled`` is True,
      * the tool being dispatched IS the configured
        ``QueryEngineConfig.expected_terminal_tool``,
      * the durable per-run latch ``engine._pre_dispatch_terminal_verify_used``
        has not yet fired, and
      * the shared corrective-turn budget
        (``rc.pre_terminal_self_verify_max_extra_turns`` vs
        ``engine._self_verify_extra_turns_used``) is not exhausted.

    It receives the engine AND the un-submitted :class:`ToolCall` (so it can
    inspect ``tool_call.arguments`` against caller-provided helper-bag state)
    and returns:

      * a non-empty corrective-instruction string → the loop VETOES the
        terminal dispatch (the tool's external side effect, e.g. an answer
        RPC, NEVER fires), appends a non-terminal error tool_result + ONE
        bounded corrective user turn, and re-drives so the model can repair
        the answer BEFORE re-submitting;
      * ``None`` / empty → no correction needed; the terminal tool dispatches
        normally.

    This is the PRE-submit counterpart to
    ``pre_terminal_self_verify_trigger`` (which runs post-dispatch and so
    cannot repair a terminal tool that submits inside its own ``run()``).
    Core never hardcodes domain-specific verification logic — the predicate is
    tenant-supplied. ``None`` (the default) means no pre-dispatch veto ever
    fires, reproducing prior behaviour.
    """

    subagent_tool_allowlist: frozenset[str] = field(default_factory=frozenset)
    """Exact set of tool names this agent declared, or empty for "no declaration".

    A subagent definition may declare the tools it is meant to use. When that
    declaration is non-empty this set is threaded to
    :meth:`~protocore.runtime.tool_permission.ToolPermissionGate.check` as
    ``subagent_whitelist``, which refuses any name outside it at DISPATCH —
    not merely at advertisement. Empty (the default, and every leader engine)
    means the agent declared nothing and the gate stage is a no-op, so a run
    that carries no declaration behaves exactly as it did before.

    The engine unions the RC tool-surface floor onto this set before handing it
    to the gate (see :attr:`QueryEngine.effective_subagent_tool_allowlist`):
    ``forced_pinned`` tools are advertised unconditionally, so denying one
    would hand the model a callable schema that deterministically fails.
    """

    tool_preconditions: tuple[ToolPrecondition, ...] = field(default_factory=tuple)
    """Ordered tools this run MUST call before the agent is free to answer.

    While an entry is outstanding,
    :mod:`protocore.runtime.run_tool_preconditions` names its tool in
    ``LLMRequest.extra['forced_tool_choice']`` so the provider's native
    ``tool_choice`` — not prompt wording — decides what the model calls first.
    Once the last entry is satisfied nothing is forced again for the rest of
    the run and the whole tool surface is back: these are preconditions, not a
    restriction.

    Order is meaningful and duplicates are meaningful: ``tool_choice`` names
    exactly one tool per request, so ``[A, B, A]`` is A, then B, then A again.

    Empty (the default) means the mechanism never engages — no state, no forced
    choice, no extra failure path — so a run that carries none behaves exactly
    as it did before the field existed.
    """

    def __post_init__(self) -> None:
        """Validate the run-mode axes and the tool-precondition bounds.

        The config is a frozen dataclass, so the ``Field(pattern=...)``
        validation (Pydantic syntax) is
        enforced here instead. Every check names the offending field in the
        message so callers (and the run-create route in the host) get an
        actionable error rather than a silent bad value flowing into the loop.

        The precondition bounds are validated here as well as at the
        caller-facing boundary on purpose: an over-long list or an inflated
        ``calls`` multiplies the forced provider calls a run spends before it
        may answer, and refusing it late is still better than truncating it
        silently — a caller who asked for a precondition and did not get one
        has been lied to.
        """
        if not self.run_id or self.run_id.strip() != self.run_id:
            raise ValueError("run_id must be a non-padded, non-empty identifier")
        if not self.root_run_id:
            if self.parent_run_id is not None or self.subagent_id is not None:
                raise ValueError("child run config requires root_run_id")
            object.__setattr__(self, "root_run_id", self.run_id)
        elif self.root_run_id.strip() != self.root_run_id:
            raise ValueError("root_run_id must be a non-padded, non-empty identifier")
        has_parent = self.parent_run_id is not None
        has_subagent = self.subagent_id is not None
        if has_parent != has_subagent:
            raise ValueError(
                "parent_run_id and subagent_id must either both be set or both be absent"
            )
        if self.root_run_id == self.run_id:
            if has_parent:
                raise ValueError("root run config must not declare parent_run_id or subagent_id")
        elif not has_parent:
            raise ValueError(
                "child run config requires both parent_run_id and subagent_id"
            )
        elif (
            not self.parent_run_id
            or self.parent_run_id.strip() != self.parent_run_id
            or not self.subagent_id
            or self.subagent_id.strip() != self.subagent_id
        ):
            raise ValueError(
                "child run config requires non-padded, non-empty parent_run_id and subagent_id"
            )
        elif self.parent_run_id == self.run_id:
            raise ValueError("child run config must not declare itself as parent_run_id")
        if self.run_depth < 0:
            raise ValueError("run_depth must not be negative")
        # The ids and the depth state one position between them; disagreement
        # means the dispatcher's count and this config came from two sources.
        if self.root_run_id == self.run_id:
            if self.run_depth != 0:
                raise ValueError("root run config must declare run_depth 0")
        elif self.run_depth == 0:
            raise ValueError("child run config must declare a run_depth below the root")
        if self.execution_profile not in {"default", "plan"}:
            raise ValueError(
                f"execution_profile must be 'default' or 'plan', got {self.execution_profile!r}"
            )
        if self.run_mode not in RUN_MODES:
            raise ValueError(
                f"run_mode must be one of {RUN_MODES!r}, got {self.run_mode!r}"
            )
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(
                "reasoning_effort must be one of "
                f"{REASONING_EFFORTS!r}, got {self.reasoning_effort!r}"
            )
        max_entries = self.rc.run_tool_precondition_max_entries
        if len(self.tool_preconditions) > max_entries:
            raise ValueError(
                f"tool_preconditions carries {len(self.tool_preconditions)} "
                f"entries, which exceeds run_tool_precondition_max_entries "
                f"({max_entries})"
            )
        max_calls = self.rc.run_tool_precondition_max_calls
        for entry in self.tool_preconditions:
            if not entry.tool:
                raise ValueError("tool_preconditions entry has an empty tool name")
            if entry.calls > max_calls:
                raise ValueError(
                    f"tool_preconditions entry {entry.tool!r} asks for "
                    f"{entry.calls} calls, which exceeds "
                    f"run_tool_precondition_max_calls ({max_calls})"
                )


class QueryEngine:
    """One instance per active run.

    Lifetime: created when executor pod consumes admission → instance lives
    across all turns of that run → destroyed at terminal phase.

    State persistence: :meth:`snapshot` ↔ :meth:`resume_from_snapshot`. Any
    executor pod can resume after another crashes.
    """

    #: Everything a :meth:`rearm` carries into the next turn.
    #:
    #: Inverted on purpose: it names what SURVIVES, and the re-arm rebuilds
    #: every other attribute the constructor sets. Naming what resets is the
    #: arrangement that fails quietly — a field added to ``__init__`` and
    #: forgotten in ``rearm`` accumulates across turns, and the agent it
    #: eventually strands looks like a model problem hundreds of turns later.
    #: Inverted, a new field resets by default, and a field that must survive
    #: says so here, in one place.
    #:
    #: ``tests/unit/runtime/test_rearm_state_inventory.py`` walks ``__init__``
    #: and fails when an attribute appears in neither this set nor the reset
    #: list it keeps, so the classification cannot be skipped.
    _REARM_PRESERVED_ATTRS: ClassVar[frozenset[str]] = frozenset(
        {
            # ── Injected collaborators and the frozen config ────────────────
            # Not run state at all. ``llm`` is included because it tracks the
            # provider chain's one-way cursor: a rung demoted mid-run has not been
            # shown healthy again by the next turn, and re-testing it costs the
            # same failed turn twice. The chain itself owns the cursor, so a fresh
            # per-turn advance budget still cannot climb back up.
            "config",
            "llm",
            "provider_chain",
            "compaction_llm",
            "tools",
            "events",
            "hooks",
            "skills",
            "blobs",
            "typed_hook_registry",
            # A primitive, not state: tool handlers acquire it around cross-call
            # appends, and swapping the object under a holder buys nothing.
            "_tool_shared_state_lock",
            # ── The continuity that makes the next turn a continuation ──────
            # The conversation and everything derived from it. ``context_manager``
            # is here for two things it holds: the compaction machinery and the
            # discovery pin LRU, which is the agent keeping the tools it went
            # looking for. Both are bounded, so neither grows without limit.
            "history",
            "compaction_state",
            "compact_checkpoint",
            "context_manager",
            "last_observed_prompt_tokens",
            "last_heartbeat_ms",
            "_pinned_tool_result_ids",
            "_skill_catalog_block",
            "active_rule_paths",
            "discovered_rules",
            "background_pool",
            # Ledgers and lanes the caller owns and reads back. These are records
            # of what the agent did, not allowances it may spend; emptying one at a
            # turn boundary destroys evidence rather than freeing a budget, and
            # nothing in the engine is the thing that bounds them.
            "lanes",
            "open_intents",
            "usage_rows",
            "spans",
            "session_grants",
            "profile_audit",
            # Live control. A queued steer or follow-up is addressed to the agent,
            # not to the turn that happened to be running when it arrived, and the
            # model/thinking overrides are the operator's standing choice.
            "_steer_queue",
            "_follow_up_queue",
            "_live_model_name",
            "_live_thinking_enabled",
            "_live_reasoning_effort",
        }
    )

    #: Attributes attached to an engine AFTER construction — by the host, or by
    #: the run loop on first use. ``vars()`` of a freshly built engine cannot see
    #: them, so :meth:`rearm` would leave every one of them in place and the
    #: inventory test, which reads ``__init__``, would never notice. They are
    #: therefore classified here by hand, and the test checks this list against
    #: what the package actually attaches.
    #:
    #: Dropped: rebuilt on next use, or a latch that belongs to one run.
    _REARM_ATTACHED_DROPPED: ClassVar[frozenset[str]] = frozenset(
        {
            # Caches a tool-error counter read out of the helper bag when it was
            # first built. Rebuilt on demand, and rebuilding is what keeps that
            # counter from going stale under a bag the host has since replaced.
            "_tool_dispatcher",
            # Fire-once warning latch. Left raised, the warning it guards is
            # emitted once in the life of the engine rather than once per run.
            "_outbound_system_normalized_warned",
        }
    )

    #: Kept: the host owns it. The bag itself survives, but the per-run cells
    #: this package keeps inside it do not — see
    #: :data:`~protocore.runtime.tool_dispatch.RUN_SCOPED_HELPER_KEYS`.
    #:
    #: ``run_work_ledger`` is deliberately NOT among those cleared: it is the run
    #: *tree's* ledger, shared with subagents that may still be reading it, so
    #: emptying it from the leader would be a decision about their budget too.
    _REARM_ATTACHED_KEPT: ClassVar[frozenset[str]] = frozenset({"_helpers"})

    def __init__(
        self,
        *,
        config: QueryEngineConfig,
        llm_provider: ILLMProvider,
        tool_registry: IToolRegistry,
        event_stream: IEventStream,
        hook_manager: IHookManager,
        skill_store: ISkillStore,
        blob_store: IBlobStore,
        compaction_provider: ILLMProvider | None = None,
        provider_chain: IProviderChain | None = None,
    ) -> None:
        self.config = config
        self.llm: ILLMProvider = llm_provider
        # The consumer's remaining model preferences. ``None`` — every caller
        # that configured no priority list, and every test — means a runtime
        # failure has nowhere to go and the loop keeps its existing recovery
        # untouched. When present, ``self.llm`` is the chain's current rung and
        # the loop rebinds BOTH on an advance, because the endpoint, the key and
        # the capability flags travel with the provider, not with its name.
        self.provider_chain: IProviderChain | None = provider_chain
        self.compaction_llm: ILLMProvider = compaction_provider or llm_provider
        self.tools = tool_registry
        self.events = event_stream
        self.hooks = hook_manager
        self.skills = skill_store
        self.blobs = blob_store

        # Mutable per-conversation state
        self.history: list[Message] = []
        self.state: LoopState = LoopState.PENDING
        self.compaction_state = CompactionState()
        self.total_usage = TokenUsage()
        # Real provider-reported prompt size (full prompt_tokens, normalised
        # into input_tokens by the adapters) for the most recent LLM call.
        # Ground truth for the compaction gate — the
        # char heuristic under-counts adversarial content, so this floors the
        # decision. 0 means "no real measurement yet" (cold start or just after
        # a compaction shrank history), in which case the gate falls back to
        # the cheap estimate until the next LLM call refreshes it.
        self.last_observed_prompt_tokens: int = 0
        self.turn_count = 0
        self.last_heartbeat_ms: int = 0
        # Optional verification state is one typed aggregate rather than a
        # collection of independent latches. It remains absent from snapshots
        # until a caller explicitly enters the lifecycle.
        self._verification_lifecycle = VerificationLifecycle()
        self._candidate_delivery_gate = CandidateDeliveryGate(
            config.verification_delivery,
            expected_run_id=config.run_id if config.verification_delivery is VerificationDelivery.gated else None,
            expected_root_run_id=(
                config.root_run_id if config.verification_delivery is VerificationDelivery.gated else None
            ),
        )
        # A gated reader envelope must not escape before its candidate is
        # released.  Keep only its opening frame temporarily; generated
        # blocks are never retained in process memory for later replay.
        #
        # Independent state events deliberately do not consume this pending
        # frame. A tool frame belongs to the same reader turn as its opening frame, so
        # it is held with that frame until the turn is known to be tool-only.
        # Providers may interleave ``tool_use_start`` and text in one assistant
        # message. Publishing a tool frame first would both expose an unmatched
        # reader envelope and leave a public reducer unable to group it.
        self._pending_public_message_start: TurnEvent | None = None
        self._pending_public_reader_turn_events: list[TurnEvent] = []
        self._holding_reader_message: bool = False

        # Per-turn block index counter (resets on message_start)
        self._block_idx = 0
        # Per-assistant-message wire round counter (#1/#4). ``turn_count``
        # advances ONCE per ``run()`` (run-level, snapshot-persisted, used by
        # the salvage-id uniqueness counter), so a multi-round Direct run would
        # otherwise emit ONE ``turn_id`` for every assistant-message round. The
        # chat per-round grouping contract (``AssistantGroup`` /
        # ``coalesceConsecutiveAssistants``) keys turns by ``turn_id`` and
        # expects a DISTINCT id per LLM round, with ``block_idx`` unique WITHIN
        # a turn. This counter advances at each ``message_start`` boundary
        # (alongside ``reset_block_idx()``) so ``turn_id()`` yields a round-local
        # wire id that is plumbed identically through every frame of the round
        # (message_start, content_block_*, tool_use_*, tool_result, message_stop,
        # and the terminal/error/compaction yields). It is NOT persisted: a
        # resumed run re-streams fresh rounds, and the run-level ``turn_count``
        # already carries durable identity. Starts at 0; the first round bumps
        # it to 1 before that round's ``message_start``.
        self._wire_round_seq = 0
        # Per-turn pending tool calls
        self._pending_tool_call_names: dict[str, str] = {}
        self._pending_approval_tool_call_id: str | None = None

        # ── Recovery state (reset per assistant message) ──
        # Only one force_compaction attempt allowed per message
        # before the run goes terminal FAILED.
        self._compaction_attempted_for_current_turn: bool = False
        # Max-output-tokens recovery: count of "Resume directly" retries
        # already issued in the current message stream.
        self._max_output_recovery_count: int = 0
        # Forced terminal backstop — set True for exactly the one final
        # assistant message granted by the forced terminal backstop on a
        # non-answer exhaustion exit. While active,
        # :meth:`reset_recovery_state` does NOT zero
        # ``_max_output_recovery_count`` (it stays at its exhausted value),
        # so the forced turn is a STRICTLY single bounded attempt: a model
        # that truncates again on the forced turn immediately re-hits the
        # exhaustion branch instead of being granted a fresh
        # ``max_output_recovery_rounds`` budget. Cleared the moment the
        # backstop turn opens (see ``reset_recovery_state``).
        self._terminal_backstop_turn_active: bool = False
        # Per-run budget for the
        # ``args_partial_truncated`` + ``finish_reason="stop"`` recovery
        # branch in :func:`~protocore.runtime.query._stream_one_assistant_message`.
        # Bound by ``rc.tool_call_max_truncation_recoveries_per_message``.
        #
        # Lifecycle: reset on every new ``engine.run()`` call AND on a
        # successful non-truncated outer iteration (the loop's
        # "recovery succeeded" path). NOT reset by
        # :meth:`reset_recovery_state` — the truncation recovery branch lives
        # in the OUTER message loop and ``continue``-s a fresh assistant
        # message every round, so per-message reset would defeat the
        # guard. The "consecutive truncations" semantics match
        # ``_consecutive_empty_responses`` — the counter must persist
        # across the recovery rounds it is bounding.
        self._tool_call_truncated_recovery_count: int = 0
        # Paths for which a Write/AppendFile chunk has ACTUALLY been written
        # successfully this run ("chunking started"). Used ONLY to make the
        # recovery message MORE directive on a repeat truncation of the SAME path
        # ("you already started chunking {path}; continue with AppendFile, do not
        # re-Write"). This set is populated by a SUCCESSFUL Write/AppendFile
        # dispatch (:func:`_record_chunk_write_success`)
        # — NOT merely by having emitted a recovery prompt — so a repeat truncation
        # BEFORE any chunk landed keeps the first-message Write(header) protocol
        # and never tells the model to AppendFile a non-existent file. Per-run,
        # in-memory; not snapshot-persisted (chunk-recovery is intra-message and a
        # resumed run re-deriving an empty set merely loses the stronger repeat
        # wording, not correctness — the call is still never dispatched and the
        # budget guard still bounds the loop).
        self._mid_chunked_write_paths: set[str] = set()
        # Per-path count of chunk-recovery prompts issued for a path that has NOT
        # yet had a successful chunk write. Drives the progressively-LOWER header
        # budget on a repeat truncation before any write lands; cleared once a
        # successful Write/AppendFile to the path is recorded. Per-run, in-memory
        # (wording/budget only, not correctness).
        self._truncation_recovery_prompt_counts: dict[str, int] = {}
        # How many rungs this engine has already stepped down its provider
        # chain. Bounded by ``rc.llm_provider_chain_max_advances``. Per-turn
        # lifecycle — NOT reset between messages, because a provider that
        # failed at message three has not been shown to be healthy again by
        # message four and re-testing it costs the same failed turn twice.
        self._provider_chain_advances: int = 0
        # Continue-prompt fallback: count of
        # consecutive assistant turns that returned empty content + populated
        # reasoning_content. Bound by ``rc.max_consecutive_empty_responses``;
        # beyond → terminal FAILED with kind=``thinking_eats_all_tokens``.
        # Per-turn lifecycle — reset on every new ``engine.run()`` call.
        self._consecutive_empty_responses: int = 0
        # Post-tool empty-response recovery counter. Counts consecutive
        # FULLY-empty assistant turns (no text, no tool calls, AND no
        # reasoning_content) that arrive immediately after a tool-result turn.
        # Distinct from ``_consecutive_empty_responses`` (which bounds the
        # empty-WITH-reasoning thinking-trap path). Bounded by
        # ``rc.max_consecutive_empty_responses``; reset on a non-empty turn
        # and on every fresh ``engine.run()``. Default-off RC keeps it inert.
        self._post_tool_empty_nudge_count: int = 0
        # Transient LLM-error retry counter. Counts consecutive in-place
        # re-stream attempts made after a 429 (``LLMRateLimitError``) or a
        # timeout/stall (``LLMTimeoutError``) when the fallback-model swap is
        # unavailable or already engaged. Bounded by
        # ``rc.llm_transient_error_retry_max_attempts``; reset after any
        # successful assistant stream so the bound applies per consecutive-
        # failure streak. In-memory per-run (a resume starts a fresh budget).
        self._transient_stream_retry_count: int = 0
        # Empty-completion guard re-drive counter. Counts bounded re-drives
        # granted when an assistant turn ends ``finish_reason='stop'`` with no
        # text, no tool calls and no reasoning while the run has produced no
        # visible answer yet. Bounded by
        # ``rc.empty_completion_guard_max_redrives``; reset on every fresh
        # ``engine.run()``. In-memory per-run.
        self._empty_completion_redrive_count: int = 0
        # Death-spiral guard: set by :func:`_emit_llm_terminal` when
        # terminal cause is an LLM-provider class error. Stop / SessionEnd
        # hook dispatchers (the host side) MUST check this flag and
        # short-circuit when ``True``. Set BEFORE the state transition so
        # any synchronous downstream consumer sees a consistent value.
        self.skip_terminal_hooks: bool = False
        # Terminal-only finalisation latch — set True by the per-turn loop
        # once the terminal-answer nudge has been appended. While True AND
        # no terminal tool result is yet in history, the tool dispatcher
        # short-circuits every non-terminal call with a structured error so
        # the model is forced to finalise via the configured terminal tool
        # instead of burning the remaining turn budget on additional
        # discovery calls. The runtime never synthesises the answer — the
        # model still chooses message / outcome / refs.
        self._terminal_only_active: bool = False

        # Wall-clock budget. Monotonic timestamp captured at
        # ``run()`` entry; the wall-clock equivalent (epoch seconds) is persisted
        # in the snapshot so a run re-driven on another pod keeps ONE budget rather
        # than restarting the clock. 0.0 means "not started yet" / no budget.
        self._run_started_monotonic: float = 0.0
        self._run_started_epoch: float = 0.0

        # Pre-terminal self-verify latch + bounded counter.
        # ``_pre_terminal_self_verify_used`` mirrors the loop's
        # ``pac_terminal_nudge_used`` fire-at-most-once pattern; the counter
        # enforces ``rc.pre_terminal_self_verify_max_extra_turns`` as an
        # explicit ceiling. Both persist across resume (snapshot/resume) so
        # the bound survives a cross-pod re-drive (horizontal-scaling rule).
        self._pre_terminal_self_verify_used: bool = False
        self._self_verify_extra_turns_used: int = 0
        # Durable at-most-once latch for the PRE-DISPATCH terminal-tool verify
        # seam. Independent of the post-dispatch ``_pre_terminal_self_verify_used``
        # latch (the two seams fire at different points and a tenant may use
        # either), but both debit the SHARED ``_self_verify_extra_turns_used``
        # counter so the total corrective turns per run stay bounded by
        # ``rc.pre_terminal_self_verify_max_extra_turns``. Persisted across
        # snapshot/resume so a crash between the veto and the model's
        # re-submission does NOT reset the latch and re-veto the corrected
        # answer (which would loop). Default False reproduces prior behaviour.
        self._pre_dispatch_terminal_verify_used: bool = False
        # Durable per-run candidate-answer preservation.
        # ``_terminal_candidate`` holds the first SUBSTANTIVE terminal-answer draft
        # that was withheld by the pre-dispatch veto, so a later repair turn that
        # regresses to an empty / 1-char body cannot silently lose it.
        # ``_terminal_candidate_reveto_used`` is the one-shot repair latch that
        # lets a regressed replacement be re-vetoed exactly once. Both persist
        # across snapshot/resume (cross-pod safe) and stay inert until a tenant
        # sets ``rc.terminal_candidate_preserve_enabled``. Default None/False
        # reproduces prior behaviour (candidate discarded).
        self._terminal_candidate: dict[str, Any] | None = None
        self._terminal_candidate_reveto_used: bool = False
        #  — durable fire-at-most-once latch for the universal
        # prose-gate. When the run is about to latch a background terminal tool
        # result but produced NO substantive visible assistant prose after its
        # latest real-work tool, the gate vetoes the terminal dispatch ONCE and
        # injects one bounded repair turn (write the answer as normal text, THEN
        # call the terminal tool). This latch ensures the gate fires at most
        # once per run — a second prose-less terminal after the repair finalises
        # rather than looping. Persisted across snapshot/resume so a cross-pod
        # re-drive cannot grant a second prose-gate (horizontal-scaling rule,
        # mirroring the candidate latches above). Default False; only
        # ever set when ``rc.finalize_prose_gate_enabled`` is True.
        self._finalize_prose_gate_used: bool = False

        # The POINTER refusal's own attempt budget, held apart from the latch
        # above on purpose. The latch bounds the substantive-answer floor, whose
        # subject is an answer too SHORT to be one; the pointer refusal's subject
        # is an answer long enough to clear any floor that nonetheless only says
        # where a file the reader cannot open lives. Sharing one shot between
        # them meant whichever fired first silenced the other for the rest of
        # the run, and it capped the pointer refusal at the single repair turn
        # measured to change nothing.
        # ``_pointer_answer_repair_attempts`` counts the repair turns this run's
        # pointer refusal has INJECTED (never reset — the bound is on the run,
        # not on a streak) and is what
        # ``rc.finalize_prose_gate_pointer_max_repair_attempts`` bounds;
        # ``_pointer_answer_repair_released`` is the at-most-once latch for the
        # warning emitted when the budget is spent and the run is left to finish
        # on the answer it has. Both are PER-RUN and SNAPSHOT-PERSISTED
        # (cross-pod resume safe): a resumed run must neither be handed a fresh
        # budget nor announce the same release twice.
        self._pointer_answer_repair_attempts: int = 0
        self._pointer_answer_repair_released: bool = False

        # Live-run guardrails / interaction. Default-empty so a snapshot
        # taken before these fields existed resumes with prior behaviour.
        self._pinned_tool_result_ids: set[str] = set()
        self._identical_tool_counts: dict[str, int] = {}
        self._loop_guard_nudge_count: int = 0
        self._steer_queue: list[dict[str, Any]] = []
        self._follow_up_queue: list[dict[str, Any]] = []
        self._live_model_name: str | None = None
        self._live_thinking_enabled: bool | None = None
        self._live_reasoning_effort: str | None = None
        self._run_settled_emitted: bool = False

        # Ordered record of the tool calls this run DISPATCHED. Written at the
        # dispatch, never derived from history, because history does not keep
        # it: compaction replaces a whole turn with prose about it and does not
        # preserve the names of the tools that turn called, so a run long
        # enough to be compacted loses the transcript of its own work. The
        # ledger is what survives that — and it is deliberately thin (an
        # ordinal, a name, whether it succeeded) so keeping it costs nothing
        # and it can never become a second copy of the arguments or results.
        self._tool_call_ledger: list[dict[str, Any]] = []
        # The ordinal keeps counting past the cap, so ``seq`` is the real
        # position of a call in the run rather than an index into this list.
        self._tool_call_ledger_seq: int = 0
        self._tool_call_ledger_truncated: bool = False

        # The run wind-down. ``_soft_stop_cause`` names which bound was reached
        # and doubles as the armed flag; ``_soft_stop_stage`` tracks how far the
        # wind-down has got. Both are durable: a run resumed on another pod
        # after its tools were withdrawn must resume with them withdrawn, or the
        # resume hands the model back the surface the stop just took away.
        self._soft_stop_cause: str | None = None
        self._soft_stop_stage: str = ""
        self.background_pool: Any = None
        self.compact_checkpoint: Any = None
        self.active_rule_paths: list[str] = []
        self.discovered_rules: list[Any] = []
        # Rule paths that the last filesystem touch newly activated, drained by
        # the tool-dispatch step into one RULES_ACTIVATED event.
        self._pending_rules_activated: list[str] = []
        self.session_grants: list[Any] = []
        self.profile_audit: list[dict[str, Any]] = []
        self.open_intents: list[Any] = []
        self.usage_rows: list[Any] = []
        self.lanes: list[Any] = []
        self.typed_hook_registry: Any = None
        self.spans: list[Any] = []

        # Has this run handed work to a subagent? Set the moment a delegation
        # call is dispatched and never cleared — a run that delegated cannot
        # un-delegate, and the fact is about the run, not the turn.
        #
        # Read by the answer-narration split, which applies only to a leader
        # that actually delegated. Deliberately NOT derived from history on
        # demand: compaction folds older turns away, so the delegation's
        # ``tool_use`` block stops being there long before the answer is
        # written, and a fact that silently flips back to False mid-run is
        # worse than no fact. Snapshot-persisted for the same reason the
        # terminal latches are — a cross-pod re-drive that forgot the run had
        # delegated would render its answer differently from the pod that
        # started it.
        self._run_delegated: bool = False

        # Repeated-tool-error circuit breaker. ``_circuit_broken_tools``
        # holds tool names HARD-STOPPED for the rest of this run after they
        # crossed ``rc.max_consecutive_tool_errors`` consecutive failures of
        # the same error class (e.g. a ``/project`` Read/Grep/Glob/List that can
        # never succeed on a non-project session). Unioned into
        # ``effective_tool_policy.blocked`` so a broken tool vanishes from the
        # advertised surface AND is denied at dispatch.
        # ``_circuit_breaker_notified_tools`` is the at-most-once latch for the
        # corrective convergence turn (one per broken tool). ``_circuit_breaker_
        # streak`` is the IN-FLIGHT pre-trip streak — ``{tool_name, error_class,
        # count}`` or ``None`` — held on the engine (NOT the per-run helper bag)
        # so it is snapshot-persisted: a cross-pod resume BEFORE the trip would
        # otherwise rebuild a fresh helper bag and reset the count, letting a run
        # exceed ``max_consecutive_tool_errors`` without tripping.
        # All three are PER-RUN and SNAPSHOT-PERSISTED (cross-pod resume safe)
        # so a re-driven run keeps the tool blocked, the streak intact, and
        # does not re-inject the corrective message.
        self._circuit_broken_tools: set[str] = set()
        self._circuit_breaker_notified_tools: set[str] = set()
        self._circuit_breaker_streak: dict[str, Any] | None = None

        # Tool-precondition progress. ``_tool_precondition_index`` points at the
        # entry of ``config.tool_preconditions`` currently being forced; it
        # reaches ``len(tool_preconditions)`` once the last entry is satisfied
        # and nothing is ever forced again for the rest of the run. An INDEX,
        # not a set of satisfied names: the list is a sequence and a repeated
        # tool (``[A, B, A]``) must be forced again the second time.
        # ``_tool_precondition_calls`` counts the SUCCESSFUL calls of the
        # current entry's tool; ``_tool_precondition_attempts`` counts the
        # consecutive unproductive forced turns spent on it (reset by a
        # success) and is what ``rc.run_tool_precondition_max_attempts``
        # bounds.
        # ``_tool_precondition_last_error`` retains the current entry's last
        # error text so the exhaustion failure can name it.
        # All four are PER-RUN and SNAPSHOT-PERSISTED (cross-pod resume safe):
        # a resumed run must not re-force an already-satisfied entry, nor be
        # granted a fresh attempt budget for a tool that keeps failing.
        self._tool_precondition_index: int = 0
        self._tool_precondition_calls: int = 0
        self._tool_precondition_attempts: int = 0
        self._tool_precondition_last_error: str | None = None

        # Declared-file read-back. A tool result may declare paths the caller
        # must open before it continues (``PENDING_READS_METADATA_KEY``); while
        # one is unread the loop forces the workspace read tool. A SET and not
        # a slot: several delegations in one turn each declare their own files.
        # ``_pending_read_paths`` keeps declaration order so the operator-facing
        # warning on release names them the way they arrived.
        # ``_pending_reads_satisfied`` is every path READ this run, so a file
        # read before a tool declared it is never forced at all;
        # ``_pending_reads_abandoned`` is every path the gate gave up on, so a
        # tool that keeps re-declaring an unreadable file cannot re-engage on
        # it. ``_pending_reads_forced_attempts`` counts the CONSECUTIVE forced
        # turns that cleared nothing (reset by any productive read) and is what
        # ``rc.pending_reads_max_forced_attempts`` bounds.
        # All four are PER-RUN and SNAPSHOT-PERSISTED (cross-pod resume safe):
        # a resumed run must neither forget a read it already owes nor re-force
        # a file it has already opened.
        self._pending_read_paths: list[str] = []
        self._pending_reads_satisfied: set[str] = set()
        self._pending_reads_abandoned: set[str] = set()
        self._pending_reads_forced_attempts: int = 0

        # large-file convergence (runtime-driven stall-aware
        # forced convergence). All three counters are PER-RUN and SNAPSHOT-
        # PERSISTED (cross-pod resume safe, like the other recovery latches
        # above) — a re-driven run must not forget how many turns it has
        # stalled or re-grant a fresh forced-append/finalize budget.
        #
        # ``_turns_since_last_byte_adding_mutation`` — turns since the
        # last Write/AppendFile tool result that actually GREW the active file
        # (``bytes_written``/``bytes_appended`` > 0). Reset to 0 on a
        # byte-adding mutation; incremented on every assistant turn that adds
        # NO bytes (prose-only OR a non-mutation tool call such as
        # Read/Grep/Bash/List). NEVER append-count, NEVER prose-keyed — both
        # are bypassed by the weak model's "header-then-idle-inspect" shape.
        self._turns_since_last_byte_adding_mutation: int = 0
        # Per-run forced-round budgets (/) — bound the driver so it
        # can NEVER spin; subordinate to ``max_turns_per_run``.
        self._longfile_forced_appends: int = 0
        self._longfile_forced_finalizes: int = 0
        #  — the path of the active large-file artifact the stall detector
        # is tracking (the first byte-adding write's resolved path). Lets the
        # forced-tool directives + tail anchor + floor checks all follow the
        # file that really exists. Snapshot-persisted (cross-pod resume safe).
        self._longfile_active_path: str | None = None
        # / — running byte SIZE of the active file, read straight from
        # the tool RESULTS (``WriteOutput.bytes_written`` is the full file size
        # after an overwrite; ``AppendFileOutput.bytes_total`` is the cumulative
        # size). Core has no direct workspace read, so the result payloads ARE
        # the universal size signal — no disk dependency, no the host hook.
        # Snapshot-persisted (cross-pod resume safe).
        self._longfile_active_file_bytes: int = 0
        # Running full-file LINE count of the active file, tracked from
        # the tool RESULTS (``AppendFileOutput.line_count_total`` / a Write's
        # content lines) so the INCOMPLETE continue message reports the file's
        # REAL line count, not the 200-char tail's lines.
        # Snapshot-persisted (cross-pod resume safe).
        self._longfile_active_file_lines: int = 0
        # Bytes added by each successful byte-adding mutation to the
        # active path this run, used for the byte-plateau finalize trigger.
        # Snapshot-persisted so the plateau read survives a cross-pod re-drive.
        self._longfile_mutation_deltas: list[int] = []
        # True when the most recent byte-adding mutation to the active
        # path was a TRUNCATED write (the tail sits mid-content), so the file
        # is NOT plausibly complete no matter how many bytes are on disk.
        # Cleared by a clean (non-truncated) byte-adding mutation. The gate
        # that separates "big" from "complete". Snapshot-persisted.
        self._longfile_last_mutation_truncated: bool = False
        # True once a FinalizeFile has SUCCESSFULLY sealed the active
        # file this run. The driver then stops (no point re-forcing a finalize
        # on an already-sealed file, even when FinalizeFile is not the tenant's
        # terminal tool so the loop continues). Snapshot-persisted.
        self._longfile_finalized: bool = False
        # Set of file paths on which an output-cap truncation was DETECTED this
        # run. This is the LOAD-BEARING engage gate: the driver may force
        # AppendFile/FinalizeFile for the active path ONLY if that path has had a
        # truncation event. A below-floor file written WITHOUT a truncation is
        # NEVER engaged, so the driver is provably inert on every non-truncated
        # file. UNLIKE ``_longfile_last_mutation_truncated`` (which
        # ``observe_tool_result`` clears on a clean byte-adding write to track
        # "big vs complete"), this set is STICKY per-run-per-path: once a path
        # truncated it stays a large-file-in-progress path, so a forced clean
        # append never drops the engage gate and re-strands the run.
        # Snapshot-persisted (cross-pod resume safe).
        self._longfile_truncated_paths: set[str] = set()
        # Per-path TOTAL append count (forced + voluntary) for the active
        # large-file artifact this run. The per-path append circuit-breaker
        # (``longfile_max_appends_per_path``) caps it so a run can never self-loop
        # unbounded appends to one path. Snapshot-persisted.
        self._longfile_appends_per_path: dict[str, int] = {}
        # Monotonic per-run counter that makes every synthetic salvage
        # ``tool_call_id`` UNIQUE. Salvage ids derive from THIS counter (not
        # ``turn_id()``) on purpose: two salvages in the SAME assistant-message
        # round share one ``turn_id`` (the #1/#4 round seq advances per round, not
        # per salvage), so reusing ``turn_id()`` would mint the SAME id and the
        # outbound pairing repair would drop the later duplicate (hiding a real
        # workspace mutation from the model). Snapshot-persisted so a cross-pod
        # resume never reuses an id that already paired in durable history.
        self._longfile_salvage_seq: int = 0
        # TERMINAL SEAL — one-shot latch. At turn-budget exhaustion, if a
        # VOLUNTARY SEAL — one-shot latch. At a VOLUNTARY run completion
        # seam (the model calls the run-terminal tool, or finishes with a prose
        # ``end_turn``) a truncation-gated file can be left UNSEALED (after a
        # recovery the model appends chunks then ends the run without
        # FinalizeFile). When ``longfile_convergence.terminal_seal_required`` is
        # True the runtime dispatches ONE deterministic synthetic FinalizeFile
        # for ``_longfile_active_path`` BEFORE the run completes (no LLM call, no
        # extra turn). This latch caps the synthetic seal to ONCE per run.
        # Snapshot-persisted + restored with a safe default (False).
        self._longfile_voluntary_seal_used: bool = False

        # Concurrency control
        self._stop_requested = asyncio.Event()
        self._current_turn_task: asyncio.Task[Any] | None = None
        # Shared async lock for tool handlers that perform cross-call atomic
        # appends (e.g. observed-state collections written from read handlers).
        # Under the parallel ``asyncio.gather`` fan-out those appends can
        # interleave; handlers MUST acquire this lock around an append before a
        # state-mutating tenant enables
        # ``rc.parallel_read_tools_enabled``, otherwise a "correct refs zeroed"
        # race can reappear. Core exposes the primitive; it never forces the host
        # to use it.
        self._tool_shared_state_lock = asyncio.Lock()

        # Skill catalog block (the enabled account catalog + project pins
        # rendered into a ``<system-reminder>``): RUN-STABLE — built at most
        # ONCE per run by ``_ensure_run_skill_catalog`` and reused byte-for-byte
        # across turns, the inner agent loop, and every recovery context
        # rebuild, so the cached system-prompt prefix stays stable. ``None`` is
        # the not-yet-built sentinel; an empty string is a valid built value
        # (no skills / no store).
        self._skill_catalog_block: str | None = None
        # ``<command-name>`` trigger-loaded skill bodies: PER-TURN — rebuilt
        # every turn from that turn's user message (a Layer-3 prepend), so a
        # trigger in a later turn still force-loads its body. NOT cached like
        # the catalog block.
        self._skill_loaded_bundles: list[SkillBundle] = []

        # Context manager — rebuild on each turn for fresh RC
        self.context_manager = ContextManager(
            rc=config.rc,
            blob_store=blob_store,
            compaction_llm=self.compaction_llm,
        )

    # ------------------------------------------------------------------
    # Turn boundary — shared by every entry that opens a turn
    # ------------------------------------------------------------------

    def _reset_per_turn_state(self) -> None:
        """Put the per-turn counters and latches back to their turn-start value.

        Named for what it contains rather than for the whole of "state a turn
        must not inherit": ``_consecutive_empty_responses`` is also per-turn in
        spirit and is deliberately NOT reset here, because the empty-budget
        path leaves it at its exhausted value on purpose.  Adding to this set
        is a decision about that behaviour, not a tidy-up.

        Called by EVERY entry that opens a turn, because the state below is
        private: no caller outside the engine can put it back itself, however
        carefully it prepares the engine. There are two such entries —
        :meth:`run` and :func:`~protocore.runtime.query.query` — and they reach
        the same private generator, so a reset owned by one of them is a reset
        the other silently skips. It lives here, called from both, rather than
        inside that generator: :meth:`run` persists a snapshot immediately
        after opening the turn, and a snapshot taken while these still held
        last turn's values would carry them across a pod.

        Idempotent — every statement is an assignment to a constant, so a
        caller that opens a turn through both entries is no different from one
        that opens it through either.
        """
        # Per-run budget counters that the outer per-message loop must NOT
        # reset (otherwise their guards never fire). Reset at the same
        # lifecycle boundary as ``_consecutive_empty_responses``.
        self._tool_call_truncated_recovery_count = 0
        # Reset the post-tool empty-response nudge counter at the same per-run
        # lifecycle boundary as the other recovery counters.
        self._post_tool_empty_nudge_count = 0
        # Reset the empty-completion guard re-drive counter per run so a fresh
        # run gets its full re-drive budget.
        self._empty_completion_redrive_count = 0
        # Lower the terminal-only finalisation latch at the same boundary. The
        # latch means "the terminal-tool nudge has fired IN THIS TURN": it is
        # armed inside the per-turn loop, and it is read by the narration
        # suppressor, the output-reserve floor and the dispatch guard, each of
        # which asks a question about the turn now streaming. Nothing lowered
        # it, so a turn opened on an engine nudged in an EARLIER turn began
        # already finalising: its visible text was classified as post-answer
        # narration and dropped from both the live stream and history, so the
        # turn answered and delivered nothing.
        #
        # The snapshot round-trip keeps its meaning. Its cross-pod consumer is
        # ``resume_approved_tool``, which dispatches the approved call directly
        # and opens no turn at all, so a re-driven approval still rejects
        # non-terminal tools under a latch armed on the pod that died.
        #
        # The wind-down's own latch is NOT cleared here and must not be: it is a
        # fact about the RUN (a bound was reached and the run was told to close)
        # rather than about the turn, and clearing it at a turn boundary would
        # hand the model back the tools the stop took away.
        self._terminal_only_active = False
        # Reset the transient-stream-error (rate-limit / timeout) retry counter
        # per run for the same reason: a reused engine must start each run with
        # its full retry budget, not one left exhausted by a prior run that
        # terminated (or completed via the preserve-answer path) mid-streak.
        self._transient_stream_retry_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self, initial_message: Message | None = None
    ) -> AsyncIterator[TurnEvent]:
        """Run one turn end-to-end. Yields events as they happen.

        The caller (executor pod) iterates and forwards events to its
        :class:`IEventStream` adapter.

        ``initial_message`` is appended to ``history`` before the loop drives,
        as the user input that opens this turn. Pass ``None`` to generate an
        assistant turn against the EXISTING ``history`` as-is, with nothing
        appended — a continuation run. The existing ``history`` MUST then
        already be non-empty and end with a ``user`` message (the input the
        turn answers); a ``ValueError`` is raised otherwise. No empty/blank
        message is ever fabricated to stand in for the missing input: an
        empty prompt drives the model with no question to answer, which it
        reliably fills with confabulated content.
        """
        # Lazy import to break circular dependency between query_engine and
        # the per-turn async generator.
        from protocore.runtime.query import _query_raw

        if initial_message is None:
            if not self.history:
                raise ValueError(
                    "run(initial_message=None) needs a non-empty history to "
                    "continue from, but history is empty."
                )
            if self.history[-1].role is not MessageRole.user:
                raise ValueError(
                    "run(initial_message=None) continues against the existing "
                    "history, whose last message must have role 'user'; the "
                    f"last message is role {self.history[-1].role.value!r}."
                )
        else:
            self.history.append(initial_message)
        self.turn_count += 1
        self._reset_per_turn_state()
        # Stamp the run-start clock ONCE for the wall-clock budget. A resumed run
        # keeps the start it was rehydrated with
        # (``resume_from_snapshot`` set ``_run_started_monotonic`` from the
        # persisted epoch); only a genuinely fresh run (no prior start)
        # stamps now.
        #
        # The two timestamps are preserved independently. A host prelude
        # that runs work BEFORE ``run()`` may pre-stamp these so the
        # ``agent_max_seconds`` budget (measured from ``_run_started_monotonic``
        # by :func:`~protocore.runtime.query._terminal_deadline_reached`) spans
        # that pre-``run()`` time instead of restarting a fresh clock here. Each
        # guard is checked separately so a caller that pre-stamps only the monotonic
        # clock still gets a real wall-clock epoch persisted into the snapshot
        # (needed by ``resume_from_snapshot`` to re-anchor on a cross-pod re-drive)
        # — and vice versa. Behaviour is bit-identical when NOTHING pre-stamps:
        # both are 0.0 at entry, both stamp now. It is also inert for any tenant
        # with ``agent_max_seconds <= 0`` because the deadline predicate returns
        # False regardless of when the clock was stamped.
        if self._run_started_monotonic == 0.0:
            self._run_started_monotonic = time.monotonic()
        if self._run_started_epoch == 0.0:
            self._run_started_epoch = time.time()
        await self._persist_snapshot()

        # #6 — record the task driving this run so ``stop()`` can HARD-cancel an
        # in-flight ``await`` (e.g. the subagent ``driver_task`` blocked inside
        # the synchronous ``Agent`` tool) instead of only setting the
        # cooperative flag. Previously ``_current_turn_task`` was never assigned,
        # so ``stop()``'s ``.cancel()`` arm was dead code. A
        # cancel that arrives via ``cancel_event`` / the tool-dispatch race
        # remains the primary path; this makes ``engine.stop()`` actually
        # interrupt when an EXTERNAL task (the subagent heartbeat watchdog, the
        # executor cancel path) calls it. ``stop()`` is self-cancel-safe (it
        # skips ``.cancel()`` when called from THIS task — e.g. the in-loop
        # cancel poll — so it never injects ``CancelledError`` into itself
        # mid-iteration; the cooperative checkpoints handle that case).
        self._current_turn_task = asyncio.current_task()
        try:
            async for evt in _query_raw(self):
                for projected in self._project_public_turn_event(evt):
                    yield projected
        finally:
            self._current_turn_task = None
            await self._persist_snapshot()

    def _project_public_turn_event(self, event: TurnEvent) -> tuple[TurnEvent, ...]:
        """Apply the authoritative public delivery boundary to one turn event.

        Every public event iterator delegates to this boundary before a
        ``TurnEvent`` reaches its consumer.  In gated mode we defer a reader
        envelope until its terminal shape is known.  Once a content block
        appears, the entire reader envelope is discarded; a tool-only envelope
        is released in its original, well-formed order at ``message_stop``.
        Independent state progress still passes through.  A durable delivery
        coordinator may later publish a verified typed projection.

        The run's outcome is stamped onto the terminal ``message_stop`` here
        because here is the one place every public event iterator passes
        through — ``run`` and ``query`` both delegate to it — and because the
        loop reaches a terminal stop from eighteen different places. Stamping
        at each of them is eighteen chances to add a nineteenth that forgets.
        """
        if event.type is EventType.MESSAGE_STOP:
            event = self._stamp_run_outcome(event)
        if not self._candidate_delivery_gate.is_gated:
            return (event,)

        if event.type is EventType.MESSAGE_START:
            # A malformed producer may send consecutive starts.  Neither
            # envelope is proven tool-only, so fail closed for the unfinished
            # one rather than exposing a reader frame before a later content
            # block makes it unauthorized.
            if self._pending_public_message_start is not None:
                self._pending_public_message_start = event
                self._pending_public_reader_turn_events.clear()
                self._holding_reader_message = False
                return ()
            self._pending_public_message_start = event
            self._pending_public_reader_turn_events.clear()
            return ()

        if event.type in {
            EventType.CONTENT_BLOCK_START,
            EventType.CONTENT_BLOCK_DELTA,
            EventType.CONTENT_BLOCK_STOP,
        }:
            self._pending_public_message_start = None
            self._pending_public_reader_turn_events.clear()
            self._holding_reader_message = True
            return ()

        if event.type is EventType.MESSAGE_STOP:
            if self._holding_reader_message:
                self._holding_reader_message = False
                self._pending_public_message_start = None
                self._pending_public_reader_turn_events.clear()
                return ()
            if self._pending_public_message_start is not None:
                pending = self._pending_public_message_start
                self._pending_public_message_start = None
                pending_turn_events = tuple(self._pending_public_reader_turn_events)
                self._pending_public_reader_turn_events.clear()
                return (pending, *pending_turn_events, event)
            return (event,) if self._candidate_delivery_gate.permits(event) else ()

        if event.type in _READER_TURN_INTERIOR_EVENT_TYPES:
            if self._pending_public_message_start is not None:
                self._pending_public_reader_turn_events.append(event)
                return ()
            if self._holding_reader_message:
                return ()

        if self._candidate_delivery_gate.permits(event):
            return (event,)
        return ()

    def _stamp_run_outcome(self, event: TurnEvent) -> TurnEvent:
        """Add ``has_final_answer`` to a TERMINAL ``message_stop`` payload.

        The question no other field on the wire answers: was the user actually
        answered? ``status`` says the loop finished cleanly, ``stop_reason``
        says why it stopped, and a run can end ``completed`` / ``end_turn``
        having produced nothing the user can read — which is the shape the
        history-hygiene rule downstream keys on.

        A run the wind-down closed also carries WHICH bound it hit. Without it
        every wind-down looks the same from outside, and "the runs are stopping
        early" is not a question anyone can answer.

        Only terminal stops are stamped. ``stop_reason="tool_use"`` closes one
        assistant round mid-run, and a consumer that latched a mid-run "no
        answer yet" would read it as the run's verdict.
        """
        if event.payload.get("stop_reason") == "tool_use":
            return event
        from protocore.runtime import soft_stop as _soft_stop

        payload = dict(event.payload)
        payload["has_final_answer"] = self.has_final_answer
        windup_cause = _soft_stop.cause(self)
        if windup_cause is not None:
            payload["soft_stop_cause"] = windup_cause
        return event.model_copy(update={"payload": payload})

    def stop(self) -> None:
        """Request graceful stop. ``query()`` checks ``_stop_requested`` between phases.

        Sets the cooperative ``_stop_requested`` flag, and — when called from a
        DIFFERENT task than the one driving :meth:`run` — hard-cancels the run
        task so a blocking ``await`` (a long tool / subagent dispatch) is
        interrupted now (#6). Calling ``stop()`` from within the run task itself
        (the in-loop cancel poll) only sets the flag: a self-``cancel()`` would
        inject ``CancelledError`` into the current frame mid-iteration, so the
        cooperative checkpoints / the caller's own ``break`` finish the unwind.
        """
        self._stop_requested.set()
        task = self._current_turn_task
        if task is None or task.done():
            return
        try:
            current = asyncio.current_task()
        except RuntimeError:  # pragma: no cover - no running loop
            current = None
        if task is current:
            return
        task.cancel()

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested.is_set()

    @property
    def tool_call_ledger(self) -> list[dict[str, Any]]:
        """The dispatched tool calls, oldest first, as plain JSON-ready dicts.

        Each entry is ``{"seq": int, "name": str, "ok": bool}`` and nothing
        else. A copy, so a consumer cannot mutate the run's record of itself.
        """
        return [dict(entry) for entry in self._tool_call_ledger]

    @property
    def tool_call_ledger_truncated(self) -> bool:
        """True once the run made more tool calls than the ledger will hold.

        The ledger keeps the HEAD and drops the tail, so a truncated ledger
        still describes how the run started; this flag is what stops a reader
        from mistaking those entries for the whole of it.
        """
        return self._tool_call_ledger_truncated

    def record_tool_call(self, name: str, *, ok: bool) -> None:
        """Append one dispatched tool call to the run's ledger.

        Called once per call, in transcript order, from the dispatch path —
        both the serial one and the transcript-order replay the parallel
        branch runs after its gather. Past
        :attr:`RuntimeConstants.run_tool_call_ledger_max_entries` the entry is
        dropped and :attr:`tool_call_ledger_truncated` latches; the ordinal
        still advances so ``seq`` keeps meaning "the Nth call this run made".
        """
        self._tool_call_ledger_seq += 1
        limit = self.config.rc.run_tool_call_ledger_max_entries
        if len(self._tool_call_ledger) >= limit:
            self._tool_call_ledger_truncated = True
            return
        self._tool_call_ledger.append(
            {"seq": self._tool_call_ledger_seq, "name": name, "ok": bool(ok)}
        )

    @property
    def has_final_answer(self) -> bool:
        """True iff this run has produced a visible assistant answer in prose.

        The question a caller downstream of the run actually has: was the user
        answered? Neither ``status`` nor ``stop_reason`` carries it — a run can
        end COMPLETED having said nothing the user can read, and a run that
        ended in error may have delivered its answer before the failure. Runs
        seeded from earlier turns of the session, and the runtime's own
        recovery scaffolding, are excluded: an earlier run's fluent reply is
        the exact thing that would make an unanswered run look answered.
        """
        # Deferred import: ``query`` imports this module, and the rule about
        # what counts as this run's own words is stated there, next to the
        # other run-scoped predicates that share it.
        from protocore.runtime.query import run_has_final_answer

        return run_has_final_answer(self)

    @property
    def effective_tool_policy(self) -> ToolVisibilityPolicy:
        """Per-turn surface policy with the RC core tool-surface floor applied.

        ``RuntimeConstants.tool_surface_forced_pins``
        is the UNIVERSAL floor (default: Agent plus the six core file tools). The core
        engine itself merges it into ``ToolVisibilityPolicy.forced_pinned`` here
        so the floor is active on the live ``compute_effective_surface`` path
        WITHOUT every external caller having to copy the tuple into the policy.
        Any explicit ``forced_pinned`` the caller already set is unioned, never
        dropped. ``blocked`` precedence is still honoured downstream in
        :meth:`ToolRegistry._floored_visible_tools`. Without this wiring a
        default ``QueryEngineConfig`` would carry ``forced_pinned=frozenset()``
        and a Russian prompt could still collapse the surface to zero tools.
        """
        base = self.config.tool_visibility_policy
        rc_floor = frozenset(self.config.rc.tool_surface_forced_pins)
        merged_floor = base.forced_pinned | rc_floor
        dynamic_pins = frozenset(self.context_manager.pinned_tool_names())
        merged_pins = set(base.pinned) | set(dynamic_pins)
        # Union the per-run circuit-broken tools into ``blocked`` so a
        # tool that crossed ``max_consecutive_tool_errors`` is removed from the
        # advertised surface (``ToolRegistry._floored_visible_tools`` lets
        # ``blocked`` win even over ``forced_pinned``) AND denied at dispatch
        # (``ToolPermissionGate.check`` Stage-1 reads ``policy.blocked``). Empty
        # for every run that never tripped the breaker ⇒ no-op.
        merged_blocked = base.blocked | frozenset(self._circuit_broken_tools)
        if (
            merged_floor == base.forced_pinned
            and merged_pins == base.pinned
            and merged_blocked == base.blocked
        ):
            policy = base
        else:
            policy = base.model_copy(
                update={
                    "forced_pinned": merged_floor,
                    "pinned": merged_pins,
                    "blocked": merged_blocked,
                }
            )
        from protocore.runtime.execution_profile import apply_execution_profile

        policy = apply_execution_profile(
            policy,
            profile=self.config.execution_profile,
            rc=self.config.rc,
        )
        # The wind-down has the last word, and it has to. Everything above this
        # line is a mechanism for keeping a tool on the surface — the RC floor
        # bypasses the retrieval clip, the discovery pins bypass it too, the
        # profile mask only ever intersects. A withdrawal expressed at any of
        # those layers is a withdrawal something else re-admits. Expressed here,
        # after all of them, it is the surface the model actually receives; and
        # because this same property feeds the dispatch permission gate, a tool
        # that is no longer advertised is also no longer callable.
        from protocore.runtime import soft_stop as _soft_stop

        if _soft_stop.tools_withdrawn(self):
            policy = _soft_stop.restricted_policy(
                policy, _soft_stop.terminal_surface(self)
            )
        return policy

    @property
    def effective_subagent_tool_allowlist(self) -> frozenset[str] | None:
        """The declared tool set this run's dispatch gate enforces, or ``None``.

        ``None`` — the value for every run whose agent declared nothing — turns
        the gate's allow-list stage off entirely, which is the behaviour every
        run had before the stage was wired.

        When a declaration exists the RC tool-surface floor
        (``ToolVisibilityPolicy.forced_pinned``) is unioned onto it. The floor is
        advertised to the model unconditionally by
        ``ToolRegistry.compute_effective_surface``, so a declaration that omits a
        floor tool must not make that tool undispatchable: the model would
        receive a schema it can call and every call would fail the gate. The
        declaration narrows what the agent may reach BEYOND the floor.
        """
        declared = self.config.subagent_tool_allowlist
        if not declared:
            return None
        return frozenset(declared) | self.effective_tool_policy.forced_pinned

    def rearm(self) -> None:
        """Return a settled engine to PENDING so it can take another turn.

        A conversational run settles once: it answers, and the engine is
        terminal for good. An agent that *lives* — a character in a simulation,
        a watcher that never finishes — instead takes an unbounded number of
        turns on one history, and the run-scoped budgets have to start over
        with each of them, because every one of them was sized for a single
        question.

        Resetting only :attr:`state` is not enough, and it fails *silently*.
        The cumulative tool-call ledger, the run's output-token total, the run
        clock and the wind-down latch all survive a bare assignment. The engine
        keeps working until the first of those bounds is reached — around
        eighty dispatched calls under the default
        ``leader_tool_call_soft_cap`` — and then arms the wind-down
        **permanently**: every later turn starts with the tools already
        withdrawn, so the agent can only try to finalise, forever. It looks
        alive and does nothing.

        Naming those bounds one at a time was the same failure a level up. Each
        one left behind is its own quiet ending, and they are not rare. The
        repeated-error circuit breaker is on for everyone, and its block list is
        unioned into the visible surface, so a tool that failed for a reason
        that has since passed is withdrawn for good. The cooperative stop flag
        has no lowering seam at all, so an agent interrupted once is mute for
        the rest of its life. Where ``loop_guard_enabled`` is set, the
        identical-tool guard counts a fingerprint for the life of the engine
        rather than of the turn its limit is documented to bound, so an agent
        that opens every turn with the same observing call is refused that call
        once the limit is reached. So the reset is expressed the other way round:
        :attr:`_REARM_PRESERVED_ATTRS` names what survives, and everything else
        the constructor sets is rebuilt from it — :attr:`state` included, which
        is why no assignment to it appears below.

        What survives is deliberate. History, the compaction state, live-control
        queues and lanes are the continuity that makes the next turn a
        *continuation* rather than a new run; the injected collaborators are not
        run state at all; and every per-run allowance, latch, streak and
        recovery budget starts over, because each was sized for one question.

        Raises :class:`~protocore.runtime.loop_state.InvalidStateTransitionError`
        when the engine is not terminal — that is a caller re-arming a turn
        that is still in flight.
        """
        if not self.is_terminal:
            raise InvalidStateTransitionError(self.state, LoopState.PENDING)

        # Build the fresh state on a throwaway engine rather than assigning it
        # in place. Construction either completes or raises before anything is
        # copied across, so a re-arm cannot leave a live engine half-reset —
        # which is a worse state than the one this method exists to fix. The
        # collaborators handed to it are this engine's own, so the copy is a
        # reset and never a rebind.
        fresh = QueryEngine(
            config=self.config,
            llm_provider=self.llm,
            tool_registry=self.tools,
            event_stream=self.events,
            hook_manager=self.hooks,
            skill_store=self.skills,
            blob_store=self.blobs,
            compaction_provider=self.compaction_llm,
            provider_chain=self.provider_chain,
        )
        for name, value in vars(fresh).items():
            if name not in self._REARM_PRESERVED_ATTRS:
                setattr(self, name, value)

        # The constructor is not the only thing that puts state on an engine, and
        # ``vars(fresh)`` only knows what the constructor did. What the host and
        # the run loop attach afterwards has to be handled by name.
        for name in self._REARM_ATTACHED_DROPPED:
            if hasattr(self, name):
                delattr(self, name)

        # The helper bag stays — the host owns it — but the per-run streaks and
        # one-shot signals kept inside it are allowances like any other.
        clear_run_scoped_helpers(getattr(self, "_helpers", None))

    def transition_to(self, new_state: LoopState) -> None:
        """Validate then apply a state transition.

        Raises :class:`protocore.runtime.loop_state.InvalidStateTransitionError`
        if the transition violates the legal table.
        """
        assert_transition(self.state, new_state)
        self.state = new_state

    def turn_id(self) -> str:
        """Wire turn identifier for the current in-flight assistant-message round.

        Incorporates the per-round ``_wire_round_seq`` (#1/#4) so each LLM round
        inside one ``run()`` gets a DISTINCT id — the contract the chat per-round
        grouping (``AssistantGroup``) keys on. The run-level ``turn_count`` is
        kept in the id so durable identity (and the salvage-id derivation) is
        unchanged; the trailing round index is what makes consecutive rounds
        coalesce instead of collapsing into one accumulated turn. Before the
        first round (``_wire_round_seq == 0``: pre-loop terminals such as
        ``stop_before_start`` / compaction-failed / hook-deny) the id collapses
        to the legacy ``turn-{run}-{turn_count}`` shape.
        """
        if self._wire_round_seq <= 0:
            return f"turn-{self.config.run_id}-{self.turn_count}"
        return f"turn-{self.config.run_id}-{self.turn_count}-{self._wire_round_seq}"

    def begin_wire_round(self) -> None:
        """Advance to the next assistant-message wire round and reset block idx.

        Called from :mod:`protocore.runtime.query`: ROUND 1 just before the Deep
        loop-strategy 4b step (so the Deep ``REASONING_STEP`` and round 1's
        ``message_start`` share the suffixed id — the reducer merge requirement),
        and rounds 2+ at the ``message_start`` boundary inside
        :func:`~protocore.runtime.query._stream_one_assistant_message`. Each call
        makes ``turn_id()`` return a distinct round-local id and restarts
        ``block_idx`` at 0 within that round (the "block_idx unique within a turn"
        frontend contract). The run-start ``reset_block_idx()`` in ``query()``
        still zeroes block_idx for the pre-loop terminals, which run BEFORE the
        first ``begin_wire_round()`` and therefore keep the legacy id.
        """
        self._wire_round_seq += 1
        self._block_idx = 0

    def next_block_idx(self) -> int:
        """Allocate and return the next block index within the current turn."""
        idx = self._block_idx
        self._block_idx += 1
        return idx

    def reset_block_idx(self) -> None:
        self._block_idx = 0

    def reset_recovery_state(self) -> None:
        """Reset per-message recovery flags.

        Called from :func:`_stream_one_assistant_message` between
        message boundaries. Per-message flags reset every time:

        * ``_compaction_attempted_for_current_turn`` — a run that ate two
          distinct PTLs in two separate model calls still gets one recovery
          attempt each.
        * ``_max_output_recovery_count`` — only consecutive
          truncations within one message exhaust the budget.

        ``_provider_chain_advances`` is NOT reset — a demotion holds for the
        whole turn (per ``engine.run()``), and the cursor never walks back.

        ``_tool_call_truncated_recovery_count`` is NOT reset here either —
        the truncation recovery branch lives in the OUTER message loop
        (continues a new assistant message every round), so per-message reset
        would defeat the budget guard. It is reset on ``engine.run()`` entry
        and on a successful non-truncated outer iteration in
        ``_stream_one_assistant_message``.

        Forced terminal backstop exception: when
        ``_terminal_backstop_turn_active`` is set (the OUTER loop is about
        to open the single final message the backstop granted on an
        output-budget exhaustion exit), ``_max_output_recovery_count`` is
        preserved at its exhausted value instead of being zeroed. This
        keeps the forced turn a STRICTLY single bounded attempt — the
        model does not get a fresh ``max_output_recovery_rounds`` budget on
        the backstop turn. The flag is consumed (cleared) here so the
        constraint applies to exactly that one message.
        """
        self._compaction_attempted_for_current_turn = False
        if self._terminal_backstop_turn_active:
            self._terminal_backstop_turn_active = False
        else:
            self._max_output_recovery_count = 0

    def remember_tool_name(self, tool_call_id: str, tool_name: str) -> None:
        self._pending_tool_call_names[tool_call_id] = tool_name

    def tool_name_for(self, tool_call_id: str) -> str:
        return self._pending_tool_call_names.get(tool_call_id, "")

    def forget_tool_name(self, tool_call_id: str) -> None:
        self._pending_tool_call_names.pop(tool_call_id, None)

    def mark_pending_approval(self, tool_call_id: str) -> None:
        self._pending_approval_tool_call_id = tool_call_id

    def pending_approval_tool_call_id(self) -> str | None:
        return self._pending_approval_tool_call_id

    @property
    def verification_lifecycle(self) -> VerificationLifecycle:
        """Return the immutable optional verification runtime state."""
        return self._verification_lifecycle

    def replace_verification_lifecycle(self, value: VerificationLifecycle) -> None:
        """Install a restored lifecycle after validating its run binding.

        Controlled collection and sealing should use the methods below.  This
        replacement seam remains for durable restore and orchestration state
        installation, and therefore validates every attached record rather
        than deriving provenance from history or output text.
        """
        if value.candidate is not None and value.candidate.run_id != self.config.run_id:
            raise ValueError("verification candidate run_id does not match engine run_id")
        if value.ledger is not None:
            owner = value.ledger.attempt_owner
            if value.candidate is None:
                if not self._evidence_origin_belongs_to_exact_engine(owner):
                    raise ValueError("open verification evidence attempt owner does not match engine identity")
            elif not self._evidence_origin_belongs_to_engine_tree(owner):
                raise ValueError("verification evidence attempt owner is outside the candidate run tree")
            # A record is admitted on tree membership whether the ledger is
            # open or sealed.  Demanding the exact engine identity of an open
            # ledger's records would reject a run resuming with evidence a
            # descendant produced, which it can neither re-observe nor drop.
            # The ledger contract binds its records to its owner's tree too;
            # this boundary re-checks them against the engine rather than
            # inheriting whatever the contract happens to enforce.
            for record in value.ledger.records:
                if not self._evidence_origin_belongs_to_engine_tree(record.origin):
                    raise ValueError("verification evidence origin is outside this engine's run tree")
        self._verification_lifecycle = value

    def _evidence_origin_belongs_to_exact_engine(self, origin: RunTreeOrigin) -> bool:
        """Return whether evidence has the current engine's complete identity."""
        return (
            origin.run_id == self.config.run_id
            and origin.root_run_id == self.config.root_run_id
            and origin.depth == self.config.run_depth
            and origin.parent_run_id == self.config.parent_run_id
            and origin.subagent_id == self.config.subagent_id
        )

    def _evidence_origin_belongs_to_engine_tree(self, origin: RunTreeOrigin) -> bool:
        """Return whether typed evidence belongs to this engine's full tree.

        The immutable root binding is supplied by orchestration and remains
        independent from answer text, process-local state, and compaction.
        Immediate parent and subagent attribution remain in the evidence record.
        """
        return origin.belongs_to_root(self.config.root_run_id)

    def _engine_evidence_origin(self) -> RunTreeOrigin:
        """Return this execution attempt's immutable typed identity."""
        return RunTreeOrigin(
            run_id=self.config.run_id,
            root_run_id=self.config.root_run_id,
            depth=self.config.run_depth,
            parent_run_id=self.config.parent_run_id,
            subagent_id=self.config.subagent_id,
        )

    def begin_evidence_collection(self, *, ledger_id: str) -> None:
        """Open one new append-only evidence ledger for this execution attempt."""
        if not ledger_id or ledger_id.strip() != ledger_id:
            raise ValueError("ledger_id must not be empty or padded")
        current = self._verification_lifecycle
        if current.state not in {
            VerificationState.not_requested,
            VerificationState.repair_requested,
        }:
            raise ValueError("evidence collection can begin only for a new execution attempt")
        self._verification_lifecycle = VerificationLifecycle(
            state=VerificationState.executing,
            ledger=EvidenceLedger(ledger_id=ledger_id, attempt_owner=self._engine_evidence_origin()),
            repair_cycles=current.repair_cycles,
        )

    def append_tool_evidence(
        self,
        records: Sequence[EvidenceRecord],
        *,
        producer: EvidenceProducerBinding | None = None,
    ) -> None:
        """Atomically append trusted typed observations to the open ledger.

        The method deliberately accepts records only from this exact engine
        identity.  Aggregating evidence from descendant engines is a separate
        orchestration concern; no ancestry, subject, or citation is inferred
        from text-shaped state here.
        """
        lifecycle = self._verification_lifecycle
        if lifecycle.state is not VerificationState.executing or lifecycle.ledger is None:
            raise ValueError("tool evidence requires an open executing ledger")
        if lifecycle.candidate is not None:
            raise ValueError("tool evidence cannot be appended after candidate sealing")
        if lifecycle.ledger.attempt_owner != self._engine_evidence_origin():
            raise ValueError("open evidence ledger attempt owner does not match engine identity")

        new_ids: set[str] = set()
        existing_ids = {record.record_id for record in lifecycle.ledger.records}
        for record in records:
            origin = record.origin
            if origin.run_id != self.config.run_id:
                raise ValueError("evidence origin run_id does not match engine run_id")
            if origin.root_run_id != self.config.root_run_id:
                raise ValueError("evidence origin root_run_id does not match engine root_run_id")
            if origin.depth != self.config.run_depth:
                raise ValueError("evidence origin depth does not match engine run_depth")
            if origin.parent_run_id != self.config.parent_run_id:
                raise ValueError("evidence origin parent_run_id does not match engine parent_run_id")
            if origin.subagent_id != self.config.subagent_id:
                raise ValueError("evidence origin subagent_id does not match engine subagent_id")
            if producer is not None and (
                record.producer_id != producer.producer_id
                or record.producer_revision != producer.producer_revision
            ):
                raise ValueError("evidence producer does not match registered tool binding")
            if record.record_id in existing_ids or record.record_id in new_ids:
                raise ValueError(f"evidence record already exists: {record.record_id}")
            new_ids.add(record.record_id)

        # Build the replacement first: validation failures above leave the
        # current immutable ledger untouched, including a duplicate mid-batch.
        updated = lifecycle.ledger
        for record in records:
            updated = updated.append(record)
        self._verification_lifecycle = lifecycle.model_copy(update={"ledger": updated})

    def seal_candidate(self, candidate: CandidateBundle) -> None:
        """Pin a candidate to the exact current evidence ledger and seal it."""
        lifecycle = self._verification_lifecycle
        if lifecycle.state is not VerificationState.executing or lifecycle.ledger is None:
            raise ValueError("candidate sealing requires an open executing ledger")
        if candidate.run_id != self.config.run_id:
            raise ValueError("verification candidate run_id does not match engine run_id")
        if lifecycle.ledger.attempt_owner != self._engine_evidence_origin():
            raise ValueError("open evidence ledger attempt owner does not match engine identity")
        reference = candidate.evidence_ledger
        if reference.ledger_id != lifecycle.ledger.ledger_id or reference.digest != lifecycle.ledger.digest:
            raise ValueError("candidate evidence ledger reference does not match current ledger")
        self._verification_lifecycle = VerificationLifecycle(
            state=VerificationState.candidate_ready,
            ledger=lifecycle.ledger,
            candidate=candidate,
            repair_cycles=lifecycle.repair_cycles,
        )

    async def seal_candidate_and_persist(self, candidate: CandidateBundle) -> None:
        """Seal a typed candidate and durably snapshot it before verification.

        Reapplying the exact sealed candidate is safe after a caller retries a
        persistence boundary.  A different candidate is never substituted once
        the lifecycle is candidate-ready.  Candidate selection stays with the
        execution-to-delivery owner: this engine receives an explicit typed
        bundle and never searches message text or history for one.
        """
        lifecycle = self._verification_lifecycle
        if lifecycle.state is VerificationState.candidate_ready:
            if lifecycle.candidate != candidate:
                raise ValueError("a different verification candidate is already sealed")
        else:
            self.seal_candidate(candidate)
        await self._persist_snapshot()

    async def persist_verification_lifecycle(self) -> None:
        """Durably checkpoint the current typed verification lifecycle.

        Verification owners use this narrow operation after an explicit
        lifecycle mutation.  They do not need access to the engine's private
        snapshot mechanism or to model history in order to preserve evidence,
        a sealed candidate, or a terminal lifecycle state.
        """
        await self._persist_snapshot()

    def candidate_released_projection(self) -> CandidateReleasedProjection:
        """Return the exact typed projection of the terminal candidate.

        A reader-facing stream must pass the engine lifecycle to
        :class:`CandidateDeliveryGate` to obtain a publishable release event.
        The engine intentionally does not manufacture a generic
        ``TurnEvent``: such envelopes are otherwise freely constructible and
        cannot be accepted as authority at the gated delivery boundary.
        """
        return CandidateReleasedProjection.from_lifecycle(self._verification_lifecycle)

    def clear_pending_approval(self, tool_call_id: str) -> None:
        if self._pending_approval_tool_call_id == tool_call_id:
            self._pending_approval_tool_call_id = None

    # ------------------------------------------------------------------
    # Snapshot / resume
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Serialise engine state for Redis Hash ``run:{id}``.

 1 + .5.
 ``last_heartbeat_ms`` is surfaced as a discrete field for the
 stuck-runs-reaper fast scan.
 """
        snapshot = {
            "run_id": self.config.run_id,
            "tenant_id": self.config.tenant_id,
            "session_id": self.config.session_id,
            "subagent_id": self.config.subagent_id,
            "parent_run_id": self.config.parent_run_id,
            "root_run_id": self.config.root_run_id,
            # This binding decides whether provider frames are reader-visible.
            # It is therefore durable admission state, not an executor-local
            # default that a recovering process may choose independently.
            "verification_delivery": (
                self.config.verification_delivery.value
                if self.config.verification_delivery is not None
                else None
            ),
            "model_name": self.config.model_name,
            "history": [m.model_dump(mode="json") for m in self.history],
            "state": self.state.value,
            "turn_count": self.turn_count,
            "pending_approval_tool_call_id": self._pending_approval_tool_call_id,
            "usage": self.total_usage.to_dict(),
            "last_observed_prompt_tokens": self.last_observed_prompt_tokens,
            "compaction": {
                "retry_count": self.compaction_state.retry_count,
                "summarised_turn_ids": list(self.compaction_state.summarised_turn_ids),
                "blob_refs_created": list(self.compaction_state.blob_refs_created),
            },
            "last_heartbeat_ms": self.last_heartbeat_ms,
            # Persist the terminal-only latch so an executor pod that
            # resumes a run after the terminal nudge fired still rejects
            # non-terminal tool dispatches. Latch lives in per-engine memory
            # otherwise, which violates the horizontal-scaling rule that
            # correctness-affecting runtime state must not rely on per-pod
            # memory.
            "terminal_only_active": self._terminal_only_active,
            # Persist "this run delegated" so a run re-driven on another pod
            # still renders its answer the way the pod that dispatched the
            # subtask would have. Same horizontal-scaling rule as the latches
            # around it: a correctness-affecting run fact must not live only in
            # per-pod memory.
            "run_delegated": self._run_delegated,
            # Persist the run-start epoch so a run re-driven on another pod keeps
            # ONE wall-clock budget instead of restarting the deadline clock.
            # Monotonic clocks are per-process so we persist epoch seconds and
            # re-derive a monotonic anchor on resume. Consumed duration is
            # persisted separately: a future-relative epoch must not restart
            # elapsed at zero on the next re-drive.
            "run_started_epoch": self._run_started_epoch,
            "run_deadline_elapsed_seconds": consumed_deadline_seconds(
                started_monotonic=self._run_started_monotonic,
                now_monotonic=time.monotonic(),
            ),
            # Persist the self-verify latch + bounded counter so the at-most-once
            # corrective turn survives a cross-pod re-drive (same rationale as
            # ``terminal_only_active``).
            "pre_terminal_self_verify_used": self._pre_terminal_self_verify_used,
            "self_verify_extra_turns_used": self._self_verify_extra_turns_used,
            # Persist the PRE-DISPATCH verify latch so a run resumed after a veto
            # does not re-veto the model's corrected re-submission. Same rationale
            # as the post-dispatch self-verify latch above.
            "pre_dispatch_terminal_verify_used": self._pre_dispatch_terminal_verify_used,
            # Persist the preserved terminal candidate + its one-shot re-veto
            # latch so a run resumed on another pod still remembers the
            # substantive draft it withheld and does not regress across a
            # cross-pod re-drive.
            "terminal_candidate": self._terminal_candidate,
            "terminal_candidate_reveto_used": self._terminal_candidate_reveto_used,
            # Persist the one-shot prose-gate latch so a run resumed after a
            # prose-gate veto does NOT re-veto the model's prose+terminal
            # re-submission (which would loop).
            "finalize_prose_gate_used": self._finalize_prose_gate_used,
            # Persist the pointer refusal's attempt budget separately: it is a
            # different test with a different bound, and a resume that reset it
            # would grant the run a second full budget of repair turns.
            "pointer_answer_repair_attempts": self._pointer_answer_repair_attempts,
            "pointer_answer_repair_released": self._pointer_answer_repair_released,
            # Persist the circuit-breaker state so a run re-driven on another pod
            # keeps the broken tool blocked AND does not re-inject the corrective
            # convergence turn. Sorted lists for deterministic snapshots (JSON has
            # no set type).
            "circuit_broken_tools": sorted(self._circuit_broken_tools),
            "circuit_breaker_notified_tools": sorted(
                self._circuit_breaker_notified_tools
            ),
            # Persist the IN-FLIGHT pre-trip streak too so a resume before the
            # trip does not reset the count (which would let a run exceed
            # ``max_consecutive_tool_errors`` without tripping).
            "circuit_breaker_streak": self._circuit_breaker_streak,
            # Persist tool-precondition progress so a run re-driven on another
            # pod neither re-forces a satisfied entry nor gets a fresh attempt
            # budget for one that keeps failing.
            "tool_precondition_index": self._tool_precondition_index,
            "tool_precondition_calls": self._tool_precondition_calls,
            "tool_precondition_attempts": self._tool_precondition_attempts,
            "tool_precondition_last_error": self._tool_precondition_last_error,
            # Persist the declared-file read-back state so a run re-driven on
            # another pod still owes the reads it owed, still remembers the
            # files it has already opened, and does not get a fresh attempt
            # budget for a file it has been unable to read. The two set-valued
            # fields are sorted on the way out so a snapshot is stable.
            "pending_read_paths": list(self._pending_read_paths),
            "pending_reads_satisfied": sorted(self._pending_reads_satisfied),
            "pending_reads_abandoned": sorted(self._pending_reads_abandoned),
            "pending_reads_forced_attempts": self._pending_reads_forced_attempts,
            # persist the large-file convergence state so a run
            # re-driven on another pod keeps ONE stall clock + ONE forced-round
            # budget (cross-pod safe, same rationale as ``terminal_only_active``
            # and the candidate latches). Without this a resume would zero the
            # stall counter and re-grant a fresh forced-append/finalize budget,
            # letting the driver re-fire past its per-run caps.
            "turns_since_last_byte_adding_mutation": (
                self._turns_since_last_byte_adding_mutation
            ),
            "longfile_forced_appends": self._longfile_forced_appends,
            "longfile_forced_finalizes": self._longfile_forced_finalizes,
            "longfile_active_path": self._longfile_active_path,
            "longfile_active_file_bytes": self._longfile_active_file_bytes,
            "longfile_active_file_lines": self._longfile_active_file_lines,
            "longfile_mutation_deltas": list(self._longfile_mutation_deltas),
            "longfile_last_mutation_truncated": (
                self._longfile_last_mutation_truncated
            ),
            "longfile_finalized": self._longfile_finalized,
            # Persist the per-path truncation latch + append counter so a cross-pod
            # re-drive keeps the engage gate and the per-path append breaker
            # (lists/dicts serialised for JSON).
            "longfile_truncated_paths": sorted(self._longfile_truncated_paths),
            "longfile_appends_per_path": dict(self._longfile_appends_per_path),
            # Persist the salvage id counter so a resume never reuses a synthetic
            # salvage tool_call_id that already paired in history.
            "longfile_salvage_seq": self._longfile_salvage_seq,
            # Persist the voluntary-seal latch so a cross-pod resume never
            # re-dispatches the synthetic run-end FinalizeFile.
            "longfile_voluntary_seal_used": self._longfile_voluntary_seal_used,
            "pinned_tool_result_ids": sorted(self._pinned_tool_result_ids),
            "identical_tool_counts": dict(self._identical_tool_counts),
            "loop_guard_nudge_count": self._loop_guard_nudge_count,
            "steer_queue": list(self._steer_queue),
            "follow_up_queue": list(self._follow_up_queue),
            "live_model_name": self._live_model_name,
            "live_thinking_enabled": self._live_thinking_enabled,
            "live_reasoning_effort": self._live_reasoning_effort,
            "run_settled_emitted": self._run_settled_emitted,
            # Persist the tool-call ledger so a run re-driven on another pod
            # continues one record rather than starting a second. It is the
            # only place the names of a compacted turn's tool calls still
            # exist, so losing it on resume loses them for good.
            "tool_call_ledger": [dict(entry) for entry in self._tool_call_ledger],
            "tool_call_ledger_seq": self._tool_call_ledger_seq,
            "tool_call_ledger_truncated": self._tool_call_ledger_truncated,
            # Persist the wind-down so a resumed run stays wound down. Without
            # it a cross-pod re-drive re-advertises every tool the stop removed
            # and the run carries on working past the bound it hit.
            "soft_stop_cause": self._soft_stop_cause,
            "soft_stop_stage": self._soft_stop_stage,
            # Whether the user has been answered. Derived, but persisted so a
            # consumer reading the snapshot does not have to re-derive the
            # run-scoping rule to find out.
            "has_final_answer": self.has_final_answer,
        }
        if self._verification_lifecycle != VerificationLifecycle():
            snapshot["verification"] = self._verification_lifecycle.snapshot()
        snapshot["open_intents"] = [
            item.to_dict() if hasattr(item, "to_dict") else item for item in self.open_intents
        ]
        snapshot["usage_rows"] = [
            item.to_dict() if hasattr(item, "to_dict") else item for item in self.usage_rows
        ]
        snapshot["lanes"] = [
            item.to_dict() if hasattr(item, "to_dict") else item for item in self.lanes
        ]
        return snapshot

    async def resume_from_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Rehydrate engine from snapshot dict.

 Called by an executor pod that picks up an orphaned run. Per
 .3 the caller is expected to also wire an
 emitter for the ``state_changed`` "resumed_from_snapshot" event.
 """
        # A delivery-mode downgrade would make held provider content visible
        # after recovery.  Treat the mode as an immutable run binding and stop
        # before restoring any state or driving the public iterator when it is
        # absent, malformed, or differs from the admitted configuration.
        persisted_delivery = snapshot.get("verification_delivery", object())
        expected_delivery = (
            self.config.verification_delivery.value
            if self.config.verification_delivery is not None
            else None
        )
        if persisted_delivery != expected_delivery:
            raise ValueError("verification delivery snapshot binding does not match engine")

        # Validate deadline fields before any engine mutation. A future epoch
        # is legal (clock skew); NaN/Inf/negative/non-numeric values are not.
        restored_deadline = restore_deadline_clock(
            persisted_epoch=snapshot.get("run_started_epoch", 0.0),
            persisted_elapsed=snapshot.get("run_deadline_elapsed_seconds", 0.0),
            now_wall=time.time(),
            now_monotonic=time.monotonic(),
        )

        history_data = snapshot.get("history", [])
        self.history = [Message.model_validate(m) for m in history_data]
        self.state = LoopState(snapshot.get("state", LoopState.PENDING.value))
        self.turn_count = int(snapshot.get("turn_count", 0))
        pending_approval_tool_call_id = snapshot.get("pending_approval_tool_call_id")
        self._pending_approval_tool_call_id = (
            pending_approval_tool_call_id if isinstance(pending_approval_tool_call_id, str) else None
        )
        self.total_usage = TokenUsage.from_dict(snapshot.get("usage", {}))
        self.last_observed_prompt_tokens = int(
            snapshot.get("last_observed_prompt_tokens", 0)
        )

        compaction = snapshot.get("compaction", {})
        self.compaction_state = CompactionState(
            retry_count=int(compaction.get("retry_count", 0)),
            summarised_turn_ids=set(compaction.get("summarised_turn_ids", [])),
            blob_refs_created=list(compaction.get("blob_refs_created", [])),
        )
        self.last_heartbeat_ms = int(snapshot.get("last_heartbeat_ms", 0))
        snapshot_root_run_id = snapshot.get("root_run_id", self.config.root_run_id)
        root_binding_matches = (
            isinstance(snapshot_root_run_id, str)
            and bool(snapshot_root_run_id)
            and snapshot_root_run_id == self.config.root_run_id
        )
        snapshot_run_id = snapshot.get("run_id")
        run_binding_matches = (
            isinstance(snapshot_run_id, str)
            and bool(snapshot_run_id)
            and snapshot_run_id == self.config.run_id
        )
        snapshot_identity_matches = (
            "parent_run_id" in snapshot
            and "subagent_id" in snapshot
            and snapshot.get("parent_run_id") == self.config.parent_run_id
            and snapshot.get("subagent_id") == self.config.subagent_id
        )
        if "verification" in snapshot:
            lifecycle = VerificationLifecycle.from_snapshot(snapshot["verification"])
            try:
                if not root_binding_matches or not run_binding_matches or not snapshot_identity_matches:
                    raise ValueError("verification snapshot run binding does not match engine")
                self.replace_verification_lifecycle(lifecycle)
            except ValueError:
                self._verification_lifecycle = VerificationLifecycle(
                    state=VerificationState.failed,
                    restore_error="verification snapshot run binding failed",
                )
        else:
            self._verification_lifecycle = VerificationLifecycle()
        # Restore the terminal-only latch. Default False so snapshots with
        # no field present resume as if the nudge had not fired.
        self._terminal_only_active = bool(
            snapshot.get("terminal_only_active", False)
        )
        # Restore the delegation fact. Default False so a snapshot taken before
        # the field existed resumes as a run that never delegated — the
        # untouched side, which is the safe one.
        self._run_delegated = bool(snapshot.get("run_delegated", False))
        # Apply the deadline clock validated before any restore mutation.
        # The synthetic monotonic start can legitimately be negative when
        # persisted elapsed exceeds this process's uptime; zero alone is the
        # unstamped sentinel, so ``run()`` preserves every non-zero re-anchor.
        self._run_started_epoch = restored_deadline.epoch
        self._run_started_monotonic = restored_deadline.monotonic_anchor
        # Restore the deadline early-finalize latch. Default False so older
        # snapshots (no field present) resume as if the deadline nudge had not
        # yet fired, matching prior behaviour.
        self._pre_terminal_self_verify_used = bool(
            snapshot.get("pre_terminal_self_verify_used", False)
        )
        self._self_verify_extra_turns_used = int(
            snapshot.get("self_verify_extra_turns_used", 0)
        )
        # Restore the PRE-DISPATCH verify latch. Default False so older snapshots
        # resume as if no veto fired.
        self._pre_dispatch_terminal_verify_used = bool(
            snapshot.get("pre_dispatch_terminal_verify_used", False)
        )
        # Restore the preserved terminal candidate + its one-shot re-veto latch.
        # Default None/False so older snapshots resume as if no candidate was
        # preserved.
        restored_candidate = snapshot.get("terminal_candidate")
        self._terminal_candidate = (
            dict(restored_candidate)
            if isinstance(restored_candidate, dict)
            else None
        )
        self._terminal_candidate_reveto_used = bool(
            snapshot.get("terminal_candidate_reveto_used", False)
        )
        # Restore the one-shot prose-gate latch. Default False so older snapshots
        # resume as if the prose-gate had not yet fired.
        self._finalize_prose_gate_used = bool(
            snapshot.get("finalize_prose_gate_used", False)
        )
        # Restore the pointer refusal's attempt budget. Defaults to an unspent
        # budget so a snapshot predating this field resumes as if the refusal
        # had not yet engaged.
        self._pointer_answer_repair_attempts = int(
            snapshot.get("pointer_answer_repair_attempts", 0)
        )
        self._pointer_answer_repair_released = bool(
            snapshot.get("pointer_answer_repair_released", False)
        )
        # Restore the circuit-breaker state. Defaults to empty so an older
        # snapshot resumes with no tool blocked / no pending notification,
        # matching prior behaviour. Coerce defensively: only string entries are
        # kept (a corrupted snapshot cannot block a non-string "tool").
        restored_broken = snapshot.get("circuit_broken_tools")
        self._circuit_broken_tools = (
            {name for name in restored_broken if isinstance(name, str)}
            if isinstance(restored_broken, list)
            else set()
        )
        restored_notified = snapshot.get("circuit_breaker_notified_tools")
        self._circuit_breaker_notified_tools = (
            {name for name in restored_notified if isinstance(name, str)}
            if isinstance(restored_notified, list)
            else set()
        )
        # Restore the in-flight pre-trip streak so a resume before the trip keeps
        # the running count. Defensive: only a well-shaped
        # ``{tool_name: str, error_class: str, count: int>=0}`` survives; anything
        # else (or an older snapshot) resumes with no streak.
        restored_streak = snapshot.get("circuit_breaker_streak")
        if (
            isinstance(restored_streak, dict)
            and isinstance(restored_streak.get("tool_name"), str)
            and isinstance(restored_streak.get("error_class"), str)
            # ``bool`` is an ``int`` subclass — reject it explicitly so a
            # malformed ``count: true`` degrades to an empty streak instead of
            # restoring a non-empty one.
            and isinstance(restored_streak.get("count"), int)
            and not isinstance(restored_streak.get("count"), bool)
            and restored_streak["count"] >= 0
        ):
            self._circuit_breaker_streak = {
                "tool_name": restored_streak["tool_name"],
                "error_class": restored_streak["error_class"],
                "count": restored_streak["count"],
            }
        else:
            self._circuit_breaker_streak = None
        # Restore tool-precondition progress. Defaults (0/0/0/None) so a
        # snapshot taken before the run carried any preconditions resumes as if
        # none had been satisfied yet — which is also the correct reading for a
        # run that has none.
        self._tool_precondition_index = int(
            snapshot.get("tool_precondition_index", 0)
        )
        self._tool_precondition_calls = int(
            snapshot.get("tool_precondition_calls", 0)
        )
        self._tool_precondition_attempts = int(
            snapshot.get("tool_precondition_attempts", 0)
        )
        restored_precondition_error = snapshot.get("tool_precondition_last_error")
        self._tool_precondition_last_error = (
            restored_precondition_error
            if isinstance(restored_precondition_error, str)
            else None
        )
        # Restore the declared-file read-back state. Defaults (empty/0) so a
        # snapshot predating this driver resumes owing nothing, which is also
        # the correct reading for a run in which no tool declared a file.
        restored_pending_reads = snapshot.get("pending_read_paths")
        self._pending_read_paths = (
            [p for p in restored_pending_reads if isinstance(p, str) and p]
            if isinstance(restored_pending_reads, list)
            else []
        )
        restored_reads_satisfied = snapshot.get("pending_reads_satisfied")
        self._pending_reads_satisfied = (
            {p for p in restored_reads_satisfied if isinstance(p, str) and p}
            if isinstance(restored_reads_satisfied, list)
            else set()
        )
        restored_reads_abandoned = snapshot.get("pending_reads_abandoned")
        self._pending_reads_abandoned = (
            {p for p in restored_reads_abandoned if isinstance(p, str) and p}
            if isinstance(restored_reads_abandoned, list)
            else set()
        )
        self._pending_reads_forced_attempts = int(
            snapshot.get("pending_reads_forced_attempts", 0)
        )
        # restore the large-file convergence state. Defaults
        # (0 / [] / None / False) so a snapshot predating this feature resumes
        # as if the driver had not yet engaged, matching prior behaviour.
        self._turns_since_last_byte_adding_mutation = int(
            snapshot.get("turns_since_last_byte_adding_mutation", 0)
        )
        self._longfile_forced_appends = int(
            snapshot.get("longfile_forced_appends", 0)
        )
        self._longfile_forced_finalizes = int(
            snapshot.get("longfile_forced_finalizes", 0)
        )
        restored_active_path = snapshot.get("longfile_active_path")
        self._longfile_active_path = (
            restored_active_path if isinstance(restored_active_path, str) else None
        )
        self._longfile_active_file_bytes = int(
            snapshot.get("longfile_active_file_bytes", 0)
        )
        self._longfile_active_file_lines = int(
            snapshot.get("longfile_active_file_lines", 0)
        )
        restored_deltas = snapshot.get("longfile_mutation_deltas")
        self._longfile_mutation_deltas = (
            [int(d) for d in restored_deltas]
            if isinstance(restored_deltas, list)
            else []
        )
        self._longfile_last_mutation_truncated = bool(
            snapshot.get("longfile_last_mutation_truncated", False)
        )
        self._longfile_finalized = bool(snapshot.get("longfile_finalized", False))
        # Restore the per-path truncation latch + append counter. Defaults (empty
        # set/dict) so an older snapshot resumes as if no path had truncated and
        # no path had any forced appends.
        restored_truncated_paths = snapshot.get("longfile_truncated_paths")
        self._longfile_truncated_paths = (
            {p for p in restored_truncated_paths if isinstance(p, str)}
            if isinstance(restored_truncated_paths, list)
            else set()
        )
        restored_appends = snapshot.get("longfile_appends_per_path")
        self._longfile_appends_per_path = (
            {
                str(k): int(v)
                for k, v in restored_appends.items()
                if isinstance(v, int) and not isinstance(v, bool)
            }
            if isinstance(restored_appends, dict)
            else {}
        )
        self._longfile_salvage_seq = int(snapshot.get("longfile_salvage_seq", 0))
        # TERMINAL SEAL — restore the one-shot latch; default False so a snapshot
        # predating this seam resumes as if the run-end seal had not yet fired.
        # VOLUNTARY SEAL — restore the one-shot latch; default False so a
        # snapshot predating this seam resumes as if the voluntary seal had not
        # yet fired.
        self._longfile_voluntary_seal_used = bool(
            snapshot.get("longfile_voluntary_seal_used", False)
        )
        restored_pins = snapshot.get("pinned_tool_result_ids")
        self._pinned_tool_result_ids = (
            {str(x) for x in restored_pins if isinstance(x, str)}
            if isinstance(restored_pins, list)
            else set()
        )
        restored_ident = snapshot.get("identical_tool_counts")
        self._identical_tool_counts = (
            {
                str(k): int(v)
                for k, v in restored_ident.items()
                if isinstance(v, int) and not isinstance(v, bool)
            }
            if isinstance(restored_ident, dict)
            else {}
        )
        self._loop_guard_nudge_count = int(
            snapshot.get("loop_guard_nudge_count", 0)
        )
        restored_steer = snapshot.get("steer_queue")
        self._steer_queue = (
            [item for item in restored_steer if isinstance(item, dict)]
            if isinstance(restored_steer, list)
            else []
        )
        restored_follow = snapshot.get("follow_up_queue")
        self._follow_up_queue = (
            [item for item in restored_follow if isinstance(item, dict)]
            if isinstance(restored_follow, list)
            else []
        )
        live_model = snapshot.get("live_model_name")
        self._live_model_name = live_model if isinstance(live_model, str) else None
        if "live_thinking_enabled" in snapshot:
            thinking_flag = snapshot.get("live_thinking_enabled")
            self._live_thinking_enabled = (
                bool(thinking_flag) if thinking_flag is not None else None
            )
        else:
            self._live_thinking_enabled = None
        live_effort = snapshot.get("live_reasoning_effort")
        self._live_reasoning_effort = (
            live_effort if isinstance(live_effort, str) else None
        )
        self._run_settled_emitted = bool(snapshot.get("run_settled_emitted", False))
        restored_ledger = snapshot.get("tool_call_ledger") or []
        self._tool_call_ledger = [
            {
                "seq": int(entry.get("seq", 0)),
                "name": str(entry.get("name", "")),
                "ok": bool(entry.get("ok", False)),
            }
            for entry in restored_ledger
            if isinstance(entry, dict)
        ]
        self._tool_call_ledger_seq = int(
            snapshot.get("tool_call_ledger_seq", len(self._tool_call_ledger))
        )
        self._tool_call_ledger_truncated = bool(
            snapshot.get("tool_call_ledger_truncated", False)
        )
        restored_cause = snapshot.get("soft_stop_cause")
        self._soft_stop_cause = (
            restored_cause if isinstance(restored_cause, str) and restored_cause else None
        )
        restored_stage = snapshot.get("soft_stop_stage")
        self._soft_stop_stage = restored_stage if isinstance(restored_stage, str) else ""
        from protocore.runtime.intent import IntentRecord
        from protocore.runtime.lanes import Lane
        from protocore.runtime.usage_ledger import UsageRow

        restored_intents = snapshot.get("open_intents") or []
        self.open_intents = [
            IntentRecord(
                operation_id=str(item.get("operation_id", "")),
                tool_name=str(item.get("tool_name", "")),
                tool_call_id=str(item.get("tool_call_id", "")),
                reserved_result_ids=list(item.get("reserved_result_ids") or []),
                replay=item.get("replay") or "safe",
                status=item.get("status") or "open",
                result=item.get("result"),
            )
            for item in restored_intents
            if isinstance(item, dict)
        ]
        restored_usage = snapshot.get("usage_rows") or []
        self.usage_rows = [
            UsageRow(
                seq=int(item.get("seq", 0)),
                kind=str(item.get("kind", "inference")),
                run_id=str(item.get("run_id", "")),
                operation_id=item.get("operation_id"),
                input_tokens=int(item.get("input_tokens", 0)),
                output_tokens=int(item.get("output_tokens", 0)),
                success=bool(item.get("success", True)),
            )
            for item in restored_usage
            if isinstance(item, dict)
        ]
        restored_lanes = snapshot.get("lanes") or []
        self.lanes = [
            Lane(
                lane_id=str(item.get("lane_id", "main")),
                cursor=int(item.get("cursor", 0)),
                model=str(item.get("model", "")),
                toolset=tuple(item.get("toolset") or ()),
                locked_by=item.get("locked_by"),
                diverged=bool(item.get("diverged", False)),
            )
            for item in restored_lanes
            if isinstance(item, dict)
        ]

    async def _persist_snapshot(self) -> None:
        """Write the snapshot via :class:`IEventStream`.

 The call lives on the engine so test doubles can intercept it through
 the in-memory :class:`IEventStream`. A real host writes the snapshot to
 its own hot store — a Redis ``HSET run:{run_id}``, for instance.
 """
        # The IEventStream protocol exposes ``emit`` / ``subscribe``
        # / ``trim``; durable snapshot persistence belongs to the host. The
        # core's contract is: the engine emits a ``state_changed`` envelope
        # carrying the snapshot payload, and the host's adapter writes it
        # durably. This keeps the core self-sufficient against the in-memory
        # doubles.
        from protocore.contracts.types import Event as DurableEvent

        # ``snapshot()`` builds
        # ``history: [m.model_dump(mode="json") for m in self.history]`` —
        # a pure-CPU serialization of the FULL engine history that runs at
        # MANY turn/tool boundaries via ``_persist_snapshot``. On a single
        # executor event loop shared by several runs, a large-history dump
        # is a synchronous CPU section that can starve a neighbour run's
        # pending provider socket read and produce a false
        # ``provider stream produced no data`` stall. Offload the
        # serialization to a worker thread. ``snapshot()`` only READS
        # engine state (no event-loop-only objects are touched), and
        # ``await asyncio.to_thread`` blocks THIS run's own coroutine until
        # the thread completes, so no concurrent history mutation can race
        # the dump — the result is observationally identical to the
        # previous synchronous call.
        snapshot = await asyncio.to_thread(self.snapshot)
        await self.events.emit(
            DurableEvent(
                run_id=self.config.run_id,
                name="state_snapshot",
                payload={
                    "tenant_id": self.config.tenant_id,
                    "snapshot": snapshot,
                },
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def latest_user_message(self) -> Message | None:
        for msg in reversed(self.history):
            if msg.role is MessageRole.user:
                return msg
        return None

    @property
    def effective_model_name(self) -> str:
        return self._live_model_name or self.config.model_name

    @property
    def effective_thinking_enabled(self) -> bool:
        if self._live_thinking_enabled is None:
            return self.config.thinking_enabled
        return self._live_thinking_enabled

    @property
    def effective_reasoning_effort(self) -> str:
        return self._live_reasoning_effort or self.config.reasoning_effort

    def apply_live_controls(
        self,
        *,
        model_name: str | None = None,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        """Apply mid-run model/thinking for the next provider call."""
        from protocore.runtime.live_control import validate_thinking_for_mode

        next_thinking = (
            self.effective_thinking_enabled
            if thinking_enabled is None
            else thinking_enabled
        )
        validate_thinking_for_mode(self.config.run_mode, next_thinking)
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError("invalid_reasoning_effort")
        if model_name is not None:
            self._live_model_name = model_name
        if thinking_enabled is not None:
            self._live_thinking_enabled = thinking_enabled
        if reasoning_effort is not None:
            self._live_reasoning_effort = reasoning_effort

    def pin_tool_result(self, tool_call_id: str) -> None:
        self._pinned_tool_result_ids.add(tool_call_id)

    @property
    def is_terminal(self) -> bool:
        return is_terminal(self.state)

    def needs_compaction(self) -> bool:
        return self.context_manager.needs_compaction(
            self.history,
            observed_prompt_tokens=self.last_observed_prompt_tokens,
        )

    def needs_emergency_compaction(self) -> bool:
        """Return True when history exceeds the emergency cliff (proactive force)."""
        return self.context_manager.needs_emergency_compaction(
            self.history,
            observed_prompt_tokens=self.last_observed_prompt_tokens,
        )

    def history_snapshot(self) -> Sequence[Message]:
        return tuple(self.history)

    def new_tool_call_id(self) -> str:
        return f"toolu_{uuid.uuid4().hex[:12]}"


__all__ = [
    "REASONING_EFFORTS",
    "RUN_MODES",
    "QueryEngine",
    "QueryEngineConfig",
]
