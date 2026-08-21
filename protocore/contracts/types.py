"""Pure-core type system for Protocore.

Every conversation primitive flows through these types. All Pydantic v2
``BaseModel`` (or ``@dataclass(frozen=True, slots=True)`` for value
objects). Multilingual safe — Cyrillic-in-JSON-escape regression covered
in :mod:`tests.unit.test_types_serialization`.
"""
from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from protocore.constants import (
    MAX_ARTIFACTS,
    MAX_DATA_NESTING_DEPTH,
    MAX_DELEGATE_TASK_CHARS,
    MAX_ENVELOPE_PAYLOAD_CHARS,
    MAX_LLM_CALL_DETAILS,
    MAX_REPORT_EVENTS,
    MAX_SUBAGENT_RUNS,
    MAX_TOOL_CALL_ARGUMENT_BYTES,
    MAX_TOOL_CALL_DETAILS,
    MAX_WARNINGS,
)
from protocore.contracts.attempt_ledger import (
    AttemptLedger,
    DeliverableDeclaration,
)

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _reject_non_finite_floats(value: dict[str, Any], *, field: str) -> dict[str, Any]:
    """Reject ``nan``/``inf``/``-inf`` anywhere inside a metadata/payload dict.

    ``Message.metadata`` / :class:`ToolResultBlock.metadata` / ``Event.payload``
    are ``dict[str, Any]`` and the unit invariant is that a model
    "round-trips losslessly (esp. metadata)". Pydantic v2 serialises a
    non-finite ``float`` to the JSON token ``null`` (the only legal JSON
    representation) and parses it back as ``None`` — a SILENT lossy round-trip
    (``{'x': inf}`` → ``{'x': None}``) that corrupts the value with no error.

    These dicts are documented to hold bool/string/int/float control flags and
    structured side-channels, never a sentinel non-finite float, so we fail
    CLOSED: a non-finite float raises at construction (and on
    ``model_validate``/``model_validate_json``) instead of being silently
    nulled. This makes the lossy case an explicit, observable rejection. The
    scan recurses through nested ``dict``/``list``/``tuple`` so a non-finite
    value buried in a nested structure is also caught.
    """
    _scan_for_non_finite(value, field=field, path=field)
    return value


def _scan_for_non_finite(value: Any, *, field: str, path: str) -> None:
    """Walk ``value`` depth-first rejecting non-finite floats and runaway nesting.

    Iterative on an explicit stack rather than recursive, and depth-bounded by
    :data:`~protocore.constants.MAX_DATA_NESTING_DEPTH`. Both properties are
    about the same failure: this validator runs on ``Message.metadata`` and
    ``ToolResultBlock.metadata``, i.e. on every streamed event, over a structure
    the model supplies. A recursive walk over a payload nested a few thousand
    levels deep raises ``RecursionError`` from inside Pydantic validation, which
    unwinds through the run and arrives at the loop's catch-all carrying no hint
    of where it came from. An explicit stack cannot exhaust the interpreter at
    all, and the depth bound turns the pathological payload into a named
    rejection naming the path it gave up on.
    """
    stack: list[tuple[Any, str, int]] = [(value, path, 0)]
    while stack:
        item, item_path, depth = stack.pop()
        # bool is an int subclass but never non-finite; check float explicitly
        # (and exclude bool so True/False are untouched).
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(
                f"{field} may not contain non-finite floats "
                f"(nan/inf/-inf) — found {item!r} at {item_path}; JSON has no "
                f"lossless representation (it would silently become null/None)"
            )
        if not isinstance(item, (dict, list, tuple)):
            continue
        if depth >= MAX_DATA_NESTING_DEPTH:
            raise ValueError(
                f"{field} is nested deeper than {MAX_DATA_NESTING_DEPTH} "
                f"levels — gave up at {item_path}; a structure this deep is "
                "not a value this field carries and walking it would exhaust "
                "the interpreter stack"
            )
        if isinstance(item, dict):
            for key, child in item.items():
                stack.append((child, f"{item_path}.{key}", depth + 1))
        else:
            for index, child in enumerate(item):
                stack.append((child, f"{item_path}[{index}]", depth + 1))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MessageRole(StrEnum):
    """Anthropic-aligned message roles."""

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class StopReason(StrEnum):
    """Why a turn terminated. Drives loop policy."""

    end_turn = "end_turn"
    tool_use = "tool_use"
    max_tokens = "max_tokens"
    max_turns = "max_turns"
    stop_sequence = "stop_sequence"
    error = "error"
    cancelled = "cancelled"


TERMINAL_TOOL_METADATA_KEY = "protocore.terminal_tool"
"""ToolResult metadata flag that marks a successful tool as loop-terminal."""

TOOL_RESULT_COUNT_AS_ERROR_METADATA_KEY = "protocore.count_as_tool_error"
"""ToolResult metadata flag (bool) gating the per-run ``tool_errors_count``.

Defaults to ``True`` (absent ⇒ count) so every historical soft/hard error
path keeps incrementing the counter. A tool that returns
``ToolResult(is_error=True)`` for a *non-failure* condition (e.g. ``Bash``
surfacing an ordinary nonzero process exit such as ``grep -q`` no-match,
``test``/``[`` false, or a ``diff`` difference — exit status used as DATA,
not a tool failure) sets this ``False`` so the dispatcher does NOT call
``_record_tool_error`` for it. That keeps the run from being downgraded to
``partial``. The model still sees ``success=false`` + ``exit_code`` in the
rendered content."""

TOOL_RESULT_CONSECUTIVE_CAP_ELIGIBLE_METADATA_KEY = (
    "protocore.consecutive_error_cap_eligible"
)
"""ToolResult metadata flag (bool) gating the generic consecutive-error cap.

Defaults to ``True`` (absent ⇒ eligible). Set ``False`` by a tool whose
soft ``is_error=True`` is not a genuine failure (see
:data:`TOOL_RESULT_COUNT_AS_ERROR_METADATA_KEY`) so a repeated benign
nonzero (e.g. polling ``grep -q`` in a loop) cannot trip
``_apply_consecutive_error_cap`` and force a spurious intervention."""

PENDING_READS_METADATA_KEY = "protocore.pending_reads"
"""ToolResult metadata field (list of paths) the CALLER must open before it
continues.

The protocol for a tool whose real output is FILES rather than the text it
returns — a delegation that writes its findings to disk, a job that renders an
export. The result body names the paths; this key says they are an OBLIGATION,
not a courtesy. While one of them is unread the loop names the workspace read
tool in ``LLMRequest.extra['forced_tool_choice']``, so the caller cannot answer
from the one-line pointer alone; the gate releases itself the moment the last
path has been read. Enforced by :mod:`protocore.runtime.pending_reads`.

Any tool may set it and the runtime knows nothing about which — a bare list of
non-empty strings, anything else ignored. Absent (the common case) leaves the
run bit-identical."""

TERMINAL_TOOL_STATUS_METADATA_KEY = "protocore.terminal_status"
"""Optional ToolResult metadata field describing the terminal outcome."""

TERMINAL_TOOL_STATUS_COMPLETED = "completed"
"""Terminal-tool status value for a completed external workflow."""

PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY = "protocore.partial_assistant_attempt"
"""``Message.metadata`` flag for an incomplete assistant stream attempt.

The runtime may persist text that was already delivered before a provider retry,
fallback, or output-cap recovery. It belongs in durable history so reload matches
the live stream, but it is not the run's final answer. Persistence adapters and
session projections use this explicit marker instead of guessing from text or
message order. Absence keeps historical and ordinarily completed turns unchanged.
"""

SYNTHETIC_RECOVERY_METADATA_KEY = "protocore.synthetic_recovery"
"""``Message.metadata`` flag marking a turn as runtime-created recovery
scaffolding (NOT genuine model output).

Some recovery paths synthesise assistant turns that are NOT the model's own
words — the post-tool empty-response nudge appends a synthetic
``assistant(post_tool_empty_nudge_assistant_text)`` (default the literal
``(empty)`` marker), and the guaranteed-terminal backstop appends a synthetic
assistant ``tool_use`` turn. Such turns MUST be excluded from
``_latest_durable_answer_text`` so the runtime never submits its own
scaffolding (e.g. the empty-marker) as the model's durable answer
(never-fabricate). The value names the source
(``"post_tool_empty_nudge"`` / ``"guaranteed_terminal_scaffold"``) for
observability."""

SYNTHETIC_RECOVERY_POST_TOOL_EMPTY_NUDGE = "post_tool_empty_nudge"
"""``SYNTHETIC_RECOVERY_METADATA_KEY`` value for the post-tool empty nudge."""

SYNTHETIC_RECOVERY_TRUNCATION_CONTINUE = "tool_call_truncation_recovery"
"""``SYNTHETIC_RECOVERY_METADATA_KEY`` value for truncation recovery user nudges."""

SYNTHETIC_RECOVERY_MAX_OUTPUT_CONTINUE = "max_output_token_recovery"
"""``SYNTHETIC_RECOVERY_METADATA_KEY`` value for max-output resume user nudges."""

SYNTHETIC_RECOVERY_THINKING_CONTINUE = "thinking_continue_prompt"
"""``SYNTHETIC_RECOVERY_METADATA_KEY`` value for thinking-only continue nudges."""

SYNTHETIC_RECOVERY_PRE_TERMINAL_SELF_VERIFY = "pre_terminal_self_verify"
"""``SYNTHETIC_RECOVERY_METADATA_KEY`` value for pre-terminal self-verify nudges."""

SYNTHETIC_RECOVERY_LONGFILE_CONTINUE = "longfile_continue"
"""``SYNTHETIC_RECOVERY_METADATA_KEY`` value for long-file continue nudges."""

SYNTHETIC_RECOVERY_TERMINAL_REPAIR = "terminal_candidate_repair"
"""``SYNTHETIC_RECOVERY_METADATA_KEY`` value for terminal-candidate repair nudges."""

SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR = "finalize_prose_gate_repair"
"""``SYNTHETIC_RECOVERY_METADATA_KEY`` value for the prose-gate repair nudge —
injected when a run is about to latch a BACKGROUND terminal tool result without
any substantive visible assistant prose. The bounded user turn asks the model to
emit the final answer as normal text first, then call the terminal tool. It is
runtime scaffolding (not user-authored), so it must be filtered from the durable
transcript like the other synthetic user nudges."""

SYNTHETIC_RECOVERY_PRE_DISPATCH_TERMINAL_VERIFY = "pre_dispatch_terminal_verify"
"""``SYNTHETIC_RECOVERY_METADATA_KEY`` value for pre-dispatch terminal verification."""

SYNTHETIC_RECOVERY_CIRCUIT_BREAKER = "tool_error_circuit_breaker"
"""``SYNTHETIC_RECOVERY_METADATA_KEY`` value for the repeated-tool-error
circuit-breaker corrective turn — injected ONCE when a tool crosses
``RuntimeConstants.max_consecutive_tool_errors`` consecutive failures of the same
error class and is hard-stopped for the rest of the run. The bounded user turn
tells the model the tool has been disabled and to answer/finalize from the
conversation instead of retrying. Runtime scaffolding (not user-authored), so it
is filtered from the durable transcript like the other synthetic user nudges."""

SYNTHETIC_RECOVERY_SANDBOX_DOWN_NUDGE = "sandbox_down_inline_strategy"
"""``SYNTHETIC_RECOVERY_METADATA_KEY`` value for sandbox-down strategy nudges."""

SYNTHETIC_RECOVERY_GUARANTEED_TERMINAL = "guaranteed_terminal_scaffold"
"""``SYNTHETIC_RECOVERY_METADATA_KEY`` value for guaranteed-terminal scaffold."""

SYNTHETIC_RECOVERY_TERMINAL_TOOL_NUDGE = "terminal_tool_nudge"
"""``SYNTHETIC_RECOVERY_METADATA_KEY`` value for the terminal-tool repair nudge."""

SYNTHETIC_RECOVERY_LONGFILE_SALVAGE = "longfile_truncation_salvage"
"""``SYNTHETIC_RECOVERY_METADATA_KEY`` value for the large-file truncation salvage
— the runtime synthesises a clean assistant ``tool_use`` for the recovered
partial ``content`` so the bytes land on disk and the convergence driver can
engage. NOT the model's own words → excluded from durable-answer scans."""

SYNTHETIC_RECOVERY_LONGFILE_TERMINAL_SEAL = "longfile_terminal_seal"
"""``SYNTHETIC_RECOVERY_METADATA_KEY`` value for the run-end terminal seal —
when a run is about to complete VOLUNTARILY while a truncation-gated large file
is complete-enough but UNSEALED, the runtime synthesises a clean assistant
``FinalizeFile`` ``tool_use`` for the active path and dispatches it BEFORE the
run completes (deterministic seal; NO LLM call, NO extra turn). NOT the model's
own words → excluded from durable-answer scans."""

COMPACTION_SUMMARY_METADATA_KEY = "protocore.compaction_summary"
"""``Message.metadata`` flag marking a turn as a Tier-2 compaction summary.

Set ``True`` on the ``<compacted-turn>`` message produced by
``run_tier2_summarisation``. That message is deliberately **user-role**, not
system: it replaces an aged turn in the MIDDLE of history, and vLLM rejects
any ``system`` message past index 0. This flag — not the role — is therefore
the ONLY reliable identity of a compaction summary, and every boundary that
re-serialises a message (snapshot/resume, durable persistence, projection)
must carry it. Survives snapshot/resume via
``model_dump``/``model_validate`` so a re-driven run recognises an
already-summarised turn by its DURABLE flag instead of the prior process-local
``str(id(obj))`` key (which minted a new value on every resume → re-summarise
churn + summary-of-summary decay)."""

COMPACTION_REFERENCE_METADATA_KEY = "protocore.compaction_reference"
"""``Message.metadata`` flag (bool) marking a non-tool message as a FROZEN
reference block — e.g. the executor's ``<environment_context>`` /
``<memory-context>`` bootstrap. When set, Tier-1 may compact the block to a
blobbed placeholder once it ages past the recent window AND exceeds the
truncation threshold, the same way it sheds large tool results. The original
task user turn is NOT tagged, so it is never shed by this path."""

SESSION_HISTORY_SEED_METADATA_KEY = "protocore.session_history_seed"
"""``Message.metadata`` flag (bool) marking a message as a PRIOR-RUN turn that
the host executor seeded into a NEW run's engine history (cross-run
history seeding). Set ``True`` on every message the executor rehydrates from
the durable ``session_messages`` projection of earlier runs in the same
session, BEFORE the new task's user turn.

Three classes of runtime consumer read this flag:

1. Compaction, which needs the boundary as POSITIONS into the list it is
   handed rather than as a filtered copy:
   :func:`protocore.runtime.context.compaction._first_user_turn_index` skips
   seed-tagged turns (like reference blocks) so the
   ``compaction_protect_first_user_turn`` guard protects the NEW task — the
   last user turn — not a seeded prior-run user turn that now precedes it in
   history, and
   :func:`protocore.runtime.context.compaction._session_history_seed_indices`
   selects exactly the seeded turns so the lossy Tier-2 collapse can be
   withheld from them. These are the two places outside
   :func:`~protocore.runtime.query._this_run_messages` allowed to name this
   key, plus the session-memory taggers that write it and this definition
   itself; the guard test names them explicitly rather than leaving them to be
   discovered.

   That guard does NOT match this constant by identity — a rule that reads
   source text cannot. It matches every NAME the constant is reachable under:
   the imported name, an ``import … as`` alias, a re-export of such an alias
   out of another module in the package — absolute or relative, from a plain
   module or from a package ``__init__.py`` — module-qualified access, any
   module-level or class-body alias of those transitively, a name of any of
   them spelled as a string in a dynamic lookup (written in the lookup or
   bound to a constant, a local or a container first), and the key's own
   string value, which it reads from this assignment rather than copying.
   Alias and re-export chains resolve to a bounded depth, and running out of
   that budget FAILS the guard rather than shortening its answer.

   Three things it does not follow, measured rather than assumed. A value
   assembled piecewise so that neither a name nor the value is ever written
   whole — ``"protocore.session" + "_history_seed"``, ``str.join``, a split
   f-string placeholder. An alias whose right-hand side is COMPUTED from the
   key rather than being the key, even when the computation is the identity
   (``f"{SESSION_HISTORY_SEED_METADATA_KEY}"``, ``str(…)``, ``…[:]``): the
   computing line is itself reported, so this hides an alias only in a module
   whose scope the guard already authorises — which means this one, where the
   key is defined. And ``from … import *`` of a re-exported alias, which the
   guard does not see at all; that is not coverage, it is the linter, which
   rejects the star import outright.

   The limits are stated here because an earlier version of this paragraph
   claimed a width the guard did not have — twice. The import alias, the most
   idiomatic spelling of all, walked past the first version; a re-export
   through a package ``__init__.py``, which is how every package in this tree
   re-exports, walked past the second, as did the constant's own name hoisted
   out of the lookup into a named constant. Each was landed against the real
   tree with the full suite, the type checker and the linter green before it
   was closed. Widen this sentence only against a measurement.
2. The host finalization mirror filters seed-tagged turns out of the
   ``history_snapshot()`` it persists via ``append_run_messages`` so prior-run
   conversation is NOT re-written under the new ``run_id`` (no duplication / no
   exponential session-history growth across runs).
3. Every reader in :mod:`protocore.runtime.query` that asks a question about
   what THIS run did or answered. ``Message`` carries no run id, so this flag
   is the ONLY run boundary those readers have, and they all reach it through
   ONE function: :func:`~protocore.runtime.query._this_run_messages`, which is
   the only place in that module permitted to name this key.
   :func:`~protocore.runtime.query._this_run_model_turns` narrows it to the
   model's own words for the answer paths; the helpers that must also see
   ``role=tool`` results — write accounting, terminal-result detection,
   precondition rehydration — take the wider sequence.

   A guard test (``tests/unit/runtime/test_history_run_boundary.py``) enforces
   this, because a docstring cannot and four readers were written without it.
   Be precise about what the guard does: it enumerates every function in
   ``protocore/`` that asks a question of the transcript — through the
   attribute on ANY receiver, through its name spelled as a string, through a
   dynamic lookup, or handed in as a parameter annotated as a container of
   ``Message`` — and refuses one that is not declared. Changing the transcript
   is exempt, and so is a parameter whose contents are only written somewhere
   else; only questions are registered. Any entry whose reason claims "this
   reaches the whole session and yet answers about ONE run" must name a test
   that pins the claim, and that test must name the entry back
   (``tests/unit/runtime/test_history_registry_claims.py``), because such a
   reason once went false under a one-word edit with nothing failing, and a
   one-sided pin could later be dropped by reclassifying the entry.

   Be equally precise about what it does NOT do, because a limit stated too
   narrowly is worse than one not stated at all. It does not make this class of
   bug impossible. An already-declared function can still be edited into a
   run-scoped reader. The parameter route is annotation-keyed, so ``turns: Any``
   and an unannotated parameter carry nothing for it to read and are invisible
   — and the CALLER of such a helper is a finding only when that caller is not
   itself already in the registry, which is often exactly where such a helper
   is called from. A transcript reached through a value no static check can
   follow back to a name is invisible too. And it stops at the edge of this
   package — the seeding splice and at least one same-shaped helper live in the
   service repository, where it does not reach.

   Both directions of forgetting the flag are live failures, and neither
   announces itself. On the answer paths an earlier run's prose is attributed
   to the current one and reaches the user as a fluent, plausible answer to a
   question they did not ask. On the "has this run finished?" paths — a seeded
   terminal tool result, a seeded file write, a seeded satisfied precondition —
   an earlier run's success answers for the current one, and the guards that
   exist to stop a run ending unanswered stand down for a run that has produced
   nothing. Stored history reaches these readers the same way, since every
   rehydrated row is tagged on the way in: data written before a reader was
   scoped is excluded once it is, but until then that data could disarm a
   later run.

Survives snapshot/resume via ``model_dump``/``model_validate``. Tier-1 shedding
preserves it (``model_copy`` keeps ``metadata``); seeded turns are added to the
Tier-2 protected set so a summary never silently drops the tag."""


class RunStatus(StrEnum):
    """Persistent run state. Mirrors PG ``runs.status`` column.

    ``partial`` is the terminal status assigned to a run that completed its
    agent loop but accumulated one or more tool dispatch errors
    (``runs.tool_errors_count > 0``). It is functionally terminal — distinct
    from ``completed`` (no tool errors) and from ``error`` (engine itself
    failed).
    """

    queued = "queued"
    running = "running"
    completed = "completed"
    partial = "partial"
    error = "error"
    cancelled = "cancelled"
    incomplete = "incomplete"
    paused = "paused"


class HookEvent(StrEnum):
    """The 10 hook events.

    Spec lives in core; executor (HTTP POST URL / LLM-as-hook prompt) is
    the host. 8 base events + ``subagent_start`` (sync, can DENY) +
    ``subagent_stop`` (async, observer only) for subagent observability.
    """

    pre_tool_use = "pre_tool_use"
    post_tool_use = "post_tool_use"
    user_prompt_submit = "user_prompt_submit"
    session_start = "session_start"
    session_end = "session_end"
    pre_compact = "pre_compact"
    post_compact = "post_compact"
    file_changed = "file_changed"
    subagent_start = "subagent_start"
    subagent_stop = "subagent_stop"


class ContentBlockKind(StrEnum):
    """Content block kinds. Anthropic-aligned + thinking + image_ref."""

    text = "text"
    tool_use = "tool_use"
    tool_result = "tool_result"
    thinking = "thinking"
    image_ref = "image_ref"


class BlockVisibility(StrEnum):
    """Where one content block belongs in a reader's prose stream.

    The durable transcript has always carried this judgement per render unit,
    and a client that reads history filters on it. The live stream did not, so
    the same content arrived unmarked while the run was in flight and marked
    once it was over — a reader had no way to tell the model's working from its
    answer until reload, and rendered a message bubble (with its own feedback
    controls) for every intermediate narration.

    The vocabulary is deliberately identical to the durable one so a client
    applies ONE rule to both sources:

    ``PUBLIC``
        Prose addressed to the person who asked. Open a bubble.
    ``COLLAPSED``
        Real content that is not prose — the model's narration between tool
        calls, its reasoning, a tool's result body. Renderable as a chip or an
        expandable detail, never as a message of its own.
    ``HIDDEN``
        Content the reader is not shown at all (redacted reasoning).
    ``DEBUG``
        Diagnostics for an operator surface, never for an end user.

    ``PUBLIC`` is the default everywhere, so a consumer that does not know the
    field behaves exactly as one that predates it.

    It lives in ``contracts`` rather than with the stream event types because
    both consumers need it and neither owns it: the live stream states it per
    ``content_block_start`` / ``content_block_stop``, and a durable
    :class:`TextBlock` can carry it as its own settled property.
    """

    PUBLIC = "public"
    COLLAPSED = "collapsed"
    HIDDEN = "hidden"
    DEBUG = "debug"


# ---------------------------------------------------------------------------
# Content blocks — Anthropic-style structured content
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TextBlock(BaseModel):
    """Plain text content block."""

    model_config = ConfigDict(frozen=True)

    kind: Literal[ContentBlockKind.text] = ContentBlockKind.text
    text: str
    visibility: BlockVisibility = BlockVisibility.PUBLIC
    """Where this block belongs in the reader's prose stream.

    Almost every reader-facing judgement about a text block is made from the
    STRUCTURE of the row it sits in — text sharing a message with a tool call
    that continues the run is narration, and both the live stream and the
    durable projection reach that verdict independently from the same fact.
    Those callers leave this at ``PUBLIC`` and the structural rule decides.

    The field exists for the one case structure cannot express: a single text
    block whose head is narration and whose body is the answer. Splitting it in
    two leaves two blocks the row-level rule cannot tell apart, so the block
    itself has to say which is which, and a reader that trusts the mark gets
    the same answer live and after a reload.

    Serialised only when it is NOT ``PUBLIC`` — see
    :meth:`_omit_default_visibility`.
    """

    @model_serializer(mode="wrap")
    def _omit_default_visibility(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        """Leave ``visibility`` out of the dump when it is the default.

        The blocks that have something to say about where they belong are the
        rare ones: a narration prefix cut off the answer behind it. Every other
        text block ever written — every user turn, every ordinary answer —
        carries ``PUBLIC``, which is exactly what a reader assumes when the key
        is absent. Emitting it anyway would rewrite the stored bytes of every
        block in every durable blob to say nothing.

        Done here rather than with ``exclude_defaults`` at a dump site because
        the dump sites are shared: ``kind`` is itself a defaulted field on every
        block class, so excluding defaults would strip the discriminator the
        durable JSONL is read back by.
        """

        data: dict[str, Any] = handler(self)
        if data.get("visibility") == BlockVisibility.PUBLIC:
            del data["visibility"]
        return data


class ThinkingBlock(BaseModel):
    """Model 'thinking' reasoning — usually stripped before persistence."""

    model_config = ConfigDict(frozen=True)

    kind: Literal[ContentBlockKind.thinking] = ContentBlockKind.thinking
    text: str


class ImageRefBlock(BaseModel):
    """Image reference (blob ref) — content lives in IBlobStore."""

    model_config = ConfigDict(frozen=True)

    kind: Literal[ContentBlockKind.image_ref] = ContentBlockKind.image_ref
    blob_ref: str
    mime_type: str = "image/png"


class ToolUseBlock(BaseModel):
    """Assistant-emitted tool invocation."""

    model_config = ConfigDict(frozen=True)

    kind: Literal[ContentBlockKind.tool_use] = ContentBlockKind.tool_use
    tool_call_id: str
    name: str
    arguments_json: str

    @field_validator("arguments_json")
    @classmethod
    def _cap_arguments(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_TOOL_CALL_ARGUMENT_BYTES:
            raise ValueError(
                f"tool-call arguments exceed {MAX_TOOL_CALL_ARGUMENT_BYTES} bytes"
            )
        return value


class ToolResultBlock(BaseModel):
    """Tool-call result returned to the model."""

    model_config = ConfigDict(frozen=True)

    kind: Literal[ContentBlockKind.tool_result] = ContentBlockKind.tool_result
    tool_call_id: str
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _reject_non_finite_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _reject_non_finite_floats(value, field="ToolResultBlock.metadata")


ContentBlock = TextBlock | ThinkingBlock | ImageRefBlock | ToolUseBlock | ToolResultBlock


# ---------------------------------------------------------------------------
# Message — sole conversation primitive
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """Single conversation turn (role-scoped).

 v2 baseline: assistant turns carry ``content_blocks`` (text + tool_use
 + thinking interleaved). User/system/tool turns are typically a single
 text block. ``content`` is a convenience read-only join.

 ``reasoning_content`` (H8) persists the model's chain-of-thought
 from thinking-capable providers (DeepSeek-R1, Kimi K2 Thinking,
 Anthropic extended thinking, GLM-4.5/5). Stored separately from
 ``content_blocks`` so that:

 1. Providers like DeepSeek/Kimi can re-inject the prior turn's
 reasoning_content on the next API call (some upstream providers
 400 without it when ``tool_calls`` are present).
 2. The dashboard "show thinking" toggle reads a single canonical
 field instead of walking every ThinkingBlock.
 3. The compaction layer can drop reasoning_content while keeping
 ``ThinkingBlock`` content blocks if both are present.

 See ``
 Bug 6 for the production failure mode (multi-turn DeepSeek loses
 chain-of-thought; provider 400s on follow-up tool calls).
 """

    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content_blocks: list[ContentBlock] = Field(default_factory=list)
    reasoning_content: str | None = None
    """Persisted chain-of-thought for thinking-capable providers.

    ``None`` for providers / models that do not emit a reasoning stream.
    Adapters populate this on assistant-role messages built from a stream
    that emitted ``ProviderDeltaKind.thinking`` events; the value is the
    accumulated reasoning text for the turn. Re-emitted to the provider on
    the next turn iff the provider requires it (DeepSeek, Kimi K2).
    """

    created_at: datetime = Field(default_factory=_utcnow)

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Out-of-band per-turn annotations that are NOT sent to the model.

    Used by the runtime to tag turns it synthesised itself — e.g.
    :data:`SYNTHETIC_RECOVERY_METADATA_KEY` marks recovery scaffolding (the
    post-tool empty nudge's ``(empty)`` turn, the guaranteed-terminal tool-use
    turn) so :func:`protocore.runtime.query._latest_durable_answer_text` can
    exclude it and never submit runtime scaffolding as the model's answer
    Survives snapshot/resume via ``model_dump`` / ``model_validate``
    (``QueryEngine.snapshot``). Default empty → no annotations, no wire effect."""

    @field_validator("metadata")
    @classmethod
    def _reject_non_finite_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _reject_non_finite_floats(value, field="Message.metadata")

    @model_validator(mode="after")
    def _validate_blocks(self) -> Self:
        # System/user/tool messages: at most one block. Assistant: free.
        if self.role in (MessageRole.system, MessageRole.user) and len(self.content_blocks) > 1:
            raise ValueError(f"role={self.role} permits at most one content block")
        # reasoning_content only meaningful on assistant turns. Non-assistant
        # roles MUST NOT carry it — otherwise we silently waste tokens
        # re-injecting it on the next turn.
        if self.reasoning_content is not None and self.role is not MessageRole.assistant:
            raise ValueError(
                f"reasoning_content is only valid on assistant messages, "
                f"got role={self.role}"
            )
        return self

    @property
    def text(self) -> str:
        """Concatenate visible text blocks (skip thinking & tool blocks)."""
        return "".join(b.text for b in self.content_blocks if isinstance(b, TextBlock))


# ---------------------------------------------------------------------------
# ToolCall / ToolResult — adapter-facing primitives
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """LLM-emitted tool invocation surfaced to :class:`Tool.invoke`."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    truncated_by_output_cap: bool = False
    """``True`` when the provider terminated the assistant turn with
    ``finish_reason="length"`` while these tool-call args were still being
    streamed (incomplete JSON, closed only by brace balancing in the SSE
    parser). Drives the mid-tool-call recovery branch in
    :func:`protocore.runtime.query._stream_one_assistant_message`: the loop
    synthesises a resume nudge naming the truncated tool(s) and re-streams up
    to :attr:`RuntimeConstants.max_output_recovery_rounds` times rather than
    dispatching the partial call (which would silently corrupt large-file
    writes)."""

    args_partial_truncated: bool = False
    """``True`` when the SSE parser had to synthesise braces to close an
    incomplete args JSON stream, REGARDLESS of the ``finish_reason``.
    Distinct from :attr:`truncated_by_output_cap` (which fires only on
    ``finish_reason="length"``); this flag also fires on the local-model
    case where ``finish_reason="stop"`` arrives after only ``{`` of the
 args, leaving the tool call orphaned. The loop checks this flag in
 combination with the stream's ``finish_reason`` to surface a
 synthetic error tool result back to the agent so it can re-chunk
 the output instead of silently failing."""


class ToolResult(BaseModel):
    """Result of a single tool invocation.

    ``evidence_records`` is a trusted runtime side channel.  It is deliberately
    not copied to :class:`ToolResultBlock`, result metadata, or the public
    event payload: those representations are model-visible and may be compacted.
    The dispatch runtime validates and forwards records to the engine-owned
    evidence ledger instead.
    """

    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    evidence_records: tuple[Any, ...] = ()

    @model_validator(mode="after")
    def _validate_evidence_records(self) -> Self:
        """Accept only immutable typed evidence from successful tools.

        ``EvidenceRecord`` lives in ``contracts.verification`` which imports
        the conversation block types in this module.  Resolving it lazily here
        preserves that acyclic contract layering while still rejecting raw
        mappings, text, and arbitrary objects at the tool boundary.
        """
        if self.is_error and self.evidence_records:
            raise ValueError("an error ToolResult must not contain evidence records")
        if not self.evidence_records:
            return self
        from protocore.contracts.verification import EvidenceRecord

        if any(not isinstance(record, EvidenceRecord) for record in self.evidence_records):
            raise ValueError("ToolResult evidence_records must contain EvidenceRecord values")
        return self


class ToolPrecondition(BaseModel):
    """One tool a run MUST call before the agent is free to answer.

    Carried as an ORDERED tuple on
    :attr:`~protocore.runtime.query_engine.QueryEngineConfig.tool_preconditions`
    and enforced by :mod:`protocore.runtime.run_tool_preconditions`, which
    names :attr:`tool` in the provider's native ``tool_choice`` until the entry
    is satisfied. Prompt wording only makes a first tool call likely; this
    makes it a property of the run.

    NOT the per-tool dependency DAG in
    :mod:`protocore.runtime.tool_preconditions` (:attr:`ToolDefinition.
    preconditions`), which decides whether a tool the model chose may run at
    all. This is a RUN-level obligation the caller states up front, and it
    forces rather than blocks.

    The order is load-bearing rather than cosmetic: ``tool_choice`` names
    exactly ONE tool per request, so entries can only be satisfied in
    sequence. For the same reason a repeated :attr:`tool` is meaningful —
    ``[A, B, A]`` means A, then B, then A again — which is why progress is an
    index into the tuple and never a set of satisfied names.
    """

    model_config = ConfigDict(frozen=True)

    tool: str
    """The tool's exact registered name, as advertised to the model.

    Forcing a name the request does not advertise makes the provider reject
    the whole request, so the caller-facing layer validates this against the
    run's resolved tool surface (after policy, clipping and discovery) rather
    than against the bundled registry.
    """

    calls: int = Field(default=1, ge=1)
    """How many SUCCESSFUL calls satisfy this entry.

    A call that errored did not run, so it never counts towards this total —
    it only spends an attempt against
    :attr:`RuntimeConstants.run_tool_precondition_max_attempts`. The upper
    bound is :attr:`RuntimeConstants.run_tool_precondition_max_calls`,
    enforced where the tuple meets the engine config rather than here, because
    a per-tenant runtime constant cannot be a class-level field bound.
    """


# ---------------------------------------------------------------------------
# Event envelope — emitted via IEventStream + EventBus
# ---------------------------------------------------------------------------


class Event(BaseModel):
    """In-flight event envelope.

    Anthropic-aligned event names
    plus Protocore extensions (``sandbox_starting``/``subagent_spawn``/
    ``hook_fired``/``tool_call_pending``). The enum lives in
    :mod:`protocore.events`.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    name: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("payload")
    @classmethod
    def _reject_non_finite_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _reject_non_finite_floats(value, field="Event.payload")


# ---------------------------------------------------------------------------
# Run — PG-row mirror
# ---------------------------------------------------------------------------


class Run(BaseModel):
    """Persistent run record. Mirrors PG ``runs`` schema."""

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: str
    session_id: str
    status: RunStatus = RunStatus.queued
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    detail_blob_ref: str | None = None
    """Reference to S3 zstd blob (messages + events + tool_calls bundle)."""


class RunState(BaseModel):
    """Ephemeral run state held in Redis hash ``run:{id}``.

    Distinct from :class:`Run` (PG durable) — this is the hot working set.
    """

    model_config = ConfigDict()

    run_id: str
    tenant_id: str
    status: RunStatus
    current_turn: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    last_event_id: str | None = None
    started_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Session — multi-turn conversation root
# ---------------------------------------------------------------------------


class Session(BaseModel):
    """User conversation session — never deleted (durable by design)."""

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: str
    title: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    last_message_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Todo — TodoWrite tool persistence shape
# ---------------------------------------------------------------------------


class TodoStatus(StrEnum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    blocked = "blocked"


class Todo(BaseModel):
    """Single todo item written via TodoWrite tool."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    status: TodoStatus = TodoStatus.pending
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Skill — skill manifest + index entry
# ---------------------------------------------------------------------------


class SkillManifest(BaseModel):
    """Frontmatter + body description of a skill.

    Body markdown is stored separately (S3 blob); manifest is the index entry.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str
    tenant_id: str | None = None
    bundle_ref: str | None = None  # IBlobStore reference for body


# ---------------------------------------------------------------------------
# Subagent — registry + dispatch
# ---------------------------------------------------------------------------


class SubagentDef(BaseModel):
    """Subagent registry entry (per tenant, dashboard-managed)."""

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: str
    name: str
    description: str
    system_prompt: str
    tool_whitelist: list[str] = Field(default_factory=list)
    pinned_tools: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=list)
    tool_call_soft_caps: dict[str, int] = Field(default_factory=dict)
    skill_whitelist: list[str] = Field(default_factory=list)


class SubagentResult(BaseModel):
    """Result of synchronous subagent dispatch.

 / A4 adds ``declared_deliverables`` so the
 orchestrator can populate the run's :class:`AttemptLedger` from the
 subagent's own claim of what it produced. The list shape mirrors the
 natural wire form (path + kind + required); the orchestrator merges
 them into the ledger via :meth:`AttemptLedger.declare`. Defaults to an
 empty list so analytic / scratchpad subagents that produce no artifact
 still validate.
 """

    model_config = ConfigDict(frozen=True)

    subagent_id: str
    parent_run_id: str
    output: str
    success: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    declared_deliverables: list[DeliverableDeclaration] = Field(
        default_factory=list,
        description=(
            "Per / A4: each entry is a "
            "DeliverableDeclaration the subagent claims it produced. The "
            "orchestrator independently verifies them via the finalization "
            "gate before classifying the run's terminal status."
        ),
    )


# ---------------------------------------------------------------------------
# Blob — IBlobStore reference + metadata
# ---------------------------------------------------------------------------


class BlobMetadata(BaseModel):
    """Blob index entry — small enough for PG ``blob_index`` row."""

    model_config = ConfigDict(frozen=True)

    ref: str
    tenant_id: str
    content_type: str
    size_bytes: int
    sha256: str
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Envelope — ingress contract (`parse_envelope`)
# ---------------------------------------------------------------------------


class EnvelopeKind(StrEnum):
    task = "task"
    control = "control"
    result = "result"
    error = "error"


class AgentEnvelope(BaseModel):
    """Single ingress contract for cross-component messaging."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    kind: EnvelopeKind
    payload: str
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload")
    @classmethod
    def _cap_payload(cls, value: str) -> str:
        if len(value) > MAX_ENVELOPE_PAYLOAD_CHARS:
            raise ValueError(
                f"envelope payload exceeds {MAX_ENVELOPE_PAYLOAD_CHARS} chars"
            )
        return value


# ---------------------------------------------------------------------------
# ExecutionReport — bounded telemetry
# ---------------------------------------------------------------------------


class LLMCallRecord(BaseModel):
    """Single LLM call telemetry entry."""

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    stop_reason: StopReason | None = None
    error: str | None = None


class ToolCallRecord(BaseModel):
    """Single tool-call telemetry entry."""

    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    name: str
    arguments_json: str
    success: bool
    duration_ms: int = 0
    error: str | None = None

    @field_validator("arguments_json")
    @classmethod
    def _cap_arguments(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_TOOL_CALL_ARGUMENT_BYTES:
            raise ValueError(
                f"tool-call record arguments exceed {MAX_TOOL_CALL_ARGUMENT_BYTES} bytes"
            )
        return value


class ExecutionReport(BaseModel):
    """Per-run summary with structural caps.

    Memory-safety caps live in :mod:`protocore.constants`.

    The optional ``attempt_ledger`` field lets a host's run-detail telemetry
    surface the declared deliverables, the verifications, and the gate's
    terminal verdict alongside the tool/LLM/subagent rollup. ``None`` when the
    finalization gate is disabled, or when ``declared_deliverables`` was empty
    across the run.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: RunStatus
    events: list[Event] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    subagent_runs: list[SubagentResult] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    attempt_ledger: AttemptLedger | None = Field(
        default=None,
        description=(
            "Snapshot of the per-run :class:`AttemptLedger`. Populated by "
            "the executor when the finalization gate is enabled. ``None`` "
            "when the gate is disabled or no declarations were made."
        ),
    )

    @model_validator(mode="after")
    def _enforce_caps(self) -> Self:
        if len(self.events) > MAX_REPORT_EVENTS:
            raise ValueError(f"events exceed {MAX_REPORT_EVENTS}")
        if len(self.tool_calls) > MAX_TOOL_CALL_DETAILS:
            raise ValueError(f"tool_calls exceed {MAX_TOOL_CALL_DETAILS}")
        if len(self.llm_calls) > MAX_LLM_CALL_DETAILS:
            raise ValueError(f"llm_calls exceed {MAX_LLM_CALL_DETAILS}")
        if len(self.warnings) > MAX_WARNINGS:
            raise ValueError(f"warnings exceed {MAX_WARNINGS}")
        if len(self.subagent_runs) > MAX_SUBAGENT_RUNS:
            raise ValueError(f"subagent_runs exceed {MAX_SUBAGENT_RUNS}")
        if len(self.artifacts) > MAX_ARTIFACTS:
            raise ValueError(f"artifacts exceed {MAX_ARTIFACTS}")
        return self


# ---------------------------------------------------------------------------
# CompactionSourceRef — wire format placeholder
# ---------------------------------------------------------------------------


class CompactionSourceRef(BaseModel):
    """Pointer to a compacted tool-result blob.

    Persisted as a wire-format placeholder string in the message stream.
    (The v1 ``recall_artifact`` recall tool was not ported to v2; the ref's
    content is re-read through the regular file tools when needed.)

    ``tool_name`` + ``preview`` make a compacted result recoverable — the
    model can see which tool produced the shed content and a short head/tail
    excerpt, so it knows what was lost and can re-fetch it through the normal
    tools instead of fabricating. Both default to empty for backward-compatible
    (and disabled-preview) placeholders.
    """

    model_config = ConfigDict(frozen=True)

    blob_ref: str
    original_tokens: int
    sha256: str
    label: str = "tool_result"
    tool_name: str = ""
    """Originating tool name. Empty when unknown / disabled."""
    preview: str = ""
    """Short head/tail preview of the original content.

    RC-bounded (``compaction_placeholder_preview_chars``). Empty when the
    preview is disabled (cap 0) or the ref predates the enrichment.
    """


# ---------------------------------------------------------------------------
# ToolDefinition / parameter schema
# ---------------------------------------------------------------------------


class ToolParameterSchema(BaseModel):
    """JSON Schema fragment for a tool parameter list (per-tool)."""

    model_config = ConfigDict(frozen=True)

    type: Literal["object"] = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    additional_properties: bool | None = Field(
        default=None,
        description=(
            "Optional JSON-Schema ``additionalProperties`` flag. ``None`` "
            "(default) omits the key so adapters render the tool exactly as "
            "today (permissive). Set ``False`` for STRICT forced tools — the "
            "deep-mode ``plan`` tool sets this so the "
            "schema the loop puts on ``LLMRequest.tools`` carries "
            "``additionalProperties: false`` and the model cannot inject "
            "fields outside the declared shape. Provider adapters that "
            "recognise the field render it at the wire level."
        ),
    )
    chunkable_content_mutation: bool | None = Field(
        default=None,
        description=(
            "Explicit opt-in marking a tool as a CHUNKABLE content-mutation "
            "tool (a large ``content`` body the LLM may write in chunks via "
            "Write->AppendFile->FinalizeFile). ``None`` (default) "
            "omits the marker. Only a tool with this flag ``True`` (or one on "
            "the built-in allowlist — see "
            ":data:`~protocore.contracts.tool_chunking.CHUNKABLE_CONTENT_MUTATION_ALLOWLIST`) "
            "whose required ``content`` field was cut at the output cap is routed "
            "into the runtime chunk-recovery protocol; an unknown/dynamic tool "
            "that merely happens to declare a ``content`` field gets the generic "
            "tool-call resume instead. Lets a per-tenant content-mutation tool "
            "with an append path opt into chunk-recovery without a code change."
        ),
    )


class ToolDefinition(BaseModel):
    """Public surface of a registered :class:`~protocore.contracts.tools.Tool`.

    Used by the LLM provider to render the tool surface.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: ToolParameterSchema
    preconditions: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of prerequisite tool-call patterns that must "
            "appear in the per-run satisfied-precondition set before this "
            "tool can be invoked. Patterns use the v1 DAG-precondition "
            "format: ``\"tool_name\"`` (bare), ``\"tool_name:{param}\"`` "
            "(parameterised with current-call argument substitution) or "
            "``\"tool_name:path_prefix*\"`` (prefix-match). "
            "The check is performed by "
            ":func:`protocore.runtime.tool_preconditions.check_preconditions` "
            "inside :class:`~protocore.runtime.tool_dispatch.ToolDispatcher` "
            "when ``RuntimeConstants.tool_preconditions_enabled`` is True; "
            "an unmet precondition returns a "
            "``[PRECONDITION NOT MET: ...]`` tool-error envelope without "
            "dispatching."
        ),
    )
    path_fields: list[str] | None = Field(
        default=None,
        description=(
            "Optional override of the path-like argument field names that "
            "satisfy future ``tool:{path}`` preconditions for this tool. "
            "Defaults to ``[\"path\", \"file_path\", \"source_path\", "
            "\"destination_path\", \"target_path\"]`` when unset. "
        ),
    )


# ---------------------------------------------------------------------------
# Subagent task envelope (for IAgentDispatch)
# ---------------------------------------------------------------------------


class SubagentTask(BaseModel):
    """Payload submitted to a subagent dispatch."""

    model_config = ConfigDict(frozen=True)

    subagent_id: str
    parent_run_id: str
    task_prompt: str

    @field_validator("task_prompt")
    @classmethod
    def _cap_task(cls, value: str) -> str:
        if len(value) > MAX_DELEGATE_TASK_CHARS:
            raise ValueError(f"task_prompt exceeds {MAX_DELEGATE_TASK_CHARS} chars")
        return value


__all__ = [
    "TERMINAL_TOOL_METADATA_KEY",
    "TERMINAL_TOOL_STATUS_COMPLETED",
    "TERMINAL_TOOL_STATUS_METADATA_KEY",
    "TOOL_RESULT_CONSECUTIVE_CAP_ELIGIBLE_METADATA_KEY",
    "TOOL_RESULT_COUNT_AS_ERROR_METADATA_KEY",
    "AgentEnvelope",
    "AttemptLedger",
    "BlobMetadata",
    "CompactionSourceRef",
    "ContentBlock",
    "ContentBlockKind",
    "DeliverableDeclaration",
    "EnvelopeKind",
    "Event",
    "ExecutionReport",
    "HookEvent",
    "ImageRefBlock",
    "LLMCallRecord",
    "Message",
    "MessageRole",
    "Run",
    "RunState",
    "RunStatus",
    "Session",
    "SkillManifest",
    "StopReason",
    "SubagentDef",
    "SubagentResult",
    "SubagentTask",
    "TextBlock",
    "ThinkingBlock",
    "Todo",
    "TodoStatus",
    "ToolCall",
    "ToolCallRecord",
    "ToolDefinition",
    "ToolParameterSchema",
    "ToolResult",
    "ToolResultBlock",
    "ToolUseBlock",
]
