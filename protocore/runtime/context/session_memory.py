
"""Persistent, structured cross-run session memory.

PURE core module. NEVER imports the host / adapters / frontend (the import
boundary guard ``tests/test_core_import_boundary.py`` enforces this). The host
loads/stores the durable memory and injects the model-client as a summarizer
callback — core only computes.

This implements an empirically-proven persistent iterative LLM running summary
that achieves exact-fact recall at long contexts where a raw-seed approach
collapses, at HALF the prompt tokens, PLUS a structured artifact registry built
ONLY from parsed tool-call arguments.

UNIVERSALITY / NO-HEURISTICS RULE (enforced by review):
* There is ZERO regex, ZERO substring/keyword matching, ZERO hardcoded text
  patterns anywhere in this module. ``import re`` is forbidden. Semantic
  extraction (exact facts, headings, decisions, identifiers, versions) is the
  job of the LLM running summary — preserved by INSTRUCTING the model in the
  summary system prompt to keep them verbatim, multilingual (RU+EN), NOT by any
  text-scanning post-process.
* The artifact registry reads STRUCTURED tool-call data by named key only: for a
  file-writing tool call, the ``path`` argument names the file the agent
  created/edited, and the ``content`` argument is kept as a verbatim snapshot.
  That is reading the actual tool-invocation fields by key — universal +
  language-agnostic, not text matching.

Two persistent artifacts make up a session memory (:class:`SessionMemory`):

1. **Running summary** — folded ``Mᵢ = LLM(Sᵢ, Mᵢ₋₁)`` once per run from THAT
   run's local messages (delta-only: the prior summary is passed as PREVIOUS
   SUMMARY, only the new run's turns are SOURCE — the summary is NEVER
   re-summarised, so cost is O(K) not O(K²) and fidelity is monotonic). The
   summarizer is an INJECTED callback so core never calls a model client.
2. **Artifact registry** — a deterministic file registry: ``path`` -> last
   verbatim ``content`` snapshot, read by key from file-writing tool-call
   arguments (Write / Edit / AppendFile / …). Pure structured field access.

:func:`fold_run` updates a memory from a finished run's messages (the UPDATE
step). :func:`build_seed` assembles a memory + a recent raw tail + the original
head into a wire-ready message list for the NEXT run (the SEED step).

ASYNC-NATIVE FOLD:
Core does NOT call a model client and does NOT bridge to one via a thread. The
LLM running-summary call lives in the HOST on the main event loop, awaited
directly under ``asyncio.wait_for`` so a timeout cancels the awaitable cleanly
and the LLM provider releases its inflight slot via its normal async
cancellation / ``finally`` path (NO worker thread, NO fresh event loop → the
slot can NEVER leak). Core only:

* builds the summary prompt input via the pure :func:`build_summary_user_message`
  (so the host serialises the exact same multilingual SOURCE MATERIAL), and
* folds the ALREADY-COMPUTED ``new_summary_text`` into the memory in
  :func:`fold_run` (pure assembly: free artifact-ledger update + summary set +
  drift cap + ``turn_index`` advance).

LAZY FOLD: :func:`running_summary_needed` lets the host SKIP the LLM call
entirely when the whole prior session still fits in ``build_seed``'s raw tail
(the running summary would never be used) — short sessions do a ledger-only,
ZERO-LLM update.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import (
    COMPACTION_REFERENCE_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.token_counting import estimate_tokens

# ---------------------------------------------------------------------------
# Structural constants (NOT tunable thresholds → not RC fields): the
# END-OF-SUMMARY marker, render labels, the verbatim-preservation system prompt,
# and the named tool-call field keys read by the structured artifact registry.
# All numeric BUDGET thresholds live in RuntimeConstants (no inline magic
# numbers per ).
# ---------------------------------------------------------------------------

END_OF_SUMMARY_MARKER = (
    "--- END OF CONTEXT SUMMARY — the message below is the current task; "
    "respond to it, not to the summary above ---"
)
"""Sentinel after the running-summary block so a local model treats the summary
as reference DATA, not a chat turn to continue (proven on the stand)."""

#: Tool names whose call records a written/edited artifact. Compared via a
#: normalised set membership on the structured ``tool_call.name`` field (not a
#: text scan of message content): a file-writing tool, in any language, is
#: identified by its registered name, which is a fixed runtime identifier.
_FILE_WRITE_TOOLS: frozenset[str] = frozenset(
    {"write", "edit", "appendfile", "createfile", "writefile", "finalizefile", "multiedit"}
)
#: Argument keys (by NAME) that carry a written/edited file path in a tool call.
_PATH_ARG_KEYS: tuple[str, ...] = ("path", "file_path", "filename", "file", "target_path")
#: Argument keys (by NAME) that carry written file CONTENT in a tool call.
_CONTENT_ARG_KEYS: tuple[str, ...] = ("content", "new_str", "new_string", "text", "body")

#: Cap on how many files are pinned + how many chars of each content snapshot is
#: kept, so the registry block stays compact even on a huge session. These are
#: render-shape constants (not behavioural thresholds an operator tunes per
#: tenant), so they are module constants rather than RC fields.
_MAX_FILES = 60
_CONTENT_SNAPSHOT_MAX_CHARS = 4000

#: Per-block truncation caps applied ONLY when rendering a run's transcript as
#: TEXT INPUT for the summary fold (``_serialize_message``). They bound how much
#: of one tool-call's args / one tool-result appears in the summarizer's source
#: material so a single huge block cannot dominate the fold input. Like the two
#: caps above, these are render-shape constants — they shape the serialized input
#: string, not a behavioural threshold an operator tunes per tenant — so they are
#: named module constants, NOT RC fields (per →
#: "truly static values that should never vary per-scope belong only in
#: constants.py"; the LLM summary remains the sole semantic mechanism).
_SERIALIZE_TOOL_ARGS_MAX_CHARS = 240
_SERIALIZE_TOOL_RESULT_MAX_CHARS = 600

SUMMARY_SYSTEM = (
    "You are a context-compression component for an autonomous agent. You "
    "receive SOURCE MATERIAL (the session transcript to fold in — this is the "
    "newest session turns, and on the first fold of a long session it may carry "
    "the whole un-summarised history so far) and, optionally, a PREVIOUS SUMMARY "
    "of everything before it. Produce ONE updated running summary of the whole "
    "session so the agent can continue without re-reading the raw history.\n"
    "Write the summary under these FIXED SECTION HEADINGS, ALWAYS in this exact "
    "order, even if a section is empty:\n"
    "  ORIGINAL ASK:\n"
    "  CONSTRAINTS & DECISIONS:\n"
    "  KEY FACTS & IDENTIFIERS:\n"
    "  PENDING / NEXT:\n"
    "  ARTIFACTS:\n"
    "RULES:\n"
    "- PRESERVE every fact already in the PREVIOUS SUMMARY; ADD the new facts "
    "from the SOURCE MATERIAL. Carry forward EVERY prior fact VERBATIM. Never "
    "drop, paraphrase, or replace a previously-recorded fact — especially never "
    "replace a constraint, decision, or identifier with file/essay content.\n"
    "- CONSTRAINTS & DECISIONS and KEY FACTS & IDENTIFIERS are the HIGHEST "
    "priority: every rule the user/agent stated (\"X MUST …\", limits, policies), "
    "every decision, and every novel identifier/code/token/URL/version/number "
    "goes here, copied VERBATIM (exact characters). If the summary must be "
    "shortened to fit a budget, shorten ARTIFACTS first and NEVER drop a "
    "constraint, decision, or identifier — those are the facts that cannot be "
    "recovered elsewhere.\n"
    "- Capture VERBATIM, copying the exact characters: novel identifiers/codes, "
    "version numbers, URLs, created/edited filenames and paths, document "
    "headings (e.g. `## Heading`), numeric constraints/limits, and every "
    "decision the user or agent made.\n"
    "- ARTIFACTS is for created/edited files: list each path and ONE short line "
    "of what it is. Do NOT copy file BODIES / essay prose here — the exact file "
    "contents are preserved separately in the artifact registry, so re-stating "
    "them only crowds out the constraints and identifiers above.\n"
    "- Preserve exact tokens REGARDLESS OF LANGUAGE. The conversation may be in "
    "Russian, English, or mixed (Русский, English, или смешанный) — copy "
    "names/codes/paths/numbers exactly in their original script; do not "
    "translate or normalise them.\n"
    "- Output ONLY the updated summary text under the fixed headings above. Do "
    "not address the user."
)
"""Verbatim-preservation, MULTILINGUAL, SECTION-STRUCTURED summary system prompt
(RU+EN). Fixed-order headings (CONSTRAINTS & DECISIONS / KEY FACTS & IDENTIFIERS
BEFORE ARTIFACTS) so verbose file/essay content can never crowd out the
hard-to-recover prose constraints, and an explicit "shorten ARTIFACTS first,
never drop a constraint/identifier" rule for the plain (schema-less) path used by
deepseek-style providers. This prompt is the SOLE semantic-extraction mechanism
(no regex/text-mining anywhere); the artifact registry already holds file
contents verbatim, so the summary keeps file entries terse on purpose."""


# ---------------------------------------------------------------------------
# Persistent artifacts
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ArtifactLedger:
    """Structured file registry built ONLY from parsed tool-call arguments.

    ``files`` is an ORDERED ``path`` list (first-seen order, de-duplicated);
    ``content`` maps ``path`` -> the last verbatim content snapshot the agent
    wrote (capped). No text scanning, no regex — pure structured field reads.
    """

    files: list[str] = field(default_factory=list)
    content: dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.files

    def to_dict(self) -> dict[str, Any]:
        return {"files": list(self.files), "content": dict(self.content)}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ArtifactLedger:
        if not data:
            return cls()
        content = data.get("content") or {}
        return cls(
            files=[str(x) for x in (data.get("files") or [])],
            content={str(k): str(v) for k, v in content.items()} if isinstance(content, dict) else {},
        )


@dataclass(slots=True)
class SessionMemory:
    """The durable, per-session structured memory carried across runs.

    ``running_summary`` is the iterative LLM fold; ``ledger`` is the structured
    file registry. ``turn_index`` counts how many runs have been folded in
    (0 = a fresh/empty memory → SEED is a no-op, preserving the single-run path).

    ``cumulative_raw_tokens`` is the running total of every folded run's RAW
    message tokens (NOT the compressed summary size). It is the ACTUAL size of
    the session history that :func:`build_seed` must carry, and is the input the
    lazy-fold gate (:func:`running_summary_needed`) needs: the LLM summary may be
    skipped ONLY while the whole raw session still fits in the seed's head+tail
    (so the summary would never be read). Using the compressed summary's size as
    a proxy under-summarises — a ~900-token summary stays below the threshold
    forever while the raw history grows past the tail and is silently clipped
    Persisted in the existing memory JSON (NO new
    migration); ``0`` for legacy rows written before this field existed (they
    then fold conservatively until they exceed the threshold once, which is
    SAFE — it errs toward summarising, never toward loss).

    ``stale_fold_count`` counts consecutive folds that ATTEMPTED to update the
    summary and did not move it. A refused reply and a reply that reproduces the
    stored summary byte for byte are the same outcome from the session's point
    of view, and both REPEAT: the next fold is handed the same previous summary
    and the same kind of source, so it produces the same reply and fails the
    same way. Nothing else in the system distinguishes that from a healthy
    session — the run succeeds, the ledger advances, ``turn_index`` advances —
    which is why the count is carried here rather than inferred. ``0`` means the
    last attempt advanced the summary. Deliberate skips (the lazy-fold gate,
    load shedding) are not attempts and must leave this untouched, so the
    adapter that knows which kind of fold ran owns the increment. Persisted in
    the existing memory JSON (NO new migration); ``0`` for rows written before
    this field existed.
    """

    running_summary: str = ""
    ledger: ArtifactLedger = field(default_factory=ArtifactLedger)
    turn_index: int = 0
    cumulative_raw_tokens: int = 0
    stale_fold_count: int = 0

    def is_empty(self) -> bool:
        return (
            self.turn_index == 0
            and not self.running_summary
            and self.ledger.is_empty()
            and self.cumulative_raw_tokens == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "running_summary": self.running_summary,
            "ledger": self.ledger.to_dict(),
            "turn_index": self.turn_index,
            "cumulative_raw_tokens": self.cumulative_raw_tokens,
            "stale_fold_count": self.stale_fold_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SessionMemory:
        if not data:
            return cls()
        return cls(
            running_summary=str(data.get("running_summary") or ""),
            ledger=ArtifactLedger.from_dict(data.get("ledger")),
            turn_index=int(data.get("turn_index") or 0),
            cumulative_raw_tokens=int(data.get("cumulative_raw_tokens") or 0),
            stale_fold_count=int(data.get("stale_fold_count") or 0),
        )


# ---------------------------------------------------------------------------
# Message token-counting + serialisation helpers (pure; no text matching)
# ---------------------------------------------------------------------------


def _message_tokens(message: Message, rc: RuntimeConstants) -> int:
    """Estimate the token cost of one message (text + tool blocks + reasoning).

    Uses only the tokenizer (allowed deterministic op) — no content matching.
    """
    parts: list[str] = []
    for block in message.content_blocks:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            parts.append(f"{block.name}({block.arguments_json})")
        elif isinstance(block, ToolResultBlock):
            parts.append(block.content)
    if message.reasoning_content:
        parts.append(message.reasoning_content)
    return estimate_tokens("\n".join(parts), rc)


def _serialize_message(message: Message) -> str:
    """Render a message as ``[ROLE] text [calls Tool(args)] [tool result …]`` for
    summary INPUT, so the model treats it as source material, not a chat to
    continue. Tool results / args are length-capped (slicing, not matching)."""
    role = message.role.value.upper()
    pieces: list[str] = []
    for block in message.content_blocks:
        if isinstance(block, TextBlock):
            if block.text:
                pieces.append(block.text)
        elif isinstance(block, ToolUseBlock):
            args = block.arguments_json
            if len(args) > _SERIALIZE_TOOL_ARGS_MAX_CHARS:
                args = args[:_SERIALIZE_TOOL_ARGS_MAX_CHARS] + " …"
            pieces.append(f"[calls {block.name}({args})]")
        elif isinstance(block, ToolResultBlock):
            content = block.content
            if len(content) > _SERIALIZE_TOOL_RESULT_MAX_CHARS:
                content = content[:_SERIALIZE_TOOL_RESULT_MAX_CHARS] + " …[truncated]"
            pieces.append(f"[tool result: {content}]")
    return f"[{role}] " + " ".join(p for p in pieces if p).strip()


def _serialize_turns(messages: Sequence[Message]) -> str:
    return "\n".join(_serialize_message(m) for m in messages)


def build_summary_user_message(
    prev_summary: str,
    run_messages: Sequence[Message],
) -> str | None:
    """Build the running-summary fold USER message (the summary prompt body) — PURE.

    Returns the ``PREVIOUS SUMMARY``-prefixed (when present) + ``SOURCE
    MATERIAL`` string that the host feeds to :data:`SUMMARY_SYSTEM` for ONE
    running-summary fold ``Mᵢ = LLM(Sᵢ, Mᵢ₋₁)``: the prior summary is the
    PREVIOUS SUMMARY and ``run_messages`` is the SOURCE to fold into it.

    ``run_messages`` is the new session turns to summarise. In the steady state
    that is THIS run's local turns only (delta-only fold). On the ONE-TIME
    catch-up first fold of a long session (the host crosses the lazy-fold
    threshold while the running summary is still empty), the host passes the
    WHOLE un-summarised prior history + this run as ``run_messages`` so the early
    (lazy-skipped) turns are summarised before they leave the raw tail — so this
    parameter is NOT always "this run only". The label is kept accurate for both
    cases ("session transcript to fold in").

    Returns ``None`` when the source serialises to nothing (empty transcript) —
    the caller then skips the LLM call and keeps the prior summary. Core owns this
    prompt assembly + serialisation so the host (which makes the async LLM
    call) produces byte-identical input regardless of where the call lives — the
    LLM running summary stays the SOLE semantic mechanism (no text-mining).
    """
    serialized = _serialize_turns(run_messages).strip()
    if not serialized:
        return None
    return (
        (f"PREVIOUS SUMMARY:\n{prev_summary}\n\n" if prev_summary else "")
        + "SOURCE MATERIAL (session transcript to fold in):\n"
        + serialized
    )


# ---------------------------------------------------------------------------
# Artifact registry — STRUCTURED tool-call field reads ONLY (no regex/text scan)
# ---------------------------------------------------------------------------


def _parse_args(block: ToolUseBlock) -> dict[str, Any]:
    """Parse the tool-call ``arguments`` JSON into a dict (structured field data).

    JSON parsing is structured decoding, not text matching. A malformed payload
    yields an empty dict (the call contributes nothing to the registry)."""
    try:
        parsed = json.loads(block.arguments_json)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _arg_by_keys(args: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first named-key value present as a non-empty string.

    Pure dict lookup by key — NOT a substring/keyword scan of any text."""
    for key in keys:
        val = args.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def _is_file_write_tool(name: str) -> bool:
    """Identify a file-writing tool by its registered NAME (a fixed runtime
    identifier), via normalised set membership — not a content text scan."""
    normalised = name.replace("-", "").replace("_", "").lower()
    return normalised in {t.replace("-", "").replace("_", "") for t in _FILE_WRITE_TOOLS}


def extract_artifacts(
    messages: Sequence[Message],
    *,
    base: ArtifactLedger | None = None,
) -> ArtifactLedger:
    """Build the file registry from a run's messages, folding into ``base``.

    Reads ONLY structured tool-call fields by named key (the ``path`` argument of
    a file-writing tool call names the file; the ``content`` argument is the
    verbatim snapshot). Language-agnostic + heuristic-free: a path is a path in
    any language, and we never scan message text for filenames/headings/symbols.

    Order-preserving + de-duplicated; ``content`` keeps the LAST snapshot per
    path (the agent's latest write wins), capped to ``_CONTENT_SNAPSHOT_MAX_CHARS``.
    """
    ledger = ArtifactLedger(
        files=list(base.files) if base else [],
        content=dict(base.content) if base else {},
    )
    seen = set(ledger.files)

    for message in messages:
        for block in message.content_blocks:
            if not isinstance(block, ToolUseBlock):
                continue
            if not _is_file_write_tool(block.name):
                continue
            args = _parse_args(block)
            path_value = _arg_by_keys(args, _PATH_ARG_KEYS)
            if not path_value:
                continue
            if path_value not in seen and len(ledger.files) < _MAX_FILES:
                seen.add(path_value)
                ledger.files.append(path_value)
            if path_value in seen:
                content_value = _arg_by_keys(args, _CONTENT_ARG_KEYS)
                if content_value:
                    ledger.content[path_value] = content_value[:_CONTENT_SNAPSHOT_MAX_CHARS]

    return ledger


def render_ledger(ledger: ArtifactLedger) -> str:
    """Render the file registry verbatim as a compact, high-salience block."""
    if ledger.is_empty():
        return ""
    lines = ["ARTIFACT REGISTRY — files created/edited this session (verbatim, authoritative):"]
    for path in ledger.files:
        snapshot = ledger.content.get(path)
        if snapshot:
            lines.append(f"- {path}\n  current content:\n{snapshot}")
        else:
            lines.append(f"- {path}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Running summary fold (UPDATE step): Mᵢ = LLM(Sᵢ, Mᵢ₋₁) — delta-only
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FoldResult:
    """Outcome of a pure :func:`fold_run` (assembly only).

    ``summary_updated`` is True when ``new_summary_text`` replaced the prior
    running summary; False when the prior summary was kept (no new text /
    the host LLM call was skipped, timed out, or failed — the host logs the
    reason). The artifact ledger always updates and ``turn_index`` always
    advances regardless.
    """

    memory: SessionMemory
    summary_updated: bool = False
    notes: dict[str, Any] = field(default_factory=dict)


def _cap_running_summary(summary: str, rc: RuntimeConstants) -> str:
    """Drift control: truncate the carried summary if it grew past the RC cap.

    The NEXT fold is still delta-only (summary passed as PREVIOUS SUMMARY) — we
    NEVER recompute from raw, which would reintroduce O(K²) cost + non-monotonic
    fidelity. We only cap the carried artifact so it cannot itself overflow.
    """
    cap = rc.session_memory_running_summary_token_cap
    if cap <= 0:
        return summary
    if estimate_tokens(summary, rc) <= cap:
        return summary
    # Truncate by characters proportionally to the token overshoot, preserving
    # the head (where the structured original-ask/decisions live).
    current = max(estimate_tokens(summary, rc), 1)
    keep_chars = max(1, int(len(summary) * cap / current))
    return summary[:keep_chars].rstrip()


def fold_run(
    memory: SessionMemory,
    run_messages: Sequence[Message],
    new_summary_text: str | None,
    rc: RuntimeConstants,
) -> FoldResult:
    """Fold ONE finished run's messages into ``memory`` (the UPDATE step) — PURE.

    Pure assembly: core does NOT call a model client. The host computes the
    delta running summary by awaiting its async LLM provider DIRECTLY on the
    event loop (under ``asyncio.wait_for``) — so a timeout cancels the awaitable
    cleanly and the provider releases its inflight slot via its normal async
    cancellation / ``finally`` path, with NO worker thread and NO leaked slot. The ALREADY-COMPUTED summary text (or ``None`` when the LLM
    call was skipped / timed out / failed / produced nothing) is passed in here.

    Core then:

    * updates the structured artifact ledger (free, deterministic — no LLM);
    * sets ``running_summary = new_summary_text or prior`` (the prior summary is
      KEPT when ``new_summary_text`` is ``None``/blank — never crash the run /
      no drift), applying the RC drift cap;
    * accumulates THIS run's RAW message tokens into ``cumulative_raw_tokens``
      (the actual session size the lazy-fold gate reads — NOT the compressed
      summary size, which would under-summarise);
    * advances ``turn_index`` so a SEED stays valid.

    Returns a NEW :class:`SessionMemory` (the input is not mutated).
    """
    new_ledger = extract_artifacts(run_messages, base=memory.ledger)

    running = memory.running_summary
    summary_updated = False
    candidate = (new_summary_text or "").strip()
    if candidate:
        running = _cap_running_summary(candidate, rc)
        summary_updated = True
    # else: no new summary (skipped/timed-out/failed/empty) — keep the prior
    # summary intact (no drift; the registry update + turn advance still land).

    run_tokens = estimate_messages_tokens(run_messages, rc)
    new_memory = SessionMemory(
        running_summary=running,
        ledger=new_ledger,
        turn_index=memory.turn_index + 1,
        cumulative_raw_tokens=memory.cumulative_raw_tokens + run_tokens,
    )
    return FoldResult(
        memory=new_memory,
        summary_updated=summary_updated,
        notes={
            "folded_messages": len(run_messages),
            "ledger_files": len(new_ledger.files),
            "running_summary_tokens": estimate_tokens(running, rc),
            "cumulative_raw_tokens": new_memory.cumulative_raw_tokens,
        },
    )


# ---------------------------------------------------------------------------
# Lazy fold: only summarise once the session outgrows the raw tail
# ---------------------------------------------------------------------------


def summary_fold_threshold_tokens(seed_budget: int, rc: RuntimeConstants) -> int:
    """Token threshold above which a session needs the LLM running summary.

    Below this, the whole prior history still fits in :func:`build_seed`'s raw
    tail (``seed_budget * session_memory_tail_budget_fraction``), so the running
    summary would never be used — the fold can skip the LLM call entirely.

    When ``session_memory_fold_min_tokens > 0`` it is used verbatim (operator
    override); otherwise the threshold is DERIVED from the same tail budget
    ``build_seed`` will hand the next run, so the lazy gate and the seed assembly
    agree by construction. Pure: no I/O, no provider.
    """
    explicit = rc.session_memory_fold_min_tokens
    if explicit > 0:
        return explicit
    return int(max(0, seed_budget) * rc.session_memory_tail_budget_fraction)


def running_summary_needed(
    cumulative_session_tokens: int,
    seed_budget: int,
    rc: RuntimeConstants,
) -> bool:
    """Whether the running-summary LLM fold is needed for this session.

    ``cumulative_session_tokens`` MUST be the ACTUAL cumulative RAW token size of
    the whole session AFTER this run is folded in (i.e.
    ``SessionMemory.cumulative_raw_tokens + this_run_tokens``) — NOT a proxy
    derived from the compressed running summary. A compressed ~900-token summary
    can represent 15K+ tokens of source: using it as the proxy keeps short
    subsequent runs below the threshold forever, so the fold is skipped while the
    RAW history grows past :func:`build_seed`'s tail and is silently clipped —
    runs 2..N content is then permanently lost.

    When the raw cumulative does not exceed :func:`summary_fold_threshold_tokens`
    (the tail budget :func:`build_seed` will carry), the verbatim raw tail still
    covers the entire history, so the summary is dead weight — the host skips
    the LLM call (ledger-only, ZERO LLM cost). Once it exceeds the threshold the
    fold ALWAYS runs, so nothing is evicted unsummarised. Pure: no I/O.
    """
    return cumulative_session_tokens > summary_fold_threshold_tokens(seed_budget, rc)


def estimate_messages_tokens(messages: Sequence[Message], rc: RuntimeConstants) -> int:
    """Sum the estimated token cost of a message sequence (pure; tokenizer only).

    Exposed so the host can size the cumulative prior session for the lazy-fold
    gate without re-implementing the per-message estimator."""
    return sum(_message_tokens(m, rc) for m in messages)


def bound_catchup_source(
    messages: Sequence[Message],
    budget_tokens: int,
    rc: RuntimeConstants,
) -> list[Message]:
    """Bound the ONE-TIME catch-up fold's SOURCE MATERIAL to a token budget — PURE.

 The catch-up first fold (the host ``_maybe_update_session_memory``)
 summarises the WHOLE un-summarised prior history + this run so early
 lazy-skipped facts are preserved before they leave the raw tail. At the
 default-threshold crossover the source is bounded in practice (cumulative ≈
 the seed tail budget), but if an operator sets a HIGH
 ``session_memory_fold_min_tokens`` the crossover can fire when the prior
 history is large — an unbounded source could produce a very large
 summary-input string and a slow / truncated summary LLM call .

 This caps the source to ``budget_tokens`` by keeping BOTH ends and dropping
 only the MIDDLE (anti lost-in-the-middle): the OLDEST head turns (most likely
 to hold unstated early constraints/identifiers that exist nowhere else) AND
 the most-RECENT turns (this run's delta — the freshest content). When the
 middle is dropped a single explicit gap marker is inserted so the model knows
 the transcript is not contiguous; the registry + raw tail still cover the
 in-between turns elsewhere. Half the budget is reserved for each end; if the
 head alone already fits within the full budget the whole list is returned
 unchanged.

 ``budget_tokens <= 0`` means "no bound" → the input is returned unchanged
 (fail-open: a misconfigured zero budget never silently discards history).
 Pure: tokenizer only, no I/O, no provider, no text matching.
 """
    msgs = list(messages)
    if budget_tokens <= 0:
        return msgs
    if estimate_messages_tokens(msgs, rc) <= budget_tokens:
        return msgs

    half = max(1, budget_tokens // 2)
    # Oldest head turns, growing forward until half the budget is used.
    head: list[Message] = []
    used = 0
    for message in msgs:
        t = _message_tokens(message, rc)
        if used + t > half and head:
            break
        head.append(message)
        used += t
    # Most-recent turns, growing backward until the REMAINING budget is used.
    remaining = max(0, budget_tokens - used)
    tail = _tail_by_budget(msgs[len(head):], remaining, rc)

    if not tail:
        return head
    gap = Message(
        role=MessageRole.user,
        content_blocks=[
            TextBlock(
                text=(
                    "[… earlier middle turns omitted from this summary source to "
                    "fit the catch-up budget; their files are in the artifact "
                    "registry and recent turns are in the raw tail …]"
                )
            )
        ],
    )
    return [*head, gap, *tail]


# ---------------------------------------------------------------------------
# Seed assembly (SEED step): head + summary + ledger + recent raw tail
# ---------------------------------------------------------------------------


def _tail_by_budget(
    messages: Sequence[Message],
    budget_tokens: int,
    rc: RuntimeConstants,
) -> list[Message]:
    """Keep the newest messages until ``budget_tokens`` is reached, then repair
    the tool-pair boundary so the tail never opens on an orphan tool-result
    (which would 400 the provider). Token budgeting + a role check on the
    boundary message — no content text matching."""
    tail: list[Message] = []
    used = 0
    for message in reversed(messages):
        t = _message_tokens(message, rc)
        if used + t > budget_tokens and tail:
            break
        tail.append(message)
        used += t
    tail.reverse()
    # Don't open the tail on an orphan tool-role result.
    while tail and tail[0].role is MessageRole.tool:
        tail.pop(0)
    return tail


def _tag_seeded(message: Message) -> Message:
    """Tag a message as a seeded prior-run turn (so finalization + compaction
    treat it as a seed: excluded from re-persistence, protected from Tier-2
    collapse, skipped by the first-user-turn guard)."""
    return message.model_copy(
        update={"metadata": {**message.metadata, SESSION_HISTORY_SEED_METADATA_KEY: True}}
    )


def _tag_seeded_reference(message: Message) -> Message:
    """Dual-tag a synthetic seed block as BOTH a seeded prior-run turn AND a
    runtime reference block.

    The running-summary + artifact-ledger blocks are large ``role=user``
    ``TextBlock`` messages synthesised by :func:`build_seed`; they are NOT the
    user's original task. With ONLY the seed tag they were permanently immovable
    once seeded into a live run — Tier-1 sheds only ``role=tool`` results or
    :data:`COMPACTION_REFERENCE_METADATA_KEY`-tagged blocks, and Tier-2 skips
    seed-tagged turns, so on a file-heavy session these blocks consumed window
 budget with no shed path (the D3 emergency-cliff edge).

 Carrying BOTH keys:

 * :data:`SESSION_HISTORY_SEED_METADATA_KEY` keeps the host finalization
 filter excluding them from re-persistence and keeps Tier-2 from collapsing
 them into an UNtagged ``<compacted-turn>`` (which would defeat that filter).
 * :data:`COMPACTION_REFERENCE_METADATA_KEY` opens the Tier-1 reference-shed
 path: when the block exceeds the Tier-1 truncation threshold
 under budget pressure, it is blobbed to a recoverable snapshot placeholder
 (recall via ``recall_artifact``), so it is no longer permanently immovable.

 Only the synthetic summary/ledger blocks are dual-tagged — NOT the protected
 head (the original task must stay verbatim, seed-tag only).
 """
    return message.model_copy(
        update={
            "metadata": {
                **message.metadata,
                SESSION_HISTORY_SEED_METADATA_KEY: True,
                COMPACTION_REFERENCE_METADATA_KEY: True,
            }
        }
    )


def _summary_message(running_summary: str) -> Message:
    body = (
        f"[Running session summary — built incrementally across prior runs]\n"
        f"{running_summary}\n\n{END_OF_SUMMARY_MARKER}"
    )
    # Dual-tag (seed + reference) so Tier-1 A2(2) can shed this large block under
    # budget pressure while Tier-2 still protects it from lossy summarisation.
    return _tag_seeded_reference(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text=body)])
    )


def _ledger_message(ledger_text: str) -> Message:
    # Dual-tag (seed + reference): see ``_tag_seeded_reference`` — on a file-heavy
    # session the registry block can grow large; Tier-1 A2(2) must be able to
    # blob-shed it rather than leaving it permanently immovable in the window.
    return _tag_seeded_reference(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text=ledger_text)])
    )


def build_seed(
    memory: SessionMemory,
    recent_tail: Sequence[Message],
    head: Sequence[Message],
    budget: int,
    rc: RuntimeConstants,
) -> list[Message]:
    """Assemble the structured-memory seed to PREPEND before the new task.

    Ordering (high-salience anchors at the edges — anti lost-in-the-middle):

      [head (verbatim original task/system)]
      [running summary block + END-OF-SUMMARY marker]
      [pinned file registry block]     ← near the task (bottom edge)
      [recent raw tail (token-budget, tool-pair-safe)]

    Every seeded message is tagged ``SESSION_HISTORY_SEED_METADATA_KEY`` so
    the host finalization mirror excludes it from re-persistence (no
    exponential growth / no prior-run mis-attribution) and compaction protects
    the new task + the seeded turns.

    NO-OP when the memory is empty AND there is no tail to carry → returns ``[]``
    so a session's FIRST run is byte-identical to a cold start (the 99 single-run
    no-op invariant).

    ``budget`` is the token budget for the seed. The recent raw tail gets
    ``session_memory_tail_budget_fraction`` of it; the head + summary + registry
    carry the rest (they are bounded by their own RC caps / module caps).
    """
    if memory.is_empty() and not recent_tail:
        return []

    head_messages = [_tag_seeded(m) for m in head[: rc.session_memory_head_protect_messages]]

    summary_block: list[Message] = []
    if memory.running_summary.strip():
        summary_block = [_summary_message(memory.running_summary.strip())]

    ledger_block: list[Message] = []
    ledger_text = render_ledger(memory.ledger)
    if ledger_text:
        ledger_block = [_ledger_message(ledger_text)]

    head_tokens = sum(_message_tokens(m, rc) for m in head_messages)
    summary_tokens = sum(_message_tokens(m, rc) for m in summary_block)
    ledger_tokens = sum(_message_tokens(m, rc) for m in ledger_block)

    tail_budget = int(budget * rc.session_memory_tail_budget_fraction)
    # Never let the fixed blocks + tail exceed the total budget: clamp the tail
    # to whatever remains after the (bounded) head/summary/ledger.
    remaining = max(0, budget - head_tokens - summary_tokens - ledger_tokens)
    tail_budget = min(tail_budget, remaining) if remaining else 0
    tail = [_tag_seeded(m) for m in _tail_by_budget(recent_tail, tail_budget, rc)]

    return head_messages + summary_block + ledger_block + tail


__all__ = [
    "END_OF_SUMMARY_MARKER",
    "SUMMARY_SYSTEM",
    "ArtifactLedger",
    "FoldResult",
    "SessionMemory",
    "bound_catchup_source",
    "build_seed",
    "build_summary_user_message",
    "estimate_messages_tokens",
    "extract_artifacts",
    "fold_run",
    "render_ledger",
    "running_summary_needed",
    "summary_fold_threshold_tokens",
]
