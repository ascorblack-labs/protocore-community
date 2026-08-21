"""Unit tests for the pure structured session-memory.

Covers the fold (delta-only + drift cap + graceful summarizer failure), the
STRUCTURED artifact registry (files + content read ONLY from parsed tool-call
arguments by key — no regex/text mining), build_seed assembly (head-protect +
ordering + tail tool-pair safety), the empty single-run no-op, and the storage
round-trip.

NO-HEURISTICS RULE: this module + its tests must contain no regex / substring
extraction in production code; the artifact registry reads structured tool-call
fields by named key only, and semantic extraction is the LLM running summary.
"""
from __future__ import annotations

import json

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
from protocore.runtime.context.session_memory import (
    END_OF_SUMMARY_MARKER,
    SUMMARY_SYSTEM,
    ArtifactLedger,
    SessionMemory,
    bound_catchup_source,
    build_seed,
    build_summary_user_message,
    estimate_messages_tokens,
    extract_artifacts,
    fold_run,
    render_ledger,
    running_summary_needed,
    summary_fold_threshold_tokens,
)

RC = RuntimeConstants()


# --- helpers ---------------------------------------------------------------


def _user(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _assistant(text: str) -> Message:
    return Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=text)])


def _write_call(path: str, content: str, *, name: str = "Write", call_id: str = "tc1") -> Message:
    return Message(
        role=MessageRole.assistant,
        content_blocks=[
            ToolUseBlock(
                tool_call_id=call_id,
                name=name,
                arguments_json=json.dumps({"path": path, "content": content}),
            )
        ],
    )


def _tool_result(text: str, *, call_id: str = "tc1") -> Message:
    return Message(
        role=MessageRole.tool,
        content_blocks=[ToolResultBlock(tool_call_id=call_id, content=text)],
    )


# --- structured artifact registry (files + content by key) -----------------


def test_extract_file_path_from_write_tool_call() -> None:
    led = extract_artifacts([_write_call("src/app.py", "def main():\n    pass\n")])
    assert led.files == ["src/app.py"]
    assert led.content["src/app.py"] == "def main():\n    pass\n"


def test_extract_is_language_agnostic_cyrillic_path() -> None:
    # A cyrillic path is read verbatim from the structured ``path`` arg — no text
    # scanning, no language assumption.
    content = "Документация процедуры отката. Версия 3.12."
    led = extract_artifacts([_write_call("docs/откат.md", content)])
    assert led.files == ["docs/откат.md"]
    assert led.content["docs/откат.md"] == content


def test_extract_edit_tool_file_path_key() -> None:
    edit = Message(
        role=MessageRole.assistant,
        content_blocks=[
            ToolUseBlock(
                tool_call_id="e1",
                name="Edit",
                arguments_json=json.dumps(
                    {"file_path": "main.py", "new_string": "def fixed():\n    pass"}
                ),
            )
        ],
    )
    led = extract_artifacts([edit])
    assert led.files == ["main.py"]
    assert led.content["main.py"] == "def fixed():\n    pass"


def test_extract_ignores_non_file_tools() -> None:
    # A non-file tool (e.g. Bash) with a path-looking arg is NOT recorded: we key
    # on the structured tool NAME, not text shape.
    bash = Message(
        role=MessageRole.assistant,
        content_blocks=[
            ToolUseBlock(
                tool_call_id="b1", name="Bash",
                arguments_json=json.dumps({"command": "cat src/app.py"}),
            )
        ],
    )
    led = extract_artifacts([bash])
    assert led.files == []


def test_extract_dedup_keeps_latest_content() -> None:
    base = ArtifactLedger(files=["a.py"], content={"a.py": "old"})
    led = extract_artifacts([_write_call("a.py", "new")], base=base)
    assert led.files == ["a.py"]  # not duplicated, order preserved
    assert led.content["a.py"] == "new"  # latest write wins


def test_extract_no_files_from_plain_prose() -> None:
    # No tool call → nothing in the registry (text is never scanned for facts).
    led = extract_artifacts([_user("Please write notes about https://x.com v3.12 `BUILD::`")])
    assert led.files == []
    assert led.content == {}


def test_extract_content_snapshot_capped() -> None:
    huge = "x" * 100_000
    led = extract_artifacts([_write_call("big.txt", huge)])
    assert len(led.content["big.txt"]) <= 4000


# --- summary prompt assembly (pure; the input to the host's async LLM) ---


def test_build_summary_user_message_delta_only_prior_summary() -> None:
    user = build_summary_user_message("PRIOR", [_user("new question"), _assistant("answer")])
    assert user is not None
    assert "PREVIOUS SUMMARY:\nPRIOR" in user
    assert "SOURCE MATERIAL" in user
    assert "new question" in user


def test_build_summary_user_message_first_run_no_prior() -> None:
    user = build_summary_user_message("", [_user("task"), _assistant("done")])
    assert user is not None
    assert "PREVIOUS SUMMARY" not in user
    assert "task" in user


def test_build_summary_user_message_empty_transcript_is_none() -> None:
    # An empty run (no messages) serialises to nothing → no LLM call needed.
    assert build_summary_user_message("PRIOR", []) is None


def test_build_summary_user_message_label_is_accurate_for_catchup(
) -> None:
    # : the SOURCE MATERIAL label must NOT claim "the most recent run's
    # transcript" — on the catch-up first fold ``run_messages`` carries the whole
    # un-summarised history, not just the most recent run. The label is the
    # case-agnostic "session transcript to fold in".
    user = build_summary_user_message("", [_user("turn 1 fact"), _assistant("ok")])
    assert user is not None
    assert "SOURCE MATERIAL (session transcript to fold in):" in user
    assert "most recent run's transcript" not in user


def test_summary_system_prompt_label_is_accurate_for_catchup() -> None:
    # : the SUMMARY_SYSTEM SOURCE MATERIAL description must be accurate for
    # BOTH the catch-up first fold (whole history) and the steady-state delta.
    assert "most recent run's transcript" not in SUMMARY_SYSTEM
    assert "session transcript to fold in" in SUMMARY_SYSTEM


def test_summary_system_prompt_is_multilingual_verbatim() -> None:
    # the SOLE semantic-extraction mechanism: a verbatim-preservation prompt that
    # is explicitly multilingual (RU+EN). Owned by core, used by the host.
    assert "VERBATIM" in SUMMARY_SYSTEM
    assert "REGARDLESS OF LANGUAGE" in SUMMARY_SYSTEM
    assert "Русский" in SUMMARY_SYSTEM


def test_summary_system_prompt_prioritises_constraints_over_artifacts() -> None:
    # Crowding-out fix: the sections must place
    # CONSTRAINTS & DECISIONS and KEY FACTS & IDENTIFIERS BEFORE ARTIFACTS so a
    # verbose file/essay body (already preserved verbatim in the artifact ledger)
    # can never crowd out the hard-to-recover prose constraints on the plain
    # (schema-less) path used by deepseek-style providers.
    assert "CONSTRAINTS & DECISIONS" in SUMMARY_SYSTEM
    assert "KEY FACTS & IDENTIFIERS" in SUMMARY_SYSTEM
    assert "ARTIFACTS" in SUMMARY_SYSTEM
    # ordering: constraints + identifiers come before artifacts in the prompt
    assert SUMMARY_SYSTEM.index("CONSTRAINTS & DECISIONS") < SUMMARY_SYSTEM.index(
        "ARTIFACTS:"
    )
    assert SUMMARY_SYSTEM.index("KEY FACTS & IDENTIFIERS") < SUMMARY_SYSTEM.index(
        "ARTIFACTS:"
    )
    # explicit "carry forward every prior fact verbatim; never replace facts with
    # file contents" instruction (the plain rung sends no schema).
    assert "Carry forward EVERY prior fact" in SUMMARY_SYSTEM
    assert "shorten ARTIFACTS first" in SUMMARY_SYSTEM


# --- catch-up source bound (— guard a large catch-up fold input) -----


def test_bound_catchup_source_returns_input_when_within_budget() -> None:
    msgs = [_user("a"), _assistant("b"), _user("c")]
    # A budget far above the tiny source → unchanged (no gap marker inserted).
    out = bound_catchup_source(msgs, 100_000, RC)
    assert out == msgs


def test_bound_catchup_source_zero_budget_is_no_bound() -> None:
    # 0 (or negative) budget means "no bound" — fail-open: never silently drop
    # history on a misconfigured zero budget.
    msgs = [_user("a"), _assistant("b")]
    assert bound_catchup_source(msgs, 0, RC) == msgs
    assert bound_catchup_source(msgs, -5, RC) == msgs


def test_bound_catchup_source_caps_and_keeps_oldest_head_and_newest_tail() -> None:
    # An over-budget source is capped: the OLDEST head turn (early facts) AND the
    # most-RECENT turn (this run's delta) survive; the middle is dropped behind a
    # single explicit gap marker, and the result fits the budget.
    big = "word " * 400  # ~hundreds of tokens each
    msgs = [
        _user("OLDEST early constraint " + big),
        _assistant("middle one " + big),
        _user("middle two " + big),
        _assistant("middle three " + big),
        _user("NEWEST delta this run " + big),
    ]
    full = estimate_messages_tokens(msgs, RC)
    budget = full // 2  # force a cap that cannot hold every turn
    out = bound_catchup_source(msgs, budget, RC)

    joined = "\n".join(m.text for m in out)
    assert "OLDEST early constraint" in joined  # head preserved (early facts)
    assert "NEWEST delta this run" in joined  # most-recent preserved (delta)
    assert "earlier middle turns omitted" in joined  # explicit gap marker present
    # The bounded source genuinely shrank and is at most ~the budget (the head's
    # last whole message can edge slightly over its half, but never the full size).
    assert len(out) < len(msgs)
    assert estimate_messages_tokens(out, RC) <= full


def test_bound_catchup_source_always_keeps_oldest_and_newest_under_tiny_budget() -> None:
    # Even under a tiny budget (well below one message) the bound never crashes and
    # never drops BOTH ends: the OLDEST message (early facts) and the NEWEST
    # message (this run's delta) are always present (each end keeps at least one
    # whole message), with the gap marker between them.
    big = "word " * 2000
    msgs = [_user("OLDEST " + big), _user("MID-DROP " + big), _user("NEWEST " + big)]
    out = bound_catchup_source(msgs, 1, RC)  # tiny budget → maximal cap
    joined = "\n".join(m.text for m in out)
    assert "OLDEST" in joined
    assert "NEWEST" in joined
    assert "earlier middle turns omitted" in joined
    assert "MID-DROP" not in joined  # the middle turn was dropped


# --- fold (UPDATE step) — pure assembly over already-computed summary text --


def test_fold_sets_new_summary_and_advances_turn() -> None:
    mem = SessionMemory(running_summary="PRIOR", turn_index=1)
    res = fold_run(mem, [_user("new question"), _assistant("answer")], "M1", RC)
    assert res.memory.running_summary == "M1"
    assert res.summary_updated is True
    assert res.memory.turn_index == 2


def test_fold_none_summary_keeps_prior_but_updates_ledger() -> None:
    # ``None`` summary text == the host LLM call was skipped/timed-out/failed
    # — keep the prior summary, but the deterministic registry still updates and
    # the turn still advances (no LLM needed for the ledger).
    mem = SessionMemory(running_summary="KEEP-ME", turn_index=2)
    res = fold_run(mem, [_write_call("x.py", "y = 1")], None, RC)
    assert res.summary_updated is False
    assert res.memory.running_summary == "KEEP-ME"  # prior preserved
    assert "x.py" in res.memory.ledger.files
    assert res.memory.turn_index == 3


def test_fold_blank_summary_keeps_prior() -> None:
    mem = SessionMemory(running_summary="PRIOR", turn_index=1)
    res = fold_run(mem, [_user("q")], "   ", RC)
    assert res.memory.running_summary == "PRIOR"
    assert res.summary_updated is False


def test_fold_drift_cap_truncates_oversized_summary() -> None:
    rc = RuntimeConstants(session_memory_running_summary_token_cap=10)
    huge = "word " * 500
    # default (large) cap → kept whole (modulo the strip()).
    res = fold_run(SessionMemory(), [_user("q")], huge, RC)
    assert res.memory.running_summary == huge.strip()
    # tiny cap → truncated well below the original length.
    res2 = fold_run(SessionMemory(), [_user("q")], huge, rc)
    assert len(res2.memory.running_summary) < len(huge.strip())


def test_fold_drift_cap_disabled_when_zero() -> None:
    rc = RuntimeConstants(session_memory_running_summary_token_cap=0)
    huge = "word " * 500
    res = fold_run(SessionMemory(), [_user("q")], huge, rc)
    assert res.memory.running_summary == huge.strip()


def test_fold_does_not_mutate_input_memory() -> None:
    mem = SessionMemory(running_summary="OLD", turn_index=1)
    fold_run(mem, [_user("q")], "NEW", RC)
    assert mem.running_summary == "OLD"
    assert mem.turn_index == 1


# --- lazy fold gate (running_summary_needed) -------------------------------


def test_lazy_fold_threshold_derived_from_tail_budget() -> None:
    rc = RuntimeConstants(
        session_memory_fold_min_tokens=0, session_memory_tail_budget_fraction=0.5
    )
    # derived: budget * tail_fraction.
    assert summary_fold_threshold_tokens(1000, rc) == 500


def test_lazy_fold_threshold_explicit_override() -> None:
    rc = RuntimeConstants(session_memory_fold_min_tokens=123)
    # a positive RC overrides the derived value verbatim.
    assert summary_fold_threshold_tokens(1000, rc) == 123


def test_running_summary_needed_skips_short_session() -> None:
    rc = RuntimeConstants(session_memory_fold_min_tokens=0, session_memory_tail_budget_fraction=0.5)
    # below the derived threshold (500) → not needed (tail covers the history).
    assert running_summary_needed(100, 1000, rc) is False
    # above it → needed (the raw tail can no longer hold everything).
    assert running_summary_needed(900, 1000, rc) is True


def test_fold_accumulates_cumulative_raw_tokens() -> None:
    """``fold_run`` accumulates THIS run's RAW message tokens into
    ``cumulative_raw_tokens`` — the actual session size the lazy gate reads (NOT
    the compressed summary size)."""
    mem = SessionMemory()
    assert mem.cumulative_raw_tokens == 0
    run1 = [_user("word " * 100), _assistant("ok " * 100)]
    r1 = fold_run(mem, run1, "small summary", RC)
    t1 = r1.memory.cumulative_raw_tokens
    assert t1 > 0
    # A second run accumulates ON TOP of the first (monotonic raw total).
    run2 = [_user("more " * 100), _assistant("done " * 100)]
    r2 = fold_run(r1.memory, run2, "small summary", RC)
    assert r2.memory.cumulative_raw_tokens > t1


def test_lazy_gate_does_not_under_summarize_via_cumulative_raw_tokens() -> None:
    """Phase D2 review CRITICAL — no silent content loss.

    A session whose CUMULATIVE RAW history has grown PAST the tail budget MUST
    trigger a summary fold even though the COMPRESSED running summary is tiny and
    the latest run is short. The old proxy (``run_tokens + summary_tokens``) saw a
    small summary + small run and skipped FOREVER; the cumulative-raw tracking
    crosses the threshold and forces the fold, so runs 2..N are never lost.
    """
    rc = RuntimeConstants(
        session_memory_fold_min_tokens=0, session_memory_tail_budget_fraction=0.30
    )
    seed_budget = 1000  # → derived threshold = 300 tokens
    threshold = summary_fold_threshold_tokens(seed_budget, rc)
    assert threshold == 300

    # Established session: a TINY compressed summary, but the raw history already
    # exceeds the tail budget (e.g. many prior short runs accumulated to 5000).
    mem = SessionMemory(
        running_summary="port 8080",  # ~2 tokens compressed
        turn_index=20,
        cumulative_raw_tokens=5000,  # >> threshold (300)
    )
    short_run = [_user("ok"), _assistant("done")]  # tiny new run
    run_tokens = estimate_messages_tokens(short_run, rc)
    assert run_tokens < threshold  # this run alone would NOT trip the old proxy

    # The gate uses the ACTUAL cumulative-after-this-run, which is >> threshold:
    cumulative_after = mem.cumulative_raw_tokens + run_tokens
    assert running_summary_needed(cumulative_after, seed_budget, rc) is True

    # And the fold keeps accumulating so it never silently stops summarising.
    res = fold_run(mem, short_run, "updated summary with ok/done", rc)
    assert res.summary_updated is True
    assert res.memory.running_summary  # non-empty → early content represented
    assert res.memory.cumulative_raw_tokens == 5000 + run_tokens


# --- build_seed (SEED step) ------------------------------------------------


def test_build_seed_empty_memory_no_tail_is_noop() -> None:
    # The 99 single-run no-op: a fresh memory with no tail seeds nothing.
    assert build_seed(SessionMemory(), [], [], 40_000, RC) == []


def test_build_seed_ordering_and_tags() -> None:
    head = [
        Message(role=MessageRole.system, content_blocks=[TextBlock(text="SYS")]),
        _user("ORIGINAL TASK"),
        _assistant("ok"),
    ]
    mem = SessionMemory(
        running_summary="the running summary",
        ledger=ArtifactLedger(files=["a.py"], content={"a.py": "x = 1"}),
        turn_index=2,
    )
    tail = [_user("recent question"), _assistant("recent answer")]
    seed = build_seed(mem, tail, head, 40_000, RC)

    assert all(m.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True for m in seed)
    texts = [m.text for m in seed]
    assert texts[0] == "SYS"
    assert "ORIGINAL TASK" in texts[1]
    summary_idx = next(i for i, t in enumerate(texts) if "running summary" in t)
    ledger_idx = next(i for i, t in enumerate(texts) if "ARTIFACT REGISTRY" in t)
    tail_idx = next(i for i, t in enumerate(texts) if "recent question" in t)
    assert summary_idx < ledger_idx < tail_idx
    assert END_OF_SUMMARY_MARKER in texts[summary_idx]
    # the registry block carries the verbatim path + content snapshot.
    assert "a.py" in texts[ledger_idx]
    assert "x = 1" in texts[ledger_idx]


def test_build_seed_dual_tags_summary_and_ledger_only() -> None:
    """The synthetic summary + ledger seed blocks carry
    BOTH the seed tag AND the compaction-reference tag (so Tier-1 A2(2) can shed
    them under budget pressure), while the protected HEAD carries ONLY the seed
    tag (the original task must never be reference-shed)."""
    head = [
        Message(role=MessageRole.system, content_blocks=[TextBlock(text="SYS")]),
        _user("ORIGINAL TASK"),
        _assistant("ok"),
    ]
    mem = SessionMemory(
        running_summary="the running summary",
        ledger=ArtifactLedger(files=["a.py"], content={"a.py": "x = 1"}),
        turn_index=2,
    )
    tail = [_user("recent question"), _assistant("recent answer")]
    seed = build_seed(mem, tail, head, 40_000, RC)

    # Every seed message is still seed-tagged (unchanged invariant).
    assert all(m.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True for m in seed)

    def _is_summary(m: Message) -> bool:
        return "running summary" in m.text

    def _is_ledger(m: Message) -> bool:
        return "ARTIFACT REGISTRY" in m.text

    summary_msg = next(m for m in seed if _is_summary(m))
    ledger_msg = next(m for m in seed if _is_ledger(m))
    # The two large synthetic user-text blocks DUAL-TAG the reference key.
    assert summary_msg.metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is True
    assert ledger_msg.metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is True

    # The protected head (system / original task / first assistant) and the raw
    # tail are NOT reference-tagged — only seed-tagged.
    non_synthetic = [m for m in seed if not _is_summary(m) and not _is_ledger(m)]
    assert non_synthetic, "head + tail must be present"
    assert all(
        m.metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is not True for m in non_synthetic
    )
    # In particular the original task is NOT reference-shed-eligible.
    task_msg = next(m for m in seed if "ORIGINAL TASK" in m.text)
    assert task_msg.metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is not True


def test_build_seed_head_protect_count_respected() -> None:
    rc = RuntimeConstants(session_memory_head_protect_messages=2)
    head = [_user("a"), _assistant("b"), _user("c"), _assistant("d")]
    mem = SessionMemory(running_summary="s", turn_index=1)
    seed = build_seed(mem, [], head, 40_000, rc)
    head_texts = [m.text for m in seed if "summary" not in m.text and "REGISTRY" not in m.text]
    assert head_texts == ["a", "b"]


def test_build_seed_tail_does_not_open_on_orphan_tool_result() -> None:
    rc = RuntimeConstants(session_memory_tail_budget_fraction=1.0)
    big = "x" * 4000
    tail = [_assistant("call"), _write_call("a.py", "y=1", call_id="z"), _tool_result(big, call_id="z")]
    mem = SessionMemory(running_summary="s", turn_index=1)
    seed = build_seed(mem, tail, [], 200, rc)
    tail_msgs = [m for m in seed if "summary" not in m.text and "REGISTRY" not in m.text]
    if tail_msgs:
        assert tail_msgs[0].role is not MessageRole.tool


def test_build_seed_only_summary_when_ledger_empty() -> None:
    mem = SessionMemory(running_summary="just a summary", turn_index=1)
    seed = build_seed(mem, [], [], 40_000, RC)
    assert any("just a summary" in m.text for m in seed)
    assert not any("ARTIFACT REGISTRY" in m.text for m in seed)


# --- render + round-trip ---------------------------------------------------


def test_render_ledger_verbatim() -> None:
    led = ArtifactLedger(files=["a.py", "b.md"], content={"a.py": "code", "b.md": "# Title"})
    rendered = render_ledger(led)
    assert "a.py" in rendered and "b.md" in rendered
    assert "code" in rendered
    assert "# Title" in rendered


def test_render_empty_ledger_is_blank() -> None:
    assert render_ledger(ArtifactLedger()) == ""


def test_session_memory_round_trip() -> None:
    mem = SessionMemory(
        running_summary="s",
        ledger=ArtifactLedger(files=["x.py"], content={"x.py": "def f(): ..."}),
        turn_index=3,
        cumulative_raw_tokens=4242,
    )
    restored = SessionMemory.from_dict(mem.to_dict())
    assert restored.running_summary == "s"
    assert restored.ledger.files == ["x.py"]
    assert restored.ledger.content == {"x.py": "def f(): ..."}
    assert restored.turn_index == 3
    # Phase D2 review CRITICAL — cumulative raw token total persists in the
    # existing memory JSON (NO migration).
    assert restored.cumulative_raw_tokens == 4242


def test_session_memory_from_empty_dict() -> None:
    assert SessionMemory.from_dict(None).is_empty()
    assert SessionMemory.from_dict({}).is_empty()
    # A legacy row written before the field existed → 0 (folds conservatively
    # until it exceeds the threshold once — safe, never under-summarises).
    assert SessionMemory.from_dict({"running_summary": "s", "turn_index": 2}).cumulative_raw_tokens == 0


def test_artifact_ledger_round_trip_via_json() -> None:
    led = ArtifactLedger(files=["b"], content={"b": "c"})
    raw = json.dumps(led.to_dict())
    restored = ArtifactLedger.from_dict(json.loads(raw))
    assert restored.to_dict() == led.to_dict()


def test_session_memory_carries_a_stale_fold_count_across_runs() -> None:
    """The stuck-memory counter survives the round trip through storage.

    It rides in the same JSON envelope as the rest of the memory, so a session
    that has been stuck for three folds still knows that after a pod restart —
    a counter that reset on reload would never reach any threshold worth
    reporting.
    """
    memory = SessionMemory(running_summary="s", turn_index=4, stale_fold_count=3)
    assert SessionMemory.from_dict(memory.to_dict()).stale_fold_count == 3


def test_session_memory_stale_fold_count_defaults_to_zero_for_legacy_rows() -> None:
    """Rows written before the field existed load as "not stuck"."""
    legacy = {"running_summary": "s", "ledger": {}, "turn_index": 2}
    assert SessionMemory.from_dict(legacy).stale_fold_count == 0
