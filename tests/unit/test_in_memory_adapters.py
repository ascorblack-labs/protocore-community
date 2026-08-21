"""Tests covering the in-memory adapter implementations."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from protocore.contracts.agent_dispatch import SubagentNotFoundError
from protocore.contracts.blob import BlobNotFoundError
from protocore.contracts.hooks import HookResult, HookSpec
from protocore.contracts.llm import LLMRequest
from protocore.contracts.run import RunNotFoundError
from protocore.contracts.search import IndexDoc
from protocore.contracts.session import SessionNotFoundError
from protocore.contracts.skills import SkillUpsertInput
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.types import (
    Event,
    HookEvent,
    Message,
    MessageRole,
    Run,
    RunStatus,
    Session,
    SkillManifest,
    StopReason,
    SubagentDef,
    SubagentResult,
    SubagentTask,
    TextBlock,
    Todo,
)
from protocore.testing import build_in_memory_runtime

# ----- BlobStore -----


async def test_blob_store_put_get() -> None:
    rt = build_in_memory_runtime()
    md = await rt.blob.put(rt.tenant_id, b"hello")
    assert md.size_bytes == 5
    fetched = await rt.blob.get(rt.tenant_id, md.ref)
    assert fetched == b"hello"


async def test_blob_store_missing_raises() -> None:
    rt = build_in_memory_runtime()
    with pytest.raises(BlobNotFoundError):
        await rt.blob.get(rt.tenant_id, "nonexistent")


async def test_blob_store_delete() -> None:
    rt = build_in_memory_runtime()
    md = await rt.blob.put(rt.tenant_id, b"x")
    assert await rt.blob.delete(rt.tenant_id, md.ref) is True
    assert await rt.blob.delete(rt.tenant_id, md.ref) is False


async def test_blob_store_stream_chunks() -> None:
    rt = build_in_memory_runtime()
    md = await rt.blob.put(rt.tenant_id, b"abcdef")
    chunks: list[bytes] = []
    async for chunk in rt.blob.get_stream(rt.tenant_id, md.ref):
        chunks.append(chunk)
    assert b"".join(chunks) == b"abcdef"


# ----- SessionStore -----


async def test_session_create_and_get() -> None:
    rt = build_in_memory_runtime()
    session = Session(id="s1", tenant_id=rt.tenant_id)
    await rt.sessions.create(session)
    fetched = await rt.sessions.get("s1", rt.tenant_id)
    assert fetched.id == "s1"


async def test_session_missing_raises() -> None:
    rt = build_in_memory_runtime()
    with pytest.raises(SessionNotFoundError):
        await rt.sessions.get("missing", rt.tenant_id)


async def test_session_append_message() -> None:
    rt = build_in_memory_runtime()
    await rt.sessions.create(Session(id="s1", tenant_id=rt.tenant_id))
    msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    await rt.sessions.append_message("s1", rt.tenant_id, msg)
    msgs = await rt.sessions.list_messages("s1", rt.tenant_id)
    assert len(msgs) == 1
    assert msgs[0].text == "hi"


# ----- RunStore -----


async def test_run_create_get_status() -> None:
    rt = build_in_memory_runtime()
    run = Run(id="r1", tenant_id=rt.tenant_id, session_id="s1")
    await rt.runs.create(run)
    fetched = await rt.runs.get("r1", rt.tenant_id)
    assert fetched.status is RunStatus.queued
    await rt.runs.update_status("r1", rt.tenant_id, RunStatus.running)
    refreshed = await rt.runs.get("r1", rt.tenant_id)
    assert refreshed.status is RunStatus.running


async def test_run_missing_raises() -> None:
    rt = build_in_memory_runtime()
    with pytest.raises(RunNotFoundError):
        await rt.runs.get("missing", rt.tenant_id)


async def test_run_flush_terminal_records_ref() -> None:
    rt = build_in_memory_runtime()
    await rt.runs.create(Run(id="r1", tenant_id=rt.tenant_id, session_id="s1"))
    await rt.runs.flush_terminal("r1", rt.tenant_id, "blob-ref-abc")
    run = await rt.runs.get("r1", rt.tenant_id)
    assert run.detail_blob_ref == "blob-ref-abc"


# ----- EventStream -----


async def test_event_emit_and_stream() -> None:
    rt = build_in_memory_runtime()
    event = Event(
        run_id="r1",
        name="tool_call_start",
        payload={"tenant_id": rt.tenant_id, "name": "Bash"},
    )
    await rt.events.emit(event)
    backlog = rt.events.stream_for(rt.tenant_id, "r1")
    assert len(backlog) == 1
    assert backlog[0].name == "tool_call_start"


async def test_event_trim() -> None:
    rt = build_in_memory_runtime()
    for i in range(5):
        await rt.events.emit(
            Event(
                run_id="r1",
                name="turn_start",
                payload={"tenant_id": rt.tenant_id, "i": i},
            )
        )
    await rt.events.trim("r1", rt.tenant_id, max_len=2)
    assert len(rt.events.stream_for(rt.tenant_id, "r1")) == 2


# ----- SkillStore -----


async def test_skill_upsert_and_list() -> None:
    rt = build_in_memory_runtime()
    manifest = SkillManifest(
        id="sk1",
        name="Helper",
        description="Helps with stuff.",
        tenant_id=rt.tenant_id,
    )
    await rt.skills.upsert(rt.tenant_id, manifest, body="# Body")
    entries = await rt.skills.list(rt.tenant_id)
    assert len(entries) == 1
    assert entries[0].id == "sk1"
    bundle = await rt.skills.load(rt.tenant_id, "sk1")
    assert bundle.body == "# Body"


async def test_skill_list_subset_matches_by_name() -> None:
    """The in-memory ``list_subset`` matches by bare account-scoped name
    like the real ``PgSkillStore``. Only the requested names surface."""
    rt = build_in_memory_runtime()
    await rt.skills.create(
        rt.tenant_id,
        SkillUpsertInput(name="foo", description="foo skill", body_md="# f"),
    )
    await rt.skills.create(
        rt.tenant_id,
        SkillUpsertInput(name="bar", description="bar skill", body_md="# b"),
    )

    only_foo = await rt.skills.list_subset(rt.tenant_id, ["foo"])
    assert len(only_foo) == 1
    assert only_foo[0].name == "foo"
    assert only_foo[0].description == "foo skill"

    both = await rt.skills.list_subset(rt.tenant_id, ["foo", "bar"])
    assert {e.name for e in both} == {"foo", "bar"}

    # An unknown name resolves to nothing.
    assert await rt.skills.list_subset(rt.tenant_id, ["missing"]) == []


async def test_skill_list_enabled_subset_drops_disabled_winner() -> None:
    """The in-memory ``list_enabled_subset`` mirrors the real store: it applies
    the same bare-name matching as ``list_subset`` PLUS an ``enabled = TRUE``
    filter, so a disabled skill is dropped from prompt surfacing while
    ``list_subset`` (whitelist resolution) still returns it."""
    rt = build_in_memory_runtime()
    entry = await rt.skills.create(
        rt.tenant_id,
        SkillUpsertInput(name="gated", description="toggle me", body_md="# g"),
    )
    # Whitelist + enabled-subset both surface it while enabled.
    assert len(await rt.skills.list_subset(rt.tenant_id, ["gated"])) == 1
    assert len(await rt.skills.list_enabled_subset(rt.tenant_id, ["gated"])) == 1

    await rt.skills.set_enabled(rt.tenant_id, entry.id, enabled=False)

    # Whitelist resolution ignores the toggle (existing subagent-dispatch
    # caller semantics — unchanged).
    assert len(await rt.skills.list_subset(rt.tenant_id, ["gated"])) == 1
    # Prompt-surfacing path drops the disabled skill (disable gates beat pins).
    assert await rt.skills.list_enabled_subset(rt.tenant_id, ["gated"]) == []


# ----- AgentDispatch -----


async def test_agent_dispatch_register_and_dispatch() -> None:
    rt = build_in_memory_runtime()
    rt.agents.register(
        SubagentDef(
            id="sa1",
            tenant_id=rt.tenant_id,
            name="Helper",
            description="Help",
            system_prompt="Be helpful.",
        )
    )
    listed = await rt.agents.list_subagents(rt.tenant_id)
    assert len(listed) == 1
    result = await rt.agents.dispatch(
        SubagentTask(subagent_id="sa1", parent_run_id="r1", task_prompt="hi")
    )
    assert result.success


async def test_agent_dispatch_missing_raises() -> None:
    rt = build_in_memory_runtime()
    with pytest.raises(SubagentNotFoundError):
        await rt.agents.get(rt.tenant_id, "missing")


# ----- Todo -----


async def test_todo_write_and_read() -> None:
    rt = build_in_memory_runtime()
    todo = Todo(content="do thing")
    await rt.todos.write("s1", rt.tenant_id, [todo])
    todos = await rt.todos.read("s1", rt.tenant_id)
    assert len(todos) == 1
    assert todos[0].content == "do thing"


# ----- Search -----


async def test_search_index_and_match() -> None:
    rt = build_in_memory_runtime()
    await rt.search.index(
        IndexDoc(doc_id="d1", tenant_id=rt.tenant_id, fields={"text": "Hello world"})
    )
    hits = await rt.search.search("hello", rt.tenant_id)
    assert len(hits) == 1
    assert hits[0].doc_id == "d1"


# ----- HookManager -----


async def test_hook_invoke_records_and_returns_default_allow() -> None:
    rt = build_in_memory_runtime()
    result = await rt.hooks.invoke(
        HookEvent.pre_tool_use,
        {"tool_name": "Bash"},
        rt.tenant_id,
    )
    assert result.action == "allow"
    assert len(rt.hooks.invocations) == 1


async def test_hook_queue_action_overrides_default() -> None:
    rt = build_in_memory_runtime()
    rt.hooks.queue_action(
        HookEvent.pre_tool_use,
        HookResult(action="deny", reason="blocked by test"),
    )
    result = await rt.hooks.invoke(
        HookEvent.pre_tool_use,
        {"tool_name": "Bash"},
        rt.tenant_id,
    )
    assert result.action == "deny"


# ----- ToolRegistry -----


def test_tool_registry_list_for_tenant() -> None:
    rt = build_in_memory_runtime()
    policy = ToolVisibilityPolicy()
    surface = rt.tools.compute_effective_surface(rt.tenant_id, policy)
    assert surface == []


# ----- InMemoryHookManager: register / unregister / list -----


async def test_hook_register_list_unregister_roundtrip() -> None:
    rt = build_in_memory_runtime()
    spec = HookSpec(
        id="h1",
        tenant_id=rt.tenant_id,
        event=HookEvent.pre_tool_use,
        executor="http",
        config={"url": "https://example.invalid/hook"},
    )
    await rt.hooks.register(spec)
    listed = await rt.hooks.list(rt.tenant_id)
    assert len(listed) == 1
    assert listed[0].id == "h1"

    filtered = await rt.hooks.list(rt.tenant_id, event=HookEvent.pre_tool_use)
    assert len(filtered) == 1
    none_match = await rt.hooks.list(rt.tenant_id, event=HookEvent.session_start)
    assert none_match == []

    await rt.hooks.unregister("h1", rt.tenant_id)
    assert await rt.hooks.list(rt.tenant_id) == []


async def test_hook_unregister_other_tenant_is_noop() -> None:
    rt = build_in_memory_runtime()
    spec = HookSpec(
        id="h1",
        tenant_id=rt.tenant_id,
        event=HookEvent.pre_tool_use,
        executor="http",
    )
    await rt.hooks.register(spec)
    await rt.hooks.unregister("h1", tenant_id="other-tenant")
    # Still registered.
    assert len(await rt.hooks.list(rt.tenant_id)) == 1


# ----- LLM: stream + structured + counts -----


async def test_llm_stream_emits_message_lifecycle() -> None:
    rt = build_in_memory_runtime()
    rt.llm.queue_response(text="hi", input_tokens=2, output_tokens=1)
    events = []
    async for ev in rt.llm.stream_with_tools(LLMRequest(model="test-model", messages=[])):
        events.append(ev.name)
    assert events[0] == "message_start"
    assert events[-1] == "message_stop"
    assert "content_block_delta" in events
    # Calls captured for downstream assertions.
    assert len(rt.llm.calls) == 1


async def test_llm_stream_empty_queue_yields_minimal_sequence() -> None:
    rt = build_in_memory_runtime()
    events = []
    async for ev in rt.llm.stream_with_tools(LLMRequest(model="test-model", messages=[])):
        events.append(ev.name)
    assert events == ["message_start", "message_stop"]


async def test_llm_complete_structured_pops_queue() -> None:
    rt = build_in_memory_runtime()
    rt.llm.queue_response(text="ok", stop_reason=StopReason.tool_use)
    response = await rt.llm.complete_structured(LLMRequest(model="test-model", messages=[]), {"type": "object"})
    assert response.stop_reason is StopReason.tool_use


async def test_llm_complete_structured_empty_queue_returns_blank() -> None:
    rt = build_in_memory_runtime()
    response = await rt.llm.complete_structured(LLMRequest(model="test-model", messages=[]), {})
    assert response.stop_reason is StopReason.end_turn
    assert response.message.content_blocks == []


async def test_llm_complete_text_pops_queue() -> None:
    rt = build_in_memory_runtime()
    rt.llm.queue_response(text="folded summary", stop_reason=StopReason.end_turn)
    response = await rt.llm.complete_text(LLMRequest(model="test-model", messages=[]))
    assert response.stop_reason is StopReason.end_turn
    assert response.message.content_blocks[0].text == "folded summary"


async def test_llm_complete_text_empty_queue_returns_blank() -> None:
    rt = build_in_memory_runtime()
    response = await rt.llm.complete_text(LLMRequest(model="test-model", messages=[]))
    assert response.stop_reason is StopReason.end_turn
    assert response.message.content_blocks == []


async def test_llm_complete_text_records_the_call() -> None:
    """Plain completions land in the same call log as streamed / structured
    ones so tests can assert on the request the fold actually sent."""
    rt = build_in_memory_runtime()
    request = LLMRequest(model="test-model", messages=[])
    await rt.llm.complete_text(request)
    assert rt.llm.calls[-1] is request


def test_llm_count_tokens_heuristic() -> None:
    rt = build_in_memory_runtime()
    assert rt.llm.count_tokens("") == 0
    assert rt.llm.count_tokens("x") == 1
    # 1 per ~4 chars heuristic — "abcdefghij" = 10 chars -> 2.
    assert rt.llm.count_tokens("abcdefghij") >= 2


# ----- Blob: head / exists / list_prefix -----


async def test_blob_head_and_exists() -> None:
    rt = build_in_memory_runtime()
    md = await rt.blob.put(rt.tenant_id, b"abc", content_type="text/plain")
    assert await rt.blob.exists(rt.tenant_id, md.ref) is True
    assert await rt.blob.exists(rt.tenant_id, "missing") is False
    headed = await rt.blob.head(rt.tenant_id, md.ref)
    assert headed.content_type == "text/plain"


async def test_blob_head_missing_raises() -> None:
    rt = build_in_memory_runtime()
    with pytest.raises(BlobNotFoundError):
        await rt.blob.head(rt.tenant_id, "missing")


async def test_blob_list_prefix_filters_by_prefix() -> None:
    rt = build_in_memory_runtime()
    await rt.blob.put(rt.tenant_id, b"a")
    await rt.blob.put(rt.tenant_id, b"b")
    results = await rt.blob.list_prefix(rt.tenant_id, prefix=rt.tenant_id, limit=10)
    assert len(results) == 2
    none_results = await rt.blob.list_prefix(rt.tenant_id, prefix="zzzz")
    assert none_results == []


# ----- Search: filters + delete -----


async def test_search_with_filter_predicate_excludes_mismatch() -> None:
    rt = build_in_memory_runtime()
    await rt.search.index(
        IndexDoc(doc_id="d1", tenant_id=rt.tenant_id, fields={"text": "hello", "kind": "a"})
    )
    await rt.search.index(
        IndexDoc(doc_id="d2", tenant_id=rt.tenant_id, fields={"text": "hello", "kind": "b"})
    )
    hits = await rt.search.search("hello", rt.tenant_id, filters={"kind": "a"})
    assert [h.doc_id for h in hits] == ["d1"]


async def test_search_other_tenant_isolated() -> None:
    rt = build_in_memory_runtime()
    await rt.search.index(
        IndexDoc(doc_id="d1", tenant_id="other", fields={"text": "hello"})
    )
    hits = await rt.search.search("hello", rt.tenant_id)
    assert hits == []


async def test_search_delete_returns_bool() -> None:
    rt = build_in_memory_runtime()
    await rt.search.index(IndexDoc(doc_id="d1", tenant_id=rt.tenant_id, fields={"x": "y"}))
    assert await rt.search.delete("d1", rt.tenant_id) is True
    assert await rt.search.delete("d1", rt.tenant_id) is False


# ----- AgentDispatch: queued result + missing list -----


async def test_agent_dispatch_queued_result() -> None:
    rt = build_in_memory_runtime()
    rt.agents.register(
        SubagentDef(
            id="sa1",
            tenant_id=rt.tenant_id,
            name="Helper",
            description="Help",
            system_prompt="Be helpful.",
        )
    )
    rt.agents.queue_result(
        SubagentResult(
            subagent_id="sa1",
            parent_run_id="r1",
            output="custom",
            success=True,
        )
    )
    result = await rt.agents.dispatch(
        SubagentTask(subagent_id="sa1", parent_run_id="r1", task_prompt="hi")
    )
    assert result.output == "custom"


# ----- Session: append/list errors + since filter -----


async def test_session_append_missing_raises() -> None:
    rt = build_in_memory_runtime()
    with pytest.raises(SessionNotFoundError):
        await rt.sessions.append_message(
            "missing",
            rt.tenant_id,
            Message(role=MessageRole.user, content_blocks=[]),
        )


async def test_session_list_messages_missing_raises() -> None:
    rt = build_in_memory_runtime()
    with pytest.raises(SessionNotFoundError):
        await rt.sessions.list_messages("missing", rt.tenant_id)


async def test_session_list_messages_since_filter() -> None:
    rt = build_in_memory_runtime()
    await rt.sessions.create(Session(id="s1", tenant_id=rt.tenant_id))
    past = datetime.now(UTC) - timedelta(days=1)
    future_cutoff = datetime.now(UTC) + timedelta(days=1)
    msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    await rt.sessions.append_message("s1", rt.tenant_id, msg)
    # With cutoff in the past, message is returned.
    after_past = await rt.sessions.list_messages("s1", rt.tenant_id, since=past)
    assert len(after_past) == 1
    # With cutoff in the future, nothing is returned.
    after_future = await rt.sessions.list_messages("s1", rt.tenant_id, since=future_cutoff)
    assert after_future == []


# ----- Run: list filters + tenant isolation -----


async def test_run_list_filters_by_status() -> None:
    rt = build_in_memory_runtime()
    await rt.runs.create(Run(id="r1", tenant_id=rt.tenant_id, session_id="s1"))
    await rt.runs.create(Run(id="r2", tenant_id=rt.tenant_id, session_id="s1"))
    await rt.runs.update_status("r1", rt.tenant_id, RunStatus.running)
    running = await rt.runs.list(rt.tenant_id, filters={"status": RunStatus.running})
    assert len(running) == 1
    assert running[0].id == "r1"


async def test_run_list_no_filters_returns_all_for_tenant() -> None:
    rt = build_in_memory_runtime()
    await rt.runs.create(Run(id="r1", tenant_id=rt.tenant_id, session_id="s1"))
    await rt.runs.create(Run(id="r2", tenant_id="other", session_id="s1"))
    listed = await rt.runs.list(rt.tenant_id)
    assert {r.id for r in listed} == {"r1"}


# ----- EventStream: subscribe replay + from_event_id -----


async def test_event_subscribe_replays_backlog() -> None:
    rt = build_in_memory_runtime()
    for i in range(3):
        await rt.events.emit(
            Event(
                id=f"e{i}",
                run_id="r1",
                name="turn_start",
                payload={"tenant_id": rt.tenant_id, "i": i},
            )
        )
    seen: list[str] = []
    async for ev in rt.events.subscribe("r1", rt.tenant_id):
        seen.append(ev.id)
        if len(seen) == 3:
            break
    assert seen == ["e0", "e1", "e2"]


async def test_event_subscribe_from_event_id_skips_backlog() -> None:
    rt = build_in_memory_runtime()
    for i in range(3):
        await rt.events.emit(
            Event(
                id=f"e{i}",
                run_id="r1",
                name="turn_start",
                payload={"tenant_id": rt.tenant_id, "i": i},
            )
        )
    seen: list[str] = []
    async for ev in rt.events.subscribe("r1", rt.tenant_id, from_event_id="e0"):
        seen.append(ev.id)
        if len(seen) == 2:
            break
    assert seen == ["e1", "e2"]


async def test_event_trim_below_threshold_is_noop() -> None:
    rt = build_in_memory_runtime()
    await rt.events.emit(
        Event(run_id="r1", name="turn_start", payload={"tenant_id": rt.tenant_id})
    )
    # max_len greater than stream length -> nothing trimmed.
    await rt.events.trim("r1", rt.tenant_id, max_len=10)
    assert len(rt.events.stream_for(rt.tenant_id, "r1")) == 1
    # Unknown key trim -> noop.
    await rt.events.trim("missing", rt.tenant_id, max_len=1)


# ----- ToolRegistry: visibility + top_k -----


def test_tool_registry_top_k_truncates() -> None:
    rt = build_in_memory_runtime()
    policy = ToolVisibilityPolicy()
    # Surface is empty in default mock; top_k applied to empty list still empty.
    surface = rt.tools.compute_effective_surface(rt.tenant_id, policy, top_k=5)
    assert surface == []
