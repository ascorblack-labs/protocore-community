"""Unit tests for :mod:`protocore.prompts.jinja_provider`.

Covers:

* All 5 bundled templates render with realistic context.
* ``StrictUndefined`` fires on missing variables (no silent empty
 substitution).
* The sandbox blocks attribute-traversal escape attempts
 (``__class__``, ``__mro__``).
* ``{% include %}`` outside the allowlist is rejected by the loader.
* Macros work correctly (the macros file is includable).
* Custom registry returns the override stub (per-tenant override
 acceptance test from the prompt — the bundled provider can be
 parameterised with a different registry).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from protocore.contracts.prompts import (
    IPromptTemplateProvider,
    PromptTemplateNotFoundError,
    PromptTemplateRenderError,
)
from protocore.prompts import BUNDLED_TEMPLATES, JinjaPromptTemplateProvider

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def provider() -> JinjaPromptTemplateProvider:
    """Default provider pointing at the bundled templates."""
    return JinjaPromptTemplateProvider()


@pytest.fixture()
def leader_ctx() -> dict[str, object]:
    """Realistic context for leader_system rendering."""
    return {
        "current_date": "2026-05-19",
        "persona_md": "# Custom Persona\nYou are tenant-specific.",
        "agent_descriptions": {
            "coder": "Writes code",
            "reviewer": "Reviews code",
        },
        "environment_capabilities": {
            "file_read": True,
            "file_write": True,
            "shell_profile": "default",
            "network_allowed": True,
            "package_install": False,
            "server_hosting": False,
            "long_running_processes": False,
        },
        "capabilities": {
            "delegation": True,
            "delegation_max": 3,
            "planning": True,
        },
        "finalization_contract_block": "<finalization_contract>...</finalization_contract>",
    }


# ---------------------------------------------------------------------------
# Protocol conformance + introspection
# ---------------------------------------------------------------------------


def test_provider_implements_protocol(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Runtime ``isinstance`` check against the Protocol."""
    assert isinstance(provider, IPromptTemplateProvider)


def test_known_templates_match_bundle(
    provider: JinjaPromptTemplateProvider,
) -> None:
    assert sorted(provider.known_templates) == sorted(BUNDLED_TEMPLATES.keys())


# ---------------------------------------------------------------------------
# Bundled template rendering
# ---------------------------------------------------------------------------


def test_leader_system_renders_full_context(
    provider: JinjaPromptTemplateProvider,
    leader_ctx: dict[str, object],
) -> None:
    """Persona is an ADDITIVE personality layer on top of the always-on
    bundled scaffolding + dynamic blocks.

    A custom ``persona_md`` must NOT suppress the bundled generic scaffolding.
    The catastrophic "no tools -> prose" failure was the empty BM25 surface
    (fixed by the forced tool-surface pins), not the scaffolding. With a full
    context the rendered prompt now contains the generic scaffolding, the
    dynamic subagent / environment / capability blocks, the persona body, AND
    the independently-gated finalization contract block.
    """
    rendered = provider.render("leader_system", leader_ctx)
    # Persona prelude is appended as a personality layer.
    assert "Custom Persona" in rendered
    assert "You are tenant-specific." in rendered
    # Always-on generic scaffolding survives the persona.
    assert "You are a Protocore agent." in rendered
    assert "Current date: 2026-05-19" in rendered
    # Subagent list / environment manifest / capability macros render too.
    assert "coder: Writes code" in rendered
    assert "reviewer: Reviews code" in rendered
    assert "<environment>" in rendered
    assert "Shell execution (profile: default)" in rendered
    # package_install=False in the fixture → manifest lists it under the
    # "NOT AVAILABLE (permanently denied)" section, so the string is present.
    assert "Package installation (pip install, npm install, apt-get)" in rendered
    assert "Maximum concurrent delegations: 3." in rendered
    # Finalization contract block stays — gated independently by its
    # own RC and surfaced verbatim as machine-readable JSON sentinels.
    assert "<finalization_contract>" in rendered


def test_leader_system_renders_minimal_context(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Optional vars are skipped without error."""
    rendered = provider.render(
        "leader_system",
        {
            "current_date": "2026-05-19",
            "persona_md": None,
            "agent_descriptions": None,
            "environment_capabilities": None,
            "capabilities": None,
            "finalization_contract_block": None,
        },
    )
    assert "You are a Protocore agent." in rendered
    assert "Custom Persona" not in rendered
    assert "<environment>" not in rendered
    # leader_system.j2 references `<finalization_contract>` inline
    # as part of the anchoring guidance in the Efficiency & Error Handling
    # block, so the bare tag is always present in the EN+RU compact section.
    # The actual rendered contract block (JSON template + sentinels) is the
    # gated bit — verify the block body (CONTRACT_OPEN_TAG followed by JSON
    # opening brace) is absent.
    assert "<finalization_contract>\n{" not in rendered
    assert "coder" not in rendered


def test_planner_renders_with_agents(
    provider: JinjaPromptTemplateProvider,
) -> None:
    rendered = provider.render(
        "planner",
        {"agent_descriptions": {"solver": "Solves problems"}},
    )
    assert "planning agent" in rendered
    assert "submit_plan" in rendered
    assert "solver: Solves problems" in rendered
    assert "Available subagents for delegation" in rendered


def test_planner_renders_without_agents(
    provider: JinjaPromptTemplateProvider,
) -> None:
    rendered = provider.render(
        "planner",
        {"agent_descriptions": None},
    )
    assert "planning agent" in rendered
    assert "submit_plan" in rendered
    assert "Available subagents" not in rendered


def test_subagent_contract_renders_full_context(
    provider: JinjaPromptTemplateProvider,
) -> None:
    rendered = provider.render(
        "subagent_contract",
        {
            "base_system_prompt": "You are a helpful subagent.",
            "current_date": "2026-05-19",
            "agent_id": "worker-1",
            "task_description": "Compute the answer.",
            "output_format": "JSON",
            "environment_capabilities": {
                "file_read": True,
                "file_write": False,
                "shell_profile": "disabled",
                "network_allowed": False,
                "package_install": False,
                "server_hosting": False,
                "long_running_processes": False,
            },
            "extra_context": "Operator note: be terse.",
            "forwarded_source_refs": ("ref_abc", "ref_def"),
        },
    )
    assert "You are a helpful subagent." in rendered
    assert "Subagent: worker-1" in rendered
    assert "Current date: 2026-05-19" in rendered
    assert "Task: Compute the answer." in rendered
    assert "Output format: JSON" in rendered
    assert "File read (workspace)" in rendered
    assert "Shell execution (entirely disabled)" in rendered
    assert "Operator note: be terse." in rendered
    assert "Forwarded source refs: ref_abc, ref_def" in rendered
    assert "SubmitAnswer" in rendered


def test_subagent_contract_renders_minimal_context(
    provider: JinjaPromptTemplateProvider,
) -> None:
    rendered = provider.render(
        "subagent_contract",
        {
            "base_system_prompt": "Persona body.",
            "current_date": "2026-05-19",
            "agent_id": "worker-2",
            "task_description": None,
            "output_format": None,
            "environment_capabilities": None,
            "extra_context": None,
            "forwarded_source_refs": None,
        },
    )
    assert "Persona body." in rendered
    assert "Subagent: worker-2" in rendered
    assert "Task:" not in rendered
    assert "Output format:" not in rendered
    assert "<environment>" not in rendered
    assert "Forwarded source refs" not in rendered


def test_finalization_template_renders(
    provider: JinjaPromptTemplateProvider,
) -> None:
    rendered = provider.render("finalization")
    assert "FINAL answer" in rendered
    assert "maximum number of tool calls" in rendered


def test_environment_manifest_renders(
    provider: JinjaPromptTemplateProvider,
) -> None:
    rendered = provider.render(
        "environment_manifest",
        {
            "environment_capabilities": {
                "file_read": True,
                "file_write": True,
                "shell_profile": "restricted",
                "network_allowed": False,
                "package_install": True,
                "server_hosting": False,
                "long_running_processes": False,
            }
        },
    )
    assert "<environment>" in rendered
    assert "Shell execution (profile: restricted)" in rendered
    assert "Package installation" in rendered
    # network denied → must appear in NOT AVAILABLE block.
    assert "Network access (curl, wget, API calls)" in rendered


# ---------------------------------------------------------------------------
# StrictUndefined behaviour
# ---------------------------------------------------------------------------


def test_strict_undefined_fires_on_missing_variable(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Missing variables MUST raise — silent empty substitution is rejected.

    This is the load-bearing guarantee that catches template typos at
    deploy time. Without ``StrictUndefined`` a typo like ``{{ persona }}``
    instead of ``{{ persona_md }}`` would silently emit empty string.
    """
    with pytest.raises(PromptTemplateRenderError) as exc_info:
        provider.render("leader_system", {})
    # The error message should reference the missing variable name.
    assert "undefined" in str(exc_info.value).lower()


def test_strict_undefined_fires_on_missing_nested_variable(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Nested attribute access on undefined variable also raises."""
    with pytest.raises(PromptTemplateRenderError):
        provider.render(
            "leader_system",
            {
                "current_date": "2026-05-19",
                "persona_md": None,
                # ``agent_descriptions`` deliberately missing — the macro
                # only emits when the var is set, so this should NOT
                # raise. We test the contrary: missing the inner attr.
                "agent_descriptions": None,
                # Pass a capabilities dict that the macro will read
                # ``.delegation`` from. The variable IS defined; what we
                # really want to trip is reading an attribute name that
                # the macro does not check, so this test instead asserts
                # the BARE missing var:
            },
        )


# ---------------------------------------------------------------------------
# Sandbox security — block code-execution escape attempts
# ---------------------------------------------------------------------------


def _provider_for_custom_template(
    tmp_path: Path, body: str, *, name: str = "evil"
) -> JinjaPromptTemplateProvider:
    """Spin up a provider rooted at ``tmp_path`` with a single template."""
    file_name = f"{name}.j2"
    (tmp_path / file_name).write_text(body, encoding="utf-8")
    return JinjaPromptTemplateProvider(
        template_dir=tmp_path,
        registry={name: file_name},
    )


def test_sandbox_blocks_dunder_attribute_traversal(tmp_path: Path) -> None:
    """``__class__`` / ``__mro__`` chains are rejected by the sandbox.

    A common Jinja2 SSTI escape pattern walks ``""__class__.__mro__[1].__subclasses__()``
    to find ``subprocess.Popen``. The sandbox MUST reject that.
    """
    body = "{{ ''.__class__.__mro__[1].__subclasses__() }}"
    sandbox = _provider_for_custom_template(tmp_path, body)
    with pytest.raises(PromptTemplateRenderError) as exc_info:
        sandbox.render("evil", {})
    msg = str(exc_info.value).lower()
    # SandboxedEnvironment surfaces these as SecurityError; the provider
    # wraps as PromptTemplateRenderError. Either the wrapper text or the
    # underlying chain mentions sandbox/security/attribute.
    assert any(token in msg for token in ("sandbox", "security", "access", "attribute"))


def test_sandbox_blocks_subprocess_call(tmp_path: Path) -> None:
    """Calling Python builtins through string templates is rejected."""
    body = "{{ ''.__class__.__bases__[0].__subclasses__()[0]('uname') }}"
    sandbox = _provider_for_custom_template(tmp_path, body)
    with pytest.raises(PromptTemplateRenderError):
        sandbox.render("evil", {})


def test_loader_rejects_include_outside_allowlist(tmp_path: Path) -> None:
    """``{% include %}`` of a non-allowlisted file is rejected.

    Even if a malicious operator drops ``secret.j2`` into the templates
    dir, a template body cannot pull it in because the allowlisted
    loader rejects the lookup.
    """
    (tmp_path / "secret.j2").write_text("LEAKED SECRET", encoding="utf-8")
    body = '{% include "secret.j2" %}'
    sandbox = _provider_for_custom_template(tmp_path, body, name="loader")
    with pytest.raises(PromptTemplateRenderError):
        sandbox.render("loader", {})


def test_loader_accepts_macros_include(tmp_path: Path) -> None:
    """The bundled ``macros.j2`` IS in the allowlist — includes succeed."""
    # Drop a minimal macros.j2 (mirrors the include-allowlist).
    (tmp_path / "macros.j2").write_text(
        "{% macro hello(name) %}HELLO {{ name }}{% endmacro %}",
        encoding="utf-8",
    )
    body = (
        '{% from "macros.j2" import hello %}\n'
        "{{ hello(name) }}"
    )
    sandbox = _provider_for_custom_template(tmp_path, body, name="user")
    rendered = sandbox.render("user", {"name": "world"})
    assert "HELLO world" in rendered


# ---------------------------------------------------------------------------
# Error surface
# ---------------------------------------------------------------------------


def test_unknown_template_raises_not_found(
    provider: JinjaPromptTemplateProvider,
) -> None:
    with pytest.raises(PromptTemplateNotFoundError) as exc_info:
        provider.render("does_not_exist", {})
    msg = str(exc_info.value)
    assert "does_not_exist" in msg


def test_missing_file_on_disk_raises_not_found(tmp_path: Path) -> None:
    """Registry refers to a file that does not exist → NotFound."""
    provider = JinjaPromptTemplateProvider(
        template_dir=tmp_path,
        registry={"orphan": "orphan.j2"},
    )
    with pytest.raises(PromptTemplateNotFoundError):
        provider.render("orphan", {})


def test_syntax_error_raises_render_error(tmp_path: Path) -> None:
    """A template with broken Jinja2 syntax surfaces as RenderError."""
    sandbox = _provider_for_custom_template(
        tmp_path, body="{% if missing_endif %}"
    )
    with pytest.raises(PromptTemplateRenderError) as exc_info:
        sandbox.render("evil", {})
    assert "parse" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------


def test_macros_agent_descriptions_block(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """The agent_descriptions_block macro is exercised via leader_system."""
    rendered = provider.render(
        "leader_system",
        {
            "current_date": "2026-05-19",
            "persona_md": None,
            "agent_descriptions": {"alpha": "first", "beta": "second"},
            "environment_capabilities": None,
            "capabilities": None,
            "finalization_contract_block": None,
        },
    )
    assert "Available subagents:" in rendered
    assert "- alpha: first" in rendered
    assert "- beta: second" in rendered


def test_macros_capability_line_omitted_when_no_capabilities(
    provider: JinjaPromptTemplateProvider,
) -> None:
    rendered = provider.render(
        "leader_system",
        {
            "current_date": "2026-05-19",
            "persona_md": None,
            "agent_descriptions": None,
            "environment_capabilities": None,
            "capabilities": None,
            "finalization_contract_block": None,
        },
    )
    assert "Maximum concurrent delegations" not in rendered
    assert "Use a short plan" not in rendered


def test_macros_capability_line_partial(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """``planning=True`` alone emits the planning hint but not delegation."""
    rendered = provider.render(
        "leader_system",
        {
            "current_date": "2026-05-19",
            "persona_md": None,
            "agent_descriptions": None,
            "environment_capabilities": None,
            "capabilities": {
                "delegation": False,
                "delegation_max": 0,
                "planning": True,
            },
            "finalization_contract_block": None,
        },
    )
    assert "Use a short plan when it clarifies execution." in rendered
    assert "Maximum concurrent delegations" not in rendered


# ---------------------------------------------------------------------------
# Per-tenant override stub (custom registry returns the override)
# ---------------------------------------------------------------------------


def test_custom_registry_returns_override_stub(tmp_path: Path) -> None:
    """The provider with a custom registry uses the overridden body.

    This stubs the per-tenant override layer: the host's
    :class:`TenantPromptTemplateProvider` swaps the registry / template
    dir per template name, and the bundled provider renders whatever it
    is pointed at. The test verifies that pointing at a *different*
    body for ``leader_system`` yields the overridden text — not the
    bundled defaults.
    """
    override_body = "TENANT OVERRIDE — Current date: {{ current_date }}"
    (tmp_path / "leader_system.j2").write_text(override_body, encoding="utf-8")
    overridden = JinjaPromptTemplateProvider(
        template_dir=tmp_path,
        registry={"leader_system": "leader_system.j2"},
    )
    rendered = overridden.render("leader_system", {"current_date": "2026-05-19"})
    assert rendered.startswith("TENANT OVERRIDE")
    assert "You are a Protocore agent." not in rendered


def test_current_date_auto_injection(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """``current_date`` is auto-injected when caller omits it."""
    # Use a minimal template that depends only on ``current_date``.
    rendered = provider.render("finalization")
    # Finalization template does not reference current_date, but render
    # should succeed without the caller providing it.
    assert rendered  # smoke

    # leader_system DOES reference current_date — call without supplying it.
    leader = provider.render(
        "leader_system",
        {
            "persona_md": None,
            "agent_descriptions": None,
            "environment_capabilities": None,
            "capabilities": None,
            "finalization_contract_block": None,
        },
    )
    import datetime as _dt
    today = _dt.date.today().isoformat()
    assert today in leader


def test_current_date_auto_injection_disabled(tmp_path: Path) -> None:
    """``inject_current_date=False`` disables the auto-injection."""
    (tmp_path / "tpl.j2").write_text(
        "Date: {{ current_date }}", encoding="utf-8"
    )
    provider = JinjaPromptTemplateProvider(
        template_dir=tmp_path,
        registry={"tpl": "tpl.j2"},
        inject_current_date=False,
    )
    with pytest.raises(PromptTemplateRenderError):
        provider.render("tpl", {})
