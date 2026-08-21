"""IPromptTemplateProvider Protocol + error hierarchy.

The leader / planner / subagent contract / finalization / environment manifest
prompts flow through a Jinja2 template engine with per-tenant overrides.

This Protocol decouples the template engine from the engine wiring: the
core ships :class:`protocore.prompts.JinjaPromptTemplateProvider` as the
default ``SandboxedEnvironment``-backed implementation; the host wraps
it with the per-tenant override layer (S3 blob lookup → fall through to
bundled defaults). Subagent runner, executor leader prompt assembly, and
admin preview routes all consume the same Protocol so a future swap
(Liquid? Pebble?) only replaces the adapter, never the call sites.

Reference impl: ``protocore.prompts.jinja_provider.JinjaPromptTemplateProvider``.
A host that wants per-tenant template overrides wraps that provider in its own.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class PromptTemplateError(Exception):
    """Base class for prompt-template domain errors."""


class PromptTemplateNotFoundError(PromptTemplateError):
    """Requested template name is not in the registry.

    Raised when the caller asks for a template the provider does not
    recognise. The bundled provider knows its own template names from the
    catalogue (``leader_system``, ``planner``, ``subagent_contract``,
    ``finalization``, ``environment_manifest``); the tenant-override layer
    raises this when a name is absent from both the override table AND
    the bundled set.
    """


class PromptTemplateRenderError(PromptTemplateError):
    """Template body parsed but rendering produced an error.

    Wraps Jinja2 ``UndefinedError`` (caller passed an incomplete context
    dict) and ``SecurityError`` (template body attempted a forbidden
    operation in the sandbox). The bundled provider re-raises these as
    :class:`PromptTemplateRenderError` so call sites need only catch one
    type.
    """


class PromptTemplateValidationError(PromptTemplateError):
    """Template source failed validation before being stored.

    Raised by admin override CRUD: the operator submitted a body that
    fails Jinja2 syntax parsing or references a forbidden identifier.
    Never raised at render time — render-time issues surface as
    :class:`PromptTemplateRenderError` instead.
    """


@runtime_checkable
class IPromptTemplateProvider(Protocol):
    """Resolves named prompt templates and renders them with context.

    Contract:
        - ``render(name, context)`` returns the rendered string for the
          named template after substituting variables from ``context``.
        - Missing-variable references MUST raise
          :class:`PromptTemplateRenderError` (Jinja2 ``StrictUndefined``
          semantics) — silent empty-string substitution hides template
          typos until they reach the model. The bundled
          :class:`protocore.prompts.JinjaPromptTemplateProvider`
          enforces this.
        - Sandbox security: implementations MUST prevent template bodies
          from executing arbitrary Python (no ``__class__`` traversal,
          no arbitrary ``import`` / ``extends`` / ``include`` outside
          a curated set). Bundled provider uses Jinja2's
          ``SandboxedEnvironment`` + a fixed include allowlist
          (currently only ``macros.j2``).
        - ``render`` is synchronous — template rendering is CPU-bound +
          fast; async indirection is unnecessary noise. Per-tenant
          override lookup MAY be async, so a wrapping provider is free to
          expose an async render alongside this one.
    """

    def render(self, name: str, context: dict[str, Any] | None = None) -> str:
        """Render template ``name`` with ``context``.

        Raises:
            PromptTemplateNotFoundError: ``name`` is not in the registry.
            PromptTemplateRenderError: rendering failed (missing variable,
                sandboxed expression rejected, template syntax error).
        """
        ...


__all__ = [
    "IPromptTemplateProvider",
    "PromptTemplateError",
    "PromptTemplateNotFoundError",
    "PromptTemplateRenderError",
    "PromptTemplateValidationError",
]
