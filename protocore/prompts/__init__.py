"""Jinja2-backed prompt template provider + bundled defaults.

Public surface:
 JinjaPromptTemplateProvider — :class:`IPromptTemplateProvider`
 implementation backed by :mod:`jinja2.sandbox.SandboxedEnvironment`
 with ``StrictUndefined`` + a curated include-allowlist.
 BUNDLED_TEMPLATES — frozen mapping of template-name → file name for
 the templates shipped in this package.
 BUNDLED_TEMPLATE_DIR — pathlib.Path to the bundled templates dir
 (consumers wiring per-tenant overrides may pass it explicitly).

A host that needs per-tenant templates wraps this provider in an
override layer of its own; core ships only the bundled-defaults path.
"""

from __future__ import annotations

from protocore.prompts.jinja_provider import (
    BUNDLED_TEMPLATE_DIR,
    BUNDLED_TEMPLATES,
    JinjaPromptTemplateProvider,
)

__all__ = [
    "BUNDLED_TEMPLATES",
    "BUNDLED_TEMPLATE_DIR",
    "JinjaPromptTemplateProvider",
]
