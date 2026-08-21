"""HookManager (pluggy-based) — typed spec dispatcher for 8 events.

Spec lives in core; HTTP / LLM-as-hook executors live in the host .
"""
from __future__ import annotations

from protocore.hooks.manager import HookManager
from protocore.hooks.specs import AgentHookSpecs, hookimpl, hookspec

__all__ = ["AgentHookSpecs", "HookManager", "hookimpl", "hookspec"]
