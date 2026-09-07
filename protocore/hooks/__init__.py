"""The in-process lifecycle registry.

The contract is :mod:`protocore.contracts.middleware`;
:class:`~protocore.hooks.manager.HookManager` is the core's implementation of
it. Executors that leave the process — an HTTP endpoint, a model asked to
judge — belong to the host and reach the same seam as ordinary registrations.
"""
from __future__ import annotations

from protocore.hooks.manager import HookManager, refuse_lifecycle_when_disabled

__all__ = ["HookManager", "refuse_lifecycle_when_disabled"]
