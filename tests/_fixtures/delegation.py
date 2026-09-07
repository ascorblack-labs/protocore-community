"""What a test double has to implement to be a delegation tool.

The loop recognises delegation by contract — a tool that can say how many child
runs one call starts and whether the caller waits for them — so a double that
only set a marker attribute is exactly the case the contract exists to reject.
This mixin is that contract, driven from the call's own arguments so a test pins
the batch width and the foreground/background choice per call.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class DelegationContract:
    """Implements :class:`protocore.contracts.agent_dispatch.IDelegationTool`."""

    @staticmethod
    def child_run_count(arguments: Mapping[str, Any]) -> int:
        """One call, one child run — the shape a double has unless it says.

        A tool whose one call starts a batch overrides this; the contract's own
        answer stays one so a double that never thought about batching is not
        silently charged for one.
        """
        return 1

    @staticmethod
    def is_background_call(arguments: Mapping[str, Any]) -> bool:
        return bool(arguments.get("background", False))


__all__ = ["DelegationContract"]
