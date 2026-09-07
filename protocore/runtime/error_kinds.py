"""The names a run's terminal failure is reported under.

One module because two of them are read from both sides of the turn-policy
seam, and a string that means "the loop crashed" must not have a second
spelling on the policy's side of it.
"""
from __future__ import annotations

from typing import Final

#: Terminal ``kind`` for a failure that is this process, not the upstream.
#:
#: Every exception outside the LLM-error family lands here — a parser bug, a
#: ``RecursionError``, an ``AttributeError`` in the loop. Kept distinct from
#: ``llm_provider_error`` because the two demand opposite responses: a
#: provider failure is a reason to try a different endpoint, and this is a
#: reason to read a traceback. Conflating them made the provider-failure
#: metric count our own bugs, and made the one record of a real crash say the
#: upstream had failed. It is never recovered from: a best-effort turn armed
#: over one would let a later terminal complete "successfully" and swallow the
#: crash. ``stop_reason`` is unaffected — an internal error still ends the run
#: in ``error``.
INTERNAL_ERROR_KIND: Final[str] = "internal_error"

__all__ = ["INTERNAL_ERROR_KIND"]
