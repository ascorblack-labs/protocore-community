"""Memory-safety caps and protocol-level constants — NOT runtime-tunable.

These values exist to bound memory growth in pathological cases (a 10M-char
tool argument crashes the executor). They are NOT user-tunable via dashboard;
that is the job of :mod:`protocore.contracts.runtime_constants`. Everything
here is a compile-time invariant.
"""
from __future__ import annotations

from typing import Final

# Single tool-call argument payload cap. Raised from 32 KiB to 128 KiB to
# unblock long Write payloads.
# Industry baselines: OpenAI Realtime API 64 KiB, OpenAI Submit Tool Output
# 512 KiB, Anthropic messages.content.text 1 MiB. 128 KiB is a conservative
# 4-8x headroom — well below any provider cap while still defending against
# pathological megabyte JSON blobs from misbehaving LLM outputs.
MAX_TOOL_CALL_ARGUMENT_BYTES: Final[int] = 128 * 1024

# Single envelope payload cap (`AgentEnvelope.payload`). 25 000 chars.
MAX_ENVELOPE_PAYLOAD_CHARS: Final[int] = 25_000

# Single subagent task payload cap. 12 000 chars.
MAX_DELEGATE_TASK_CHARS: Final[int] = 12_000

# Structural caps for ExecutionReport — prevents pathological 100k-iter
# agents from producing multi-MB telemetry payloads.
MAX_REPORT_EVENTS: Final[int] = 500
MAX_TOOL_CALL_DETAILS: Final[int] = 1000
MAX_LLM_CALL_DETAILS: Final[int] = 500
MAX_WARNINGS: Final[int] = 200
MAX_SUBAGENT_RUNS: Final[int] = 200
MAX_ARTIFACTS: Final[int] = 500

# Defensive JSON parser caps.
MAX_STRUCTURED_JSON_CHARS: Final[int] = 1_000_000

# Nesting-depth ceiling for model-supplied data structures: tool-call argument
# JSON, message/tool-result metadata, helper-bag snapshots. Every walk over one
# of those is depth-bounded so a pathologically nested payload raises a named,
# catchable error instead of exhausting the interpreter stack — a ``RecursionError``
# thrown from inside a Pydantic validator or a streaming JSON parser unwinds
# through the whole run and is indistinguishable, at the point it is caught, from
# a crash with no cause. 200 levels is far past any structure a model legitimately
# produces and far below CPython's default 1000-frame limit, so the guard fires
# while there is still stack left to raise on.
#
# This is the STRUCTURAL floor, used where no RuntimeConstants snapshot is in
# scope (Pydantic field validators, the pure JSON utilities). It is also the
# default of ``RuntimeConstants.max_data_nesting_depth``, which is what the
# engine-driven paths read so the ceiling stays dashboard-tunable.
MAX_DATA_NESTING_DEPTH: Final[int] = 200

# Protocol marker for cross-pod compacted tool-result placeholders.
# Wire format with a byte-deterministic renderer per
# `runtime/wire_format.py`.
PROTOCOL_COMPACTED_TOOL_RESULT_V1: Final[str] = "PROTOCOL_COMPACTED_TOOL_RESULT_V1"

# Default model name used in test fixtures and bench. NOT a runtime default.
DEFAULT_MODEL: Final[str] = "qwen3.6-35b-a3b"

# Protocol version string surfaced in envelopes. Bumped whenever wire format breaks.
PROTOCOL_VERSION: Final[str] = "2.0.0"


__all__ = [
    "DEFAULT_MODEL",
    "MAX_ARTIFACTS",
    "MAX_DATA_NESTING_DEPTH",
    "MAX_DELEGATE_TASK_CHARS",
    "MAX_ENVELOPE_PAYLOAD_CHARS",
    "MAX_LLM_CALL_DETAILS",
    "MAX_REPORT_EVENTS",
    "MAX_STRUCTURED_JSON_CHARS",
    "MAX_SUBAGENT_RUNS",
    "MAX_TOOL_CALL_ARGUMENT_BYTES",
    "MAX_TOOL_CALL_DETAILS",
    "MAX_WARNINGS",
    "PROTOCOL_COMPACTED_TOOL_RESULT_V1",
    "PROTOCOL_VERSION",
]
