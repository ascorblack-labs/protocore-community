"""The conformance suite for the request-manifest sink.

Its own module rather than a class in :mod:`protocore.conformance.suites`
because the contract it covers is the one a host is most likely to implement
LAST: a run works perfectly without a manifest sink, so an adapter written for
it is written after everything else is already green, against a core the host
has been running for a while. That is exactly the situation the suites exist
for — nothing calls the adapter until a run does, and by then the call is
inside somebody's turn.

The suite body is the shared one. What it adds is the reason the shape matters
here: the sink is called on the hot path, before the first delta of the request
it describes, and its one method is asynchronous because a host stores the
oversized bodies through its blob store. An adapter that wrote it synchronously
would hand the core a value where it awaits one, at the worst possible moment —
between the request being assembled and the provider being asked.

It also adds the one behavioural case in this package, because the contract's
timing requirement cannot be read off a signature: the core awaits this call
before it opens the stream, and a sink that takes its time there is latency
every user of the run sees. An adapter that blocks the event loop or takes
longer than the budget below is failed here rather than in production.
"""
from __future__ import annotations

import time
from typing import Any, ClassVar

from protocore.conformance.suite import ContractSuite
from protocore.contracts.llm import LLMRequest
from protocore.contracts.observability import (
    IRequestManifestSink,
    build_request_manifest,
)
from protocore.contracts.types import Message, MessageRole, TextBlock


class RequestManifestSinkConformance(ContractSuite):
    """Where the record of every provider call the core makes is handed over."""

    protocol = IRequestManifestSink

    #: How long the core's hot path is willing to wait for one handoff. A host
    #: whose store is slower than this hands the work to its own queue and
    #: returns; the contract says so, and this is where that is checked.
    handoff_budget_seconds: ClassVar[float] = 0.25

    @staticmethod
    def _sample_manifest() -> tuple[Any, dict[str, bytes]]:
        request = LLMRequest(
            model="a-model",
            messages=[
                Message(
                    role=MessageRole.user,
                    content_blocks=[TextBlock(text="hello")],
                )
            ],
        )
        return build_request_manifest(
            request=request,
            attempt_scope="run/turn/purpose",
            constants_sha256="0" * 64,
            inline_value_max_bytes=4096,
        )

    async def test_the_handoff_returns_promptly(self, subject: Any) -> None:
        """The core awaits this before it opens the stream it describes.

        Two ways an adapter breaks that, and both look like a healthy run from
        the outside. It can block the event loop — a synchronous store call
        inside an async method — which stalls every other task in the process,
        not just this run. Or it can simply take its time awaiting a store,
        which is time the user spends watching nothing arrive. Either way the
        budget below is the whole of the core's patience: a host that needs to
        do real work writes it to its own queue and returns.
        """
        manifest, bodies = self._sample_manifest()
        started = time.perf_counter()
        await subject.record_request_manifest(
            manifest=manifest,
            manifest_id=manifest.manifest_id,
            bodies=bodies,
        )
        elapsed = time.perf_counter() - started

        assert elapsed <= self.handoff_budget_seconds, (
            f"{type(subject).__name__}.record_request_manifest took "
            f"{elapsed:.3f}s; the core awaits it between assembling a request "
            "and asking the provider, so that is latency every user of the run "
            f"sees. The budget is {self.handoff_budget_seconds}s — hand the "
            "storing to your own queue and return."
        )


__all__ = ["RequestManifestSinkConformance"]
