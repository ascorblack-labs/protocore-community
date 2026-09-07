"""Every field of the run's state is either carried across a resume, or not — on purpose.

``to_snapshot`` writes two allowances and the resume reads them back. Nothing
made that a decision: a twenty-third field could be added tomorrow, be durable
in every sense that matters, and be dropped by a pickup with no test failing and
no reader the wiser. The engine-level inventory beside this one already settles
that question for engine attributes; this settles it for the state the engine
carries, which is where the run's allowances actually live now.

So every field is either read by ``to_snapshot`` or named below with the reason
it cannot travel. Adding a field to the state becomes a decision taken once, at
the time it is added.
"""
from __future__ import annotations

import dataclasses

from protocore.contracts.run_state import RunScopedState

# Fields a resume deliberately does NOT carry, and why. A live handle, an object
# bound to one event loop, a compartment whose owner is not this package, or a
# value the receiving process rebuilds for itself.
_PROCESS_LOCAL: dict[str, str] = {
    "rc": (
        "the constants snapshot is resolved by whoever composes the run, and "
        "the receiving process resolves its own before it builds the state."
    ),
    "root_run_id": (
        "the identity of the tree is stated by the caller that starts the "
        "pickup, not recovered from what the previous process wrote."
    ),
    "cancel_event": (
        "an asyncio.Event belongs to one event loop; a resumed run is given a "
        "fresh one by whoever can actually set it."
    ),
    "tool_shared_state_lock": (
        "a lock is a live primitive of one process, and serialising against a "
        "dead process's lock serialises nothing."
    ),
    "subagent_tree_permit": (
        "a permit is a slot held in a live semaphore; no process can release "
        "one taken in a process that is gone."
    ),
    "tool_call_soft_caps": (
        "the advisory limits are stated by the caller that composes the run, "
        "and are re-stated on every pickup."
    ),
    "tool_call_soft_cap_state": (
        "the counts are advisory guidance for one process's fan-out, and the "
        "lock in them cannot travel at all."
    ),
    "consecutive_error": (
        "an error streak measures how the current process is going; a pickup "
        "starts by trying again rather than by inheriting despair."
    ),
    "transport_down": (
        "a transport streak describes the provider connection of the process "
        "that saw it, and the new process has its own."
    ),
    "transport_down_injection_pending": (
        "the one-shot signal is consumed within the turn that raised it, so "
        "there is never a pending one to carry."
    ),
    "string_type": (
        "the streak measures repetition inside a process's own fan-out and is "
        "rebuilt from the transcript the pickup replays."
    ),
    "satisfied_preconditions": (
        "the satisfied set is derived from the run's messages on the way in, "
        "so persisting it would be a second answer to one question."
    ),
    "session_grants": (
        "the grants a person gave belong to the session store the host owns, "
        "and are re-read for the run that picks the session up."
    ),
    "tool_error_counter": (
        "a telemetry sink is a handle the host wires into the process that is "
        "running, not a value about the run."
    ),
    "adaptive_safety_band": (
        "the band is a telemetry handle the host wires per process, exactly as "
        "the error counter above is."
    ),
    "run_metadata": (
        "the per-run envelope is supplied by the caller on every drive, so a "
        "pickup receives it rather than recovering it."
    ),
    "host": (
        "the compartment belongs to whoever embeds this package; only its "
        "owner knows which of its slots are durable."
    ),
}


def _snapshot_keys() -> set[str]:
    return set(RunScopedState().to_snapshot())


def test_every_field_is_classified() -> None:
    """A field either travels with the run, or says why it cannot."""
    fields = {f.name for f in dataclasses.fields(RunScopedState)}
    unclassified = sorted(fields - _snapshot_keys() - set(_PROCESS_LOCAL))
    assert not unclassified, (
        "these fields of the run's state are neither carried by to_snapshot() "
        f"nor named as process-local: {unclassified}. Decide it here, once: "
        "either write the field into the snapshot and read it back in "
        "apply_snapshot(), or say in _PROCESS_LOCAL why a resumed run cannot "
        "be given it."
    )


def test_no_field_is_both_carried_and_declared_process_local() -> None:
    both = sorted(_snapshot_keys() & set(_PROCESS_LOCAL))
    assert not both, f"declared process-local but carried by to_snapshot(): {both}"


def test_no_stale_process_local_entries() -> None:
    fields = {f.name for f in dataclasses.fields(RunScopedState)}
    stale = sorted(set(_PROCESS_LOCAL) - fields)
    assert not stale, f"named process-local but no longer a field: {stale}"


def test_every_process_local_entry_gives_a_reason() -> None:
    """A one-word entry is a list someone filled in to make a test pass."""
    thin = sorted(
        name for name, reason in _PROCESS_LOCAL.items() if len(reason.split()) < 6
    )
    assert not thin, f"these entries do not say why the value cannot travel: {thin}"


def test_the_snapshot_names_only_real_fields() -> None:
    fields = {f.name for f in dataclasses.fields(RunScopedState)}
    unknown = sorted(_snapshot_keys() - fields)
    assert not unknown, f"to_snapshot() writes keys that are not fields: {unknown}"
