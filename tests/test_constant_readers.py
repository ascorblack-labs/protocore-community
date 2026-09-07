"""Guard test: a constant in the core's model must have a core reader.

The rule this file enforces is one sentence long — *every field of the
constants model is read somewhere in* ``protocore/`` — and it is deliberately
enforced from the day the model still breaks it, because the alternative is to
enforce it on the day the cleanup finishes, by which point nothing stops the
next field from arriving.

So the rule ships with a ledger of the fields that break it today
(:data:`ALLOWED_UNREAD_PATH`). A field on that list is permitted to have no
reader; a field that is not on it is not. The ledger only shrinks — see the
header of the file itself — and the test refuses the two ways a ledger rots:
an entry that has since acquired a reader, and an entry naming a field that no
longer exists.

What counts as reading is not obvious, and getting it wrong is expensive in a
particular direction. Under-count, and this test demands the removal of a field
that a live configuration channel depends on. Over-count, and a field nothing
consumes keeps its place forever, because "it has a reader" is the sentence
that ends the argument. :mod:`tests.constant_reader_scan` carries the
definition and the reasoning; this file states the properties of it that the
guard depends on, so a later narrowing of the scan fails here rather than
quietly turning the ledger into a formality.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.constant_reader_scan import Scan, getattr_readings, scan

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: The fields allowed, for now, to have no reader in the core.
ALLOWED_UNREAD_PATH = _REPO_ROOT / "tests" / "host_owned_constants.txt"

#: The smallest model this test will believe in. A scan that found nothing
#: reports a perfectly clean tree — no unread field is outside the ledger when
#: there are no fields — so a collapse of the model, or of the walk that finds
#: it, would read exactly like compliance. The model holds several hundred
#: fields; the floor sits far below ordinary attrition and far above zero.
_MIN_FIELDS = 150

#: Likewise for the readers: the ledger's entries are proven stale by finding a
#: reader, so a scan that finds no readers at all cannot be told from a tree
#: where every remaining field is genuinely host-owned.
_MIN_READ_FIELDS = 100

#: Metadata keys the memory tools read out of ``ToolContext.metadata``. The
#: layer above fills them from its own constants, so the core never touches an
#: attribute of the constants object for any of them — the only evidence they
#: are alive is the string literal. They are named here, ahead of the ledger,
#: because a scan that lost the literal pattern would call them unread and this
#: test would then demand the deletion of the memory configuration channel.
#: Two of the seven address values that were never fields of the model.
_MEMORY_CONTEXT_KEYS = (
    "memory_default_scope",
    "memory_default_scope_key",
    "memory_allowed_scopes",
    "memory_scope_keys",
    "memory_enabled",
    "memory_write_similarity_threshold",
    "memory_max_records_per_scope",
)

#: How many fields the core reaches only through ``getattr(rc, "<name>", ...)``
#: — measured, then floored. The number matters because this is the other half
#: of the literal pattern: were the memory keys the only thing keeping it in the
#: scan, narrowing the pattern to those seven names would look harmless.
_MIN_GETATTR_FIELDS = 14

#: Attribute reads whose owner is not the constants object — a field of the
#: same name on some other object. Each is a judgement about two same-named
#: attributes, so each is settled by hand, once, and written down here. An
#: unlisted name fails the test rather than defaulting either way.
AMBIGUOUS_DECISIONS: dict[str, str] = {}


@pytest.fixture(scope="module")
def core_scan() -> Scan:
    return scan(_REPO_ROOT)


def _allowed_unread() -> list[str]:
    """The ledger, in file order, comments and blank lines dropped."""
    lines = ALLOWED_UNREAD_PATH.read_text(encoding="utf-8").split("\n")
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def test_the_scan_found_a_model_worth_checking(core_scan: Scan) -> None:
    """A collapsed scan must not read as a clean tree."""
    assert len(core_scan.fields) >= _MIN_FIELDS, (
        f"the constants model scanned to {len(core_scan.fields)} fields, below the floor of "
        f"{_MIN_FIELDS}; this guard cannot tell a clean tree from a broken scan"
    )
    assert len(core_scan.read) >= _MIN_READ_FIELDS, (
        f"only {len(core_scan.read)} fields were found to have a core reader, below the floor "
        f"of {_MIN_READ_FIELDS}; the scan, not the tree, is the likely cause"
    )


def test_every_constant_in_the_model_has_a_core_reader(core_scan: Scan) -> None:
    """The rule itself, less the fields the ledger still excuses."""
    unexcused = sorted(core_scan.unread - set(_allowed_unread()))
    assert not unexcused, (
        "these constants are declared in the core's model and read nowhere in the core:\n  "
        + "\n  ".join(unexcused)
        + "\n\nA constant the loop never reads belongs to the layer above: declare it there "
        "and remove the field. Nothing is added to "
        f"{ALLOWED_UNREAD_PATH.name} — that list only shrinks."
    )


def test_the_ledger_holds_no_field_that_has_since_gained_a_reader(core_scan: Scan) -> None:
    """An entry excusing a field that is now read is an entry to strike."""
    stale = sorted(set(_allowed_unread()) & core_scan.read)
    assert not stale, (
        f"{ALLOWED_UNREAD_PATH.name} still excuses constants the core now reads:\n  "
        + "\n  ".join(f"{field} — read at {core_scan.readers_of(field)[0].location}" for field in stale)
        + "\n\nRemove those lines."
    )


def test_the_ledger_holds_no_field_that_no_longer_exists(core_scan: Scan) -> None:
    """An entry naming a removed field is an entry to strike."""
    unknown = sorted(set(_allowed_unread()) - set(core_scan.fields))
    assert not unknown, (
        f"{ALLOWED_UNREAD_PATH.name} names constants the model no longer declares:\n  "
        + "\n  ".join(unknown)
        + "\n\nRemove those lines in the change that removed the fields."
    )


def test_the_ledger_names_each_field_once(core_scan: Scan) -> None:
    entries = _allowed_unread()
    duplicated = sorted({entry for entry in entries if entries.count(entry) > 1})
    assert not duplicated, f"{ALLOWED_UNREAD_PATH.name} repeats: {duplicated}"


def test_the_memory_context_keys_are_not_mistaken_for_dead_constants(core_scan: Scan) -> None:
    """The keys the memory tools read out of the call's metadata are alive.

    Their values never pass through an attribute of the constants object, so a
    scan reduced to attribute access would report every one of them unread and
    this guard would then ask for the deletion of a working channel.
    """
    known = [key for key in _MEMORY_CONTEXT_KEYS if key in core_scan.fields]
    assert known, "none of the memory context keys name a field of the model any more"
    unread = sorted(key for key in known if key not in core_scan.read)
    assert not unread, (
        "the memory tools read these out of the call metadata, and the scan missed them: "
        f"{unread}"
    )
    excused = sorted(set(known) & set(_allowed_unread()))
    assert not excused, (
        f"{ALLOWED_UNREAD_PATH.name} excuses live memory configuration: {excused}"
    )


def test_the_scan_still_credits_getattr_with_a_string_name() -> None:
    """The other branch of the literal pattern, stated as its own property."""
    by_field = getattr_readings(_REPO_ROOT)
    assert len(by_field) >= _MIN_GETATTR_FIELDS, (
        f"only {len(by_field)} constants are reached through getattr with a string name, below "
        f"the measured floor of {_MIN_GETATTR_FIELDS}; if the scan narrowed rather than the core, "
        "every one of these fields is about to be reported unread"
    )
    scanned = scan(_REPO_ROOT)
    missed = sorted(field for field in by_field if field not in scanned.read)
    assert not missed, f"reached by getattr, yet the scan calls them unread: {missed}"


def test_every_ambiguous_attribute_read_has_a_recorded_decision(core_scan: Scan) -> None:
    """A same-named attribute on another object is settled by hand, not by rule.

    A false "this field has a reader" is the expensive direction: it is the
    sentence that keeps a host-owned field in the core's model forever.
    """
    undecided = sorted(set(core_scan.ambiguous) - set(AMBIGUOUS_DECISIONS))
    assert not undecided, (
        "these constants are named by an attribute read whose owner is not the constants "
        "object, and nothing else in the core reads them:\n  "
        + "\n  ".join(
            f"{field} — {', '.join(r.location for r in core_scan.ambiguous[field])}"
            for field in undecided
        )
        + "\n\nDecide each by hand and record the decision in AMBIGUOUS_DECISIONS. Counting a "
        "name coincidence as a read would make the field unremovable."
    )
    settled = sorted(set(AMBIGUOUS_DECISIONS) - set(core_scan.ambiguous) - set(core_scan.fields))
    assert not settled, (
        f"AMBIGUOUS_DECISIONS records names the model no longer declares: {settled}"
    )


def test_a_field_decided_host_owned_is_carried_by_the_ledger(core_scan: Scan) -> None:
    """A decision of "host-owned" is a debt, and the ledger is where debt goes."""
    for field, decision in AMBIGUOUS_DECISIONS.items():
        if field not in core_scan.fields or not decision.startswith("host-owned"):
            continue
        assert field in set(_allowed_unread()), (
            f"{field} was decided host-owned but is absent from {ALLOWED_UNREAD_PATH.name}, so "
            "nothing tracks its removal"
        )
