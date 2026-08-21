"""Unit tests for the pure-core evidence-ref canonicalizer.

Covers each canonicalization rule
in :func:`protocore.contracts.references.normalize_ref`, the NEVER-MUTATE
invariant (the projection does not alter the model's submitted string in place),
idempotence (a canonical form is a fixed point), and the SYMMETRY invariant
(record-side and compare-side produce the same key for the same input + flags).

HTML-entity decoding was removed from the canonicalizer. Refs compare LITERALLY apart from RFC-3986 percent/path
canonicalization. The adversarial sweep below proves the property the three
prior entity-patch rounds kept failing to guarantee: NO entity ref of ANY form
(decimal/hex/named, with or without the trailing semicolon, or PUA-encoded) can
truncate the membership key, false-match a real ``/``-separated path, or
cross-collide with a different entity in the same slot — because no entity is
ever decoded or specially handled at all.
"""
from __future__ import annotations

import collections

import pytest

from protocore.contracts.references import SUBLOCATOR_RE, normalize_ref

# ---------------------------------------------------------------------------
# Each canonical rule
# ---------------------------------------------------------------------------


def test_strip_whitespace() -> None:
    assert normalize_ref("  /docs/x.md  ") == "/docs/x.md"


@pytest.mark.parametrize(
    "raw",
    ['"/docs/x.md"', "'/docs/x.md'", "`/docs/x.md`"],
)
def test_strip_enclosing_quotes(raw: str) -> None:
    assert normalize_ref(raw) == "/docs/x.md"


def test_inner_quotes_preserved() -> None:
    # Only ENCLOSING quotes are stripped; an interior quote stays.
    assert normalize_ref("/docs/o'brien.md") == "/docs/o'brien.md"


def test_percent_decode_unreserved() -> None:
    # %2D is '-', an unreserved char, so it decodes.
    assert normalize_ref("/docs/a%2Db.md") == "/docs/a-b.md"


def test_percent_encoded_separator_preserved() -> None:
    # %2F is an encoded '/'; decoding it would forge structure, so it must NOT
    # become a path separator. It is left encoded (kept structural-safe).
    out = normalize_ref("/docs/a%2Fb.md")
    assert out.count("/") == 2  # the two literal separators only
    assert "%2F" in out or "%2f" in out


# ---------------------------------------------------------------------------
# No entity decoding — refs compare LITERALLY. An entity
# form of a structural delimiter (or of any char) is left verbatim, so a
# fabricated ref using entity-encoded delimiters can NEVER normalize onto a
# DISTINCT real path, nor be truncated by a structural pass, nor cross-collide
# with a different entity in the same slot. This is the adversarial set that
# beat three prior entity-handling rounds.
# ---------------------------------------------------------------------------

# Every entity FORM that previously had to be specially guarded: decimal,
# decimal-without-semicolon, lowercase-hex, hex-without-semicolon, uppercase-hex,
# named, and a PUA-sentinel-bracketed payload (the forge that beat round 3).
_ADVERSARIAL_ENTITIES: list[str] = [
    # slash variants
    "&#47;", "&#47", "&#x2f;", "&#x2f", "&#X2F;", "&sol;",
    # hash / number sign (the literal char SUBLOCATOR_RE truncates on)
    "&#35;", "&#35", "&num;",
    # colon (the other SUBLOCATOR_RE delimiter)
    "&#58;", "&#58", "&colon;",
    # dot (would feed remove_dot_segments / extension-strip)
    "&#46;", "&#46", "&period;",
    # backslash
    "&#92;", "&bsol;",
    # question mark
    "&#63;", "&quest;",
    # percent (would forge a %XX escape downstream)
    "&#37;", "&percnt;",
    # alphanumeric forms (must not forge a path-segment char)
    "&#112;", "&#x70;", "&#48;", "&#65;",
    # PUA-sentinel-bracketed payload (the prior masking sentinels)
    "0", "&#57344;0&#57345;",
]


@pytest.mark.parametrize("entity", _ADVERSARIAL_ENTITIES)
def test_entity_never_truncates_key(entity: str) -> None:
    # Property: an entity embedded mid-path must NOT cause the normalized key to
    # lose its tail. The full path body after the entity must survive, and the
    # key must not collapse to a dangling ``&`` / entity prefix.
    raw = f"/proc/payments{entity}pay_007"
    out = normalize_ref(raw)
    assert "pay_007" in out, f"{entity!r} truncated the key: {out!r}"
    assert not out.endswith("&"), f"{entity!r} left a dangling '&': {out!r}"
    assert not out.rstrip("&") != out, f"{entity!r} '&'-suffix truncation: {out!r}"


@pytest.mark.parametrize("entity", _ADVERSARIAL_ENTITIES)
def test_entity_does_not_false_match_real_path(entity: str) -> None:
    # Property: a fabricated ref using an entity must NEVER normalize onto the
    # real ``/``-separated path. Compared literally, it stays distinct.
    fabricated = normalize_ref(f"/proc/payments{entity}pay_007")
    real = normalize_ref("/proc/payments/pay_007")
    assert fabricated != real, f"{entity!r} false-matched the real path: {fabricated!r}"


def test_entities_in_same_slot_do_not_cross_collide() -> None:
    # Property (the cross-collision the masking BLOCKER created): DIFFERENT
    # entities in the SAME slot must each yield a DISTINCT key — they are kept
    # literal, so no two distinct entity texts collapse together.
    keys = [normalize_ref(f"/proc/payments{e}pay_007") for e in _ADVERSARIAL_ENTITIES]
    dups = [k for k, c in collections.Counter(keys).items() if c > 1]
    assert dups == [], f"entities cross-collided: {dups!r}"


@pytest.mark.parametrize("entity", _ADVERSARIAL_ENTITIES)
@pytest.mark.parametrize("strip_extension", [False, True])
@pytest.mark.parametrize("casefold", [False, True])
def test_entity_idempotent_all_flags(
    entity: str, strip_extension: bool, casefold: bool
) -> None:
    # Property: the literal projection is a fixed point under every flag combo
    # (no decode pass means nothing changes on a second application).
    raw = f"/proc/x{entity}y/z-2025-06-22.md"
    once = normalize_ref(raw, strip_extension=strip_extension, casefold=casefold)
    twice = normalize_ref(once, strip_extension=strip_extension, casefold=casefold)
    assert once == twice


def test_entity_in_body_does_not_block_extensionless_grounding() -> None:
    # Property: a literal entity in the BODY does not prevent a clean
    # extensionless match elsewhere (the entity text is identical on both sides).
    record = normalize_ref("/proc/a&#47;b/report.md", strip_extension=True)
    cited = normalize_ref("/proc/a&#47;b/report", strip_extension=True)
    assert record == cited


def test_identical_entity_strings_still_match_symmetrically() -> None:
    # Symmetric: neither side is decoded, so two refs that LITERALLY contain the
    # same entity text still match each other (a real recorded ref that itself
    # carried that entity).
    assert normalize_ref("/proc/a&#47;b") == normalize_ref("/proc/a&#47;b")
    assert normalize_ref("/proc/a&amp;b") == normalize_ref("/proc/a&amp;b")


def test_amp_entity_is_not_decoded() -> None:
    # The benign ``&amp;`` is NOT decoded — refs compare literally, so a
    # literal-``&`` ref and an ``&amp;`` ref are DISTINCT keys (a malformed ref
    # is not silently grounded onto a different literal one).
    assert normalize_ref("/proc/a&amp;b") != normalize_ref("/proc/a&b")


def test_remove_dot_segments() -> None:
    assert normalize_ref("/docs/./a/../b.md") == "/docs/b.md"
    assert normalize_ref("/a/b/../../c.md") == "/c.md"


def test_collapse_double_slash() -> None:
    assert normalize_ref("/docs//a///b.md") == "/docs/a/b.md"


def test_strip_trailing_slash_not_root() -> None:
    assert normalize_ref("/docs/sub/") == "/docs/sub"
    assert normalize_ref("/") == "/"  # root slash preserved


def test_strip_sublocator_fragment() -> None:
    assert normalize_ref("/docs/x.md#row=12") == "/docs/x.md"
    assert normalize_ref("/docs/x.md#L12-30") == "/docs/x.md"
    assert normalize_ref("/docs/x.md:12") == "/docs/x.md"
    assert normalize_ref("/docs/x.md:12:30") == "/docs/x.md"


def test_sublocator_regex_does_not_eat_plain_path() -> None:
    # A path with no fragment / line suffix is untouched by the regex.
    assert SUBLOCATOR_RE.sub("", "/docs/x.md") == "/docs/x.md"


def test_sublocator_regex_ignores_entity_hash() -> None:
    # The '#' of a '&#NNN;' numeric charref is NOT a fragment delimiter, so it
    # must not truncate the key (the literal entity stays whole).
    assert SUBLOCATOR_RE.sub("", "/proc/payments&#47;pay_007") == (
        "/proc/payments&#47;pay_007"
    )
    # But a GENUINE trailing fragment after such literal text is still stripped.
    assert SUBLOCATOR_RE.sub("", "/proc/a&#47;b#row=5") == "/proc/a&#47;b"


# ---------------------------------------------------------------------------
# strip_extension (extensionless case) — flag-gated, UNIVERSAL framing
# ---------------------------------------------------------------------------


def test_strip_extension_off_by_default() -> None:
    assert normalize_ref("/docs/2025-06-22-reporting-nuts-bolts.md") == (
        "/docs/2025-06-22-reporting-nuts-bolts.md"
    )


def test_strip_extension_on() -> None:
    # The extensionless case: observed with .md, cited without.
    observed = normalize_ref(
        "/docs/2025-06-22-reporting-nuts-bolts.md", strip_extension=True
    )
    cited = normalize_ref(
        "/docs/2025-06-22-reporting-nuts-bolts", strip_extension=True
    )
    assert observed == cited == "/docs/2025-06-22-reporting-nuts-bolts"


def test_strip_extension_only_last_segment() -> None:
    assert normalize_ref("/a.b/c.md", strip_extension=True) == "/a.b/c"


def test_strip_extension_single_suffix() -> None:
    # "drop the suffix after the last dot" — single extension only.
    assert normalize_ref("/a/b.tar.gz", strip_extension=True) == "/a/b.tar"


def test_strip_extension_dotfile_preserved() -> None:
    assert normalize_ref("/a/.env", strip_extension=True) == "/a/.env"


def test_strip_extension_no_dot_unchanged() -> None:
    assert normalize_ref("/a/b", strip_extension=True) == "/a/b"


def test_strip_extension_directory_unchanged() -> None:
    # Trailing-slash directory has no filename to strip; trailing slash is
    # removed by rule 5 first, so this is a no-ext segment.
    assert normalize_ref("/docs/sub/", strip_extension=True) == "/docs/sub"


def test_distinct_ids_stay_distinct_with_extension_strip() -> None:
    # Two distinct resource ids must not collapse under extension-strip.
    assert normalize_ref(
        "/p/pay_007.json", strip_extension=True
    ) != normalize_ref("/p/pay_008.json", strip_extension=True)


# ---------------------------------------------------------------------------
# casefold — flag-gated, default OFF
# ---------------------------------------------------------------------------


def test_casefold_off_by_default() -> None:
    assert normalize_ref("/Docs/Foo.MD") == "/Docs/Foo.MD"


def test_casefold_on() -> None:
    assert normalize_ref("/Docs/Foo.MD", casefold=True) == "/docs/foo.md"


# ---------------------------------------------------------------------------
# Totality / edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", None, 123, [], {}])
def test_empty_or_nonstring_yields_empty(bad: object) -> None:
    assert normalize_ref(bad) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# NEVER-MUTATE invariant
# ---------------------------------------------------------------------------


def test_never_mutates_input() -> None:
    # str is immutable in Python, but assert identity semantics: the function
    # returns a NEW canonical string and the original binding is unchanged.
    original = "  '/docs/./X.md#row=1'  "
    snapshot = str(original)
    projected = normalize_ref(original, strip_extension=True, casefold=True)
    assert original == snapshot  # original binding untouched
    assert projected == "/docs/x"  # canonical projection differs
    assert projected is not original


# ---------------------------------------------------------------------------
# Idempotence — a canonical form is a fixed point under every flag combo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "/docs/x.md",
        "  '/docs/./a/../b.md#L1'  ",
        "/docs//A//B.MD/",
        "/docs/a&amp;b.md:12:30",
        "/",
        "relative/path.md",
    ],
)
@pytest.mark.parametrize("strip_extension", [False, True])
@pytest.mark.parametrize("casefold", [False, True])
def test_idempotent(raw: str, strip_extension: bool, casefold: bool) -> None:
    once = normalize_ref(raw, strip_extension=strip_extension, casefold=casefold)
    twice = normalize_ref(once, strip_extension=strip_extension, casefold=casefold)
    assert once == twice


# ---------------------------------------------------------------------------
# SYMMETRY invariant — record-side and compare-side produce the same key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("record_form", "cited_form"),
    [
        # Same logical resource expressed two ways; with normalized (no
        # strip_extension) they collapse when only cosmetic differences exist.
        ("/docs/report.md", "  /docs/report.md  "),
        ("/docs/report.md", "'/docs/report.md'"),
        # An identical literal entity on BOTH sides still matches (no decode).
        ("/docs/a&amp;b.md", "/docs/a&amp;b.md"),
        ("/docs/sub/file.md", "/docs/./sub/file.md"),
        ("/docs//sub/file.md", "/docs/sub/file.md"),
        ("/docs/file.md", "/docs/file.md#row=3"),
    ],
)
def test_symmetry_record_equals_cited_normalized(
    record_form: str, cited_form: str
) -> None:
    # The record side stores raw, the compare side projects; both project to the
    # same key, so a previously-recorded ref matches its normalized citation.
    assert normalize_ref(record_form) == normalize_ref(cited_form)


def test_symmetry_requires_extensionless() -> None:
    # The extensionless case (cited without .md) does NOT collapse under plain
    # normalized, only under the extensionless tier — proving the tier is
    # load-bearing for exactly that case and nothing broader.
    record = "/docs/2025-06-22-reporting-nuts-bolts.md"
    cited = "/docs/2025-06-22-reporting-nuts-bolts"
    assert normalize_ref(record) != normalize_ref(cited)
    assert normalize_ref(record, strip_extension=True) == normalize_ref(
        cited, strip_extension=True
    )


# ---------------------------------------------------------------------------
# Behaviorally-neutral claim: a previously exact-matching ref still matches
# after normalization (required verification).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref",
    [
        "/docs/exact.md",
        "/AGENTS.md",
        "/proc/catalog/items.json",
        "/",
    ],
)
def test_exact_match_preserved_post_normalization(ref: str) -> None:
    # If cited == observed (raw exact) today, the normalized projections are
    # also equal, so enabling exact+normalized never DROPS a match that exact
    # used to make.
    assert normalize_ref(ref) == normalize_ref(ref)
    # And the projection of an already-canonical ref is the ref itself
    # (canonical inputs are fixed points), so exact == normalized for them.
    assert normalize_ref(ref) == ref
