"""What the evidence contract keeps, and why the list is written down.

Most of this module's declarations have no caller inside this package. That
reads, from a symbol survey, exactly like dead code, and the survey is right
about the count and wrong about the conclusion: every one of them is the type
of a field or an argument of a model the loop does build and does hand out.
Deleting one would not remove an unused name — it would remove the type of a
live field.

So the remainder is accepted as what it is, and fixed here. The list only ever
shrinks: a name added to it is a declaration that arrived without a caller,
which is the thing this file exists to refuse.
"""

from __future__ import annotations

import ast
from pathlib import Path

from protocore.contracts import evidence

#: Declarations reachable only as the type of a field or argument of a model
#: this package builds. Each is named by the class that holds it.
REACHED_ONLY_AS_A_TYPE = frozenset(
    {
        "ArtifactDeclaration",
        "CitationSpan",
        "DeliveryMode",
        "EvidenceLedgerReference",
        "InheritedEvidencePrefix",
        "InvalidVerificationTransitionError",
        "ReleaseDecision",
        "RequirementsManifest",
        "VerificationCheckResult",
        "VerificationCheckStatus",
        "VerificationFinding",
        "VerificationReport",
        "VerificationResourceUse",
        "VerificationSeverity",
        "assert_verification_transition",
    }
)

#: The models and helpers this package names directly.
CALLED_BY_NAME = frozenset(
    {
        "CandidateBundle",
        "CandidateReleasedProjection",
        "EvidenceLedger",
        "EvidenceRecord",
        "RunTreeOrigin",
        "ToolEvidenceContext",
        "VerificationDelivery",
        "VerificationLifecycle",
        "VerificationState",
        "canonical_frozen_object",
        "freeze_json",
        "freeze_json_object",
        "require_nonempty_identifier",
        "thaw_json",
        "thaw_json_object",
    }
)


def test_the_module_declares_exactly_these_two_sets() -> None:
    assert set(evidence.__all__) == REACHED_ONLY_AS_A_TYPE | CALLED_BY_NAME


def test_no_name_is_in_both_sets() -> None:
    assert not (REACHED_ONLY_AS_A_TYPE & CALLED_BY_NAME)


def test_every_retained_name_is_really_reachable_as_a_type() -> None:
    """Each retained name is written somewhere in the module's own source.

    A name kept for its type has to be *used* as one. Were it to lose its last
    mention, it would be an ordinary unused declaration, and the argument for
    keeping it would be gone with it.
    """

    source = Path(evidence.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    mentions: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            mentions[node.id] = mentions.get(node.id, 0) + 1
        elif isinstance(node, ast.Attribute):
            mentions[node.attr] = mentions.get(node.attr, 0) + 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for token in node.value.replace("[", " ").replace("]", " ").split():
                mentions[token] = mentions.get(token, 0) + 1
    for name in REACHED_ONLY_AS_A_TYPE:
        assert mentions.get(name, 0) >= 1, f"{name} is no longer used as a type"


def test_every_retained_name_is_importable() -> None:
    for name in REACHED_ONLY_AS_A_TYPE | CALLED_BY_NAME:
        assert getattr(evidence, name, None) is not None
