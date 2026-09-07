"""Guard test: a name the docs present as a runtime constant must exist.

A field removed from the constants model leaves the prose behind. The doc
still names the knob, an operator still goes looking for it, and nothing in
the tree says the sentence stopped being true — the published documentation is
copied out verbatim, so the claim travels further than the field ever did.

The rule enforced here is narrow on purpose: only the two places where a
backticked snake_case token *means* "a field of the constants model" are
scanned — the runtime-constants page in full, and the tunable column of the
technology table in the architecture page. Everywhere else a token of that
shape is far more often a method, an event or an attribute, and a guard that
cannot tell them apart is a guard that gets an allow-list instead of a fix.

The identifiers that legitimately appear in those two scopes without being
fields are listed in :data:`NOT_A_CONSTANT`, each with the reason it is there,
so the list cannot quietly grow into a way of silencing the check.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_PATH = _REPO_ROOT / "protocore" / "contracts" / "runtime_constants.py"

#: Pages scanned in full: everything they backtick in this shape is a claim
#: about the constants model.
_WHOLE_PAGES = ("docs/runtime-constants.md", "docs/ru/runtime-constants.md")

#: Pages where only the tunable column of the technology table is scanned.
_TABLE_PAGES = ("docs/architecture.md", "docs/ru/architecture.md")

#: A token of the constant shape: lowercase, at least one underscore.
_TOKEN = re.compile(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`")

#: Identifiers of the constant shape that appear in the scanned scopes and are
#: deliberately not fields. Each maps to why it is written there.
NOT_A_CONSTANT: dict[str, str] = {
    "tenant_id": "the argument of the provider Protocol, not a tunable",
    "register_policy": "the safety-policy registration function",
    "allowed_values": "a field of ConstantSpec — the descriptor of a knob, not a knob",
    "zero_means_unlimited": "a field of ConstantSpec — the descriptor of a knob, not a knob",
    "not_a_lever": "a field of ConstantSpec — the descriptor of a knob, not a knob",
    "group_from_model": "the function that reflects a declaring model into a group",
}


def _declared_fields() -> frozenset[str]:
    """Every field name declared on the constants model."""

    tree = ast.parse(_MODEL_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "LoopConstants":
            return frozenset(
                statement.target.id
                for statement in node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            )
    raise AssertionError("the constants model was not found in its own module")


def _table_cells(text: str) -> list[tuple[int, str]]:
    """The tunable column of the technology table, with line numbers."""

    cells: list[tuple[int, str]] = []
    inside = False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("| Technology |"):
            inside = True
            continue
        if inside:
            if not line.startswith("|"):
                inside = False
                continue
            columns = line.split("|")
            if len(columns) > 3:
                cells.append((number, columns[3]))
    return cells


def _claims() -> list[tuple[str, int, str]]:
    """Every token the scanned scopes present as a constant name."""

    found: list[tuple[str, int, str]] = []
    for relative in _WHOLE_PAGES:
        text = (_REPO_ROOT / relative).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            found.extend(
                (relative, number, match.group(1)) for match in _TOKEN.finditer(line)
            )
    for relative in _TABLE_PAGES:
        text = (_REPO_ROOT / relative).read_text(encoding="utf-8")
        for number, cell in _table_cells(text):
            found.extend(
                (relative, number, match.group(1)) for match in _TOKEN.finditer(cell)
            )
    return found


def test_the_scan_still_finds_the_documentation() -> None:
    """A scan that found nothing would pass every other test in this file."""

    claims = _claims()
    assert len(claims) > 40, f"the doc scan collapsed to {len(claims)} tokens"
    assert any(relative.startswith("docs/ru/") for relative, _, _ in claims)
    assert any(relative == "docs/architecture.md" for relative, _, _ in claims)


def test_every_documented_constant_is_declared() -> None:
    """No page names a tunable the model does not carry."""

    fields = _declared_fields()
    stale = [
        f"{relative}:{number} names `{token}`"
        for relative, number, token in _claims()
        if token not in fields and token not in NOT_A_CONSTANT
    ]
    assert not stale, "documentation names constants the model does not declare:\n" + (
        "\n".join(stale)
    )


@pytest.mark.parametrize("name", sorted(NOT_A_CONSTANT))
def test_the_exception_list_does_not_rot(name: str) -> None:
    """An exception that became a field, or left the docs, is a stale entry."""

    assert name not in _declared_fields(), (
        f"`{name}` is a field of the model now — drop it from NOT_A_CONSTANT"
    )
    assert any(token == name for _, _, token in _claims()), (
        f"`{name}` is no longer written in the scanned pages — drop the entry"
    )
