"""Find, for every runtime constant, the places in the core that read it.

A constant with no reader in this package is not a core constant. It is a knob
belonging to the layer above, sitting in the core's model because that is where
the model happened to be written, and it costs on every axis at once: it is
carried in every snapshot the core builds, it has to be defaulted and validated
here, and an operator editing it sees no effect because nothing in the loop ever
looks at it. The scan below is what makes that condition observable, so a guard
test can refuse it.

Reading is two patterns, not one
--------------------------------

**Attribute access** — ``rc.<field>`` — is the obvious one, and on its own it
under-counts badly. **A string literal equal to a field name** is the second,
and it covers two unrelated mechanisms:

* ``getattr(rc, "<field>", <default>)``, which the core uses wherever it wants
  a value to survive a constants object that predates the field;
* the ``*_CONTEXT_KEY`` names of the memory tools, whose values arrive in
  ``ToolContext.metadata`` — filled in by the layer above from its own constants
  — and never through an attribute of the constants object at all.

A scan that knows only attribute access calls both of those groups unread, and
a guard built on it would invite the deletion of live configuration channels.

Attribute access is credited only when the owner is the constants object
-----------------------------------------------------------------------

``self.run_mode`` is not a read of ``LoopConstants.run_mode`` merely because
the attribute names match — the engine's own configuration object declares a
field of the same name on a completely different axis of values. So an
attribute is counted only when the expression it is read from resolves to the
constants object: a bare name, or a trailing attribute, of ``rc`` / ``_rc`` /
``*_rc`` / ``constants`` / ``runtime_constants`` / ``defaults``.

Fields whose *only* evidence is an attribute access with some other owner are
not counted as read. They are reported separately as ambiguous, because the
answer is a judgement about two same-named attributes and belongs to a person,
recorded once, rather than to a regular expression. Getting this wrong is
expensive in one specific direction: a field falsely credited with a reader can
never be removed.

Declaring a constant is not reading it
--------------------------------------

The module that defines the model, and the modules that describe the tunable
surface — a descriptor, a group, a cross-field relationship — name fields in
order to *say something about them*. So does a conformance suite. None of that
is a read: if every such mention counted, a field would acquire a reader by the
act of being declared, and the guard would pass for a knob nothing consumes.
Those surfaces are excluded by path.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

#: The core package, relative to the repository root.
CORE_PACKAGE = "protocore"

#: Where the constants model is defined.
DEFINITION_MODULE = "protocore/contracts/runtime_constants.py"

#: The class whose fields are the tunable surface.
CONSTANTS_CLASS = "LoopConstants"

#: Paths whose mentions of a field name describe it rather than consume it. A
#: prefix ending in ``/`` covers a whole tree; anything else is one file.
DECLARATION_PATHS: tuple[str, ...] = (
    DEFINITION_MODULE,
    # Descriptors, groups and cross-field relationships: a statement *about* a
    # knob, made by naming it.
    "protocore/contracts/config.py",
    # Suites a host runs against its own adapters. They name a field to assert
    # the contract that surrounds it, and consume no value of their own.
    "protocore/conformance/",
)

#: Fixtures, not core code.
EXCLUDED_TREES: tuple[str, ...] = (
    "protocore/tests_support/",
    "__pycache__",
)

#: An attribute is a read of a constant only when it hangs off one of these.
_CONSTANTS_OWNER = re.compile(r"^(_?rc|.*_rc|constants|runtime_constants|defaults)$")

#: Identifier-shaped tokens in a prompt template.
_TEMPLATE_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

Pattern = Literal["attribute", "literal", "template"]


@dataclass(frozen=True, slots=True)
class Reading:
    """One place a field name was found, and the pattern that found it."""

    field: str
    path: str
    lineno: int
    pattern: Pattern

    @property
    def location(self) -> str:
        return f"{self.path}:{self.lineno}"


@dataclass(frozen=True, slots=True)
class Scan:
    """The whole picture: which fields exist, and what reads each of them."""

    #: Every field of the constants class, mapped to its line in the model.
    fields: dict[str, int]
    #: Field -> the readings that prove it is read by the core.
    readers: dict[str, tuple[Reading, ...]]
    #: Field -> attribute accesses whose owner is not the constants object,
    #: for fields that have no accepted reading at all.
    ambiguous: dict[str, tuple[Reading, ...]]

    @property
    def read(self) -> frozenset[str]:
        return frozenset(self.readers)

    @property
    def unread(self) -> frozenset[str]:
        return frozenset(self.fields) - self.read

    def readers_of(self, field: str) -> tuple[Reading, ...]:
        return self.readers.get(field, ())

    def by_pattern(self, pattern: Pattern) -> dict[str, tuple[Reading, ...]]:
        """Only the readings found by ``pattern``, fields with none dropped."""
        found = {
            field: tuple(r for r in readings if r.pattern == pattern)
            for field, readings in self.readers.items()
        }
        return {field: readings for field, readings in found.items() if readings}


def constant_fields(repo_root: Path) -> dict[str, int]:
    """Every annotated field of the constants class, mapped to its line."""
    definition = repo_root / DEFINITION_MODULE
    module = ast.parse(definition.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == CONSTANTS_CLASS:
            return {
                statement.target.id: statement.lineno
                for statement in node.body
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
            }
    raise AssertionError(f"{DEFINITION_MODULE} declares no class {CONSTANTS_CLASS}")


def _owner_name(value: ast.expr) -> str:
    """The trailing name of the expression an attribute is read from."""
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    if isinstance(value, ast.Call):
        return _owner_name(value.func)
    return ""


def _is_constants_owner(value: ast.expr) -> bool:
    return bool(_CONSTANTS_OWNER.match(_owner_name(value)))


def _scanned_files(repo_root: Path) -> Iterator[tuple[Path, str]]:
    """Every core file the scan reads, with its repository-relative path."""
    package = repo_root / CORE_PACKAGE
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".j2"}:
            continue
        relative = path.relative_to(repo_root).as_posix()
        if any(tree in relative for tree in EXCLUDED_TREES):
            continue
        if any(
            relative == declaration or relative.startswith(declaration)
            for declaration in DECLARATION_PATHS
        ):
            continue
        yield path, relative


def scan(repo_root: Path) -> Scan:
    """Scan the core package for readers of every runtime constant."""
    fields = constant_fields(repo_root)
    readers: dict[str, list[Reading]] = {}
    ambiguous: dict[str, list[Reading]] = {}

    for path, relative in _scanned_files(repo_root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix == ".j2":
            for lineno, line in enumerate(text.split("\n"), 1):
                for word in _TEMPLATE_WORD.findall(line):
                    if word in fields:
                        readers.setdefault(word, []).append(
                            Reading(word, relative, lineno, "template")
                        )
            continue
        try:
            module = ast.parse(text)
        except SyntaxError:  # pragma: no cover - a core module that cannot parse
            continue
        for node in ast.walk(module):
            if isinstance(node, ast.Attribute) and node.attr in fields:
                bucket = readers if _is_constants_owner(node.value) else ambiguous
                pattern: Pattern = "attribute"
                bucket.setdefault(node.attr, []).append(
                    Reading(node.attr, relative, node.lineno, pattern)
                )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in fields:
                readers.setdefault(node.value, []).append(
                    Reading(node.value, relative, node.lineno, "literal")
                )

    return Scan(
        fields=fields,
        readers={field: tuple(found) for field, found in sorted(readers.items())},
        ambiguous={
            field: tuple(found)
            for field, found in sorted(ambiguous.items())
            if field not in readers
        },
    )


def getattr_readings(repo_root: Path) -> dict[str, tuple[Reading, ...]]:
    """Fields reached by ``getattr(rc, "<field>", ...)``, by field.

    Isolated from :func:`scan` so a test can state the branch of the literal
    pattern it depends on, instead of trusting a total that the other branch —
    the memory context keys — could keep green on its own.
    """
    fields = constant_fields(repo_root)
    found: dict[str, list[Reading]] = {}
    for path, relative in _scanned_files(repo_root):
        if path.suffix != ".py":
            continue
        try:
            module = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:  # pragma: no cover - a core module that cannot parse
            continue
        for node in ast.walk(module):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id != "getattr" or len(node.args) < 2:
                continue
            owner, name = node.args[0], node.args[1]
            if not isinstance(name, ast.Constant) or name.value not in fields:
                continue
            if not _is_constants_owner(owner):
                continue
            found.setdefault(name.value, []).append(
                Reading(name.value, relative, node.lineno, "literal")
            )
    return {field: tuple(readings) for field, readings in sorted(found.items())}
