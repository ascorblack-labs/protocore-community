"""Guard test: ``protocore/`` must never import a package that sits above it.

Fails CI immediately if a reverse import appears anywhere in the
:mod:`protocore` package tree.

The rule is the namespace, not a list of names. Everything the core is
allowed to import from itself lives under the single ``protocore`` package;
a host distribution puts its adapters, service layer, and frontends in
sibling distributions whose import names all begin ``protocore_``. Naming
the shape instead of enumerating today's siblings means a package added
after this file was written is caught on the day it appears, which an
allowlist of known names never is.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

#: The one top-level name the core may import from.
_CORE_PACKAGE = "protocore"

#: Anything above the core shares the core's name with an underscore after it.
#: ``protocore`` itself is the core, and an unrelated third-party package that
#: merely starts with the same letters (``protocorenetwork``) is not a sibling
#: — the underscore is what makes the prefix a namespace rather than a
#: substring.
_UPWARD_PREFIX = f"{_CORE_PACKAGE}_"


def _is_upward_package(top_level: str) -> bool:
    """Whether ``top_level`` names a package that sits above the core."""
    return top_level.startswith(_UPWARD_PREFIX)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CORE_ROOT = _REPO_ROOT / "protocore"

#: The smallest package this test will believe in. The check reports a clean
#: tree by finding no violations, so a scope that collapsed to nothing reports
#: exactly the same thing as a package with no reverse imports — and this is the
#: boundary the project treats as inviolable, which makes a silent pass over
#: zero files the most expensive one available. The package has ninety modules;
#: the floor sits far enough below that ordinary deletion does not trip it, and
#: far enough above zero that a collapse cannot pass for compliance.
_MIN_CORE_MODULES = 60


def _tracked_python_files() -> list[Path] | None:
    """Every tracked ``protocore/*.py``, or ``None`` where there is no index.

    The pathspec is ``protocore/*.py`` — a git pathspec ``*`` already matches
    across ``/``, so the single star is the recursive form and returns all of
    them. ``protocore/**/*.py`` is the NARROWER pattern despite looking like the
    more thorough one: it requires an intervening directory and so drops every
    top-level module in the package.

    A non-zero exit means there is no repository to ask — a source archive, or
    an image stage that copies the tree without ``.git``. That is a different
    answer from "this branch tracks nothing", which is a zero exit with no
    output, and the caller keeps them apart.
    """
    try:
        listing = subprocess.run(
            [
                "git",
                "-C",
                str(_REPO_ROOT),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--deduplicate",
                "--",
                "protocore/*.py",
            ],
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if listing.returncode != 0:
        return None
    return [_REPO_ROOT / name for name in listing.stdout.split("\0") if name]


def _walked_python_files(root: Path) -> list[Path]:
    """Every Python file under ``root``, skipping bytecode and dot-directories.

    The fallback for a tree with no index. The skip is asked of the path
    RELATIVE TO ``root``: a dot in the absolute prefix is a fact about where the
    checkout is parked — a tool's worktrees often live under a dotted home
    directory — and reading it would drop the entire package instead of the ``.git`` and
    ``.worktrees`` entries it is meant to drop.
    """
    return [
        path
        for path in sorted(root.rglob("*.py"))
        if not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(root).parts)
    ]


def _collect_python_files(root: Path) -> list[Path]:
    """The package's modules, from the index where there is one.

    A walk scans whatever is on disk, which in a working checkout includes
    nested worktrees — this repository keeps them under ``.worktrees/``, some of
    them INSIDE ``protocore/``, each a complete copy of the package on another
    branch. Enumerating the index removes that dependency rather than relying on
    nobody creating one: another branch's modules are untracked here whatever
    the directory is called.

    A tracked path deleted in the working tree is still in the index, so what
    cannot be read is dropped rather than raising.
    """
    tracked = _tracked_python_files()
    files = [path for path in tracked if path.is_file()] if tracked is not None else _walked_python_files(root)
    if len(files) < _MIN_CORE_MODULES:
        pytest.fail(
            f"the sweep found {len(files)} module(s) under {root}, fewer than "
            f"the {_MIN_CORE_MODULES} this package must have. This check reports "
            "a clean tree by finding no violations, so a collapsed scope would "
            "otherwise be reported as a core that imports nothing upward — the "
            "one result nobody would think to question."
        )
    return files


def _extract_top_level_imports(source: str) -> list[str]:
    """Return top-level module names referenced by import statements."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module.split(".")[0])
    return names


def _find_violations() -> list[tuple[Path, str]]:
    violations: list[tuple[Path, str]] = []
    for py_file in _collect_python_files(_CORE_ROOT):
        source = py_file.read_text(encoding="utf-8")
        for top_level in _extract_top_level_imports(source):
            if _is_upward_package(top_level):
                violations.append((py_file, top_level))
    return violations


def test_the_sweep_is_the_index_and_not_the_directory() -> None:
    """What is judged is this branch's package, not what shares its directory.

    A checkout may hold a full copy of the source for every branch beside it —
    this repository keeps worktrees under ``.worktrees/``, including inside
    ``protocore/`` — and a walk reads all of them, judging another branch's
    code as this branch's. Stated as a test because the difference is invisible
    in a checkout that happens to hold no populated worktree today, which is
    exactly when someone would replace the enumeration with an ``rglob`` for
    being shorter.
    """
    tracked = _tracked_python_files()
    assert tracked is not None, "the suite runs in a checkout; git must answer"

    swept = _collect_python_files(_CORE_ROOT)
    assert swept, "the sweep is empty, so this guard proves nothing"
    assert len({path.resolve() for path in swept}) == len(swept), (
        "the same module was swept twice, which is what a walk into a nested checkout looks like"
    )
    on_index = {path.resolve() for path in tracked}
    for path in swept:
        assert path.resolve() in on_index, f"{path} is swept but not tracked"


def test_the_sweep_refuses_to_report_a_clean_tree_over_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collapsed scope fails here instead of passing as a clean boundary.

    ``if violations: fail`` cannot tell an empty scope from a compliant one, and
    that indistinguishability is not theoretical: the sibling guard in
    ``tests/unit/runtime/`` applied its dot-directory skip to the ABSOLUTE path
    and swept zero of ninety modules from any checkout parked under a dotted
    ancestor, reporting a pass every time.
    """
    monkeypatch.setattr(__name__ + "._tracked_python_files", lambda: [], raising=True)
    with pytest.raises(pytest.fail.Exception, match="fewer than"):
        _collect_python_files(_CORE_ROOT)


def test_the_walk_fallback_does_not_depend_on_where_the_checkout_is_parked(
    tmp_path: Path,
) -> None:
    """Without an index, the same tree still sweeps the same under a dotted path."""

    def plant(root: Path) -> Path:
        package = root / "protocore"
        (package / "sub").mkdir(parents=True)
        for index in range(5):
            (package / f"mod_{index}.py").write_text("x = 1\n")
        (package / ".worktrees" / "copy").mkdir(parents=True)
        (package / ".worktrees" / "copy" / "mod_0.py").write_text("x = 1\n")
        (package / "sub" / "__pycache__").mkdir()
        (package / "sub" / "__pycache__" / "mod_0.py").write_text("x = 1\n")
        return package

    plain = plant(tmp_path / "plain")
    dotted = plant(tmp_path / ".dotted" / "nested")

    assert [path.relative_to(plain) for path in _walked_python_files(plain)] == [
        path.relative_to(dotted) for path in _walked_python_files(dotted)
    ]
    # And the skip still drops the nested copy and the bytecode cache.
    assert len(_walked_python_files(plain)) == 5


def test_protocore_does_not_import_upward_packages() -> None:
    """``protocore/`` must not import any package that sits above it."""
    violations = _find_violations()
    if violations:
        lines = [f"  {path.relative_to(_CORE_ROOT.parent)}: imports {pkg!r}" for path, pkg in violations]
        pytest.fail(
            "Reverse imports from protocore/ into upward packages detected:\n"
            + "\n".join(lines)
            + "\n\nThe core package must remain independent of all upward layers."
        )
