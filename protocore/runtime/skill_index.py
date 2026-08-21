"""Skill catalog renderer.

The skill catalog is a compact ``<system-reminder>`` block listing every
enabled skill (name + one-line description) for the account. It is built
ONCE per run and placed in the static system-prompt prefix so it stays
byte-stable across turns (preserving the prompt cache).

Pure-core: no PG, no HTTP, no token-budget I/O beyond a caller-supplied
counter. Consumes :class:`~protocore.contracts.skills.SkillIndexEntry`
rows (the read path of an :class:`~protocore.contracts.skills.ISkillStore`
adapter).
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable

from protocore.contracts.skills import SkillIndexEntry

# Format-level invariant of the catalog block.
# Qwen-Code / SkillsBench prior treats "- name: desc" bullets as a
# directory index and then Glob/Read /workspace/.skills/. Emit the
# actual call shape instead.
SYSTEM_REMINDER_HEADER = (
    "Skills are tools, not files. There is no /workspace/.skills/ and "
    "Read/Glob/Bash cannot open a skill. To use one, call exactly "
    'Skill(skill="<name>").'
)

# Token counter signature: ``async def count(text: str) -> int``
TokenCounter = Callable[[str], Awaitable[int]]


def _wrap(lines: list[str]) -> str:
    body = "\n".join(lines)
    return (
        "<system-reminder>\n"
        f"{SYSTEM_REMINDER_HEADER}\n\n"
        f"{body}\n"
        "</system-reminder>"
    )


async def render_skills_catalog(
    entries: Iterable[SkillIndexEntry],
    *,
    token_counter: TokenCounter,
    budget_tokens: int,
) -> str:
    """Render the skill catalog block for ``entries``.

    Emits one ``Skill(skill="{name}") — {description}`` line per skill,
    alphabetical by name. Returns an empty string when there are no entries.

    Budget guard: when the full block exceeds ``budget_tokens`` the catalog
    deterministically degrades to call shapes only
    (``Skill(skill="{name}")``, still alphabetical). The degrade decision
    is binary and content-stable for a given entry set + budget, so callers
    can compute it once per run without re-rendering mid-run.
    ``budget_tokens <= 0`` is treated as unbounded.
    """
    ordered = sorted(entries, key=lambda e: e.name)
    if not ordered:
        return ""

    full_lines = [
        (
            f'Skill(skill="{entry.name}") — {entry.description.strip()}'
            if entry.description.strip()
            else f'Skill(skill="{entry.name}")'
        )
        for entry in ordered
    ]
    full_block = _wrap(full_lines)

    if budget_tokens <= 0:
        return full_block

    if await token_counter(full_block) <= budget_tokens:
        return full_block

    # Over budget — deterministic call-shape-only degrade.
    names_only = [f'Skill(skill="{entry.name}")' for entry in ordered]
    return _wrap(names_only)


def derive_skill_index_budget_tokens(
    *,
    model_context_window: int,
    skill_index_budget_ratio: float,
) -> int:
    """RuntimeConstants-derived budget for the skill catalog block.

    Defaults to 1% of ``model_context_window``.
    Returns an integer token count, floored ≥ 0.
    """

    return max(0, int(model_context_window * skill_index_budget_ratio))


__all__ = [
    "SYSTEM_REMINDER_HEADER",
    "TokenCounter",
    "derive_skill_index_budget_tokens",
    "render_skills_catalog",
]
