"""Default shell safety policy — deny patterns for Bash tool.

Battle-tested deny patterns: ``sudo``/``doas``, ``rm -rf /``, fork bomb,
SUID, base64 decode, ``eval``/``exec``, interpreter ``-c`` injection.

Uses :func:`~protocore.runtime.chain_parser.parse_chain` to apply policy
to every sub-command in a chain (including ``$()`` substitutions).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from protocore.runtime.chain_parser import parse_chain


class ShellSafetyError(Exception):
    """Raised when a shell command violates the safety policy."""


class ShellPolicyVerdict(StrEnum):
    """Policy decision verdict."""

    allow = "allow"
    deny = "deny"


@dataclass(frozen=True, slots=True)
class ShellPolicyDecision:
    """Result of policy evaluation."""

    verdict: ShellPolicyVerdict
    reason: str = ""
    matched_pattern: str = ""


# Command-position prefix. Matches
# the start of a command position: start of line, after a chain operator
# (``;`` ``&&`` ``||`` ``|``), after ``$(`` opening a substitution, after a
# newline (inside heredocs / multi-line scripts) or backtick. Permits
# optional ``env`` and benign wrapper prefixes (``nohup``/``setsid``/
# ``time``) — those are already denied elsewhere when used in attack form,
# so the anchor stays consistent without re-introducing false positives.
_CMDPOS: Final[str] = (
    r"(?:^|[;&|\n`]|\$\()"
    r"\s*"
    r"(?:env\s+(?:\w+=\S*\s+)*)?"
    r"(?:(?:nohup|setsid|time)\s+)*"
    r"\s*"
)


# Per-segment deny patterns — compiled lazily.
# Restored from v1 ``safety/shell.py`` ``_DENY_PATTERNS``. v2 kept the
# original minimal set; this expansion covers battle-tested attack vectors
# that escaped the original trim:
#
#   - heredoc / process-substitution injection
#   - source/dot file injection
#   - shell wrapper injection (``bash -c``, ``$VAR -c``)
#   - su -c privilege escalation
#   - SUID chmod via symbolic notation (``chmod u+s``, ``chmod +s``)
#   - server hosting (long-running processes that should never spawn in a
#     sandboxed agent context — uvicorn, npm start, nohup, tmux, etc.)
#   - bare interpreter as pipeline target (``... | python``)
#   - destructive system control (shutdown / reboot)
#   - additional interpreter injection: node/deno/bun ``-e/--eval``, php ``-r``
#   - ANSI-C quoting (``$'...'``) used for obfuscation
#   - carriage-return injection (split-line attack)
#
# Relaxations following sandbox policy audit:
#   - ``python[N]? -c`` removed: legitimate verification (``py_compile``,
#     ``ast.parse``, ``compile()``, ``help()``, ``pytest.main``) is the
#     dominant agent use; sandbox per-session Pod + zero-perm SA +
#     NetworkPolicy + ``automountServiceAccountToken: false`` is the real
#     isolation boundary, and the same execution surface already passes
#     when invoked via a file or ``python -m``. Defense-in-depth: the
#     PreToolUse hook consuming ``dangerous_commands.py`` is the higher
#     fidelity catch (``code_execution``, ``reverse_shell``, etc.).
#   - ``eval`` / ``exec`` anchored to command position via ``_CMDPOS``:
#     the bare-word match fired inside quoted strings and on Python's
#     ``compile(..., 'exec')`` mode argument. Bash builtins only act at
#     command position; matching anywhere else was structurally wrong.
#   - ``<<-?`` heredoc removed: heredocs to interpreters are a normal
#     shell pattern for inline scripts; payloads like
#     ``bash <<EOF\n curl|sh \nEOF`` are still caught by the
#     ``curl|sh`` pattern (the full-command match walks the heredoc
#     body). Process substitution (``<(...)`` / ``>(...)``) stays
#     blocked — it has no legitimate agent workflow analogue.
_DENY_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    # --- carriage-return / command-substitution / variable-expansion ----
    (r"\r", "carriage return injection forbidden"),
    # --- privilege escalation -------------------------------------------
    (r"\bsudo\b", "privileged escalation: sudo forbidden"),
    (r"\bdoas\b", "privileged escalation: doas forbidden"),
    (r"\bsu\b[^;|&]*-c\b", "privileged escalation: su -c forbidden"),
    # --- destructive commands -------------------------------------------
    (r"\brm\s+-[a-z]*r[a-z]*f?[a-z]*\s+/(\s|$)", "destructive: rm -rf / forbidden"),
    (r"\brm\s+-[a-z]*r[a-z]*f?[a-z]*\s+~(\s|$)", "destructive: rm -rf ~ forbidden"),
    (r":\s*\(\s*\)\s*{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "fork bomb forbidden"),
    (r"\bdd\s+if=", "dd forbidden"),
    (r"\bmkfs\b", "filesystem create forbidden"),
    # System-control verbs must only fire when used as shell commands, not
    # inside Python/method bodies or quoted strings. The bare ``\b`` anchor
    # over-matched on legitimate heredoc bodies like ``httpd.shutdown()`` and
    # string literals like ``echo "Please run shutdown"``. The ``_CMDPOS``
    # prefix narrows the match to start-of-line, post-chain-operator, or
    # substitution-opener positions — exactly where a shell command can
    # legitimately begin.
    (
        _CMDPOS + r"(?:shutdown|reboot|poweroff|halt)\b",
        "system control forbidden",
    ),
    # --- SUID / setgid privilege bits -----------------------------------
    (r"\bchmod\s+[0-7]*[4567][0-7]*\s+", "SUID/SGID chmod forbidden"),
    (r"\bchmod\s+[^;|&]*\+s\b", "SUID symbolic chmod forbidden"),
    # --- obfuscated payloads --------------------------------------------
    (r"\bbase64\s+-d\b", "obfuscated payload: base64 decode forbidden"),
    (r"\bbase64\s+--decode\b", "obfuscated payload: base64 decode forbidden"),
    (r"\$'", "obfuscated syntax: ANSI-C quoting forbidden"),
    # Locale/translation quoting ``$"..."`` is another quoting form an attacker
    # can use to assemble flags/words that evade the literal-text deny patterns
    # (e.g. ``$"-rf"`` → ``-rf``). Mirrors the ANSI-C ``$'`` rule above and
    # Matched on the full command — the opening ``$"`` is unambiguous
    # regardless of position.
    (r'\$"', "obfuscated syntax: locale quoting forbidden"),
    # ``$IFS`` / ``${...IFS...}`` word-split injection: bash word-splits on
    # ``$IFS`` at runtime, so ``rm${IFS}-rf${IFS}/`` reconstructs ``rm -rf /``
    # without emitting literal whitespace and defeats every whitespace-anchored
    # deny pattern. Flag ALL IFS usage (including parameter-expansion variants
    # like ``${IFS:0:1}`` / ``${#IFS}``). Anchored to the full command —
    # IFS can appear anywhere.
    (r"\$IFS|\$\{[^}]*IFS", "obfuscated syntax: IFS word-split injection forbidden"),
    # --- inline evaluation (command-position only) ----------------------
    # Anchored to ``_CMDPOS`` so the bare word does not match inside quoted
    # strings (``echo "exec is a word"``), grep argument lists
    # (``grep -r exec ./code``), or Python's ``compile(..., 'exec')`` mode
    # argument. Bash builtins ``eval`` / ``exec`` only act at command
    # position; matching the bare word anywhere else is structurally wrong.
    (_CMDPOS + r"eval\b", "eval forbidden"),
    (_CMDPOS + r"exec\b", "exec forbidden"),
    # --- interpreter -c / -e / --eval injection -------------------------
    # ``python[N]? -c`` was dropped because agents legitimately need it for
    # verification (``py_compile``, ``ast.parse``, ``help``, ``pytest.main``).
    # The sandbox isolation boundary (per-session Pod + NetworkPolicy +
    # zero-perm SA) is the real defense.
    (r"\bperl\s+-[ce]\b", "interpreter -ce injection forbidden"),
    (r"\bruby\s+-e\b", "interpreter -e injection forbidden"),
    (r"\b(bash|sh|zsh|dash)\s+-c\b", "shell wrapper -c forbidden"),
    (r"\$\w+\s+-c\b", "shell wrapper via variable -c forbidden"),
    (r"\b(node|deno|bun)\s+(-e|--eval|-p|--print)\b", "interpreter eval forbidden"),
    (r"\bphp\s+(-r|-a)\b", "interpreter eval forbidden"),
    # --- pipe-to-shell remote execution (curl/wget) ---------------------
    # Order matters: this is checked BEFORE the bare-interpreter pipeline
    # rule so the more-specific reason ("pipe-to-shell") surfaces first.
    (r"\bcurl\b.*\|\s*(sh|bash|zsh|dash)\b", "pipe-to-shell forbidden"),
    (r"\bwget\b.*\|\s*(sh|bash|zsh|dash)\b", "pipe-to-shell forbidden"),
    # --- pipeline ending in bare interpreter (stdin execution) ---------
    # ``(?m)`` enables ``re.MULTILINE`` for this pattern only so the ``$``
    # anchor matches end-of-line inside a multi-line heredoc body. Without
    # multiline, ``bash <<EOF\ncat file | sh\nEOF`` slips through because
    # ``$`` only matches end-of-string (heredoc bypass from security review,
    # 2026-05-20).
    (
        r"(?m)\|\s*(bash|sh|zsh|dash|fish|python[23]?|perl|ruby|node)\s*$",
        "pipe to bare interpreter forbidden",
    ),
    # --- network listeners / netcat reverse shells ----------------------
    (r"\b(nc|netcat|ncat)\b.*-l", "network listener forbidden"),
    (r"\bsocat\b", "network listener forbidden"),
    # --- source / dot-file injection ------------------------------------
    # ``(?m)`` + ``\n`` in the operator class make the command-position anchor
    # newline-aware: a ``source`` / ``.`` on any non-first line of a
    # multi-line script is now caught. The old ``(^|[;&|`])`` anchor used
    # ``^`` (string-start only, no ``(?m)``) and omitted ``\n`` from its
    # class, so ``chain_parser.parse_chain`` — which never splits on newline
    # — left a newline-separated ``source``/``.`` inside one segment where
    # neither anchor matched and the deny silently missed. Same newline fix
    # the pipe-to-interpreter rule and the system-control verbs already
    # received. ``$(`` is deliberately NOT
    # added to the class (unlike ``_CMDPOS``): a real ``$(source …)`` is
    # already caught by the chain-walker descending into the substitution
    # body, and adding it would mis-fire on the single-quoted LITERAL
    # ``echo '$(source …)'`` (not executed by bash).
    (r"(?m)(^|[;&|\n`])\s*source\s+\S", "source injection forbidden"),
    (r"(?m)(^|[;&|\n`])\s*\.\s+\S", "dot-file source injection forbidden"),
    # --- process substitution -------------------------------------------
    # Heredocs (``<<EOF``) are allowed; the body is part of the
    # full-command match so payloads like ``bash <<EOF\n curl|sh \nEOF``
    # are still caught by the ``curl|sh`` pattern above. Process
    # substitution (``<(...)``, ``>(...)``) has no legitimate agent
    # workflow analogue and stays blocked.
    (r"<\(", "process substitution forbidden"),
    (r">\(", "process substitution forbidden"),
    # --- long-running server / listener processes ----------------------
    (r"\bpython[23]?\s+-m\s+(http\.server|SimpleHTTPServer)\b", "server hosting forbidden"),
    (r"\b(flask\s+run|uvicorn|gunicorn|hypercorn|daphne)\b", "server hosting forbidden"),
    (r"\b(npm|pnpm|yarn)\s+(start|run\s+(dev|start|serve))\b", "server hosting forbidden"),
    (r"\bnpx\s+serve\b", "server hosting forbidden"),
    (r"\b(tmux|screen|nohup)\b", "long-running process forbidden"),
)


_COMPILED: Final[tuple[tuple[re.Pattern[str], str], ...]] = tuple(
    (re.compile(pattern, re.IGNORECASE), reason) for pattern, reason in _DENY_PATTERNS
)


class DefaultShellSafetyPolicy:
    """Conservative deny policy applied to every sub-command of a chain.

    Used by the host ``Bash`` tool. Tenant-configurable policies
    (allowlist, approval-required) extend this baseline.
    """

    def evaluate(self, command: str) -> ShellPolicyDecision:
        """Return :class:`ShellPolicyDecision` for a full command line.

        Evaluation strategy:
            1. Apply deny patterns to the FULL command (catches multi-segment
               attacks like ``curl X | sh`` or fork-bomb literals).
            2. Walk every sub-command and ``$()`` / backtick substitution
               body. First match → deny.
        """
        if not command or not command.strip():
            return ShellPolicyDecision(verdict=ShellPolicyVerdict.allow)
        full_verdict = self._evaluate_segment(command)
        if full_verdict.verdict is ShellPolicyVerdict.deny:
            return full_verdict
        chain = parse_chain(command)
        for segment in chain:
            verdict = self._evaluate_segment(segment.raw)
            if verdict.verdict is ShellPolicyVerdict.deny:
                return verdict
            for sub in segment.substitutions:
                sub_verdict = self.evaluate(sub)
                if sub_verdict.verdict is ShellPolicyVerdict.deny:
                    return sub_verdict
        return ShellPolicyDecision(verdict=ShellPolicyVerdict.allow)

    def _evaluate_segment(self, segment: str) -> ShellPolicyDecision:
        for compiled, reason in _COMPILED:
            if compiled.search(segment):
                return ShellPolicyDecision(
                    verdict=ShellPolicyVerdict.deny,
                    reason=reason,
                    matched_pattern=compiled.pattern,
                )
        return ShellPolicyDecision(verdict=ShellPolicyVerdict.allow)


__all__ = [
    "DefaultShellSafetyPolicy",
    "ShellPolicyDecision",
    "ShellPolicyVerdict",
    "ShellSafetyError",
]
