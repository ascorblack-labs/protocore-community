"""``ToolPermissionGate`` — the 4-stage permission decision pipeline.

md`. Orders the gating concerns:

1. Tenant tool whitelist (or subagent restricted scope).
2. Per-side-effect-class safety policy
 (``DefaultShellSafetyPolicy`` for Bash, DNS allowlist for WebFetch,
 denied-paths for workspace tools).
3. Rate limit (host-side — left as a Protocol seam in core; the gate
 only knows how to ASK).
4. ``PreToolUse`` hook fires last and may override (allow / deny /
 require_approval / mutate input). The hook is the **highest-leverage
 point** for tenants who want an LLM-as-a-policy gate.

Core ships the *default* safety policies (shell deny patterns are in
:mod:`protocore.safety.shell`; DNS allowlist is a static-set check).
The host can stack additional policies by composing via
:meth:`ToolPermissionGate.register_policy` — keeps the core API frozen.

Result envelope: :class:`ToolPermissionDecision` carries an outcome
(allow / deny / require_approval), the optional rewritten input args
(``modify`` action from hooks), the reason string, and the originating
stage for telemetry/audit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Protocol, runtime_checkable
from urllib.parse import urlparse

from protocore.contracts.hooks import HookActionKind, HookResult, IHookManager
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import HookEvent
from protocore.logging_utils import get_logger
from protocore.safety.shell import (
    DefaultShellSafetyPolicy,
    ShellPolicyVerdict,
)

_logger = get_logger(__name__)

_HELPERS_METADATA_KEY = "protocore.helpers"


def _session_grant_covers(ctx: ToolContext, arguments: dict[str, Any]) -> bool:
    """A session grant skips approval only; denials still stand."""
    command = str((arguments or {}).get("command") or "")
    if not command:
        return False
    metadata = getattr(ctx, "metadata", None) or {}
    bag = metadata.get(_HELPERS_METADATA_KEY) if isinstance(metadata, dict) else None
    grants = list((bag or {}).get("session_grants") or [])
    if not grants:
        return False
    from protocore.runtime.permission_widen import grant_covers

    return any(grant_covers(grant, command) for grant in grants)


# Side-effect class strings used to fan out to per-class safety policies.
# Defined as constants here rather than RC because they are protocol
# values — not user-tunable. Matches the tool catalog in ``docs/tools.md``.
SIDE_EFFECT_SANDBOX: Final[str] = "sandbox"
SIDE_EFFECT_HTTP: Final[str] = "http"
SIDE_EFFECT_WORKSPACE: Final[str] = "workspace"
SIDE_EFFECT_STATE_ONLY: Final[str] = "state_only"

# Tool name → side-effect class, per the catalog in ``docs/tools.md``.
# Not an RC field — protocol invariant; new tools declare their class
# via the `Tool.side_effect_class` ClassVar (a host's adapters supply it).
_DEFAULT_SIDE_EFFECT_MAP: Final[dict[str, str]] = {
    "Bash": SIDE_EFFECT_SANDBOX,
    "PythonExec": SIDE_EFFECT_SANDBOX,
    "WebFetch": SIDE_EFFECT_HTTP,
    "Write": SIDE_EFFECT_WORKSPACE,
    "Edit": SIDE_EFFECT_WORKSPACE,
    "Read": SIDE_EFFECT_STATE_ONLY,
    "Grep": SIDE_EFFECT_STATE_ONLY,
    "Glob": SIDE_EFFECT_STATE_ONLY,
    "Skill": SIDE_EFFECT_STATE_ONLY,
    "Agent": SIDE_EFFECT_STATE_ONLY,
    "ToolSearch": SIDE_EFFECT_STATE_ONLY,
    "TodoWrite": SIDE_EFFECT_STATE_ONLY,
}


class ToolPermissionOutcome(StrEnum):
    """Permission gate decision outcome."""

    allow = "allow"
    deny = "deny"
    require_approval = "require_approval"


class PermissionStage(StrEnum):
    """Stage at which a decision was reached (for telemetry)."""

    whitelist = "whitelist"
    safety_policy = "safety_policy"
    rate_limit = "rate_limit"
    hook = "hook"
    default = "default"


@dataclass(frozen=True, slots=True)
class ToolPermissionDecision:
    """Outcome envelope from :meth:`ToolPermissionGate.check`.

    Attributes
    ----------
    outcome:
        ``allow`` — proceed with execute.
        ``deny`` — abort, surface as ``tool_result(success=false)``.
        ``require_approval`` — pause turn (``AWAITING``), emit
        ``tool_call_pending``, resume via user approval.
    reason:
        Human-readable explanation (logged + surfaced in audit + payload
        for ``deny`` results).
    stage:
        Which gate produced the verdict — drives telemetry/buckets.
    modified_input:
        ``None`` if input passes through unchanged; otherwise the
        mutated dict produced by a ``MODIFY``-action ``PreToolUse`` hook.
    approval_token:
        Opaque token issued by the hook manager when an approval is
        required; carried back via the user-resume payload.
    """

    outcome: ToolPermissionOutcome
    reason: str = ""
    stage: PermissionStage = PermissionStage.default
    modified_input: dict[str, Any] | None = None
    approval_token: str | None = None

    @property
    def allowed(self) -> bool:
        return self.outcome is ToolPermissionOutcome.allow

    @property
    def denied(self) -> bool:
        return self.outcome is ToolPermissionOutcome.deny

    @property
    def requires_approval(self) -> bool:
        return self.outcome is ToolPermissionOutcome.require_approval


# ---------------------------------------------------------------------------
# Per-side-effect-class safety policies
# ---------------------------------------------------------------------------


@runtime_checkable
class IToolSafetyPolicy(Protocol):
    """Per-side-effect-class safety policy Protocol.

    Implementations check tool input args against per-class rules
    (denied paths for workspace tools, deny patterns for sandbox tools,
    DNS allowlist for HTTP tools). Stateless — config is bound at
    construction time.
    """

    def applies_to(self, side_effect_class: str) -> bool:
        """Return ``True`` if this policy handles ``side_effect_class``."""
        ...

    def evaluate(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> ToolPermissionDecision:
        """Evaluate the call; return :class:`ToolPermissionDecision`."""
        ...


# Argument keys that carry the shell command for sandbox-class tools.
# The gate runs on the RAW ``tool_call.arguments`` BEFORE the tool's input
# model validates/normalises them, so the safety policy must recognise every
# alias the model may emit — the host ``BashInput.command`` field
# declares ``validation_alias=AliasChoices("command", "cmd", "shell")`` and the
# alias is only resolved to ``command`` inside ``Tool.invoke`` (after this
# gate). Reading only ``command`` here lets a ``{"cmd": "sudo rm -rf /"}`` /
# ``{"shell": ...}`` call skip the deny patterns and still execute. These are
# protocol-level alias names (mirrors the host field), not user-tunable —
# hence a module constant rather than an RC.
_SHELL_COMMAND_ARG_ALIASES: Final[tuple[str, ...]] = ("command", "cmd", "shell")

# Argument keys that carry the workspace path for workspace-class tools.
# The host ``WriteInput.path`` / ``EditInput.path`` / ``AppendFileInput.
# path`` fields declare ``validation_alias=AliasChoices("path", "file_path")``
# (path is canonical, file_path is the legacy alias) and the alias is only
# resolved to ``path`` inside ``Tool.invoke`` (after this gate). Reading only
# ``file_path`` here lets a model emit ``{"path": "/denied/x"}`` (the field
# name its tool description instructs it to use) skip the
# ``denied_path_prefixes`` deny patterns and still execute — same root cause
# as the shell-command alias bypass above. .
_WORKSPACE_PATH_ARG_ALIASES: Final[tuple[str, ...]] = ("file_path", "path")


@dataclass(frozen=True, slots=True)
class ShellSafetyPolicyAdapter(IToolSafetyPolicy):
    """Wraps :class:`DefaultShellSafetyPolicy` for the sandbox class.

    Reads the shell command from any of :data:`_SHELL_COMMAND_ARG_ALIASES`
    (``command``/``cmd``/``shell`` — the Bash input shape and its validation
    aliases) and evaluates via the v1-gold deny-pattern policy.
    """

    policy: DefaultShellSafetyPolicy = field(default_factory=DefaultShellSafetyPolicy)

    def applies_to(self, side_effect_class: str) -> bool:
        return side_effect_class == SIDE_EFFECT_SANDBOX

    def evaluate(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> ToolPermissionDecision:
        del tool, ctx
        command = self._resolve_command(arguments)
        if not isinstance(command, str):
            return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)
        verdict = self.policy.evaluate(command)
        if verdict.verdict is ShellPolicyVerdict.deny:
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.deny,
                reason=verdict.reason,
                stage=PermissionStage.safety_policy,
            )
        return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)

    @staticmethod
    def _resolve_command(arguments: dict[str, Any]) -> str | None:
        """Return the shell command from the first present alias.

        The single logical ``command`` field can arrive under any of its
        validation aliases (:data:`_SHELL_COMMAND_ARG_ALIASES`) because the
        gate sees raw arguments before the tool's input model resolves them.
        First alias whose value is a ``str`` wins; non-string / absent values
        yield ``None`` (treated as nothing to evaluate by the caller).
        """
        for alias in _SHELL_COMMAND_ARG_ALIASES:
            value = arguments.get(alias)
            if isinstance(value, str):
                return value
        return None


@dataclass(frozen=True, slots=True)
class HttpDnsAllowlistPolicy(IToolSafetyPolicy):
    """DNS allowlist policy for the ``http`` side-effect class.

 Reads ``arguments['url']`` (WebFetch input shape). Empty allowlist
 means "allow any host" (default) — the host registers a populated
 allowlist via :meth:`ToolPermissionGate.register_policy`.
 Blocked hosts always reject regardless of allowlist.

 NOTE: this is the *core* policy and is intentionally lightweight —
 it does NOT perform DNS resolution (no network in pure core).
 Full SSRF guard (private-IP / cloud-metadata / link-local resolution)
 lives in the host adapter and stacks on top of this gate.

 Host matching is **case-insensitive** (RFC 4343): the configured
 ``allowed_hosts`` / ``blocked_hosts`` are ASCII-lowercased at
 construction so a mixed-case operator entry (e.g. ``"Evil.com"``) still
 matches the always-lowercase host that :func:`urllib.parse.urlparse`
 produces . We use ``str.lower`` (not ``str.casefold``) to mirror
 ``urlparse``'s own ASCII-lowercasing exactly — ``casefold`` would
 over-fold some non-ASCII characters (e.g. ``faß`` → ``fass``) and could
 silently widen the allowlist to distinct hostnames.
 """

    allowed_hosts: frozenset[str] = field(default_factory=frozenset)
    blocked_hosts: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Normalise host sets once at construction so comparisons against
        # ``urlparse().hostname`` (always ASCII-lowercase) are
        # case-insensitive. The dataclass is frozen+slots, so mutate via
        # object.__setattr__.
        object.__setattr__(
            self,
            "allowed_hosts",
            frozenset(h.lower() for h in self.allowed_hosts),
        )
        object.__setattr__(
            self,
            "blocked_hosts",
            frozenset(h.lower() for h in self.blocked_hosts),
        )

    def applies_to(self, side_effect_class: str) -> bool:
        return side_effect_class == SIDE_EFFECT_HTTP

    def evaluate(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> ToolPermissionDecision:
        del tool, ctx
        url = arguments.get("url")
        if not isinstance(url, str):
            return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)
        parsed = urlparse(url)
        # ``urlparse`` ASCII-lowercases the hostname; lowercase again (no-op
        # for ASCII) so both sides of the membership test agree without the
        # over-folding ``casefold`` would introduce for non-ASCII hosts.
        host = (parsed.hostname or "").lower()
        if not host:
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.deny,
                reason="url missing host component",
                stage=PermissionStage.safety_policy,
            )
        if host in self.blocked_hosts:
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.deny,
                reason=f"host {host!r} is on the blocklist",
                stage=PermissionStage.safety_policy,
            )
        if self.allowed_hosts and host not in self.allowed_hosts:
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.deny,
                reason=f"host {host!r} not in DNS allowlist",
                stage=PermissionStage.safety_policy,
            )
        return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)


@dataclass(frozen=True, slots=True)
class WorkspacePathPolicy(IToolSafetyPolicy):
    """Path-prefix denial policy for the ``workspace`` side-effect class.

 Reads the path from any of :data:`_WORKSPACE_PATH_ARG_ALIASES`
 (``file_path`` / ``path`` — the Write/Edit/AppendFile input shape and
 its validation aliases). Denies any resolved path with a prefix in
 ``denied_path_prefixes``. Empty set means "allow all".

 """

    denied_path_prefixes: frozenset[str] = field(default_factory=frozenset)

    def applies_to(self, side_effect_class: str) -> bool:
        return side_effect_class == SIDE_EFFECT_WORKSPACE

    def evaluate(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> ToolPermissionDecision:
        del tool, ctx
        path = self._resolve_path(arguments)
        if not isinstance(path, str):
            return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)
        for prefix in self.denied_path_prefixes:
            if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
                return ToolPermissionDecision(
                    outcome=ToolPermissionOutcome.deny,
                    reason=f"path {path!r} is denied by tenant policy",
                    stage=PermissionStage.safety_policy,
                )
        return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)

    @staticmethod
    def _resolve_path(arguments: dict[str, Any]) -> str | None:
        """Return the workspace path from the first present alias.

 The single logical ``path`` field can arrive under either of its
 validation aliases (:data:`_WORKSPACE_PATH_ARG_ALIASES`) because
 the gate sees raw arguments before the tool's input model resolves
 them. First alias whose value is a ``str`` wins; non-string /
 absent values yield ``None`` (treated as nothing to evaluate by
 the caller). .
 """
        for alias in _WORKSPACE_PATH_ARG_ALIASES:
            value = arguments.get(alias)
            if isinstance(value, str):
                return value
        return None


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class ToolPermissionGate:
    """4-stage permission decision pipeline.

 Stage ordering **Whitelist** — :class:`ToolVisibilityPolicy` (and optional
 subagent narrowing whitelist) must permit the tool name.
 2. **Safety policies** — per-side-effect-class checks via
 :class:`IToolSafetyPolicy` chain.
 3. **Rate limit** — host-only; baseline returns
 :class:`ToolPermissionOutcome.allow`. Subclass or compose to
 inject a Redis-backed bucket.
 4. **PreToolUse hook** — final, highest-leverage stage; may flip
 allow → deny / require_approval / modify.

 The gate is async-callable (``check``) — the hook stage is async by
 nature of :class:`IHookManager.invoke`.
 """

    def __init__(
        self,
        *,
        policies: Iterable[IToolSafetyPolicy] | None = None,
        side_effect_map: dict[str, str] | None = None,
    ) -> None:
        self._policies: list[IToolSafetyPolicy] = list(policies or self._default_policies())
        self._side_effect_map: dict[str, str] = dict(side_effect_map or _DEFAULT_SIDE_EFFECT_MAP)

    @staticmethod
    def _default_policies() -> list[IToolSafetyPolicy]:
        """Default policy stack — shell deny patterns only (no DNS / path
        config in core baseline; the host stacks tenant-specific
        instances on top via :meth:`register_policy`)."""
        return [ShellSafetyPolicyAdapter()]

    def register_policy(self, policy: IToolSafetyPolicy) -> None:
        """Stack an additional policy. Evaluated after the defaults."""
        self._policies.append(policy)

    def set_side_effect_class(self, tool_name: str, side_effect_class: str) -> None:
        """Override the side-effect class for a tool name.

        Used by tests + the host registry to declare classification
        for tools not in :data:`_DEFAULT_SIDE_EFFECT_MAP`.
        """
        self._side_effect_map[tool_name] = side_effect_class

    def classify(self, tool: Tool) -> str:
        """Return the side-effect class for ``tool``.

 Falls back to ``state_only`` for unregistered tool names — the
 most permissive class. The host adapters typically declare
 their classification via a ``ClassVar`` on the concrete impl
 set via ``set_side_effect_class`` on registration.
 """
        # Honour an explicit attribute on the tool first (a host's adapters
        # declare ``side_effect_class`` as a ``ClassVar[str]``).
        attr = getattr(tool, "side_effect_class", None)
        if isinstance(attr, str):
            return attr
        return self._side_effect_map.get(tool.name, SIDE_EFFECT_STATE_ONLY)

    # ------------------------------------------------------------------
    # check — the entry point
    # ------------------------------------------------------------------

    async def check(
        self,
        *,
        tool: Tool,
        arguments: dict[str, Any],
        ctx: ToolContext,
        visibility_policy: ToolVisibilityPolicy,
        subagent_whitelist: Iterable[str] | None = None,
        hook_manager: IHookManager | None = None,
        skip_pre_tool_approval: bool = False,
    ) -> ToolPermissionDecision:
        """Run the 4-stage pipeline; return the first non-allow decision.

        Parameters
        ----------
        tool:
            The registered :class:`Tool` instance.
        arguments:
            Already-parsed input dict.
        ctx:
            :class:`ToolContext` for this dispatch.
        visibility_policy:
            Tenant-level :class:`ToolVisibilityPolicy` (visible/blocked).
        subagent_whitelist:
            Optional narrow scope when invoked inside a subagent — if
            non-None and non-empty, only these names are permitted in
            addition to the tenant policy.
        hook_manager:
            Optional :class:`IHookManager` to fire ``PreToolUse``. If
            ``None``, the hook stage is skipped (used by tests that
            isolate the policy stack).
        skip_pre_tool_approval:
            Treat a hook-level ``require_approval`` result as already
            satisfied for a previously approved resume. Earlier gate stages
            and hook denials/modifications still apply.
        """
        # Apply the whitelist.
        if visibility_policy.blocked and tool.name in visibility_policy.blocked:
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.deny,
                reason=f"tool {tool.name!r} is on the tenant blocked list",
                stage=PermissionStage.whitelist,
            )
        # The allowed set under a non-empty ``visible`` whitelist is
        # ``visible | pinned | forced_pinned``. ``forced_pinned`` is the core
        # tool-surface floor and ``pinned`` is the progressive-
        # discovery set — both are advertised to the model unconditionally
        # (``ToolRegistry.compute_effective_surface`` / ``_floored_visible_tools``),
        # so dispatch MUST permit them too, or the model receives a callable
        # tool schema that deterministically fails at execution — recreating the
        # cause-#3 collapse one step later. ``blocked`` already overrode above.
        if visibility_policy.visible:
            allowed = (
                visibility_policy.visible
                | visibility_policy.pinned
                | visibility_policy.forced_pinned
            )
            if tool.name not in allowed:
                return ToolPermissionDecision(
                    outcome=ToolPermissionOutcome.deny,
                    reason=f"tool {tool.name!r} is not in the tenant visible set",
                    stage=PermissionStage.whitelist,
                )
        if subagent_whitelist is not None:
            allow = frozenset(subagent_whitelist)
            if allow and tool.name not in allow:
                return ToolPermissionDecision(
                    outcome=ToolPermissionOutcome.deny,
                    reason=(f"tool {tool.name!r} not in subagent whitelist"),
                    stage=PermissionStage.whitelist,
                )

        # Apply per-side-effect safety policies.
        side_effect = self.classify(tool)
        for policy in self._policies:
            if not policy.applies_to(side_effect):
                continue
            decision = policy.evaluate(tool, arguments, ctx)
            if decision.outcome is not ToolPermissionOutcome.allow:
                return decision

        # Apply the rate limit (currently a no-op).
        # The host composes a RedisLuaRateLimitPolicy in via
        # :meth:`register_policy` — it is just another
        # :class:`IToolSafetyPolicy` that returns deny with
        # PermissionStage.rate_limit when the bucket is exhausted.

        # Run the PreToolUse hook.
        if hook_manager is None:
            return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)

        try:
            hook_result = await hook_manager.invoke(
                HookEvent.pre_tool_use,
                {
                    "run_id": ctx.run_id,
                    "tenant_id": ctx.tenant_id,
                    "tool_name": tool.name,
                    "tool_input": arguments,
                    "side_effect_class": side_effect,
                },
                ctx.tenant_id,
            )
        except Exception:
            _logger.warning(
                "PreToolUse hook raised for tool=%s; isolating",
                tool.name,
                exc_info=True,
            )
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.allow,
                reason="hook dispatch failed; allowing",
                stage=PermissionStage.hook,
            )

        decision = self._project_hook_result(hook_result, arguments)
        if skip_pre_tool_approval and decision.requires_approval:
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.allow,
                reason="pre-tool approval already satisfied",
                stage=PermissionStage.hook,
            )
        if decision.requires_approval and _session_grant_covers(ctx, arguments):
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.allow,
                reason="session_grant_covers",
                stage=PermissionStage.hook,
            )
        return decision

    @staticmethod
    def _project_hook_result(
        hook_result: HookResult,
        original_args: dict[str, Any],
    ) -> ToolPermissionDecision:
        """Translate a :class:`HookResult` to a :class:`ToolPermissionDecision`."""
        if hook_result.modifications.get("requires_approval"):
            token = hook_result.modifications.get("approval_token")
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.require_approval,
                reason=hook_result.reason or "",
                stage=PermissionStage.hook,
                approval_token=token if isinstance(token, str) else None,
            )

        if hook_result.action == HookActionKind.DENY:
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.deny,
                reason=hook_result.reason or "blocked by policy",
                stage=PermissionStage.hook,
            )

        if hook_result.action == HookActionKind.MODIFY:
            modified = hook_result.modifications.get("tool_input")
            if isinstance(modified, dict):
                return ToolPermissionDecision(
                    outcome=ToolPermissionOutcome.allow,
                    reason=hook_result.reason or "modified by hook",
                    stage=PermissionStage.hook,
                    modified_input=dict(modified),
                )
            # MODIFY without tool_input: fall through as allow.
            del original_args  # unchanged
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.allow,
                reason=hook_result.reason or "",
                stage=PermissionStage.hook,
            )

        return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)


__all__ = [
    "SIDE_EFFECT_HTTP",
    "SIDE_EFFECT_SANDBOX",
    "SIDE_EFFECT_STATE_ONLY",
    "SIDE_EFFECT_WORKSPACE",
    "HttpDnsAllowlistPolicy",
    "IToolSafetyPolicy",
    "PermissionStage",
    "ShellSafetyPolicyAdapter",
    "ToolPermissionDecision",
    "ToolPermissionGate",
    "ToolPermissionOutcome",
    "WorkspacePathPolicy",
]
