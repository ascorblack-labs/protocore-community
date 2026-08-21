# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `QueryEngine.rearm()` now also restarts the state that is attached to an
  engine *after* it is constructed. The re-arm rebuilds from a fresh engine, and
  `vars()` of a fresh engine cannot see what the host or the run loop attaches
  later, so three things survived every re-arm in silence: the cached tool
  dispatcher, which holds a tool-error counter read out of a helper bag the host
  may since have replaced; the fire-once warning latch for a normalised outbound
  system prompt, which made its warning fire once per engine rather than once
  per run; and the per-run streaks the dispatcher keeps inside the helper bag —
  the consecutive same-tool-same-error cell, the sandbox-down streak and its
  one-shot injection flag, the string-type streak, and the subagent soft-cap
  counts. An agent that repeats one failing call at the start of each turn could
  cross a cap documented as per-run that no single turn ever reached. The bag
  itself belongs to the host and is left alone, as is the run tree's shared work
  ledger. A test now reads the package for every `engine.x = ...` and
  `setattr(engine, "x", ...)` outside the constructor and fails when one is
  classified as neither dropped nor kept.

## [2.0.0a3]

Supersedes 2.0.0a2, whose files were removed from the index. That build carried
comments and docstrings that cited internal working documents as the authority
for public behaviour, and that kept the labels a review round leaves behind in
the code it reviewed. A reader outside the project could see that closed
documents govern this library and could read nothing of them. No functional
difference; the prose now states each reason in its own terms.

The publication scanner gained the rules that would have caught it — a document
cited as authority, a review-round trace, a work-package label, a planning or
triage label — so the class cannot come back silently.

### Fixed

- `QueryEngine.rearm()` now restarts every per-run allowance rather than a
  named twelve. An engine that takes an unbounded number of turns on one
  history carried the rest across, and each one ended the agent quietly: the
  identical-tool loop guard counted a fingerprint for the life of the engine,
  so an agent that opened every turn with the same observing call was refused
  it from the fourth turn on; the repeated-error circuit breaker's block list
  is unioned into the visible tool surface, so a tool that failed for a reason
  that had since passed was withdrawn for good; the cooperative stop flag had
  no lowering seam, so an agent interrupted once never spoke again. The reset
  is now expressed the other way round — the engine names what SURVIVES a
  re-arm (history, compaction state, live-control queues, lanes, the injected
  collaborators) and rebuilds everything else — so a field added to the
  constructor resets by default instead of quietly accumulating. A test walks
  the constructor and fails when an attribute is classified as neither.

## [2.0.0a2]

Supersedes 2.0.0a1, whose files were removed from the index. That build
carried comments naming the tooling used to write them and a pointer to a
working document that ships with nothing — no functional difference, but not
what belongs in a published artifact. Nothing else changed.

## [2.0.0a1]

First public release. Withdrawn.

Protocore has existed for some time as the closed core of an agent product.
This is that core, extracted and published under the MPL — the same code, with
the parts that only made sense inside one company's repository rewritten to
describe the boundary rather than the company.

### Added

- The protocol boundary: 20 interface `Protocol`s and an `IBlobStore` ABC in
  `protocore/contracts/`, covering the model client, run and session stores,
  the tool registry, memory, workspace, search, blobs, skills, todos, hooks,
  event transport, and subagent dispatch.
- The ReAct runtime: `QueryEngine` plus `query()`, driving one agent turn at a
  time and emitting typed `TurnEvent`s. Snapshot and resume are first-class.
- A three-layer tool surface — tenant policy, a lean clipped surface, and
  progressive discovery over BM25 retrieval — with a permission gate ahead of
  dispatch and a shell-safety policy behind a real command-chain parser.
- Two-tier context compaction, session memory folding, and a token-budget model
  that keeps a long run inside its window.
- `RuntimeConstants`: 524 per-tenant tunables as a frozen Pydantic snapshot,
  every one of them documented, with new behaviour defaulting off.
- In-memory adapters (`protocore.tests_support.adapters`) that implement the
  same protocols the real ones do, so a turn runs end to end with no external
  services.
- Documentation in English and Russian under `docs/`.
- 2964 tests, a 90% coverage floor, strict typing, lint, and a security scan,
  all gated on Python 3.12, 3.13, and 3.14.

[Unreleased]: https://github.com/ascorblack-labs/protocore-community/compare/v2.0.0a2...HEAD
[2.0.0a2]: https://github.com/ascorblack-labs/protocore-community/releases/tag/v2.0.0a2
[2.0.0a1]: https://github.com/ascorblack-labs/protocore-community/releases/tag/v2.0.0a1
