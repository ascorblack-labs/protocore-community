# Protocore Core — Documentation

`protocore` is the **pure core** of the Protocore agent runtime: a Python 3.12+
library of **contracts** (Protocols + typed models) and a **protocol-first ReAct
runtime** that drives one agent turn at a time. It is a **universal,
multi-tenant** product core with zero upward imports — it owns no database
driver, no HTTP endpoint, and no deployment logic; everything outward-facing is a
`Protocol` that the host layer implements. The runtime entry points are
`QueryEngine` (the per-run state owner) and `query()` (a **sync** function that
resets per-turn state and returns an async iterator of `TurnEvent`s), both
imported from `protocore.runtime.*`.

## Reading order

A new engineer or agent can read these end to end, in this order:

1. **[Getting Started](getting-started.md)** — install via `uv`, then a minimal
   runnable shape: build a `QueryEngine` with injected adapters and stream a turn
   with `query()`.
2. **[Architecture](architecture.md)** — the deep reference: layered structure,
   the one-turn data flow, the technology inventory, and a per-subsystem tour.
3. **[Contracts](contracts.md)** — the boundary: the interface `Protocol`s 
   the host layer implements and the core type system.
4. **[Tools](tools.md)** — the lean verb surface, the `@tool` decorator, the
   dispatch pipeline, the permission gate, and tool preconditions.
5. **[Runtime Constants](runtime-constants.md)** — how every tunable value flows
   through the frozen `RuntimeConstants` snapshot, and how to read a default.
6. **[Extending](extending.md)** — pick your seam: a `Protocol`, a hook, an RC
   toggle, or a system-prompt section — plus the rules you must not break.
7. **[Testing](testing.md)** — how to run tests, lint, types, and the security
   scan, and what the import-boundary guard test asserts.
8. **[Glossary](glossary.md)** — concise definitions for the terms used
   throughout these docs.

## Bilingual documentation

This `docs/` tree is the **English canonical** source. A full **Russian mirror**
lives at [`ru/`](ru/index.md) with a one-to-one file set and matching headings
and anchors. Any change edits the English file first, then updates its Russian
counterpart in the same change set; each Russian file records the source path and
the commit it was translated from. The two project READMEs follow the same pair:
the Russian [`../README.md`](../README.md) is primary, and the English
[`../README.en.md`](../README.en.md) is its mirror.

## All documents

| Document | Purpose |
|---|---|
| [`index.md`](index.md) | This hub — overview, reading order, and the full doc map. |
| [`getting-started.md`](getting-started.md) | Install and a minimal runnable quickstart. |
| [`architecture.md`](architecture.md) | Deep reference: structure, turn data flow, per-subsystem tour. |
| [`contracts.md`](contracts.md) | Interface `Protocol`s and the core type system. |
| [`runtime-constants.md`](runtime-constants.md) | The `RuntimeConstants` model and how to read defaults. |
| [`tools.md`](tools.md) | Lean verb surface, `@tool`, dispatch, and the permission gate. |
| [`extending.md`](extending.md) | The four extension seams and the rules you must not break. |
| [`testing.md`](testing.md) | Running tests, lint, types, security, and the boundary guard. |
| [`glossary.md`](glossary.md) | Definitions for the terms used across these docs. |

## Project root files

| File | Purpose |
|---|---|
| [`../README.md`](../README.md) | Project overview and quickstart (Russian, primary). |
| [`../README.en.md`](../README.en.md) | Project overview and quickstart (English mirror). |
| [`../LICENSE`](../LICENSE) | License terms (Mozilla Public License 2.0). |
| [`../NOTICE`](../NOTICE) | Attribution and what the MPL asks of a distributor. |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Setup, the four gates, and the conventions. |
| [`../SECURITY.md`](../SECURITY.md) | Supported versions and private vulnerability disclosure. |
