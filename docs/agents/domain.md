# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

This repo uses a **single-context** layout: one `CONTEXT.md` at the repo root, with ADRs under `docs/adr/`.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root — the project's glossary and domain language
- **`docs/adr/`** — read ADRs that touch the area you're about to work in

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The producer skill (`/grill-with-docs`) creates them lazily when terms or decisions actually get resolved.

## File structure

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-<decision-slug>.md
│   └── 0002-<decision-slug>.md
└── (source dirs: model/, tokenizer/, training/, eval/, tools_engine/, data/)
```

If this repo ever grows into a monorepo with distinct subsystems that need their own glossaries, switch to a multi-context layout: replace `CONTEXT.md` at the root with `CONTEXT-MAP.md` pointing at per-subsystem `CONTEXT.md` files, and let each subsystem keep its own `docs/adr/` for context-specific decisions.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…_
