# Shotline — Automated RAW Processing Pipeline

## Overview

A lightweight, automated photo processing tool that takes RAW files from camera to a solid 60-point baseline edit. Combines signal-level RAW decoding with aesthetic tone mapping to produce images that match or exceed camera JPEG/HEIF direct output quality — with full control over every step.

100% AI-runnable development workflow — only PR review and merge requires human intervention.

## Core Principles

- **Elegant, efficient, concise** — code and docs must not be overcomplicated or redundant
- **Leverage existing tools** — use MCP servers, skills, `gh` CLI, and established ecosystems; don't reinvent the wheel
- **Convention over configuration** — follow project conventions; when in doubt, check `docs/`
- **Minimal public API** — each function does one thing; only export when needed by CLI or other modules

## Domain Expertise

This project requires thinking as **engineer + photographer + retoucher** simultaneously:

- **Engineer**: correctness before performance; understand the math behind every transform, never blindly chain operations
- **Photographer**: understand light, exposure, optics, and what makes a technically sound image — know what the camera does and why
- **Retoucher**: understand imaging principles end-to-end — know how to make an image look better and why

## Directory Structure

```
src/shotline/             # Main package source
src/shotline/processors/  # Individual processing step modules
tests/                    # Test suite
docs/                     # Technical docs, architecture, product specs
shotline.toml             # Default processing config
.claude/                  # Claude Code settings, skills
.github/                  # CI workflows, PR/issue templates, Dependabot
```

## Common Commands

```bash
# Install     — uv sync
# Test        — uv run pytest
# Coverage    — uv run pytest --cov
# Lint        — uv run ruff check src/ tests/
# Format      — uv run ruff format src/ tests/
# Type check  — uv run basedpyright src/
# Pre-commit  — uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run basedpyright src/ && uv run pytest
```

## Key Rules

1. **Never commit directly to `main`** — all changes go through feature branches and PRs
2. **Always use `/submit` to submit PRs** — no exceptions, no manual `git push` + `gh pr create`
3. **Never commit secrets** — no `.env` files, API keys, or tokens in git
4. **No debug output in production code** — remove debug statements before committing
5. **Every feature must include tests** — no PR without corresponding test coverage. Tests must guard real behavior, not exist for the sake of coverage. Don't test language mechanics (dict assignment, return types, key existence without value checks). Do test: correctness of domain logic, physical/mathematical direction of transforms, branch coverage for distinct code paths, degradation behavior. If a test would still pass with a broken implementation, it's worthless — delete it.
6. **Don't create new abstractions or helper functions** unless called from 3+ places; prefer inlining simple logic (<5 lines)

## Verification

Every change must pass before submitting:

```bash
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run basedpyright src/ && uv run pytest
```

| Task type | Done when |
|-----------|-----------|
| Bug fix | Regression test + all checks green |
| New feature | Corresponding test file + all checks green |
| Refactor | Existing tests unchanged + all checks green |
| Docs | No broken links, no stale info |

## Compact Instructions

When compacting, preserve by priority:

1. Architecture decisions (never summarize)
2. Modified files and key changes
3. Current verification status (pass/fail)
4. Outstanding TODOs and rollback notes

## Documentation

All project documentation lives under `docs/`:

- Technical architecture and design decisions
- Product specifications and requirements
- Coding conventions and patterns
- API documentation

Reference `docs/` in code and discussions. Keep docs updated when changes affect architecture, commands, or conventions.
