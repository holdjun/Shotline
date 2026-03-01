# AI Coding Agent Template

## Overview

A language-agnostic template for AI-driven development workflows. 100% AI-runnable — only PR review and merge requires human intervention.

## Core Principles

- **Elegant, efficient, concise** — code and docs must not be overcomplicated or redundant
- **Leverage existing tools** — use MCP servers, skills, `gh` CLI, and established ecosystems; don't reinvent the wheel
- **Convention over configuration** — follow project conventions; when in doubt, check `docs/`

## Directory Structure

```
src/shotline/          # Main package source
src/shotline/processors/  # Individual processing step modules
tests/                 # Test suite
docs/                  # Technical docs, architecture, product specs
.claude/               # Claude Code settings, skills
.github/               # CI workflows, PR/issue templates, Dependabot
```

## Common Commands

```bash
# Install     — uv sync --extra dev
# Lint        — uv run ruff check src/ tests/
# Format      — uv run ruff format src/ tests/
# Test        — uv run pytest
# Type check  — uv run mypy src/shotline/
```

## Key Rules

1. **Never commit directly to `main`** — all changes go through feature branches and PRs
2. **Always use `/submit` to submit PRs** — no exceptions, no manual `git push` + `gh pr create`
3. **Never commit secrets** — no `.env` files, API keys, or tokens in git
4. **No debug output in production code** — remove debug statements before committing
5. **Every feature must include tests** — no PR without corresponding test coverage

## Git Workflow

All development must be based on the **latest remote `main`**. Always branch from `origin/main`, not local `main`:

```bash
git fetch origin main
git checkout -b <type>/<short-description> origin/main
```

Development flow:

```
1. git fetch origin main && git checkout -b feat/my-change origin/main
2. Make changes, commit incrementally
3. Run project quality checks (lint, format, test, build)
4. /submit    # handles push, PR creation, CI monitoring
```

After a PR is merged (squash-merged on GitHub), the local branch diverges from remote `main`. Always start the next branch from `origin/main` and delete the old local branch.

**Hard rules:**
- Never `git push origin main`
- Never commit on `main`
- If on `main` with uncommitted changes: stash, create a branch, then apply

## Documentation

All project documentation lives under `docs/`:

- Technical architecture and design decisions
- Product specifications and requirements
- Coding conventions and patterns
- API documentation

Reference `docs/` in code and discussions. Keep docs updated when changes affect architecture, commands, or conventions.
