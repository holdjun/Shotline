# AI Coding Agent Template

A language-agnostic template for AI-driven development. Provides the workflow infrastructure — Claude Code skills, GitHub Actions CI, git conventions, and issue/PR templates — so you can focus on building.

## What's Included

- **Claude Code skills** — `/setup` (configure GitHub repo) and `/submit` (code-to-PR workflow)
- **GitHub Actions CI** — minimal skeleton, extend per project
- **Git workflow** — branch protection, squash merge, conventional commits
- **Issue & PR templates** — structured bug reports, feature requests, and pull requests
- **Dependabot** — automated GitHub Actions version updates

## Quick Start

1. Create a new repo from this template (or fork it)
2. Clone and open with Claude Code
3. Run `/setup` to configure GitHub branch protection and merge settings
4. Start building — create a feature branch and develop
5. Run `/submit` when ready to open a PR

## Workflow

```
git fetch origin main
git checkout -b feat/my-feature origin/main
# ... develop ...
/submit
# → push, create PR, monitor CI
# → GitHub squash-merges the PR into main
```

## Customizing for Your Project

After creating a project from this template:

1. **`CLAUDE.md`** — add tech stack, common commands, project-specific rules
2. **`.claude/settings.json`** — add language-specific command permissions
3. **`.github/workflows/ci.yml`** — add build, test, and lint steps
4. **`.gitignore`** — add language-specific ignore patterns
5. **`docs/`** — add architecture, conventions, and product docs
