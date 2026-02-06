# AGENTS.md

## Introduction
This repository is documentation + configuration (Markdown-first). Optimize for readability and minimal diffs.

## Purpose and scope
- Repository purpose: [one sentence describing what this repo documents/contains]
- Allowed changes:
  - Edit Markdown docs and configuration files.
  - Add new docs in the appropriate folder (see “Project map”).
- Not allowed (unless explicitly requested):
  - Large refactors/restructures (moving many files, renaming sections globally).
  - Reformatting entire documents “for style” unless the task requests it.
  - Renaming files (case-sensitive) without a clear reason and updated references.

If requirements are ambiguous, ask before making broad changes.

## Important files (read before changing behavior/structure)
- ARCHITECTURE.md — overall design and repo structure
- STYLE.md — writing + formatting conventions
- CHANGELOG.md — user-visible changes (keep it current)

## Quick start (commands)
- Install hooks: `pre-commit install`
- Run all checks: `pre-commit run --all-files`
- [If you have any other checks, list them here exactly, e.g. markdownlint, link checker, docs build]

## Contributing: Definition of Done
WARNING THE FOLLOWING IS MANDATORY YOU MUST FOLLOW THESE STEPS BEFORE COMING BACK TO THE USER:
@@@@@@
1. Run: `pre-commit run --all-files` (mandatory; do not respond without running it unless the user explicitly says not to). Do not ask for confirmation to run it.
2. If you create new files, add them to git (e.g., `git add <file>`).
3. Run 'pytest --doctest-modules ' and other tests if you have them; do not respond without running them unless the user explicitly says not to).
4. Update CHANGELOG.md with a short entry describing what changed (and why).
5. For every satisfied request, release a new version in CHANGELOG.md (do not leave changes only under Unreleased).
6. If design/structure changed, update ARCHITECTURE.md.
7. In the PR/response, include:
   - Summary of changes (1–3 bullets)
   - Commands you ran + results
   - Any follow-ups or risks
@@@@@@



## Project map
- `/` — top-level docs and configs
- `[docs/ or other folders if present]` — what belongs there
- `[templates/ etc]` — what belongs there
-  `[test/ etc]`  — e2e test files or others

## Writing and style rules (reader-optimized)
- Prefer declarative, intention-revealing writing.
- Keep sections short; avoid repetition.
- Use consistent headings and terminology (follow STYLE.md).
- Avoid mass whitespace/format-only changes unless required.
