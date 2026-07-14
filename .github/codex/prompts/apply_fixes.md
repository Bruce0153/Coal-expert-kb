You are Codex working in the Coal Expert KB repository.

Goal:
Apply the requested pull-request fix with a minimal, reviewable diff while preserving the canonical architecture.

Rules:
- Read `docs/engineering/coding_standards.md` and `docs/engineering/acceptance_criteria.md` first.
- Use only the canonical packages under `src/coal_kb/`.
- Do not create duplicate modules, alternate implementations, migration shims, or state-suffixed filenames.
- Keep Python and Shell naming consistent with the repository conventions.
- Update every import, configuration reference, document, test, and command affected by a rename.
- Do not add model downloads or external-service calls to the offline test suite.
- Never print secrets or personal data.

Validation:
- Run `bash scripts/quality/check_repository.sh`.
- Add or update focused tests for behavior changed by the fix.
- Report changed files, checks executed, and any check that could not run.
