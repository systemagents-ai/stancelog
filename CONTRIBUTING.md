# Contributing to stancelog

Thanks for your interest in improving stancelog. This is a small project, so the process is short.

## Development setup

```bash
git clone https://github.com/systemagents-ai/stancelog
cd stancelog
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest -q
```

CI runs the suite on Python 3.10, 3.11, 3.12 and 3.13. Please make sure it passes locally before
opening a pull request.

## Design invariants

These are the guarantees the rest of the project relies on. A change that breaks one of them will
not be merged, however useful it is otherwise.

- **The `events` table is append-only.** Never `UPDATE` or `DELETE` an event (SQLite triggers
  enforce this). Corrections, forgets and other changes are new events. `beliefs`, `relations`
  and `beliefs_fts` are derived caches and must be reproducible by `rebuild()` from the event log.
- **Every write is one transaction.** Each `remember`/`forget` appends its event(s) and refreshes the
  derived rows inside a single `BEGIN IMMEDIATE … COMMIT`. A write is either fully present or fully
  absent.
- **Journal before commit.** The JSONL journal line is written and fsync'd before the SQLite
  `COMMIT`, so a returned write is always recoverable from the journal.
- **Zero core dependencies.** The library, CLI and MCP server use the standard library only
  (`dependencies = []` in `pyproject.toml`). Test-only tools belong in the `dev` extra.
- **Python >= 3.10 compatible.** Avoid 3.11+-only features such as `tomllib`, `typing.Self`,
  `except*`, `datetime.UTC` and `enum.StrEnum`.

## Pull request guidelines

- Keep PRs small and focused: one fix or change per PR.
- Add or update tests for any behaviour change, and for bug fixes add a test that fails without
  the fix.
- Update `README.md` when user-visible behaviour changes. Don't document features that don't exist yet.
- Describe what changed and why in the PR description, and call out anything that touches
  durability, recovery or the on-disk format.
- Make sure `pytest -q` passes on Python 3.10+.

By contributing, you agree that your contributions are licensed under the project's MIT license.
