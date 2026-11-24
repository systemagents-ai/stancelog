# Agent memory tool spec (from Deep Mehta)
Ask: a plug-and-play memory tool for AI agents that "can survive any issues" and solves contradictions over time. Example: user first says "don't do A", later says "maybe A is better" -> memory keeps both: A marked bad initially, latest stance A is good; current belief = A good, with full history.

Requirements:
- Python package (pip-installable, pyproject), no required external services. Default storage SQLite with WAL + atomic transactions so it survives crashes/kill -9, recovers from corrupted/partial writes, handles concurrent writers. Append-only event log (facts never destroyed) with a derived current view.
- Temporal belief versioning: each memory has subject/key, value/stance, timestamp, source, confidence. New memory on the same subject supersedes older (bi-temporal valid_from/valid_to). Contradiction detection: same subject with opposite stance -> link as supersedes, keep history. API: current(subject), history(subject), recall(query) ranked by recency+relevance, forget (soft, logged).
- Pluggable subject/contradiction extractor: simple built-in heuristic + optional LLM hook (interface only, no API key needed).
- Plug-and-play: Python API (Memory().remember(...), .recall(...)), a CLI, and an MCP stdio server exposing remember/recall/current/history/forget for Claude Code, Cursor etc. Include MCP client config example.
- pytest tests: A-bad-then-A-good scenario, crash/kill-during-write durability, concurrency, persistence across restarts. All must pass.
- README with the A example, quickstart, MCP setup, design notes. MIT license. Pick a good name.
