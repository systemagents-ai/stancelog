# stancelog

[![CI](https://github.com/systemagents-ai/stancelog/actions/workflows/ci.yml/badge.svg)](https://github.com/systemagents-ai/stancelog/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Crash-proof, contradiction-aware memory for AI agents.**

stancelog is a small Python library, CLI and MCP server that gives an agent a long-term memory of
preferences, rules and facts. It never overwrites anything. Every statement goes into an
append-only, hash-chained event log, and "what we currently believe" is derived from that log.
When the user changes their mind, the new stance supersedes the old one, an opposite stance is
linked as a contradiction, and the whole history stays available.

- **Zero dependencies.** Standard-library `sqlite3` only, including the MCP server.
- **Survives crashes.** WAL, `synchronous=FULL`, one `BEGIN IMMEDIATE` transaction per write, and an fsync'd JSONL journal (see [Durability](#durability-guarantees-and-limits) for exactly what this covers).
- **Bi-temporal.** You can ask what was true at a given time, and what was known at a given time.
- **Plug-and-play.** Use the Python API, the `stancelog` CLI, or `stancelog serve-mcp` for Claude Code, Cursor and other MCP clients.

## The "A" example

```python
from stancelog import Memory

m = Memory("memory.db")

m.remember("Don't do A")          # -> subject "a", polarity -1
m.remember("Maybe A is better")   # -> subject "a", polarity +1, confidence 0.6 (hedged)

m.current("A")                    # case-insensitive; the latest stance wins
# Belief(subject='a', polarity=1, value='Maybe A is better', stance='positive',
#        supersedes='<id of "Don't do A">', contradicts='<same id>', valid_to=None, ...)

for b in m.history("a"):
    print(b.stance, b.valid_from, "->", b.valid_to or "now", b.value)
# negative 2026-11-24T10:00:00Z -> 2026-11-24T10:05:00Z  Don't do A
# positive 2026-11-24T10:05:00Z -> now                   Maybe A is better

m.relations("a")
# [{'src': <new>, 'dst': <old>, 'kind': 'contradicts'},
#  {'src': <new>, 'dst': <old>, 'kind': 'supersedes'}]
```

Both statements are kept. The current belief is "A is good", the earlier belief "A is bad" is
closed (its `valid_to` is set), and the two are linked by `supersedes` and `contradicts`.

## Install

stancelog is not on PyPI yet. Install it from GitHub:

```bash
pip install git+https://github.com/systemagents-ai/stancelog
```

Or, for development, from a checkout:

```bash
git clone https://github.com/systemagents-ai/stancelog && cd stancelog
pip install -e ".[dev]"        # dev extra = pytest
```

Requires Python 3.10 or newer, and nothing else.

## Quickstart: Python

```python
from stancelog import Memory

with Memory("~/.stancelog/memory.db") as m:      # or Memory() -> $STANCELOG_DB or ~/.stancelog/memory.db
    m.remember("never use eval")
    m.remember("prefer pnpm over npm")            # two claims: +pnpm, -npm
    m.remember("deploy target: prod-3")           # neutral fact (polarity 0)
    m.remember("tabs are evil", subject="tabs", polarity=-1, source="style-guide", confidence=0.9)

    m.current("pnpm")                             # Belief | None
    m.all_current()                               # every subject's current belief
    m.history("tabs")                             # every version, oldest first
    m.recall("npm or pnpm?", k=5)                 # ranked by BM25 relevance + recency

    # time travel
    m.as_of("a", "2026-01-01T00:00:00Z")                          # valid time: what was true then
    m.as_of("a", "2026-06-01", known_at="2026-03-01")             # ...using only what we knew by March

    # soft forget: logged as an event, kept in history, hidden from current()/recall()
    m.forget("tabs", reason="user asked")         # a whole subject
    m.forget("3f2a9c1d07")                        # or one belief by id (a unique prefix of 8+ chars works)

    m.check()      # integrity report: quick_check, hashes, hash chain, derived view, journal
    m.rebuild()    # recompute the derived view from the event log
```

`remember()` returns a list of `Belief`s, one per extracted claim. Explicit `subject=`,
`polarity=` or `confidence=` arguments override what the extractor finds. Each `Belief` has a
`status`: `"current"`, `"superseded"` or `"forgotten"`.

## Quickstart: CLI

```bash
export STANCELOG_DB=~/.stancelog/memory.db      # or pass --db PATH (before or after the command)

stancelog remember "don't do A"
stancelog remember "maybe A is better"
stancelog current a
stancelog history a                              # every version + SUPERSEDED/CONTRADICTS links
stancelog recall "A" -k 5 [--all]
stancelog forget a --reason "no longer relevant"
stancelog current a --at 2026-01-01T00:00:00Z [--known-at ...]
stancelog check [--repair]                       # exit code 1 if problems were found
stancelog rebuild
stancelog stats
stancelog serve-mcp                              # MCP stdio server
```

Add `--json` to any command for machine-readable output. `python -m stancelog` works the same way.

## MCP setup

The server speaks MCP (JSON-RPC 2.0 over stdio, protocol versions `2025-06-18`, `2025-03-26` and
`2024-11-05`) and exposes five tools:

| tool       | what it does |
|------------|--------------|
| `remember` | `text` (required); optional `subject`, `polarity` (-1/0/1), `source`, `confidence`, `valid_from` |
| `recall`   | `query`, `k`, `include_superseded` |
| `current`  | `subject` (omit it to list every current belief), optional `at` |
| `history`  | `subject`, returns versions plus their `supersedes`/`contradicts` relations |
| `forget`   | `target` (subject or belief id), `reason` |

### Claude Code

```bash
# user-wide memory
claude mcp add stancelog -- stancelog serve-mcp

# per-project memory (writes .mcp.json; relative paths resolve against the project directory)
claude mcp add --scope project stancelog -- stancelog serve-mcp --db .stancelog/memory.db
```

Run `claude mcp list` to confirm that the server connects.

Or write `.mcp.json` in the project root by hand:

```json
{
  "mcpServers": {
    "stancelog": {
      "command": "stancelog",
      "args": ["serve-mcp"],
      "env": { "STANCELOG_DB": ".stancelog/memory.db" }
    }
  }
}
```

### Cursor

Put this in `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (per project):

```json
{
  "mcpServers": {
    "stancelog": {
      "command": "stancelog",
      "args": ["serve-mcp", "--db", "~/.stancelog/memory.db"]
    }
  }
}
```

[`examples/mcp_config.json`](examples/mcp_config.json) has a ready-to-copy config. If `stancelog`
is installed in a virtualenv, set `command` to the full path, for example
`/path/to/.venv/bin/stancelog`. Any number of MCP servers, CLIs and scripts can share one database
file at the same time.

## Extractors

An extractor turns text into `Claim(subject, polarity, value, confidence)` objects. Anything with
an `extract(text) -> list[Claim]` method works:

```python
from stancelog import Memory, Claim, LLMExtractor

# Built-in (default): HeuristicExtractor, regex based and deterministic.
#  -1: "don't do A", "never use X", "avoid A", "stop using X", "A is bad", "I hate X", "no X"
#  +1: "use A", "prefer A", "maybe A is better", "A is good", "always X", "we should X"
#  +A/-B: "prefer A over B", "use A instead of B", "A is better than B"
#   0: "editor: vim", "my editor is vim"; unmatched text becomes a neutral note
# Hedges ("maybe", "I think", "might") lower confidence to 0.6.
# Subjects are normalised: "A", "a" and "the A!" all map to "a".
# Extracted subjects are then matched against existing ones (see "Subject matching").

# LLM-backed: you supply the callable. stancelog never calls a network or needs an API key.
def my_llm(prompt: str) -> str:
    return my_client.complete(prompt)  # must return JSON: [{"subject", "polarity", "value", "confidence"}]

m = Memory("memory.db", extractor=LLMExtractor(my_llm))
```

`LLMExtractor` adds the store's existing subjects to the prompt, so the model can map a
paraphrase ("the first option") onto an existing key. This lets contradiction detection work even
when the wording changes. If the model output can't be parsed, or the callable raises, it falls
back to the heuristic extractor.

## Design notes

### Event log and derived view

```
events (source of truth, append-only)         beliefs / relations / beliefs_fts (derived)
seq | id | kind | subject | body | ...  ──►    replay(events of subject) ──► rows
      prev_hash | hash                         rebuild() = drop + replay everything
```

- **`events`** is the only table that matters. SQLite triggers make it append-only: `UPDATE`
  and `DELETE` abort. Every event stores `hash = sha256(prev_hash, seq, id, kind, subject,
  recorded_at, body)`. A modified, missing or reordered event therefore breaks verification of
  everything after it, and each journal line verifies itself.
- **`beliefs`**, **`relations`** and **`beliefs_fts`** are a cache. They come from one pure
  function (`stancelog.timeline.compute_timeline`) and are refreshed for the affected subject in
  the same transaction as each event. `rebuild()` recomputes them from scratch, and `check()`
  compares them against a fresh replay.
- Event kinds are `assert` (a stance) and `forget` (a soft delete of one belief or a subject).

### Bi-temporal model

Each belief carries both **valid time** (`valid_from`/`valid_to`, when the statement holds in the
world) and **transaction time** (`recorded_at`, when we learned it). `valid_from` defaults to
`recorded_at`, but you can backdate it (`remember(..., valid_from=...)`). Valid-time intervals are
chained over the active beliefs: each belief is valid until the next one starts. A backdated fact
slots into the past and does not override a newer belief. `as_of(subject, at)` queries valid time.
`as_of(subject, at, known_at=T)` replays only the events recorded by `T`, which answers "what did
we believe back then?".

### Contradictions

A new `assert` on subject *S* supersedes its predecessor, the active belief immediately before it
in valid time. The predecessor's `valid_to` closes and both `supersedes` and `superseded_by` are
set. If the polarities are opposite (+1 against -1), a `contradicts` relation is added as well.
Same-polarity updates and neutral value changes only supersede. Relations are historical facts and
never change afterwards. Forgetting the newest belief by id makes the previous one current again,
while forgetting a subject hides all of its beliefs.

### Subject matching

Different phrasings of the same stance often extract to slightly different subjects:
"don't use Redis for the cache" gives `redis for the cache`, while "maybe Redis is better for the
cache" gives `redis`. Before an extracted claim is stored, `Memory` matches it against the subjects
that have active beliefs, and if one matches, the claim joins that subject's timeline. That makes
the second statement supersede and contradict the first.

Matching is deterministic (`stancelog.subject_similarity`). Both subjects are reduced to content
tokens: stopwords, articles, prepositions, generic verbs such as "use" and judgement words such as
"better" are dropped, and light stemming is applied, so "deploying on Fridays" matches "deploy on
Fridays". Two subjects then match if:

- their token sets are equal, or
- one token set contains the other and the Jaccard overlap is at least 0.5 (`redis` matches
  `redis for the cache`, but `redis` does not match `redis cluster config`), or
- neither set contains the other and the Jaccard overlap is at least 0.7;
- and in every case the sets do not differ in a token that contains a digit (`python` does not
  match `python 3`).

So `redis` and `postgres`, `tabs` and `spaces`, or `tabs for indentation` and `spaces for
indentation` never merge. An exact live subject always wins. When several subjects match, the
highest similarity wins, then the most recently written subject, then alphabetical order. Two
claims from the same message are never merged into each other. When a claim is matched, the original
extracted subject is kept in `metadata["matched_from"]`. The chosen subject is written into the
event itself, so `rebuild()` replays it exactly.

Lookups (`current`, `history`, `relations`, `events`, `forget`, and the CLI/MCP equivalents) use
the same matching when no exact subject exists, so `current("redis")` finds `redis for the cache`.
Passing an explicit `subject=` to `remember()` bypasses matching, and
`Memory(..., subject_matching=False)` turns it off entirely.

### Recall ranking

FTS5 (porter stemming, subject weighted 2×) is used when the SQLite build has it. Otherwise a
pure-Python BM25 with light stemming takes over. Both produce the same blended score:

```
score = ((1 - w) * bm25 / max_bm25  +  w * 0.5 ** (age / half_life)) * (0.75 + 0.25 * confidence)
```

The defaults are `w = 0.3` and `half_life = 30 days`, and both are configurable through `Memory(...)`.
By default only current beliefs are returned. With `include_superseded=True` (CLI `--all`),
superseded and forgotten beliefs are included with their `status` set. Their score is multiplied by
`superseded_penalty` (default 0.5) and capped just below the current belief of the same subject, so
a subject's current belief always ranks above its own history.

### Durability guarantees and limits

"Survives any issues" means the following specific, tested guarantees:

| Failure | What happens | Test |
|---|---|---|
| **`kill -9` / crash mid-write** | Each `remember`/`forget` is one `BEGIN IMMEDIATE … COMMIT` transaction. A write is either fully present (event, derived rows and journal line) or fully absent. There are no partial events, and every write whose call returned is present after reopening. No recovery step is needed. | `test_kill9_mid_write_loses_nothing_committed` |
| **Power loss / OS crash** | `journal_mode=WAL` with `synchronous=FULL` fsyncs the WAL on every commit, and the journal line is fsync'd before `COMMIT`. A returned write survives on hardware that honours fsync. (`NORMAL` would also be corruption-safe in WAL mode, but it can drop the last few commits on power loss. Memory writes are rare and small, so FULL costs little.) | — |
| **Concurrent writers** (threads and processes) | SQLite's write lock (`BEGIN IMMEDIATE`, 30 s `busy_timeout` plus retry with backoff) serialises writers, so seq numbers stay gap-free and the hash chain linear. Readers never block in WAL mode. | `test_processes_and_threads_concurrently` |
| **Corrupted DB file** (garbled header, truncated, zeroed, deleted, damaged pages) | On open, `PRAGMA quick_check` runs, plus a comparison with the journal tail. If damage is found, the file and its `-wal`/`-shm` are **moved aside** to `memory.db.corrupt-<timestamp>` and kept for forensics. Hash-valid events are salvaged from the damaged file (a full scan, then a row-by-row probe), merged with the journal, and loaded into a fresh database whose derived view is rebuilt. The journal is rewritten cleanly, with a `.bak-` copy kept. | `test_corruption_recovery_from_journal[*]` |
| **DB rolled back** (an old copy restored) | The journal holds events the DB lacks, so it is merged back in. | `test_recovery_merges_partial_db_and_journal` |
| **Journal lost or truncated** | A healthy DB regenerates the journal. | `test_recovery_salvages_db_when_journal_is_lost` |
| **Torn journal line / crash between journal fsync and COMMIT** | Torn lines fail their hash and are skipped, and the next writer never appends onto a torn line. The uncommitted "ghost" line is neutralised with a `void` record on the next open, so recovery never resurrects it. | `test_ghost_journal_entry_is_voided_not_resurrected` |
| **Derived view drift** | `check()` detects it and `rebuild()` fixes it. | `test_check_detects_derived_drift` |

Limits:

- **Both copies lost.** If the DB and the journal are both destroyed, the data is gone. They are
  separate files in the same directory by default, so back up that directory, or put the journal
  on another disk with `Memory(path, journal="/other/disk/memory.journal.jsonl")`.
- **Damage in both places.** When some events are lost from both the DB and the journal, recovery
  keeps everything else. The gap shows up as an acknowledged hash-chain break (`chain_breaks` in
  the recovery report), and is not hidden.
- **A narrow ghost window.** One write was in flight at the crash, it was journaled but never
  committed, and then the DB was corrupted *before any later open*. In that case recovery may
  include that one write. In practice this is an extra copy of a write the caller never saw
  confirmed.
- **Recovery while other processes are active.** Recovery takes a cross-process lock file
  (`flock` / `msvcrt`) and assumes no other process is still writing to the damaged file. A
  process that keeps a corrupted file open needs to be restarted.
- **Corruption that `quick_check` misses.** Damage confined to free pages can go undetected,
  because `quick_check` is O(N) but less thorough than `integrity_check`. Run `stancelog check`
  for a full verification, including hashes and the derived view.
- **Unreliable storage.** Network filesystems (NFS, SMB) break SQLite's locking guarantees, and
  hardware that lies about fsync breaks the power-loss guarantee. Use a local disk.
- **Growth.** The log only grows. Forget is soft by design, so there is no compaction yet.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Contributing

Bug reports and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup,
the design invariants every change must keep, and PR guidelines.

## License

MIT © 2026 Deep Mehta
