"""Command-line interface: ``stancelog [--db PATH] <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from . import __version__
from .store import ENV_DB, Belief, Memory, default_db_path

_MARK = {1: "+", -1: "-", 0: "="}


def _fmt_belief(b: Belief, *, show_links: bool = True) -> str:
    flags = []
    if b.forgotten:
        flags.append(f"FORGOTTEN({b.forgotten_reason or ''})".replace("()", ""))
    elif b.valid_to is None:
        flags.append("CURRENT")
    if show_links and b.contradicts:
        flags.append(f"contradicts {b.contradicts[:8]}")
    elif show_links and b.supersedes:
        flags.append(f"supersedes {b.supersedes[:8]}")
    if b.score is not None:
        flags.append(f"score={b.score:.3f}")
    until = b.valid_to or "now"
    return (
        f"[{_MARK[b.polarity]}] {b.subject}: {b.value}\n"
        f"    id={b.id[:12]}  {b.valid_from} -> {until}  source={b.source}  conf={b.confidence:g}"
        + (f"\n    {'  '.join(flags)}" if flags else "")
    )


def _emit(args: argparse.Namespace, data: Any, human: str) -> None:
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(human)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=argparse.SUPPRESS,
                        help=f"database path (default: ${ENV_DB} or ~/.stancelog/memory.db)")
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="machine-readable output")

    p = argparse.ArgumentParser(
        prog="stancelog", parents=[common],
        description="Crash-proof, contradiction-aware memory for AI agents.",
    )
    p.add_argument("--version", action="version", version=f"stancelog {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="<command>")

    r = sub.add_parser("remember", parents=[common], help="record a stance or fact")
    r.add_argument("text", help='e.g. "don\'t use tabs"')
    r.add_argument("--subject", "-s")
    r.add_argument("--polarity", "-p", type=int, choices=(-1, 0, 1))
    r.add_argument("--source", default="cli")
    r.add_argument("--confidence", type=float)
    r.add_argument("--valid-from", help="ISO-8601 time the statement became true")

    q = sub.add_parser("recall", parents=[common], help="search memories (relevance + recency)")
    q.add_argument("query")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--all", action="store_true", help="include superseded beliefs")

    c = sub.add_parser("current", parents=[common], help="current belief about a subject (or all)")
    c.add_argument("subject", nargs="?")
    c.add_argument("--at", help="valid time (ISO-8601) for time travel")
    c.add_argument("--known-at", help="transaction time: only use what was recorded by then")

    h = sub.add_parser("history", parents=[common], help="full history of a subject")
    h.add_argument("subject")

    f = sub.add_parser("forget", parents=[common], help="soft-forget a belief id or a subject")
    f.add_argument("target")
    f.add_argument("--reason", default="")

    sub.add_parser("rebuild", parents=[common], help="recompute the derived view from the event log")
    ck = sub.add_parser("check", parents=[common], help="verify integrity (exit 1 if not ok)")
    ck.add_argument("--repair", action="store_true", help="run a full recovery if the check fails")
    sub.add_parser("stats", parents=[common], help="counts and paths")
    sub.add_parser("serve-mcp", parents=[common], help="run the MCP stdio server")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.db = getattr(args, "db", None) or default_db_path()
    args.json = getattr(args, "json", False)

    if args.command == "serve-mcp":
        from .mcp import serve_stdio

        serve_stdio(args.db)
        return 0

    try:
        mem = Memory(args.db)
    except Exception as e:
        print(f"stancelog: cannot open {args.db}: {e}", file=sys.stderr)
        return 2
    if mem.recovery and not args.json:
        print(f"note: database was recovered ({mem.recovery['events_recovered']} events); "
              f"backup: {mem.recovery['backup']}", file=sys.stderr)
    try:
        return _run(args, mem)
    except (ValueError, KeyError) as e:
        print(f"stancelog: {e}", file=sys.stderr)
        return 1
    finally:
        mem.close()


def _run(args: argparse.Namespace, mem: Memory) -> int:
    cmd = args.command
    if cmd == "remember":
        bs = mem.remember(args.text, subject=args.subject, polarity=args.polarity, source=args.source,
                          confidence=args.confidence, valid_from=args.valid_from)
        _emit(args, [b.to_dict() for b in bs], "\n".join("remembered " + _fmt_belief(b) for b in bs))
    elif cmd == "recall":
        hits = mem.recall(args.query, k=args.k, include_superseded=args.all)
        _emit(args, [b.to_dict() for b in hits],
              "\n".join(_fmt_belief(b) for b in hits) or "(no matching memories)")
    elif cmd == "current":
        if args.subject:
            if args.known_at:
                b = mem.as_of(args.subject, args.at or args.known_at, known_at=args.known_at)
            else:
                b = mem.current(args.subject, at=args.at)
            _emit(args, b.to_dict() if b else None, _fmt_belief(b) if b else f"(no current belief about {args.subject!r})")
            return 0 if b else 1
        bs = mem.all_current()
        _emit(args, [b.to_dict() for b in bs], "\n".join(_fmt_belief(b, show_links=False) for b in bs) or "(empty)")
    elif cmd == "history":
        bs = mem.history(args.subject)
        _emit(args, {"history": [b.to_dict() for b in bs], "relations": mem.relations(args.subject)},
              "\n".join(_fmt_belief(b) for b in bs) or f"(no history for {args.subject!r})")
    elif cmd == "forget":
        gone = mem.forget(args.target, reason=args.reason, source="cli")
        _emit(args, {"forgotten": [b.id for b in gone]},
              f"forgot {len(gone)} belief(s)" if gone else "nothing to forget")
        return 0 if gone else 1
    elif cmd == "rebuild":
        n = mem.rebuild()
        _emit(args, {"beliefs": n}, f"rebuilt derived view: {n} beliefs")
    elif cmd == "check":
        rep = mem.check()
        if not rep["ok"] and args.repair:
            rec = mem.recover(reason="check --repair")
            rep = mem.check()
            rep["repair"] = rec
        lines = [f"{'OK' if rep['ok'] else 'PROBLEMS FOUND'}: {rep['events']} events, {rep['beliefs']} beliefs",
                 f"  sqlite quick_check: {', '.join(rep['quick_check'])}",
                 f"  bad hashes: {rep['bad_hashes'] or 'none'}; chain breaks: {rep['chain_breaks'] or 'none'}",
                 f"  derived view mismatches: {rep['derived_mismatch'] or 'none'}"]
        if "journal" in rep:
            j = rep["journal"]
            lines.append(f"  journal: {j['events']} events, {j['bad_lines']} torn/bad lines, "
                         f"{j['missing_from_journal']} missing, {j['not_in_database']} not in db")
        _emit(args, rep, "\n".join(lines))
        return 0 if rep["ok"] else 1
    elif cmd == "stats":
        s = mem.stats()
        _emit(args, s, "\n".join(f"{k}: {v}" for k, v in s.items()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
