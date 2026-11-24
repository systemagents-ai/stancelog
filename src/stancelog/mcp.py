"""Minimal, dependency-free MCP server over stdio (JSON-RPC 2.0, newline-delimited).

Implements the subset of the Model Context Protocol needed for tools:
``initialize``, ``notifications/initialized``, ``ping``, ``tools/list`` and
``tools/call``. Tools: ``remember``, ``recall``, ``current``, ``history``,
``forget``.

stdout carries protocol messages only; all logging goes to stderr.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, BinaryIO, Callable

from . import __version__
from .store import Belief, Memory

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = (
    -32700, -32600, -32601, -32602, -32603,
)

INSTRUCTIONS = (
    "stancelog is a durable memory of the user's preferences, decisions and facts. "
    "Call `remember` whenever the user states a preference, rule, decision or fact worth keeping "
    "(e.g. \"don't use X\", \"prefer Y\", \"our deploy target is Z\"). Newer statements on the same "
    "subject supersede older ones; opposite stances are recorded as contradictions and the full "
    "history is kept. Call `recall` (or `current` for a known subject) before acting on something "
    "the user may have an opinion about. Use `history` to explain how a stance changed, and "
    "`forget` only when the user asks to forget something."
)

_BELIEF_NOTE = "Each belief has: id, subject, stance (positive/negative/neutral), polarity, value, valid_from/valid_to, supersedes, contradicts."

TOOLS: list[dict[str, Any]] = [
    {
        "name": "remember",
        "title": "Remember a stance or fact",
        "description": (
            "Store a durable memory. Pass the user's statement as `text`; the subject and stance "
            "(polarity +1 endorse / -1 reject / 0 neutral fact) are extracted automatically, or can be "
            "given explicitly. A newer memory on the same subject supersedes the older one; an opposite "
            "stance is linked as a contradiction. History is never lost."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The statement to remember, e.g. \"don't use tabs\"."},
                "subject": {"type": "string", "description": "Optional explicit subject key (e.g. \"tabs\")."},
                "polarity": {"type": "integer", "enum": [-1, 0, 1],
                             "description": "Optional explicit stance: 1 endorse, -1 reject, 0 neutral."},
                "source": {"type": "string", "description": "Who/what said it (default \"mcp\")."},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "valid_from": {"type": "string", "description": "ISO-8601 time the statement became true (default: now)."},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
    },
    {
        "name": "recall",
        "title": "Search memories",
        "description": "Search memories relevant to a query, ranked by relevance and recency. "
                       "Returns current beliefs only unless include_superseded is true. " + _BELIEF_NOTE,
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 100, "default": 5},
                "include_superseded": {"type": "boolean", "default": False},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "current",
        "title": "Current belief",
        "description": "The current belief about a subject (or, with no subject, every current belief). "
                       + _BELIEF_NOTE,
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "at": {"type": "string", "description": "Optional ISO-8601 valid time for time travel."},
                "limit": {"type": "integer", "minimum": 1, "default": 50},
            },
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "history",
        "title": "Belief history",
        "description": "Every recorded belief about a subject in time order, including superseded, "
                       "contradicted and forgotten ones. " + _BELIEF_NOTE,
        "inputSchema": {
            "type": "object",
            "properties": {"subject": {"type": "string"}},
            "required": ["subject"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "forget",
        "title": "Forget (soft)",
        "description": "Soft-forget a belief by id, or a whole subject. The forget is logged as an event; "
                       "history is preserved and flagged.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Belief id or subject."},
                "reason": {"type": "string"},
            },
            "required": ["target"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True},
    },
]


def _compact(b: Belief) -> dict[str, Any]:
    d = b.to_dict()
    for k in ("text", "metadata", "forgotten_at", "forgotten_reason", "superseded_by", "seq"):
        if k == "text" and d.get("text") != d.get("value"):
            continue
        if k in ("forgotten_at", "forgotten_reason") and d.get("forgotten"):
            continue
        d.pop(k, None)
    return d


class _ToolError(Exception):
    pass


class MCPServer:
    def __init__(self, memory: Memory, *, log: Callable[[str], None] | None = None) -> None:
        self.memory = memory
        self.log = log or (lambda msg: print(f"[stancelog-mcp] {msg}", file=sys.stderr, flush=True))
        self.initialized = False

    # ------------------------------------------------------------- tools
    def _tool(self, name: str, args: dict[str, Any]) -> Any:
        m = self.memory

        def req(key: str) -> Any:
            v = args.get(key)
            if v is None or (isinstance(v, str) and not v.strip()):
                raise _ToolError(f"missing required argument: {key}")
            return v

        if name == "remember":
            beliefs = m.remember(
                str(req("text")),
                subject=args.get("subject") or None,
                polarity=args.get("polarity"),
                source=str(args.get("source") or "mcp"),
                confidence=args.get("confidence"),
                valid_from=args.get("valid_from") or None,
            )
            out = []
            for b in beliefs:
                d = _compact(b)
                if b.contradicts:
                    d["note"] = f"contradicts and supersedes earlier belief {b.contradicts}"
                elif b.supersedes:
                    d["note"] = f"supersedes earlier belief {b.supersedes}"
                out.append(d)
            return {"remembered": out}
        if name == "recall":
            k = int(args.get("k") or 5)
            hits = m.recall(str(req("query")), k=max(1, min(k, 100)),
                            include_superseded=bool(args.get("include_superseded", False)))
            return {"results": [_compact(b) for b in hits]}
        if name == "current":
            subject = args.get("subject")
            if subject:
                b = m.current(str(subject), at=args.get("at") or None)
                return {"subject": subject, "belief": _compact(b) if b else None}
            return {"beliefs": [_compact(b) for b in m.all_current(limit=int(args.get("limit") or 50))]}
        if name == "history":
            hist = m.history(str(req("subject")))
            return {"subject": args["subject"], "history": [_compact(b) for b in hist],
                    "relations": m.relations(str(args["subject"]))}
        if name == "forget":
            gone = m.forget(str(req("target")), reason=str(args.get("reason") or ""), source="mcp")
            return {"forgotten": [b.id for b in gone], "count": len(gone)}
        raise KeyError(name)

    # ---------------------------------------------------------- protocol
    def handle(self, msg: Any) -> dict[str, Any] | None:
        """Handle one decoded JSON-RPC message; returns the response or None for notifications."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method", ""), str):
            if isinstance(msg, dict) and "method" not in msg and ("result" in msg or "error" in msg):
                return None  # a response to a server->client request; we never send any
            return _error(msg.get("id") if isinstance(msg, dict) else None, INVALID_REQUEST, "Invalid Request")
        method = msg.get("method")
        is_notification = "id" not in msg
        mid = msg.get("id")
        params = msg.get("params") or {}
        if method is None:
            return None if is_notification else _error(mid, INVALID_REQUEST, "missing method")
        if not isinstance(params, dict):
            return None if is_notification else _error(mid, INVALID_PARAMS, "params must be an object")

        if is_notification:
            if method == "notifications/initialized":
                self.initialized = True
            return None  # notifications/cancelled etc. are accepted and ignored

        try:
            if method == "initialize":
                requested = params.get("protocolVersion")
                version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION
                return _result(mid, {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "stancelog", "title": "stancelog memory", "version": __version__},
                    "instructions": INSTRUCTIONS,
                })
            if method == "ping":
                return _result(mid, {})
            if method == "tools/list":
                return _result(mid, {"tools": TOOLS})
            if method == "tools/call":
                name = params.get("name")
                args = params.get("arguments") or {}
                if not isinstance(name, str) or name not in {t["name"] for t in TOOLS}:
                    return _error(mid, INVALID_PARAMS, f"unknown tool: {name!r}")
                if not isinstance(args, dict):
                    return _error(mid, INVALID_PARAMS, "arguments must be an object")
                try:
                    payload = self._tool(name, args)
                except (_ToolError, ValueError, KeyError, TypeError) as e:
                    return _result(mid, {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True})
                text = json.dumps(payload, ensure_ascii=False, indent=2)
                return _result(mid, {"content": [{"type": "text", "text": text}], "isError": False})
            return _error(mid, METHOD_NOT_FOUND, f"Method not found: {method}")
        except Exception as e:  # never crash the server on a single bad request
            self.log("internal error: " + "".join(traceback.format_exception_only(type(e), e)).strip())
            return _error(mid, INTERNAL_ERROR, f"Internal error: {e}")

    def serve(self, stdin: BinaryIO, stdout: BinaryIO) -> None:
        self.log(f"serving {self.memory.path}")
        for raw in stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                self._send(stdout, _error(None, PARSE_ERROR, f"Parse error: {e}"))
                continue
            if isinstance(msg, list):  # JSON-RPC batch (2025-03-26 and earlier)
                if not msg:
                    self._send(stdout, _error(None, INVALID_REQUEST, "empty batch"))
                    continue
                responses = [r for r in (self.handle(m) for m in msg) if r is not None]
                if responses:
                    self._send(stdout, responses)
            else:
                resp = self.handle(msg)
                if resp is not None:
                    self._send(stdout, resp)

    @staticmethod
    def _send(stdout: BinaryIO, obj: Any) -> None:
        stdout.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
        stdout.flush()


def _result(mid: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _error(mid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def serve_stdio(db_path: str | None = None) -> None:
    """Entry point used by ``stancelog serve-mcp``."""
    out = sys.stdout.buffer
    # Anything that accidentally prints must not corrupt the protocol stream.
    sys.stdout = sys.stderr
    memory = Memory(db_path)
    if memory.recovery:
        print(f"[stancelog-mcp] recovered database: {json.dumps(memory.recovery)}", file=sys.stderr, flush=True)
    try:
        MCPServer(memory).serve(sys.stdin.buffer, out)
    except KeyboardInterrupt:
        pass
    finally:
        memory.close()
