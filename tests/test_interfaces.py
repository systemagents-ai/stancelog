"""MCP stdio round-trip and CLI smoke tests (real subprocesses)."""

import json
import os
import subprocess
import sys

import pytest


def _env(db):
    env = dict(os.environ)
    env["STANCELOG_DB"] = db
    return env


class MCPClient:
    def __init__(self, db):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "stancelog", "serve-mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=_env(db),
        )
        self.next_id = 0

    def send(self, obj):
        self.proc.stdin.write(json.dumps(obj).encode() + b"\n")
        self.proc.stdin.flush()

    def request(self, method, params=None):
        self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        resp = json.loads(self.proc.stdout.readline())
        assert resp["id"] == self.next_id
        return resp

    def call(self, name, **arguments):
        resp = self.request("tools/call", {"name": name, "arguments": arguments})
        res = resp["result"]
        assert res["content"][0]["type"] == "text"
        return res, (None if res["isError"] else json.loads(res["content"][0]["text"]))

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(10)
        return self.proc.stderr.read().decode()


@pytest.fixture
def mcp(db_path):
    c = MCPClient(db_path)
    yield c
    c.close()


def test_mcp_round_trip(mcp, db_path):
    init = mcp.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": "pytest", "version": "1"}})
    r = init["result"]
    assert r["protocolVersion"] == "2025-06-18"
    assert r["serverInfo"]["name"] == "stancelog"
    assert "tools" in r["capabilities"]
    mcp.send({"jsonrpc": "2.0", "method": "notifications/initialized"})  # no response expected

    assert mcp.request("ping")["result"] == {}

    tools = mcp.request("tools/list")["result"]["tools"]
    assert {t["name"] for t in tools} == {"remember", "recall", "current", "history", "forget"}
    for t in tools:
        assert t["inputSchema"]["type"] == "object"

    _, out = mcp.call("remember", text="Don't do A")
    assert out["remembered"][0]["stance"] == "negative"
    _, out = mcp.call("remember", text="Maybe A is better")
    assert out["remembered"][0]["stance"] == "positive"
    assert "contradicts" in out["remembered"][0]["note"]

    _, out = mcp.call("current", subject="A")
    assert out["belief"]["stance"] == "positive"
    _, out = mcp.call("history", subject="a")
    assert [b["stance"] for b in out["history"]] == ["negative", "positive"]
    assert {r["kind"] for r in out["relations"]} == {"supersedes", "contradicts"}
    _, out = mcp.call("recall", query="A", k=3)
    assert len(out["results"]) == 1
    _, out = mcp.call("current")
    assert len(out["beliefs"]) == 1
    _, out = mcp.call("forget", target="a", reason="test")
    assert out["count"] == 2
    _, out = mcp.call("current", subject="a")
    assert out["belief"] is None

    # errors
    res, _ = mcp.call("remember")
    assert res["isError"]
    assert mcp.request("tools/call", {"name": "nope", "arguments": {}})["error"]["code"] == -32602
    assert mcp.request("does/not/exist")["error"]["code"] == -32601
    mcp.proc.stdin.write(b"{not json\n")
    mcp.proc.stdin.flush()
    assert json.loads(mcp.proc.stdout.readline())["error"]["code"] == -32700
    assert mcp.request("ping")["result"] == {}  # still alive

    stderr = mcp.close()
    assert "serving" in stderr

    # The server's writes are durable and visible to the library.
    from stancelog import Memory

    with Memory(db_path) as m:
        assert [e["kind"] for e in m.events("a")] == ["assert", "assert", "forget"]


def test_mcp_version_negotiation_and_batch(mcp):
    r = mcp.request("initialize", {"protocolVersion": "1999-01-01", "capabilities": {}})["result"]
    assert r["protocolVersion"] == "2025-06-18"
    mcp.send([{"jsonrpc": "2.0", "id": 100, "method": "ping"},
              {"jsonrpc": "2.0", "method": "notifications/initialized"},
              {"jsonrpc": "2.0", "id": 101, "method": "tools/list"}])
    batch = json.loads(mcp.proc.stdout.readline())
    assert [x["id"] for x in batch] == [100, 101]


def cli(db, *args, check=True):
    p = subprocess.run([sys.executable, "-m", "stancelog", *args], capture_output=True, text=True,
                       env=_env(db), timeout=60)
    if check:
        assert p.returncode == 0, (p.stdout, p.stderr)
    return p


def test_cli_smoke(db_path, tmp_path):
    out = cli(db_path, "remember", "don't do A").stdout
    assert "[-] a" in out
    cli(db_path, "remember", "maybe A is better")
    assert "[+] a" in cli(db_path, "current", "a").stdout

    hist = json.loads(cli(db_path, "history", "A", "--json").stdout)
    assert [b["stance"] for b in hist["history"]] == ["negative", "positive"]
    assert hist["history"][1]["contradicts"] == hist["history"][0]["id"]

    hits = json.loads(cli(db_path, "recall", "A", "--json").stdout)
    assert len(hits) == 1 and hits[0]["polarity"] == 1

    assert "OK" in cli(db_path, "check").stdout
    assert "rebuilt" in cli(db_path, "rebuild").stdout
    assert cli(db_path, "forget", "a", "--reason", "cli test").returncode == 0
    assert cli(db_path, "current", "a", check=False).returncode == 1
    assert cli(db_path, "forget", "a", check=False).returncode == 1  # nothing left

    # --db overrides the env var and works before or after the subcommand
    other = str(tmp_path / "other.db")
    cli(db_path, "--db", other, "remember", "use B")
    cli(db_path, "current", "b", "--db", other)
    assert cli(db_path, "current", "b", check=False).returncode == 1

    stats = json.loads(cli(db_path, "stats", "--json").stdout)
    assert stats["events"] == 3


def test_console_script_installed():
    import shutil

    exe = shutil.which("stancelog", path=os.path.dirname(sys.executable))
    if exe is None:
        pytest.skip("console script not installed in this environment")
    p = subprocess.run([exe, "--version"], capture_output=True, text=True)
    assert p.returncode == 0 and "stancelog" in p.stdout
