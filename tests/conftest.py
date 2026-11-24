import os
import subprocess
import sys

import pytest

from stancelog import Memory


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "mem.db")


@pytest.fixture
def mem(db_path):
    m = Memory(db_path)
    yield m
    m.close()


@pytest.fixture(params=[True, False], ids=["fts5", "python-bm25"])
def any_search_mem(request, db_path):
    m = Memory(db_path, use_fts=request.param)
    yield m
    m.close()


def run_py(code: str, *args: str, timeout: float = 60, **kw) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("STANCELOG_DB", None)
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True,
                          timeout=timeout, env=env, **kw)
