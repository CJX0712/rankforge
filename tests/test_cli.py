import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_bench_smoke():
    # Run as a script from the project root (cli.py self-inserts the project root
    # into sys.path and uses top-level imports), which is robust regardless of cwd.
    r = subprocess.run(
        [sys.executable, "cli.py", "bench", "--queries", "10", "--seeds", "2"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "ranker" in r.stdout.lower()
