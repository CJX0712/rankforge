"""conftest.py — ensure the ``rankforge`` package is importable in tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# project root (the ``rankforge`` package dir) and its parent (workspace)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
