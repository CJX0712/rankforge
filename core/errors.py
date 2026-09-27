"""core/errors.py — RankForge error taxonomy (E100~E500).

Every error carries a stable machine code so logs and CI assertions can key on it.
"""
from __future__ import annotations


class RankForgeError(Exception):
    """Base class. Subclasses set their own ``code``."""

    code = "E000"

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        self.code = code or self.__class__.code
        super().__init__(f"[{self.code}] {message}")


class ConfigError(RankForgeError):
    code = "E100"


class DataError(RankForgeError):
    code = "E200"


class DataLeakError(RankForgeError):
    """Raised when preprocessing/HPO would touch evaluation data (leakage guard)."""

    code = "E210"


class BackendUnavailableError(RankForgeError):
    code = "E300"


class MetricError(RankForgeError):
    code = "E400"


class PipelineError(RankForgeError):
    code = "E500"
