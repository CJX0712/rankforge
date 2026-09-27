"""错误码体系（E100~E500）。每条错误带稳定 code 便于定位。"""
from __future__ import annotations


class RankForgeError(Exception):
    """基类错误。"""

    code = "E000"


class ConfigError(RankForgeError):
    code = "E100"


class DataError(RankForgeError):
    code = "E200"


class BackendError(RankForgeError):
    code = "E300"


class MetricError(RankForgeError):
    code = "E400"


class PipelineError(RankForgeError):
    code = "E500"
