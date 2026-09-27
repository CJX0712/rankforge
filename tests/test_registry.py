"""注册表单测：auto 解析 / 未知后端报错 / 工厂构建。"""
import pytest

from rankforge.core.errors import BackendError
from rankforge.ltr.registry import available_backends, list_backends, make_backend, resolve_backends


def test_list_backends():
    names = list_backends()
    for n in ("random", "pointwise", "heuristic", "ranknet", "lambdamart", "xgboost"):
        assert n in names


def test_resolve_auto():
    assert resolve_backends(["auto"]) == list_backends()


def test_resolve_unknown_raises():
    with pytest.raises(BackendError):
        resolve_backends(["does_not_exist"])


def test_make_backend():
    m = make_backend("ranknet")
    assert m.name == "ranknet"


def test_available_backends_reports_flag():
    av = dict(available_backends())
    assert "ranknet" in av and av["ranknet"] is True
