"""全局确定性入口。唯一 seed 来源，一次性设齐 numpy / random / 库级种子。"""
from __future__ import annotations

import os
import random

import numpy as np

_SEEDED = False


def set_all(seed: int) -> None:
    """设齐所有随机源，保证同 seed 可复现。"""
    global _SEEDED
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
    except Exception:  # noqa: BLE001, S110
        pass
    _SEEDED = True


def is_seeded() -> bool:
    return _SEEDED
