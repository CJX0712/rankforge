"""RankNet — 离线旗舰排序模型（Burges et al., 2005）。

纯 numpy 实现的 pairwise 神经网络。两种模式：
- linear=True：score = X·w + b（线性成对排序器，等价于 RankSVM 类）。
- linear=False（默认）：输入 -> tanh 隐层 -> 标量得分（非线性，LambdaMART 前身）。

零下载、零重型依赖。全局确定性：固定 seed 下两次训练逐位一致。
"""
from __future__ import annotations

import numpy as np

from ..core.seed import set_all
from ..core.types import LTRDataset, RankResult
from .metrics import rank_result_from


class RankNet:
    name = "ranknet"
    available = True

    def __init__(
        self,
        hidden: int = 16,
        lr: float = 0.02,
        epochs: int = 30,
        l2: float = 1e-4,
        batch_pairs: int = 512,
        seed: int = 42,
        linear: bool = False,
    ) -> None:
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.batch_pairs = batch_pairs
        self.seed = seed
        self.linear = linear
        self.W1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = 0.0
        self.w_lin = None
        self.b_lin = 0.0

    def _collect_pairs(self, train: LTRDataset):
        Xs = []
        pairs = []
        pos = 0
        for q in train.queries:
            n = q.n_docs
            Xs.append(q.X)
            yi = np.asarray(q.y)
            for i in range(n):
                for j in range(n):
                    if yi[i] > yi[j]:
                        pairs.append((pos + i, pos + j))
            pos += n
        X = np.vstack(Xs).astype(np.float64)
        pairs = np.array(pairs, dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)
        return X, pairs

    def fit(self, train: LTRDataset, val: LTRDataset | None = None) -> RankNet:
        set_all(self.seed)
        rng = np.random.default_rng(self.seed)
        X, pairs = self._collect_pairs(train)
        n_pairs = pairs.shape[0]
        d = X.shape[1]
        if n_pairs == 0:
            if self.linear:
                self.w_lin = np.zeros(d)
                self.b_lin = 0.0
            else:
                self.W1 = np.zeros((d, max(self.hidden, 1)))
                self.b1 = np.zeros(max(self.hidden, 1))
                self.w2 = np.zeros(max(self.hidden, 1))
            return self

        if self.linear:
            self.w_lin = (rng.standard_normal(d) * np.sqrt(1.0 / d)).astype(np.float64)
            self.b_lin = 0.0
            for _ in range(self.epochs):
                perm = rng.permutation(n_pairs)
                for start in range(0, n_pairs, self.batch_pairs):
                    sel = perm[start : start + self.batch_pairs]
                    if sel.size == 0:
                        continue
                    ia, ib = pairs[sel, 0], pairs[sel, 1]
                    Xa, Xb = X[ia], X[ib]
                    sa = Xa @ self.w_lin + self.b_lin
                    sb = Xb @ self.w_lin + self.b_lin
                    sig = 1.0 / (1.0 + np.exp(-(sa - sb)))
                    dd = sig - 1.0
                    g_w = Xa.T @ dd + Xb.T @ (-dd) + self.l2 * self.w_lin
                    self.w_lin -= self.lr * g_w / max(sel.size, 1)
            return self

        # 非线性模式
        h = self.hidden
        self.W1 = (rng.standard_normal((d, h)) * np.sqrt(2.0 / (d + h))).astype(np.float64)
        self.b1 = np.zeros(h, dtype=np.float64)
        self.w2 = (rng.standard_normal(h) * np.sqrt(2.0 / (h + 1))).astype(np.float64)
        self.b2 = 0.0
        for _ in range(self.epochs):
            perm = rng.permutation(n_pairs)
            for start in range(0, n_pairs, self.batch_pairs):
                sel = perm[start : start + self.batch_pairs]
                if sel.size == 0:
                    continue
                ia, ib = pairs[sel, 0], pairs[sel, 1]
                Xa, Xb = X[ia], X[ib]
                ha = np.tanh(Xa @ self.W1 + self.b1)
                hb = np.tanh(Xb @ self.W1 + self.b1)
                sa = ha @ self.w2 + self.b2
                sb = hb @ self.w2 + self.b2
                sig = 1.0 / (1.0 + np.exp(-(sa - sb)))
                dd = sig - 1.0
                g_w2 = ha.T @ dd + hb.T @ (-dd)
                da = dd[:, None] * (1.0 - ha**2) * self.w2[None, :]
                db = (-dd)[:, None] * (1.0 - hb**2) * self.w2[None, :]
                g_W1 = Xa.T @ da + Xb.T @ db
                g_b1 = da.sum(axis=0) + db.sum(axis=0)
                g_W1 += self.l2 * self.W1
                g_w2 += self.l2 * self.w2
                self.W1 -= self.lr * g_W1 / max(sel.size, 1)
                self.b1 -= self.lr * g_b1 / max(sel.size, 1)
                self.w2 -= self.lr * g_w2 / max(sel.size, 1)
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        if self.linear:
            return X @ self.w_lin + self.b_lin
        return np.tanh(X @ self.W1 + self.b1) @ self.w2 + self.b2

    def predict(self, data: LTRDataset) -> np.ndarray:
        X, _, _ = data.Xy_group()
        return self._score(X)

    def evaluate(self, data: LTRDataset, ks) -> RankResult:
        _, y, g = data.Xy_group()
        return rank_result_from(self.name, y, self.predict(data), g, ks)
