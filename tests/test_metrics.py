"""指标单测：NDCG / MAP 正确性。"""
import numpy as np

from rankforge.ltr.metrics import average_precision, ndcg_at_k


def test_ndcg_ideal_is_one():
    y = np.array([3.0, 2.0, 1.0, 0.0])
    pred = np.array([100.0, 50.0, 10.0, 0.0])  # 与真实序一致
    assert abs(ndcg_at_k(y, pred, [4], 4) - 1.0) < 1e-9


def test_ndcg_reversed_is_low():
    y = np.array([3.0, 2.0, 1.0, 0.0])
    pred = np.array([0.0, 10.0, 50.0, 100.0])  # 倒序
    v = ndcg_at_k(y, pred, [4], 4)
    assert v < 0.7


def test_ndcg_per_query_independent():
    y = np.array([3.0, 0.0, 2.0, 1.0])
    pred = np.array([1.0, 0.0, 1.0, 0.0])
    # query0: [3,0] pred[1,0]->order[0,1] idcg=dcg ->1.0
    # query1: [2,1] pred[1,0]->order[0,1] ->1.0
    assert abs(ndcg_at_k(y, pred, [2, 2], 2) - 1.0) < 1e-9


def test_map_basic():
    y = np.array([1.0, 0.0, 1.0])
    pred = np.array([5.0, 1.0, 3.0])  # order [0,2,1]
    v = average_precision(y, pred, [3])
    # 相关文档 at rank1, rank2 -> AP = (1/1 + 2/2)/2 = 1.0
    assert abs(v - 1.0) < 1e-9
