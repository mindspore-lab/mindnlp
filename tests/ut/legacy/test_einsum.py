import numpy as np
from mindspore import ops, Tensor
from mindnlp._legacy.functional import einsum_label_to_index, sumproduct_pair, einsum

def test_einsum_label_to_index():
    assert einsum_label_to_index('a') == 26


def test_sumproduct_pair():
    left_ = ops.randn(5, 6, 7, 8, 1, 2, 3, 4)
    right_ = ops.randn(1, 1, 1, 8, 10, 2, 3, 4)
    sum_dims_ = [5, 6, 7]
    keep_dim_ = False
    out = sumproduct_pair(left_, right_, sum_dims_, keep_dim_)
    assert out.shape == (5, 6, 7, 8, 10)

def test_einsum_two_operand():
    a1 = np.random.randn(1, 2, 5, 6)
    a2 = np.random.randn(1, 2, 7, 8, 9)

    equation = "abef,abghi->egh" # 5,6,7,8,10
    out = einsum(equation, Tensor(a1), Tensor(a2))
    np_out = np.einsum(equation, a1, a2)
    assert out.shape == np_out.shape
    assert np.abs(out.asnumpy() - np_out).sum() / out.size < 5e-3
