import numpy as np
import torch
from mindspore import ops, Tensor
from mindnlp._legacy.functional import einsum_label_to_index, sumproduct_pair, einsum

import mindspore
mindspore.set_context(pynative_synchronize=True)
def test_einsum_label_to_index():
    assert einsum_label_to_index('a') == 26


def test_sumproduct_pair():
    left_ = ops.randn(5, 6, 7, 8, 1, 2, 3, 4)
    right_ = ops.randn(1, 1, 1, 8, 10, 2, 3, 4)
    sum_dims_ = [5, 6, 7]
    keep_dim_ = False
    out = sumproduct_pair(left_, right_, sum_dims_, keep_dim_)
    assert out.shape == (5, 6, 7, 8, 10)


def test_einsum():
    a1 = np.random.randn(1, 2, 3, 4, 5, 6)
    a2 = np.random.randn(1, 2, 3, 4, 7, 8, 9)
    a3 = np.random.randn(1, 2, 3, 4, 8, 10)

    equation = "abcdef,abcdghi,abcdhk->efghki"  # 5,6,7,8,10
    out = einsum(equation, *(Tensor(a1), Tensor(a2), Tensor(a3)))
    pt_out = torch.einsum(equation, torch.tensor(
        a1), torch.tensor(a2), torch.tensor(a3))
    assert out.shape == pt_out.shape
    print(out.shape, (pt_out.detach().numpy()-out.asnumpy()).sum())
    print(pt_out.detach().numpy()-out.asnumpy())
    assert np.allclose(out.asnumpy(), pt_out.detach().numpy(), 1e-4)
