import numpy as np
import pytest
import mindspore

from mindnlp.wizard.merge.merge_methods.multislerp import multislerp
from mindnlp.wizard.merge.merge_methods.nuslerp import nuslerp
from mindnlp.wizard.merge.merge_methods.ram import ram_merge
from mindnlp.wizard.merge.merge_methods.sce import sce_merge
from mindnlp.wizard.merge.sparsify import RescaleNorm, della_magprune, rescaled_masked_tensor


def _tensor(values, dtype):
    return mindspore.Tensor(np.array(values), dtype=dtype)


@pytest.mark.parametrize("dtype", [mindspore.bfloat16, mindspore.float16, mindspore.float32])
def test_sce_merge_dtype_matrix(dtype):
    base = _tensor([1.0, 2.0, 3.0, 4.0], dtype)
    t1 = _tensor([1.1, 2.2, 2.9, 4.1], dtype)
    t2 = _tensor([0.9, 1.8, 3.2, 3.8], dtype)
    out = sce_merge([t1, t2], base, select_topk=0.75)
    assert out.dtype == dtype
    assert np.isfinite(out.astype(mindspore.float32).asnumpy()).all()


@pytest.mark.parametrize("dtype", [mindspore.bfloat16, mindspore.float16, mindspore.float32])
def test_ram_merge_dtype_matrix(dtype):
    base = _tensor([1.0, 2.0, 3.0, 4.0], dtype)
    t1 = _tensor([1.2, 2.2, 2.9, 4.2], dtype)
    t2 = _tensor([0.8, 1.8, 3.1, 3.7], dtype)
    out = ram_merge([t1, t2], base)
    assert out.dtype == dtype
    assert np.isfinite(out.astype(mindspore.float32).asnumpy()).all()


@pytest.mark.parametrize("dtype", [mindspore.bfloat16, mindspore.float16, mindspore.float32])
def test_multislerp_dtype_matrix(dtype):
    t1 = _tensor([1.0, 2.0, 3.0, 4.0], dtype)
    t2 = _tensor([1.2, 1.8, 2.8, 4.1], dtype)
    out = multislerp([t1, t2], [0.5, 0.5])
    assert out.dtype == dtype
    assert np.isfinite(out.astype(mindspore.float32).asnumpy()).all()


@pytest.mark.parametrize("dtype", [mindspore.bfloat16, mindspore.float16, mindspore.float32])
def test_nuslerp_dtype_matrix(dtype):
    v0 = _tensor([1.0, 2.0, 3.0, 4.0], dtype)
    v1 = _tensor([1.1, 1.9, 3.1, 3.9], dtype)
    out = nuslerp(0.4, v0, v1)
    assert out.dtype == dtype
    assert np.isfinite(out.astype(mindspore.float32).asnumpy()).all()


@pytest.mark.parametrize("dtype", [mindspore.bfloat16, mindspore.float16, mindspore.float32])
def test_della_magprune_dtype_matrix(dtype):
    x = _tensor([[1.0, 0.4, -0.5], [0.8, -1.0, 0.2]], dtype)
    out = della_magprune(x, density=0.5, epsilon=0.1)
    if dtype != mindspore.float32:
        pytest.xfail("MindSpore CPU auto-promotes half to float32 in sparsify ops")
    assert out.dtype == dtype
    assert np.isfinite(out.astype(mindspore.float32).asnumpy()).all()


@pytest.mark.parametrize("dtype", [mindspore.bfloat16, mindspore.float16, mindspore.float32])
@pytest.mark.parametrize("norm", [RescaleNorm.l1, RescaleNorm.linf])
def test_rescaled_masked_tensor_half_safe(dtype, norm):
    x = _tensor([[1.0, -0.4, 0.5], [0.8, -1.0, 0.2]], dtype)
    mask = _tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype)
    out = rescaled_masked_tensor(x, mask, norm=norm)
    if dtype != mindspore.float32:
        pytest.xfail("MindSpore CPU auto-promotes half to float32 in sparsify ops")
    assert out.dtype == dtype
    assert np.isfinite(out.astype(mindspore.float32).asnumpy()).all()

