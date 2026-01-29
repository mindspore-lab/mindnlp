"""Tests for PyBoost CPU backend."""
import pytest


def test_pyboost_add_op_exists():
    """PyBoost add op should be importable."""
    from mindtorch_v2._backends.pyboost_cpu import add_op
    assert add_op is not None


def test_pyboost_add_basic():
    """PyBoost add should work on MindSpore tensors."""
    import mindspore
    from mindtorch_v2._backends.pyboost_cpu import add_op

    a = mindspore.Tensor([1.0, 2.0, 3.0])
    b = mindspore.Tensor([4.0, 5.0, 6.0])
    result = add_op(a, b)

    expected = [5.0, 7.0, 9.0]
    assert list(result.asnumpy()) == expected


def test_get_ms_data_extracts_mindspore_tensor():
    """_get_ms_data should extract MindSpore tensor from our Tensor."""
    from mindtorch_v2 import Tensor
    from mindtorch_v2._backends.pyboost_cpu import _get_ms_data
    import mindspore

    t = Tensor([1.0, 2.0, 3.0])
    ms_t = _get_ms_data(t)

    assert isinstance(ms_t, mindspore.Tensor)
    assert list(ms_t.asnumpy()) == [1.0, 2.0, 3.0]


def test_wrap_result_creates_tensor():
    """_wrap_result should wrap MindSpore tensor in our Tensor."""
    import mindspore
    from mindtorch_v2 import Tensor
    from mindtorch_v2._backends.pyboost_cpu import _wrap_result

    ms_t = mindspore.Tensor([1.0, 2.0, 3.0])
    t = _wrap_result(ms_t)

    assert isinstance(t, Tensor)
    assert list(t.numpy()) == [1.0, 2.0, 3.0]
