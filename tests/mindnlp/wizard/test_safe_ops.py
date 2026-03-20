import mindspore

from mindnlp.wizard.merge.safe_ops import (
    safe_abs,
    safe_mul,
    safe_norm,
    safe_stack,
    safe_sum,
    safe_where,
)


def _bf16_tensors():
    a = mindspore.Tensor([1.0, -2.0, 3.0], dtype=mindspore.bfloat16)
    b = mindspore.Tensor([2.0, 4.0, -1.0], dtype=mindspore.bfloat16)
    return a, b


def test_safe_stack_preserves_output_dtype():
    a, b = _bf16_tensors()
    out = safe_stack([a, b], axis=0, out_dtype=mindspore.bfloat16)
    assert out.dtype == mindspore.bfloat16
    assert out.shape == (2, 3)


def test_safe_mul_sum_abs_norm_where_dtype_contract():
    a, b = _bf16_tensors()
    mul = safe_mul(a, b, out_dtype=mindspore.bfloat16)
    assert mul.dtype == mindspore.bfloat16

    summed = safe_sum(mul, out_dtype=mindspore.bfloat16)
    assert summed.dtype == mindspore.bfloat16

    absed = safe_abs(a, out_dtype=mindspore.bfloat16)
    assert absed.dtype == mindspore.bfloat16

    normed = safe_norm(a, out_dtype=mindspore.float32)
    assert normed.dtype == mindspore.float32

    cond = mindspore.Tensor([True, False, True])
    merged = safe_where(cond, a, b, out_dtype=mindspore.bfloat16)
    assert merged.dtype == mindspore.bfloat16


def test_safe_mul_sum_numeric_contract():
    a = mindspore.Tensor([1.0, -2.0, 3.0], dtype=mindspore.float32)
    b = mindspore.Tensor([2.0, 4.0, -1.0], dtype=mindspore.float32)
    out = safe_mul(a, b, out_dtype=mindspore.float32)
    np_out = out.asnumpy().tolist()
    assert np_out == [2.0, -8.0, -3.0]

    reduced = safe_sum(out, axis=0, out_dtype=mindspore.float32)
    assert float(reduced.asnumpy()) == -9.0

