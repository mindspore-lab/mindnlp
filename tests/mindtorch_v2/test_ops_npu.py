import math
import numpy as np
import pytest
import mindtorch_v2 as torch


def test_npu_add():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([1.0, 2.0]).to("npu")
    y = torch.tensor([3.0, 4.0]).to("npu")
    z = torch.add(x, y).to("cpu")
    assert z.storage().data.tolist() == [4.0, 6.0]





def test_npu_matmul_2d():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu", dtype=torch.float16)
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="npu", dtype=torch.float16)
    out = torch.matmul(a, b)
    assert out.device.type == "npu"
    assert np.allclose(out.to("cpu").numpy(), np.matmul(a.to("cpu").numpy(), b.to("cpu").numpy()))


def test_npu_matmul_1d_2d():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([1.0, 2.0, 3.0], device="npu", dtype=torch.float16)
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="npu", dtype=torch.float16)
    out = torch.matmul(a, b)
    assert out.shape == (2,)
    assert np.allclose(out.to("cpu").numpy(), np.matmul(a.to("cpu").numpy(), b.to("cpu").numpy()))


def test_npu_matmul_batched_broadcast():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor(
        np.arange(2 * 1 * 2 * 3, dtype=np.float16).reshape(2, 1, 2, 3),
        device="npu",
        dtype=torch.float16,
    )
    b = torch.tensor(
        np.arange(1 * 4 * 3 * 5, dtype=np.float16).reshape(1, 4, 3, 5),
        device="npu",
        dtype=torch.float16,
    )
    out = torch.matmul(a, b)
    assert out.shape == (2, 4, 2, 5)
    assert np.allclose(out.to("cpu").numpy(), np.matmul(a.to("cpu").numpy(), b.to("cpu").numpy()))

@pytest.mark.parametrize(
    "op_name, numpy_fn",
    [
        ("abs", np.abs),
        ("neg", np.negative),
        ("exp", np.exp),
        ("log", np.log),
        ("sqrt", np.sqrt),
        ("rsqrt", lambda x: 1.0 / np.sqrt(x)),
        ("sin", np.sin),
        ("cos", np.cos),
        ("tan", np.tan),
        ("tanh", np.tanh),
        ("sigmoid", lambda x: 1.0 / (1.0 + np.exp(-x))),
        ("ceil", np.ceil),
        ("floor", np.floor),
        ("round", np.round),
        ("trunc", np.trunc),
        ("frac", lambda x: x - np.trunc(x)),
        ("log2", np.log2),
        ("log10", np.log10),
        ("exp2", np.exp2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_unary_ops(op_name, numpy_fn, dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    if op_name in {"log", "log2", "log10", "sqrt", "rsqrt"}:
        data = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    else:
        data = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)

    x = torch.tensor(data, device="npu", dtype=dtype)
    op = getattr(torch, op_name)
    out = op(x)
    expected = numpy_fn(data).astype(np.float32)
    assert out.device.type == "npu"
    assert np.allclose(
        out.to("cpu").numpy().astype(np.float32),
        expected,
        atol=1e-3,
        rtol=1e-3,
    )

@pytest.mark.parametrize(
    "op_name, numpy_fn",
    [
        ("cosh", np.cosh),
        ("sinh", np.sinh),
        ("erf", lambda x: np.vectorize(math.erf)(x)),
        ("erfc", lambda x: np.vectorize(math.erfc)(x)),
        ("softplus", lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_unary_ops_extra(op_name, numpy_fn, dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)
    op = getattr(torch, op_name)
    out = op(x)
    expected = numpy_fn(data).astype(np.float32)
    assert out.device.type == "npu"
    assert np.allclose(
        out.to("cpu").numpy().astype(np.float32),
        expected,
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_clamp_ops(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)
    out = torch.clamp(x, -1.0, 1.0)
    out_min = torch.clamp_min(x, -1.0)
    out_max = torch.clamp_max(x, 1.0)
    assert np.allclose(out.to("cpu").numpy(), np.clip(data, -1.0, 1.0).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(out_min.to("cpu").numpy(), np.clip(data, -1.0, None).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(out_max.to("cpu").numpy(), np.clip(data, None, 1.0).astype(np.float32), atol=1e-3, rtol=1e-3)

    tensor_data = np.array([[-2.0, -0.5], [0.5, 2.0]], dtype=np.float32)
    min_data = np.array([[-1.0], [0.0]], dtype=np.float32)
    max_data = np.array([[0.25, 1.0]], dtype=np.float32)
    tensor_x = torch.tensor(tensor_data, device="npu", dtype=dtype)
    tensor_min = torch.tensor(min_data, device="npu", dtype=dtype)
    tensor_max = torch.tensor(max_data, device="npu", dtype=dtype)
    tensor_out = torch.clamp(tensor_x, tensor_min, tensor_max)
    tensor_out_min = torch.clamp_min(tensor_x, tensor_min)
    tensor_out_max = torch.clamp_max(tensor_x, tensor_max)
    expected = np.clip(tensor_data, min_data, max_data).astype(np.float32)
    expected_min = np.clip(tensor_data, min_data, None).astype(np.float32)
    expected_max = np.clip(tensor_data, None, max_data).astype(np.float32)
    assert np.allclose(tensor_out.to("cpu").numpy(), expected, atol=1e-3, rtol=1e-3)
    assert np.allclose(tensor_out_min.to("cpu").numpy(), expected_min, atol=1e-3, rtol=1e-3)
    assert np.allclose(tensor_out_max.to("cpu").numpy(), expected_max, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_relu6_hardtanh(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([-2.0, -0.5, 0.5, 2.0, 7.0], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)
    relu6 = torch.relu6(x)
    hardtanh = torch.hardtanh(x, -1.0, 1.0)
    assert np.allclose(relu6.to("cpu").numpy(), np.clip(data, 0.0, 6.0).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(hardtanh.to("cpu").numpy(), np.clip(data, -1.0, 1.0).astype(np.float32), atol=1e-3, rtol=1e-3)

def test_npu_isfinite_isinf_isnan_signbit():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([0.0, 1.0, -1.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=torch.float32)
    isfinite = torch.isfinite(x)
    isinf = torch.isinf(x)
    isnan = torch.isnan(x)
    signbit = torch.signbit(x)
    assert isfinite.dtype == torch.bool
    assert isinf.dtype == torch.bool
    assert isnan.dtype == torch.bool
    assert signbit.dtype == torch.bool
    assert np.array_equal(isfinite.to("cpu").numpy(), np.isfinite(data))
    assert np.array_equal(isinf.to("cpu").numpy(), np.isinf(data))
    assert np.array_equal(isnan.to("cpu").numpy(), np.isnan(data))
    assert np.array_equal(signbit.to("cpu").numpy(), np.signbit(data))


def test_npu_amin_amax():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 0.5]], device="npu")
    expected_min = np.amin(x.to("cpu").numpy(), axis=1)
    expected_max = np.amax(x.to("cpu").numpy(), axis=1)
    np.testing.assert_allclose(torch.amin(x, dim=1).to("cpu").numpy(), expected_min)
    np.testing.assert_allclose(torch.amax(x, dim=1).to("cpu").numpy(), expected_max)


def test_npu_argmax_argmin():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], device="npu")
    expected_max = np.argmax(x.to("cpu").numpy(), axis=1)
    expected_min = np.argmin(x.to("cpu").numpy(), axis=1)
    np.testing.assert_array_equal(torch.argmax(x, dim=1).to("cpu").numpy(), expected_max)
    np.testing.assert_array_equal(torch.argmin(x, dim=1).to("cpu").numpy(), expected_min)
    np.testing.assert_array_equal(
        torch.argmax(x, dim=1, keepdim=True).to("cpu").numpy(),
        expected_max.reshape(2, 1),
    )
    np.testing.assert_array_equal(
        torch.argmin(x, dim=1, keepdim=True).to("cpu").numpy(),
        expected_min.reshape(2, 1),
    )


def test_npu_all_any():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[True, False], [True, True]], device="npu", dtype=torch.bool)
    expected_all = np.all(x.to("cpu").numpy(), axis=1)
    expected_any = np.any(x.to("cpu").numpy(), axis=1)
    np.testing.assert_array_equal(torch.all(x, dim=1).to("cpu").numpy(), expected_all)
    np.testing.assert_array_equal(torch.any(x, dim=1).to("cpu").numpy(), expected_any)
    expected_keep = np.all(x.to("cpu").numpy(), axis=1, keepdims=True)
    np.testing.assert_array_equal(torch.all(x, dim=1, keepdim=True).to("cpu").numpy(), expected_keep)


def test_npu_count_nonzero():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 3.0]], device="npu")
    expected = np.count_nonzero(x.to("cpu").numpy(), axis=1)
    np.testing.assert_array_equal(torch.count_nonzero(x, dim=1).to("cpu").numpy(), expected)
    expected_keep = np.count_nonzero(x.to("cpu").numpy(), axis=1, keepdims=True)
    np.testing.assert_array_equal(
        torch.count_nonzero(x, dim=1, keepdim=True).to("cpu").numpy(),
        expected_keep,
    )


def test_npu_split_stack_family():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    a = torch.tensor([1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="npu")
    out = torch.chunk(x, 2)
    assert len(out) == 2
    np.testing.assert_allclose(out[0].to("cpu").numpy(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(out[1].to("cpu").numpy(), np.array([3.0, 4.0]))


def test_npu_hstack():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    a = torch.tensor([1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")
    expected = np.hstack([a.to("cpu").numpy(), b.to("cpu").numpy()])
    np.testing.assert_allclose(torch.hstack([a, b]).to("cpu").numpy(), expected)
    torch.npu.synchronize()


def test_npu_vstack_row_stack():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

def test_npu_column_stack():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

def test_npu_dstack():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

def test_npu_hsplit():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="npu")
    out = torch.hsplit(x, 2)
    assert len(out) == 2
    np.testing.assert_allclose(out[0].to("cpu").numpy(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(out[1].to("cpu").numpy(), np.array([3.0, 4.0]))
    torch.npu.synchronize()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_pow(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    base = torch.tensor([1.0, 2.0, 3.0], device="npu", dtype=dtype)
    exp = torch.tensor([2.0, 3.0, 0.5], device="npu", dtype=dtype)
    out = torch.pow(base, exp)
    expected = np.power(base.to("cpu").numpy(), exp.to("cpu").numpy())
    assert np.allclose(
        out.to("cpu").numpy().astype(np.float32),
        expected.astype(np.float32),
        atol=1e-3,
        rtol=1e-3,
    )

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_elementwise_batch2(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    base = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    x = torch.tensor(base, device="npu", dtype=dtype)
    y = torch.tensor(base[::-1], device="npu", dtype=dtype)

    expected_asin = np.arcsin(base).astype(np.float32)
    expected_acos = np.arccos(base).astype(np.float32)
    out_asin = torch.asin(x).to("cpu").numpy().astype(np.float32)
    out_acos = torch.acos(x).to("cpu").numpy().astype(np.float32)
    if dtype == torch.float16:
        valid = np.abs(base) <= 1.0
        assert np.allclose(out_asin[valid], expected_asin[valid], atol=1e-3, rtol=1e-3)
        assert np.allclose(out_acos[valid], expected_acos[valid], atol=1e-3, rtol=1e-3)
    else:
        assert np.allclose(out_asin, expected_asin, atol=1e-3, rtol=1e-3, equal_nan=True)
        assert np.allclose(out_acos, expected_acos, atol=1e-3, rtol=1e-3, equal_nan=True)
    assert np.allclose(torch.atan(x).to("cpu").numpy(), np.arctan(base).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.atan2(x, y).to("cpu").numpy(), np.arctan2(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.asinh(x).to("cpu").numpy(), np.arcsinh(base).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.acosh(torch.abs(x) + 1.5).to("cpu").numpy(), np.arccosh(np.abs(base) + 1.5).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.atanh(x * 0.25).to("cpu").numpy(), np.arctanh(base * 0.25).astype(np.float32), atol=1e-3, rtol=1e-3)

    assert np.allclose(torch.addcmul(x, x, y).to("cpu").numpy(), (base + base * base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.addcdiv(x, x, y).to("cpu").numpy(), (base + base / base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)

    assert np.allclose(torch.logaddexp(x, y).to("cpu").numpy(), np.logaddexp(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.logaddexp2(x, y).to("cpu").numpy(), np.logaddexp2(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.hypot(x, y).to("cpu").numpy(), np.hypot(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)

    assert np.allclose(torch.remainder(x, y).to("cpu").numpy(), np.remainder(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.fmod(x, y).to("cpu").numpy(), np.fmod(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)

    assert np.allclose(torch.fmin(x, y).to("cpu").numpy(), np.fmin(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.fmax(x, y).to("cpu").numpy(), np.fmax(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.min(x, y).to("cpu").numpy(), np.minimum(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.max(x, y).to("cpu").numpy(), np.maximum(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)

    where_cond = torch.tensor([True, False, True, False], device="npu")
    where_out = torch.where(where_cond, x, y)
    expected_where = np.where(np.array([True, False, True, False]), base, base[::-1]).astype(np.float32)
    assert np.allclose(where_out.to("cpu").numpy().astype(np.float32), expected_where, atol=1e-3, rtol=1e-3)

    lerp_out = torch.lerp(x, y, 0.25).to("cpu").numpy()
    expected_lerp = (base + 0.25 * (base[::-1] - base)).astype(np.float32)
    assert np.allclose(lerp_out, expected_lerp, atol=1e-3, rtol=1e-3)

    assert torch.allclose(x, y) == np.allclose(base, base[::-1])
    isclose_out = torch.isclose(x, y).to("cpu").numpy()
    assert np.all(isclose_out == np.isclose(base, base[::-1]))
    assert torch.equal(x, y) == np.array_equal(base, base[::-1])


    scalar_base = torch.tensor(base, device="npu", dtype=dtype)
    scalar_out = torch.pow(scalar_base, 2.0)
    scalar_expected = np.power(base, 2.0)
    assert np.allclose(
        scalar_out.to("cpu").numpy().astype(np.float32),
        scalar_expected.astype(np.float32),
        atol=1e-3,
        rtol=1e-3,
    )


def test_npu_model_dir_probe():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    ok = torch._C._npu_probe_model_dirs()
    assert ok is True


def test_npu_model_dir_selected():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    path = torch._C._npu_model_dir()
    assert path in {
        "/usr/local/Ascend/ascend-toolkit/latest/opp",
        "/home/lvyufeng/lvyufeng/acl_engine",
    }


def test_npu_aclnn_available():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    assert torch._C._npu_aclnn_available() is True


def test_aclnn_symbols_present():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    assert torch._C._npu_aclnn_symbols_ok() is True


def test_aclnn_ones_zero_symbols_present():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    assert torch._C._npu_aclnn_ones_zero_ok() is True


def test_npu_add_execute():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")
    out = a + b
    assert out.device.type == "npu"
    assert out.to("cpu").numpy().tolist() == [4.0, 6.0]


def test_npu_mul_relu():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([-1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")
    prod = a * b
    relu = a.relu()
    assert prod.device.type == "npu"
    assert relu.device.type == "npu"
    assert prod.to("cpu").numpy().tolist() == [-3.0, 8.0]
    assert relu.to("cpu").numpy().tolist() == [0.0, 2.0]


def test_npu_sum():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([[1.0, 2.0]], device="npu")
    total = a.sum()
    kept = a.sum(dim=1, keepdim=True)
    assert total.device.type == "npu"
    assert kept.device.type == "npu"
    assert total.to("cpu").numpy().tolist() == 3.0
    assert kept.to("cpu").numpy().tolist() == [[3.0]]

def test_npu_device_index_preserved():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.ones((1,), device="npu:0")
    assert out.device.type == "npu"
    assert out.device.index == 0


def test_npu_cross_device_copy():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    if torch._C._npu_device_count() < 2:
        pytest.skip("Need 2 NPUs")
    src = torch.ones((2,), device="npu:0")
    dst = src.to("npu:1")
    assert dst.device.index == 1
    assert dst.to("cpu").numpy().tolist() == [1.0, 1.0]


def test_npu_ones():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.ones((1,), device="npu")
    assert out.device.type == "npu"
    assert out.to("cpu").numpy().tolist() == [1.0]


def test_npu_zeros():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.zeros((2,), device="npu")
    assert out.device.type == "npu"
    assert out.to("cpu").numpy().tolist() == [0.0, 0.0]



def test_npu_arange():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.arange(0, 5, device="npu")
    assert out.device.type == "npu"
    np.testing.assert_allclose(out.to("cpu").numpy(), np.array([0, 1, 2, 3, 4]))


def test_npu_range():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.range(0.0, 2.0, 0.5, device="npu")
    expected = np.arange(0.0, 2.0 + 0.5, 0.5)
    assert out.device.type == "npu"
    np.testing.assert_allclose(out.to("cpu").numpy(), expected, atol=1e-6, rtol=1e-6)


def test_npu_linspace():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.linspace(0.0, 1.0, 5, device="npu")
    assert out.device.type == "npu"
    np.testing.assert_allclose(out.to("cpu").numpy(), np.linspace(0.0, 1.0, 5), atol=1e-6, rtol=1e-6)


def test_npu_logspace():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.logspace(0.0, 2.0, 3, device="npu")
    assert out.device.type == "npu"
    np.testing.assert_allclose(out.to("cpu").numpy(), np.logspace(0.0, 2.0, 3), atol=1e-6, rtol=1e-6)


def test_npu_full():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.full((2, 3), 1.5, device="npu")
    assert out.device.type == "npu"
    np.testing.assert_allclose(out.to("cpu").numpy(), np.full((2, 3), 1.5), atol=1e-6, rtol=1e-6)


def test_npu_eye():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.eye(3, 2, device="npu")
    assert out.device.type == "npu"
    np.testing.assert_allclose(out.to("cpu").numpy(), np.eye(3, 2), atol=1e-6, rtol=1e-6)


def test_npu_linspace_prefers_single_op(monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    from mindtorch_v2._backends.npu import aclnn as npu_aclnn

    if hasattr(npu_aclnn, "linspace_symbols_ok") and not npu_aclnn.linspace_symbols_ok():
        pytest.skip("aclnnLinspace not available")

    def _forbid_arange(*args, **kwargs):
        raise AssertionError("linspace should not call aclnn.arange")

    monkeypatch.setattr(npu_aclnn, "arange", _forbid_arange)
    out = torch.linspace(0.0, 1.0, 5, device="npu")
    np.testing.assert_allclose(out.to("cpu").numpy(), np.linspace(0.0, 1.0, 5), atol=1e-6, rtol=1e-6)


def test_npu_eye_prefers_single_op(monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    from mindtorch_v2._backends.npu import aclnn as npu_aclnn

    if hasattr(npu_aclnn, "eye_symbols_ok") and not npu_aclnn.eye_symbols_ok():
        pytest.skip("aclnnEye not available")

    def _forbid_arange(*args, **kwargs):
        raise AssertionError("eye should not call aclnn.arange")

    monkeypatch.setattr(npu_aclnn, "arange", _forbid_arange)
    out = torch.eye(3, 2, device="npu")
    np.testing.assert_allclose(out.to("cpu").numpy(), np.eye(3, 2), atol=1e-6, rtol=1e-6)


def test_npu_range_prefers_single_op(monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    from mindtorch_v2._backends.npu import aclnn as npu_aclnn

    if hasattr(npu_aclnn, "range_symbols_ok") and not npu_aclnn.range_symbols_ok():
        pytest.skip("aclnnRange not available")

    def _forbid_arange(*args, **kwargs):
        raise AssertionError("range should not call aclnn.arange")

    monkeypatch.setattr(npu_aclnn, "arange", _forbid_arange)
    out = torch.range(0.0, 2.0, 0.5, device="npu")
    np.testing.assert_allclose(out.to("cpu").numpy(), np.arange(0.0, 2.0 + 0.5, 0.5), atol=1e-6, rtol=1e-6)


def test_npu_flip():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1, 2], [3, 4]], device="npu")
    out = torch.flip(x, dims=(0,))
    np.testing.assert_array_equal(out.to("cpu").numpy(), np.flip(x.to("cpu").numpy(), axis=0))


def test_npu_roll():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1, 2], [3, 4]], device="npu")
    out = torch.roll(x, shifts=1, dims=0)
    np.testing.assert_array_equal(out.to("cpu").numpy(), np.roll(x.to("cpu").numpy(), shift=1, axis=0))


def test_npu_nonzero():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[0, 1], [2, 0]], device="npu")
    out = torch.nonzero(x)
    np.testing.assert_array_equal(out.to("cpu").numpy(), np.array([[0, 1], [1, 0]]))


def test_npu_cumsum():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
    out = torch.cumsum(x, dim=1)
    np.testing.assert_allclose(out.to("cpu").numpy(), np.cumsum(x.to("cpu").numpy(), axis=1), atol=1e-6, rtol=1e-6)


def test_npu_cumprod():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
    out = torch.cumprod(x, dim=1)
    np.testing.assert_allclose(out.to("cpu").numpy(), np.cumprod(x.to("cpu").numpy(), axis=1), atol=1e-6, rtol=1e-6)


def test_npu_cummax():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], device="npu")
    values, indices = torch.cummax(x, dim=1)
    expected_vals = np.maximum.accumulate(x.to("cpu").numpy(), axis=1)
    expected_idx = np.array([[0, 1, 1], [0, 0, 2]], dtype=np.int64)
    np.testing.assert_allclose(values.to("cpu").numpy(), expected_vals, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(indices.to("cpu").numpy(), expected_idx)


def test_npu_argsort():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]], device="npu")
    out = torch.argsort(x, dim=1)
    expected = np.argsort(x.to("cpu").numpy(), axis=1)
    np.testing.assert_array_equal(out.to("cpu").numpy(), expected)


def test_npu_sort():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]], device="npu")
    values, indices = torch.sort(x, dim=1)
    expected_indices = np.argsort(x.to("cpu").numpy(), axis=1)
    expected_values = np.take_along_axis(x.to("cpu").numpy(), expected_indices, axis=1)
    np.testing.assert_allclose(values.to("cpu").numpy(), expected_values, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(indices.to("cpu").numpy(), expected_indices)


def test_npu_topk():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]], device="npu")
    values, indices = torch.topk(x, k=2, dim=1, largest=True, sorted=True)
    expected_indices = np.argsort(-x.to("cpu").numpy(), axis=1)[:, :2]
    expected_values = np.take_along_axis(x.to("cpu").numpy(), expected_indices, axis=1)
    np.testing.assert_allclose(values.to("cpu").numpy(), expected_values, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(indices.to("cpu").numpy(), expected_indices)


def test_npu_tril_triu():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
    tril_out = torch.tril(x)
    triu_out = torch.triu(x)
    np.testing.assert_allclose(tril_out.to("cpu").numpy(), np.tril(x.to("cpu").numpy()), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(triu_out.to("cpu").numpy(), np.triu(x.to("cpu").numpy()), atol=1e-6, rtol=1e-6)


def test_npu_to_cpu_synchronizes(monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    calls = []

    from mindtorch_v2._backends.npu import runtime as npu_runtime
    runtime = npu_runtime.get_runtime(0)

    def fake_sync():
        calls.append("sync")

    monkeypatch.setattr(runtime, "synchronize", fake_sync)

    t = torch.ones((1,), device="npu")
    _ = t.to("cpu")
    assert "sync" in calls


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_div(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a_data = np.array([4.0, 6.0, 9.0, 10.0], dtype=np.float32)
    b_data = np.array([2.0, 3.0, 3.0, 5.0], dtype=np.float32)
    a = torch.tensor(a_data, device="npu", dtype=dtype)
    b = torch.tensor(b_data, device="npu", dtype=dtype)
    out = torch.div(a, b)
    expected = (a_data / b_data).astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_mean(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)

    # Mean along dim 1
    out = torch.mean(x, dim=1)
    expected = np.mean(data, axis=1).astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)

    # Mean along dim 0
    out0 = torch.mean(x, dim=0)
    expected0 = np.mean(data, axis=0).astype(np.float32)
    assert np.allclose(out0.to("cpu").numpy().astype(np.float32), expected0, atol=1e-3, rtol=1e-3)

    # Global mean
    out_all = torch.mean(x)
    expected_all = np.mean(data).astype(np.float32)
    assert np.allclose(out_all.to("cpu").numpy().astype(np.float32), expected_all, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_softmax(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)

    from mindtorch_v2.nn import functional as F
    out = F.softmax(x, dim=-1)

    # Compute expected using numpy
    def numpy_softmax(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    expected = numpy_softmax(data).astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)

    # Check that each row sums to ~1.0
    row_sums = np.sum(out.to("cpu").numpy().astype(np.float32), axis=1)
    assert np.allclose(row_sums, np.ones(2), atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_log_softmax(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)

    from mindtorch_v2.nn import functional as F
    out = F.log_softmax(x, dim=-1)

    def numpy_log_softmax(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return np.log(e_x / np.sum(e_x, axis=axis, keepdims=True))

    expected = numpy_log_softmax(data).astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_gelu(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)

    from mindtorch_v2.nn import functional as F
    out = F.gelu(x)

    # GELU formula: x * 0.5 * (1 + erf(x / sqrt(2)))
    from scipy.special import erf as scipy_erf
    expected = (data * 0.5 * (1.0 + scipy_erf(data / np.sqrt(2.0)))).astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_layer_norm(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    # Test with 1-row input (works)
    data_1row = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    x_1row = torch.tensor(data_1row, device="npu", dtype=dtype)

    from mindtorch_v2.nn import functional as F
    out_1row = F.layer_norm(x_1row, (3,))

    mean_val = np.mean(data_1row, axis=-1, keepdims=True)
    var_val = np.var(data_1row, axis=-1, keepdims=True)
    expected = ((data_1row - mean_val) / np.sqrt(var_val + 1e-5)).astype(np.float32)
    assert np.allclose(out_1row.to("cpu").numpy().astype(np.float32), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_layer_norm_multirow(dtype):
    """Test layer_norm with multi-row input (batch_size > 1)."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)

    from mindtorch_v2.nn import functional as F
    out = F.layer_norm(x, (3,))

    # Compute expected manually
    mean_val = np.mean(data, axis=-1, keepdims=True)
    var_val = np.var(data, axis=-1, keepdims=True)
    expected = ((data - mean_val) / np.sqrt(var_val + 1e-5)).astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_embedding(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    weight_data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    weight = torch.tensor(weight_data, device="npu", dtype=dtype)
    indices = torch.tensor([0, 2, 1], device="npu", dtype=torch.int64)

    from mindtorch_v2.nn import functional as F
    out = F.embedding(indices, weight)

    expected = weight_data[np.array([0, 2, 1])].astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_embedding_2d_indices(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    weight_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    weight = torch.tensor(weight_data, device="npu", dtype=dtype)
    indices = torch.tensor([[0, 2], [1, 3]], device="npu", dtype=torch.int64)

    from mindtorch_v2.nn import functional as F
    out = F.embedding(indices, weight)

    assert out.shape == (2, 2, 2)
    expected = weight_data[np.array([[0, 2], [1, 3]])].astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)


def test_npu_take():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
    index = torch.tensor([0, 3, 1], dtype=torch.int64, device="npu")
    expected = np.take(
        x.to("cpu").numpy().reshape(-1),
        index.to("cpu").numpy().astype(np.int64),
    )
    np.testing.assert_allclose(torch.take(x, index).to("cpu").numpy(), expected)
    neg_index = torch.tensor([-1, 0], dtype=torch.int64, device="npu")
    expected_neg = np.take(
        x.to("cpu").numpy().reshape(-1),
        neg_index.to("cpu").numpy().astype(np.int64),
    )
    np.testing.assert_allclose(torch.take(x, neg_index).to("cpu").numpy(), expected_neg)
    out_of_range = torch.tensor([4], dtype=torch.int64, device="npu")
    with pytest.raises(IndexError):
        torch.take(x, out_of_range)


def test_npu_take_along_dim():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="npu")
    indices = torch.tensor([[0, 2, 1], [2, 0, 1]], dtype=torch.int64, device="npu")
    expected = np.take_along_axis(
        x.to("cpu").numpy(),
        indices.to("cpu").numpy().astype(np.int64),
        axis=1,
    )
    np.testing.assert_allclose(
        torch.take_along_dim(x, indices, dim=1).to("cpu").numpy(),
        expected,
    )
    neg_indices = torch.tensor([[-1, 0, 1], [1, -2, 0]], dtype=torch.int64, device="npu")
    expected_neg = np.take_along_axis(
        x.to("cpu").numpy(),
        neg_indices.to("cpu").numpy().astype(np.int64),
        axis=1,
    )
    np.testing.assert_allclose(
        torch.take_along_dim(x, neg_indices, dim=1).to("cpu").numpy(),
        expected_neg,
    )
    out_of_range = torch.tensor([[3, 0, 1], [1, 2, 0]], dtype=torch.int64, device="npu")
    with pytest.raises(IndexError):
        torch.take_along_dim(x, out_of_range, dim=1)


def test_npu_index_select():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="npu")
    index = torch.tensor([2, 0], dtype=torch.int64, device="npu")
    expected = np.take(
        x.to("cpu").numpy(),
        index.to("cpu").numpy().astype(np.int64),
        axis=1,
    )
    np.testing.assert_allclose(
        torch.index_select(x, dim=1, index=index).to("cpu").numpy(),
        expected,
    )
    neg_index = torch.tensor([-1, 0], dtype=torch.int64, device="npu")
    expected_neg = np.take(
        x.to("cpu").numpy(),
        neg_index.to("cpu").numpy().astype(np.int64),
        axis=1,
    )
    np.testing.assert_allclose(
        torch.index_select(x, dim=1, index=neg_index).to("cpu").numpy(),
        expected_neg,
    )
    out_of_range = torch.tensor([3], dtype=torch.int64, device="npu")
    with pytest.raises(IndexError):
        torch.index_select(x, dim=1, index=out_of_range)
    bad_index = torch.tensor([[0, 1]], dtype=torch.int64, device="npu")
    with pytest.raises(ValueError):
        torch.index_select(x, dim=1, index=bad_index)


def test_npu_gather():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="npu")
    index = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64, device="npu")
    expected = np.take_along_axis(
        x.to("cpu").numpy(),
        index.to("cpu").numpy().astype(np.int64),
        axis=1,
    )
    np.testing.assert_allclose(
        torch.gather(x, dim=1, index=index).to("cpu").numpy(),
        expected,
    )
    neg_index = torch.tensor([[0, -1], [1, 0]], dtype=torch.int64, device="npu")
    with pytest.raises(IndexError):
        torch.gather(x, dim=1, index=neg_index)
    out_of_range = torch.tensor([[3, 0], [1, 0]], dtype=torch.int64, device="npu")
    with pytest.raises(IndexError):
        torch.gather(x, dim=1, index=out_of_range)


def test_npu_masked_select():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1, 2], [3, 4]], device="npu")
    mask = torch.tensor([[True, False], [False, True]], device="npu")
    out = torch.masked_select(x, mask)
    np.testing.assert_array_equal(out.to("cpu").numpy(), np.array([1, 4]))
