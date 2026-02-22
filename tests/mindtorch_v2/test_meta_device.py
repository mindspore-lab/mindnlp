import pytest
import mindtorch_v2 as torch


def test_meta_tensor_blocks_numpy():
    t = torch.tensor([1.0, 2.0], device="meta")
    with pytest.raises(RuntimeError, match="meta tensor has no data"):
        _ = t.numpy()


def test_meta_to_cpu_materializes():
    t = torch.tensor([1.0, 2.0], device="meta")
    out = t.to("cpu")
    assert out.device.type == "cpu"
    assert out.shape == (2,)


def test_meta_ops_shape_propagation():
    a = torch.tensor([[1.0, 2.0]], device="meta")
    b = torch.tensor([[3.0, 4.0]], device="meta")
    c = a + b
    d = c.relu()
    e = d.sum()
    assert c.device.type == "meta"
    assert d.device.type == "meta"
    assert e.device.type == "meta"
    assert c.shape == a.shape
    assert e.shape == ()


def test_meta_unary_elementwise_ops_shape():
    x = torch.tensor([1.0, 2.0], device="meta")
    for op in (
        torch.abs,
        torch.neg,
        torch.exp,
        torch.log,
        torch.sqrt,
        torch.sin,
        torch.cos,
        torch.tan,
        torch.tanh,
        torch.sigmoid,
        torch.floor,
        torch.ceil,
        torch.round,
        torch.trunc,
        torch.frac,
        torch.log2,
        torch.log10,
        torch.exp2,
        torch.rsqrt,
        torch.sign,
        torch.signbit,
        torch.isnan,
        torch.isinf,
        torch.isfinite,
        torch.sinh,
        torch.cosh,
        torch.asinh,
        torch.acosh,
        torch.atanh,
        torch.erf,
        torch.erfc,
        torch.softplus,
        torch.relu6,
    ):
        out = op(x)
        assert out.device.type == "meta"
        assert out.shape == x.shape

    for op in (
        lambda t: torch.clamp(t, 0.0, 1.0),
        lambda t: torch.clamp_min(t, 0.0),
        lambda t: torch.clamp_max(t, 1.0),
        lambda t: torch.hardtanh(t, -1.0, 1.0),
        lambda t: torch.min(t, t),
        lambda t: torch.max(t, t),
        lambda t: torch.fmin(t, t),
        lambda t: torch.fmax(t, t),
        lambda t: torch.where(t, t, t),
        lambda t: torch.atan(t),
        lambda t: torch.atan2(t, t),
        lambda t: torch.asin(t),
        lambda t: torch.acos(t),
        lambda t: torch.lerp(t, t, 0.5),
        lambda t: torch.addcmul(t, t, t, value=0.5),
        lambda t: torch.addcdiv(t, t, t, value=0.5),
        lambda t: torch.logaddexp(t, t),
        lambda t: torch.logaddexp2(t, t),
        lambda t: torch.hypot(t, t),
        lambda t: torch.remainder(t, t),
        lambda t: torch.fmod(t, t),
    ):
        out = op(x)
        assert out.device.type == "meta"
        assert out.shape == x.shape

    for op in (
        lambda t: torch.amin(t, dim=0),
        lambda t: torch.amax(t, dim=0),
        lambda t: torch.all(t, dim=0),
        lambda t: torch.any(t, dim=0),
        lambda t: torch.argmax(t, dim=0),
    ):
        out = op(x)
        assert out.device.type == "meta"
        assert out.shape == ()

    for op in (
        lambda t: torch.allclose(t, t),
        lambda t: torch.isclose(t, t),
        lambda t: torch.equal(t, t),
    ):
        out = op(x)
        assert out.device.type == "meta"
        assert out.shape in ((), x.shape)


def test_meta_pow_shape():
    x = torch.tensor([1.0, 2.0, 3.0], device="meta")
    out = torch.pow(x, 2.0)
    assert out.device.type == "meta"
    assert out.shape == x.shape
