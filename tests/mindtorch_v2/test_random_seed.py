"""Tests for random seed management and dropout."""
import pytest
import numpy as np

try:
    import mindtorch_v2 as torch
    from mindtorch_v2 import nn
    CPU_AVAILABLE = True
except ImportError:
    CPU_AVAILABLE = False

try:
    NPU_AVAILABLE = torch.npu.is_available() if CPU_AVAILABLE and hasattr(torch, 'npu') else False
except Exception:
    NPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CPU_AVAILABLE, reason="mindtorch_v2 not available")


class TestManualSeed:
    """Test manual_seed reproducibility."""

    def test_manual_seed_randn_reproducible(self):
        torch.manual_seed(42)
        x1 = torch.randn(5, 5)
        torch.manual_seed(42)
        x2 = torch.randn(5, 5)
        np.testing.assert_array_equal(x1.numpy(), x2.numpy())

    def test_manual_seed_different_seeds_differ(self):
        torch.manual_seed(42)
        x1 = torch.randn(5, 5)
        torch.manual_seed(123)
        x2 = torch.randn(5, 5)
        assert not np.array_equal(x1.numpy(), x2.numpy())

    def test_manual_seed_returns_generator(self):
        gen = torch.manual_seed(42)
        assert isinstance(gen, torch.Generator)

    def test_initial_seed(self):
        torch.manual_seed(42)
        assert torch.initial_seed() == 42

    def test_seed_function(self):
        s = torch.seed()
        assert isinstance(s, int)
        assert s > 0


class TestRNGState:
    """Test RNG state save/restore."""

    def test_get_set_rng_state(self):
        torch.manual_seed(42)
        state = torch.get_rng_state()
        x1 = torch.randn(5, 5)
        torch.set_rng_state(state)
        x2 = torch.randn(5, 5)
        np.testing.assert_array_equal(x1.numpy(), x2.numpy())

    def test_rng_state_is_tensor(self):
        state = torch.get_rng_state()
        assert hasattr(state, 'numpy')


class TestGenerator:
    """Test Generator class."""

    def test_generator_cpu(self):
        gen = torch.Generator('cpu')
        gen.manual_seed(42)
        assert gen.initial_seed() == 42

    def test_generator_state(self):
        gen = torch.Generator('cpu')
        gen.manual_seed(42)
        state = gen.get_state()
        assert hasattr(state, 'numpy')

    def test_generator_set_state(self):
        gen = torch.Generator('cpu')
        gen.manual_seed(42)
        state = gen.get_state()
        gen.manual_seed(123)  # Change seed
        gen.set_state(state)  # Restore
        # State should be restored


class TestDropoutCPU:
    """Test dropout on CPU."""

    def test_dropout_training(self):
        torch.manual_seed(42)
        x = torch.randn(100, 100)
        out = nn.functional.dropout(x, p=0.5, training=True)
        # Some elements should be zero
        out_np = out.numpy()
        assert np.any(out_np == 0.0)
        # Non-zero elements should be scaled by 1/(1-p) = 2
        nonzero_mask = out_np != 0.0
        if np.any(nonzero_mask):
            np.testing.assert_allclose(
                out_np[nonzero_mask],
                x.numpy()[nonzero_mask] * 2.0,
                rtol=1e-5
            )

    def test_dropout_eval(self):
        x = torch.randn(5, 5)
        out = nn.functional.dropout(x, p=0.5, training=False)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_dropout_p_zero(self):
        x = torch.randn(5, 5)
        out = nn.functional.dropout(x, p=0.0, training=True)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_dropout_reproducible(self):
        x = torch.randn(10, 10)
        torch.manual_seed(42)
        out1 = nn.functional.dropout(x, p=0.5, training=True)
        torch.manual_seed(42)
        out2 = nn.functional.dropout(x, p=0.5, training=True)
        np.testing.assert_array_equal(out1.numpy(), out2.numpy())

    def test_dropout_module(self):
        layer = nn.Dropout(p=0.5)
        layer.train()
        x = torch.randn(10, 10)
        torch.manual_seed(42)
        out = layer(x)
        out_np = out.numpy()
        assert np.any(out_np == 0.0)

    def test_dropout_module_eval(self):
        layer = nn.Dropout(p=0.5)
        layer.eval()
        x = torch.randn(5, 5)
        out = layer(x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
class TestDropoutNPU:
    """Test dropout on NPU."""

    def test_dropout_npu_training(self):
        torch.npu.manual_seed(42)
        x = torch.randn(100, 100, device="npu")
        out = nn.functional.dropout(x, p=0.5, training=True)
        assert out.device.type == "npu"
        out_np = out.to("cpu").numpy()
        # Some elements should be zero
        assert np.any(out_np == 0.0)

    def test_dropout_npu_eval(self):
        x = torch.randn(5, 5, device="npu")
        out = nn.functional.dropout(x, p=0.5, training=False)
        np.testing.assert_allclose(
            out.to("cpu").numpy(), x.to("cpu").numpy(), rtol=1e-5
        )

    def test_dropout_npu_p_zero(self):
        x = torch.randn(5, 5, device="npu")
        out = nn.functional.dropout(x, p=0.0, training=True)
        np.testing.assert_allclose(
            out.to("cpu").numpy(), x.to("cpu").numpy(), rtol=1e-5
        )

    def test_dropout_npu_module(self):
        layer = nn.Dropout(p=0.5)
        layer.train()
        x = torch.randn(100, 100, device="npu")
        torch.npu.manual_seed(42)
        out = layer(x)
        assert out.device.type == "npu"
        out_np = out.to("cpu").numpy()
        assert np.any(out_np == 0.0)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
class TestRandnNPU:
    """Test randn on NPU."""

    def test_randn_npu(self):
        x = torch.randn(5, 5, device="npu")
        assert x.device.type == "npu"
        assert x.shape == (5, 5)

    def test_randn_npu_dtype(self):
        x = torch.randn(5, 5, device="npu")
        x_np = x.to("cpu").numpy()
        assert x_np.dtype == np.float32

    def test_randn_npu_statistics(self):
        torch.npu.manual_seed(42)
        x = torch.randn(10000, device="npu")
        x_np = x.to("cpu").numpy()
        # Should be approximately N(0, 1)
        assert abs(x_np.mean()) < 0.1
        assert abs(x_np.std() - 1.0) < 0.1


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
class TestNPUSeed:
    """Test NPU seed management."""

    def test_npu_manual_seed(self):
        torch.npu.manual_seed(42)
        assert torch.npu._get_seed() == 42

    def test_npu_manual_seed_all(self):
        torch.npu.manual_seed_all(42)
        assert torch.npu._get_seed() == 42
