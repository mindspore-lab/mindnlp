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


class TestCPUReproducibility:
    """Test that all CPU random ops are reproducible with manual_seed."""

    def test_randn_reproducible(self):
        torch.manual_seed(42)
        a = torch.randn(5, 5)
        torch.manual_seed(42)
        b = torch.randn(5, 5)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_rand_reproducible(self):
        torch.manual_seed(42)
        a = torch.rand(5, 5)
        torch.manual_seed(42)
        b = torch.rand(5, 5)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_randint_reproducible(self):
        torch.manual_seed(42)
        a = torch.randint(0, 100, size=(5, 5))
        torch.manual_seed(42)
        b = torch.randint(0, 100, size=(5, 5))
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_randperm_reproducible(self):
        torch.manual_seed(42)
        a = torch.randperm(100)
        torch.manual_seed(42)
        b = torch.randperm(100)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_uniform_reproducible(self):
        torch.manual_seed(42)
        a = torch.empty(5, 5).uniform_()
        torch.manual_seed(42)
        b = torch.empty(5, 5).uniform_()
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_normal_reproducible(self):
        torch.manual_seed(42)
        a = torch.empty(5, 5).normal_()
        torch.manual_seed(42)
        b = torch.empty(5, 5).normal_()
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_bernoulli_reproducible(self):
        probs = torch.full((5, 5), 0.5)
        torch.manual_seed(42)
        a = torch.bernoulli(probs)
        torch.manual_seed(42)
        b = torch.bernoulli(probs)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_multinomial_reproducible(self):
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        torch.manual_seed(42)
        a = torch.multinomial(weights, 10, replacement=True)
        torch.manual_seed(42)
        b = torch.multinomial(weights, 10, replacement=True)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_different_seeds_different_results(self):
        torch.manual_seed(42)
        a = torch.randn(100)
        torch.manual_seed(123)
        b = torch.randn(100)
        assert not np.array_equal(a.numpy(), b.numpy())

    def test_sequence_of_ops_reproducible(self):
        """Multiple ops in sequence must produce identical results."""
        torch.manual_seed(42)
        a1 = torch.randn(3, 3)
        a2 = torch.rand(3, 3)
        a3 = torch.randint(0, 10, size=(3, 3))
        a4 = torch.randperm(10)

        torch.manual_seed(42)
        b1 = torch.randn(3, 3)
        b2 = torch.rand(3, 3)
        b3 = torch.randint(0, 10, size=(3, 3))
        b4 = torch.randperm(10)

        np.testing.assert_array_equal(a1.numpy(), b1.numpy())
        np.testing.assert_array_equal(a2.numpy(), b2.numpy())
        np.testing.assert_array_equal(a3.numpy(), b3.numpy())
        np.testing.assert_array_equal(a4.numpy(), b4.numpy())


class TestGeneratorFull:
    """Test Generator class functionality."""

    def test_cpu_generator_independent(self):
        """User generator should be independent from default generator."""
        g = torch.Generator('cpu')
        g.manual_seed(42)
        torch.manual_seed(999)  # set default to different seed
        a = torch.bernoulli(torch.full((100,), 0.5), generator=g)

        g2 = torch.Generator('cpu')
        g2.manual_seed(42)
        b = torch.bernoulli(torch.full((100,), 0.5), generator=g2)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_generator_state_save_restore(self):
        g = torch.Generator('cpu')
        g.manual_seed(42)
        _ = torch.bernoulli(torch.full((10,), 0.5), generator=g)  # advance state
        state = g.get_state()
        a = torch.bernoulli(torch.full((10,), 0.5), generator=g)
        g.set_state(state)
        b = torch.bernoulli(torch.full((10,), 0.5), generator=g)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_initial_seed_default(self):
        g = torch.Generator('cpu')
        assert g.initial_seed() == 67280421310721

    def test_manual_seed_returns_self(self):
        g = torch.Generator('cpu')
        result = g.manual_seed(42)
        assert result is g

    def test_npu_generator(self):
        g = torch.Generator('npu')
        g.manual_seed(42)
        assert g.initial_seed() == 42
        seed, offset = g.philox_engine_inputs(10)
        assert seed == 42 and offset == 0
        seed2, offset2 = g.philox_engine_inputs(10)
        assert seed2 == 42 and offset2 == 10


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
class TestNPUReproducibility:
    """Test that NPU random ops are reproducible with manual_seed."""

    def test_randn_npu_reproducible(self):
        torch.manual_seed(42)
        a = torch.randn(5, 5, device='npu')
        torch.manual_seed(42)
        b = torch.randn(5, 5, device='npu')
        np.testing.assert_array_equal(a.to('cpu').numpy(), b.to('cpu').numpy())

    def test_rand_npu_reproducible(self):
        torch.manual_seed(42)
        a = torch.rand(5, 5, device='npu')
        torch.manual_seed(42)
        b = torch.rand(5, 5, device='npu')
        np.testing.assert_array_equal(a.to('cpu').numpy(), b.to('cpu').numpy())

    def test_uniform_npu_reproducible(self):
        torch.manual_seed(42)
        a = torch.empty(5, 5, device='npu').uniform_()
        torch.manual_seed(42)
        b = torch.empty(5, 5, device='npu').uniform_()
        np.testing.assert_array_equal(a.to('cpu').numpy(), b.to('cpu').numpy())

    def test_normal_npu_reproducible(self):
        torch.manual_seed(42)
        a = torch.empty(5, 5, device='npu').normal_()
        torch.manual_seed(42)
        b = torch.empty(5, 5, device='npu').normal_()
        np.testing.assert_array_equal(a.to('cpu').numpy(), b.to('cpu').numpy())

    def test_dropout_npu_reproducible(self):
        torch.manual_seed(42)
        x = torch.ones(10, 10, device='npu')
        a = nn.functional.dropout(x, p=0.5, training=True)
        torch.manual_seed(42)
        b = nn.functional.dropout(x, p=0.5, training=True)
        np.testing.assert_array_equal(a.to('cpu').numpy(), b.to('cpu').numpy())

    def test_npu_rng_state_save_restore(self):
        torch.manual_seed(42)
        state = torch.npu.get_rng_state()
        a = torch.randn(5, 5, device='npu')
        torch.npu.set_rng_state(state)
        b = torch.randn(5, 5, device='npu')
        np.testing.assert_array_equal(a.to('cpu').numpy(), b.to('cpu').numpy())

    def test_npu_sequence_reproducible(self):
        """Multiple NPU ops in sequence must produce identical results."""
        torch.manual_seed(42)
        a1 = torch.randn(3, 3, device='npu')
        a2 = torch.rand(3, 3, device='npu')
        a3 = torch.empty(3, 3, device='npu').uniform_(-1, 1)
        a4 = torch.empty(3, 3, device='npu').normal_(0, 2)

        torch.manual_seed(42)
        b1 = torch.randn(3, 3, device='npu')
        b2 = torch.rand(3, 3, device='npu')
        b3 = torch.empty(3, 3, device='npu').uniform_(-1, 1)
        b4 = torch.empty(3, 3, device='npu').normal_(0, 2)

        np.testing.assert_array_equal(a1.to('cpu').numpy(), b1.to('cpu').numpy())
        np.testing.assert_array_equal(a2.to('cpu').numpy(), b2.to('cpu').numpy())
        np.testing.assert_array_equal(a3.to('cpu').numpy(), b3.to('cpu').numpy())
        np.testing.assert_array_equal(a4.to('cpu').numpy(), b4.to('cpu').numpy())


class TestGeneratorThreading:
    """Test that generator parameter is threaded through all random ops."""

    def test_randn_with_generator(self):
        g1 = torch.Generator('cpu').manual_seed(42)
        g2 = torch.Generator('cpu').manual_seed(42)
        torch.manual_seed(999)  # different default seed
        a = torch.randn(5, 5, generator=g1)
        b = torch.randn(5, 5, generator=g2)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_randn_generator_independent_of_default(self):
        g = torch.Generator('cpu').manual_seed(42)
        a = torch.randn(5, 5, generator=g)
        # Change default seed; generator result should not change
        torch.manual_seed(0)
        g2 = torch.Generator('cpu').manual_seed(42)
        b = torch.randn(5, 5, generator=g2)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_rand_with_generator(self):
        g1 = torch.Generator('cpu').manual_seed(42)
        g2 = torch.Generator('cpu').manual_seed(42)
        torch.manual_seed(999)
        a = torch.rand(5, 5, generator=g1)
        b = torch.rand(5, 5, generator=g2)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_randint_with_generator(self):
        g1 = torch.Generator('cpu').manual_seed(42)
        g2 = torch.Generator('cpu').manual_seed(42)
        torch.manual_seed(999)
        a = torch.randint(0, 100, size=(5, 5), generator=g1)
        b = torch.randint(0, 100, size=(5, 5), generator=g2)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_randperm_with_generator(self):
        g1 = torch.Generator('cpu').manual_seed(42)
        g2 = torch.Generator('cpu').manual_seed(42)
        torch.manual_seed(999)
        a = torch.randperm(100, generator=g1)
        b = torch.randperm(100, generator=g2)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_uniform_with_generator(self):
        g = torch.Generator('cpu').manual_seed(42)
        a = torch.empty(5, 5).uniform_(generator=g)
        g.manual_seed(42)
        b = torch.empty(5, 5).uniform_(generator=g)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_normal_with_generator(self):
        g = torch.Generator('cpu').manual_seed(42)
        a = torch.empty(5, 5).normal_(generator=g)
        g.manual_seed(42)
        b = torch.empty(5, 5).normal_(generator=g)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_kaiming_uniform_with_generator(self):
        g1 = torch.Generator('cpu').manual_seed(42)
        g2 = torch.Generator('cpu').manual_seed(42)
        torch.manual_seed(999)
        t1 = torch.empty(64, 32)
        t2 = torch.empty(64, 32)
        torch.nn.init.kaiming_uniform_(t1, generator=g1)
        torch.nn.init.kaiming_uniform_(t2, generator=g2)
        np.testing.assert_array_equal(t1.numpy(), t2.numpy())

    def test_kaiming_normal_with_generator(self):
        g1 = torch.Generator('cpu').manual_seed(42)
        g2 = torch.Generator('cpu').manual_seed(42)
        t1 = torch.empty(64, 32)
        t2 = torch.empty(64, 32)
        torch.nn.init.kaiming_normal_(t1, generator=g1)
        torch.nn.init.kaiming_normal_(t2, generator=g2)
        np.testing.assert_array_equal(t1.numpy(), t2.numpy())

    def test_xavier_uniform_with_generator(self):
        g1 = torch.Generator('cpu').manual_seed(42)
        g2 = torch.Generator('cpu').manual_seed(42)
        t1 = torch.empty(64, 32)
        t2 = torch.empty(64, 32)
        torch.nn.init.xavier_uniform_(t1, generator=g1)
        torch.nn.init.xavier_uniform_(t2, generator=g2)
        np.testing.assert_array_equal(t1.numpy(), t2.numpy())

    def test_xavier_normal_with_generator(self):
        g1 = torch.Generator('cpu').manual_seed(42)
        g2 = torch.Generator('cpu').manual_seed(42)
        t1 = torch.empty(64, 32)
        t2 = torch.empty(64, 32)
        torch.nn.init.xavier_normal_(t1, generator=g1)
        torch.nn.init.xavier_normal_(t2, generator=g2)
        np.testing.assert_array_equal(t1.numpy(), t2.numpy())

    def test_init_uniform_with_generator(self):
        g1 = torch.Generator('cpu').manual_seed(42)
        g2 = torch.Generator('cpu').manual_seed(42)
        t1 = torch.empty(10, 10)
        t2 = torch.empty(10, 10)
        torch.nn.init.uniform_(t1, -1, 1, generator=g1)
        torch.nn.init.uniform_(t2, -1, 1, generator=g2)
        np.testing.assert_array_equal(t1.numpy(), t2.numpy())

    def test_init_normal_with_generator(self):
        g1 = torch.Generator('cpu').manual_seed(42)
        g2 = torch.Generator('cpu').manual_seed(42)
        t1 = torch.empty(10, 10)
        t2 = torch.empty(10, 10)
        torch.nn.init.normal_(t1, 0, 1, generator=g1)
        torch.nn.init.normal_(t2, 0, 1, generator=g2)
        np.testing.assert_array_equal(t1.numpy(), t2.numpy())

    def test_trunc_normal_with_generator(self):
        g1 = torch.Generator('cpu').manual_seed(42)
        g2 = torch.Generator('cpu').manual_seed(42)
        t1 = torch.empty(10, 10)
        t2 = torch.empty(10, 10)
        torch.nn.init.trunc_normal_(t1, generator=g1)
        torch.nn.init.trunc_normal_(t2, generator=g2)
        np.testing.assert_array_equal(t1.numpy(), t2.numpy())

    def test_two_generators_independent(self):
        """Two generators with different seeds produce different results."""
        g1 = torch.Generator('cpu').manual_seed(42)
        g2 = torch.Generator('cpu').manual_seed(123)
        a = torch.randn(5, 5, generator=g1)
        b = torch.randn(5, 5, generator=g2)
        assert not np.array_equal(a.numpy(), b.numpy())

    def test_generator_does_not_affect_default(self):
        """Using a generator should not advance the default RNG state."""
        torch.manual_seed(42)
        expected = torch.randn(5, 5)

        torch.manual_seed(42)
        g = torch.Generator('cpu').manual_seed(999)
        _ = torch.randn(5, 5, generator=g)  # should not affect default
        actual = torch.randn(5, 5)
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
