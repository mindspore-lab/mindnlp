import numpy as np
import pytest

import mindtorch_v2 as torch


NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_310b_watchlist_remainder_lerp_isclose():
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0], device="npu", dtype=torch.float32)
    y = torch.tensor([2.0, 0.5, -0.5, -2.0], device="npu", dtype=torch.float32)

    rem = torch.remainder(x, y).to("cpu").numpy()
    np.testing.assert_allclose(rem, np.remainder(np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32),
                                                 np.array([2.0, 0.5, -0.5, -2.0], dtype=np.float32)),
                               atol=1e-4, rtol=1e-4)

    lerp = torch.lerp(x, y, 0.25).to("cpu").numpy()
    np.testing.assert_allclose(lerp,
                               (np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
                                + 0.25 * (np.array([2.0, 0.5, -0.5, -2.0], dtype=np.float32)
                                          - np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32))),
                               atol=1e-4, rtol=1e-4)

    close = torch.isclose(x, y, rtol=1e-5, atol=1e-8).to("cpu").numpy()
    np.testing.assert_array_equal(close, np.isclose(np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32),
                                                    np.array([2.0, 0.5, -0.5, -2.0], dtype=np.float32),
                                                    rtol=1e-5, atol=1e-8))


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_310b_watchlist_rng_reproducibility():
    torch.npu.manual_seed(42)
    a1 = torch.randn(16, device="npu")
    a2 = torch.rand(16, device="npu")
    a3 = torch.empty(16, device="npu").uniform_(-1, 1)
    a4 = torch.empty(16, device="npu").normal_(0, 2)

    torch.npu.manual_seed(42)
    b1 = torch.randn(16, device="npu")
    b2 = torch.rand(16, device="npu")
    b3 = torch.empty(16, device="npu").uniform_(-1, 1)
    b4 = torch.empty(16, device="npu").normal_(0, 2)

    np.testing.assert_array_equal(a1.to("cpu").numpy(), b1.to("cpu").numpy())
    np.testing.assert_array_equal(a2.to("cpu").numpy(), b2.to("cpu").numpy())
    np.testing.assert_array_equal(a3.to("cpu").numpy(), b3.to("cpu").numpy())
    np.testing.assert_array_equal(a4.to("cpu").numpy(), b4.to("cpu").numpy())


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_310b_watchlist_random_stats_smoke():
    torch.npu.manual_seed(2026)
    x = torch.randn(4096, device="npu", dtype=torch.float32).to("cpu").numpy()
    u = torch.rand(4096, device="npu", dtype=torch.float32).to("cpu").numpy()

    # Loose bounds for smoke-level stability checks.
    assert abs(float(x.mean())) < 0.2
    assert 0.6 < float(x.std()) < 1.4
    assert 0.0 <= float(u.min()) <= 1.0
    assert 0.0 <= float(u.max()) <= 1.0
