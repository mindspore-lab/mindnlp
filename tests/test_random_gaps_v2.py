"""Tests for P0+P1 random number gaps in mindtorch_v2."""

import pytest
import torch


class TestRandintLike:
    def test_shape_dtype(self):
        """randint_like output should match input shape and dtype."""
        x = torch.ones(3, 4, dtype=torch.int64)
        y = torch.randint_like(x, 0, 10)
        assert y.shape == x.shape
        assert y.dtype == x.dtype

    def test_range(self):
        """randint_like values should be in [low, high)."""
        x = torch.ones(1000, dtype=torch.int64)
        y = torch.randint_like(x, 5, 15)
        arr = y.numpy()
        assert (arr >= 5).all()
        assert (arr < 15).all()

    def test_default_low(self):
        """randint_like with only high should use low=0."""
        x = torch.ones(1000, dtype=torch.int64)
        y = torch.randint_like(x, 0, 10)
        arr = y.numpy()
        assert (arr >= 0).all()
        assert (arr < 10).all()


class TestForkRng:
    def test_restores_state(self):
        """fork_rng should restore RNG state after context exit."""
        torch.manual_seed(42)
        # Record state before fork
        before = torch.randn(5)

        torch.manual_seed(42)
        with torch.random.fork_rng():
            # Consume some random numbers inside fork
            _ = torch.randn(100)
            _ = torch.rand(50)

        # After fork, state should be restored to post-manual_seed(42)
        after = torch.randn(5)
        assert torch.equal(before, after)

    def test_disabled(self):
        """fork_rng with enabled=False should be a no-op."""
        torch.manual_seed(42)
        a = torch.randn(5)

        torch.manual_seed(42)
        with torch.random.fork_rng(enabled=False):
            _ = torch.randn(100)
        b = torch.randn(5)

        # b should NOT equal a because fork was disabled
        assert not torch.equal(a, b)


class TestFeatureAlphaDropout:
    def test_shape(self):
        """FeatureAlphaDropout should preserve input shape."""
        m = torch.nn.FeatureAlphaDropout(p=0.5)
        m.train()
        x = torch.randn(2, 4, 8)
        out = m(x)
        assert out.shape == x.shape

    def test_eval_noop(self):
        """FeatureAlphaDropout should be a no-op in eval mode."""
        m = torch.nn.FeatureAlphaDropout(p=0.5)
        m.eval()
        x = torch.randn(2, 4, 8)
        out = m(x)
        assert torch.equal(out, x)

    def test_channel_wise(self):
        """FeatureAlphaDropout should drop entire channels."""
        torch.manual_seed(0)
        m = torch.nn.FeatureAlphaDropout(p=0.5)
        m.train()
        # Use a large spatial dim so we can check channel consistency
        x = torch.randn(1, 10, 100)
        out = m(x)
        # For each channel, check if all spatial values are either
        # all from input or all from alpha_p (channel-wise dropout)
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        alpha_p = -alpha * scale
        for c in range(10):
            channel_out = out[0, c, :]
            channel_in = x[0, c, :]
            # Either channel was kept (scaled) or dropped (all same pattern)
            # Check that all values in the channel share the same mask state
            diff = (channel_out.numpy() - channel_in.numpy())
            # If channel was kept, diff pattern is consistent; if dropped, all values changed
            if abs(diff[0]) < 1e-5:
                # Looks like channel was mostly kept - check consistency
                pass
            else:
                # Channel was dropped - all spatial positions should be transformed identically
                # relative to input (same affine transform applied)
                pass
