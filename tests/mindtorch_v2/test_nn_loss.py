"""Tests for loss functions and nn.init."""

import numpy as np
import mindtorch_v2 as torch
import mindtorch_v2.nn as nn


class TestMSELoss:
    def test_mse_loss_mean(self):
        loss_fn = nn.MSELoss(reduction='mean')
        input = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 4.0])
        loss = loss_fn(input, target)
        # (0 + 0 + 1) / 3 = 0.333...
        expected = 1.0 / 3.0
        np.testing.assert_almost_equal(loss.item(), expected, decimal=5)

    def test_mse_loss_sum(self):
        loss_fn = nn.MSELoss(reduction='sum')
        input = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 4.0])
        loss = loss_fn(input, target)
        expected = 1.0  # 0 + 0 + 1
        np.testing.assert_almost_equal(loss.item(), expected, decimal=5)

    def test_mse_loss_none(self):
        loss_fn = nn.MSELoss(reduction='none')
        input = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 4.0])
        loss = loss_fn(input, target)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(loss.numpy(), expected)


class TestCrossEntropyLoss:
    def test_cross_entropy_basic(self):
        loss_fn = nn.CrossEntropyLoss()
        # 2 samples, 3 classes
        input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        target = torch.tensor([2, 0])  # correct class for first, wrong for second
        loss = loss_fn(input, target)
        assert loss.item() > 0  # Loss should be positive

    def test_cross_entropy_perfect_prediction(self):
        loss_fn = nn.CrossEntropyLoss()
        # High logit for correct class
        input = torch.tensor([[10.0, 0.0, 0.0]])
        target = torch.tensor([0])
        loss = loss_fn(input, target)
        # Should be very small
        assert loss.item() < 0.1


class TestBCEWithLogitsLoss:
    def test_bce_with_logits_basic(self):
        loss_fn = nn.BCEWithLogitsLoss()
        input = torch.tensor([0.0, 0.0, 0.0])
        target = torch.tensor([1.0, 0.0, 1.0])
        loss = loss_fn(input, target)
        # At logit=0, sigmoid=0.5, BCE = -0.5*log(0.5) - 0.5*log(0.5) = log(2)
        expected = np.log(2)
        np.testing.assert_almost_equal(loss.item(), expected, decimal=5)


class TestNNInit:
    def test_uniform(self):
        t = torch.zeros(10, 10)
        nn.init.uniform_(t, a=-1.0, b=1.0)
        arr = t.numpy()
        assert arr.min() >= -1.0
        assert arr.max() <= 1.0
        assert arr.std() > 0  # Not all same value

    def test_normal(self):
        t = torch.zeros(100, 100)
        nn.init.normal_(t, mean=0.0, std=1.0)
        arr = t.numpy()
        # Check mean and std are approximately correct
        np.testing.assert_almost_equal(arr.mean(), 0.0, decimal=1)
        np.testing.assert_almost_equal(arr.std(), 1.0, decimal=1)

    def test_zeros(self):
        t = torch.ones(10, 10)
        nn.init.zeros_(t)
        np.testing.assert_array_equal(t.numpy(), np.zeros((10, 10)))

    def test_ones(self):
        t = torch.zeros(10, 10)
        nn.init.ones_(t)
        np.testing.assert_array_equal(t.numpy(), np.ones((10, 10)))

    def test_xavier_uniform(self):
        t = torch.zeros(64, 32)
        nn.init.xavier_uniform_(t)
        arr = t.numpy()
        # Check values are in reasonable range
        fan_in, fan_out = 32, 64
        std = np.sqrt(2.0 / (fan_in + fan_out))
        bound = np.sqrt(3.0) * std
        assert arr.min() >= -bound - 0.1
        assert arr.max() <= bound + 0.1

    def test_kaiming_uniform(self):
        t = torch.zeros(64, 32)
        nn.init.kaiming_uniform_(t)
        arr = t.numpy()
        assert arr.std() > 0  # Not all same

    def test_kaiming_normal(self):
        t = torch.zeros(64, 32)
        nn.init.kaiming_normal_(t)
        arr = t.numpy()
        assert arr.std() > 0  # Not all same
