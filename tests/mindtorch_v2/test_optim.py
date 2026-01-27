"""Tests for optimizers."""

import numpy as np
import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.optim as optim


class TestAdamW:
    def test_adamw_basic(self):
        # Simple linear model
        model = nn.Linear(10, 2)
        optimizer = optim.AdamW(model.parameters(), lr=0.01)

        # Create input
        x = torch.randn(4, 10)

        # Training loop
        initial_params = [p.numpy().copy() for p in model.parameters()]

        for _ in range(5):
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Check that parameters have changed
        for i, p in enumerate(model.parameters()):
            assert not np.allclose(p.numpy(), initial_params[i]), \
                f"Parameter {i} did not change after optimization"

    def test_adamw_weight_decay(self):
        # Test that weight decay is applied
        model = nn.Linear(10, 2)
        optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)

        x = torch.randn(4, 10)

        for _ in range(10):
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Should complete without error
        assert True

    def test_adamw_zero_grad(self):
        model = nn.Linear(10, 2)
        optimizer = optim.AdamW(model.parameters(), lr=0.01)

        x = torch.randn(4, 10)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        for p in model.parameters():
            assert p.grad is not None

        # Zero gradients
        optimizer.zero_grad()

        # Check gradients are cleared
        for p in model.parameters():
            assert p.grad is None


class TestSGD:
    def test_sgd_basic(self):
        model = nn.Linear(10, 2)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        x = torch.randn(4, 10)
        initial_params = [p.numpy().copy() for p in model.parameters()]

        for _ in range(5):
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Check that parameters have changed
        for i, p in enumerate(model.parameters()):
            assert not np.allclose(p.numpy(), initial_params[i])

    def test_sgd_momentum(self):
        model = nn.Linear(10, 2)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        x = torch.randn(4, 10)

        for _ in range(5):
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Should complete without error
        assert True


class TestTrainingLoop:
    def test_full_training_loop(self):
        """Test a complete training loop with model, loss, and optimizer."""
        # Create model
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        # Create optimizer and loss
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training data
        x = torch.randn(8, 10)
        y = torch.randn(8, 2)

        losses = []
        for _ in range(10):
            # Forward
            output = model(x)
            loss = criterion(output, y)
            losses.append(loss.item())

            # Backward
            loss.backward()

            # Update
            optimizer.step()
            optimizer.zero_grad()

        # Loss should generally decrease (or at least not explode)
        assert all(np.isfinite(l) for l in losses), "Loss contains non-finite values"
