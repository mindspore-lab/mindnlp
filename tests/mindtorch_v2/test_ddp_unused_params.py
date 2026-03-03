"""Test DDP with find_unused_parameters=True."""

import os
import sys
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist


class ModelWithUnusedParam(nn.Module):
    """Model with a parameter that may be unused depending on input."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x, use_fc2=True):
        x = self.fc1(x)
        if use_fc2:
            x = self.fc2(x)
        x = self.fc3(x)
        return x


def run_ddp_test(rank, world_size, use_fc2):
    """Run DDP test on a single rank."""
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    # Initialize process group
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    # Create model
    model = ModelWithUnusedParam()
    ddp_model = nn.parallel.DistributedDataParallel(
        model,
        find_unused_parameters=True
    )

    # Forward pass
    x = torch.randn(4, 10)
    output = ddp_model(x, use_fc2=use_fc2)
    loss = output.sum()

    # Backward pass - should not hang even with unused parameters
    loss.backward()

    # Verify gradients exist for all parameters
    for name, param in ddp_model.named_parameters():
        if use_fc2:
            # All params should have gradients
            assert param.grad is not None, f"{name} should have gradient"
        else:
            # fc2 is unused, but should still have zero gradient
            if 'fc2' in name:
                assert param.grad is not None, f"{name} should have zero gradient"
                # Check it's actually zero
                assert torch.allclose(param.grad, torch.zeros_like(param.grad)), \
                    f"{name} gradient should be zero"
            else:
                assert param.grad is not None, f"{name} should have gradient"

    dist.destroy_process_group()
    return True


def test_ddp_unused_params_single_process():
    """Test find_unused_parameters with single process (no actual distributed)."""
    # This test doesn't require actual multi-process setup
    model = ModelWithUnusedParam()

    # Test without DDP first to understand expected behavior
    x = torch.randn(4, 10)
    output = model(x, use_fc2=False)
    loss = output.sum()
    loss.backward()

    # fc2 should have no gradient since it wasn't used
    assert model.fc2.weight.grad is None, "fc2 should not have gradient without DDP"

    # Now test with DDP and find_unused_parameters=True
    # Initialize a fake process group for single-process testing
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'

    dist.init_process_group(backend='gloo', rank=0, world_size=1)

    model2 = ModelWithUnusedParam()
    ddp_model = nn.parallel.DistributedDataParallel(
        model2,
        find_unused_parameters=True
    )

    x = torch.randn(4, 10)
    output = ddp_model(x, use_fc2=False)
    loss = output.sum()
    loss.backward()

    # With find_unused_parameters=True, fc2 should have zero gradient
    assert model2.fc2.weight.grad is not None, "fc2 should have zero gradient with find_unused_parameters"
    assert torch.allclose(model2.fc2.weight.grad, torch.zeros_like(model2.fc2.weight.grad)), \
        "fc2 gradient should be zero"

    # Used parameters should have non-zero gradients
    assert model2.fc1.weight.grad is not None
    assert not torch.allclose(model2.fc1.weight.grad, torch.zeros_like(model2.fc1.weight.grad)), \
        "fc1 should have non-zero gradient"

    dist.destroy_process_group()


def test_ddp_all_params_used():
    """Test that find_unused_parameters doesn't break normal case where all params are used."""
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'

    dist.init_process_group(backend='gloo', rank=0, world_size=1)

    model = ModelWithUnusedParam()
    ddp_model = nn.parallel.DistributedDataParallel(
        model,
        find_unused_parameters=True
    )

    x = torch.randn(4, 10)
    output = ddp_model(x, use_fc2=True)  # All params used
    loss = output.sum()
    loss.backward()

    # All parameters should have non-zero gradients
    for name, param in ddp_model.named_parameters():
        assert param.grad is not None, f"{name} should have gradient"
        # Check gradients are not all zero (at least some non-zero values)
        assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
            f"{name} should have non-zero gradient"

    dist.destroy_process_group()


if __name__ == '__main__':
    # Run tests
    print("Testing DDP with unused parameters (single process)...")
    test_ddp_unused_params_single_process()
    print("✓ Single process test passed")

    print("\nTesting DDP with all parameters used...")
    test_ddp_all_params_used()
    print("✓ All params used test passed")

    print("\n✓ All tests passed!")
