"""Test DDP gradient_as_bucket_view feature."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class ModelWithUnused(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.unused = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc1(x)


def test_gradient_as_bucket_view():
    """Verify bucket views are initialized and gradients are correct."""
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29520'

    dist.init_process_group(backend='gloo', rank=0, world_size=1)

    model = SimpleModel()
    ddp_model = nn.parallel.DistributedDataParallel(
        model,
        gradient_as_bucket_view=True,
    )

    # Verify bucket views were initialized
    assert len(ddp_model.reducer._bucket_buffers) > 0
    assert len(ddp_model.reducer._bucket_views) > 0
    assert len(ddp_model.reducer._bucket_offsets) > 0

    x = torch.randn(4, 10)
    output = ddp_model(x)
    loss = output.sum()
    loss.backward()

    # Verify gradients exist
    assert model.fc1.weight.grad is not None
    assert model.fc2.weight.grad is not None
    assert model.fc1.bias.grad is not None
    assert model.fc2.bias.grad is not None

    # Verify gradients are non-zero (model was used)
    assert not torch.allclose(model.fc1.weight.grad, torch.zeros_like(model.fc1.weight.grad))

    dist.destroy_process_group()
    print("PASS: test_gradient_as_bucket_view")


def test_bucket_view_with_static_graph():
    """Test both features together."""
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29521'

    dist.init_process_group(backend='gloo', rank=0, world_size=1)

    model = ModelWithUnused()
    ddp_model = nn.parallel.DistributedDataParallel(
        model,
        gradient_as_bucket_view=True,
        static_graph=True,
    )

    # Two forward/backward passes to exercise caching
    for i in range(2):
        x = torch.randn(4, 10)
        output = ddp_model(x)
        loss = output.sum()
        loss.backward()

    assert model.fc1.weight.grad is not None
    assert ddp_model.reducer._cached_unused_param_indices is not None

    # Unused param should have zero grad
    assert model.unused.weight.grad is not None
    assert torch.allclose(model.unused.weight.grad, torch.zeros_like(model.unused.weight.grad))

    dist.destroy_process_group()
    print("PASS: test_bucket_view_with_static_graph")


def test_bucket_view_multiple_iterations():
    """Verify gradients are correct across multiple iterations."""
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29522'

    dist.init_process_group(backend='gloo', rank=0, world_size=1)

    model = SimpleModel()
    ddp_model = nn.parallel.DistributedDataParallel(
        model,
        gradient_as_bucket_view=True,
    )

    for _ in range(3):
        x = torch.randn(4, 10)
        output = ddp_model(x)
        loss = output.sum()
        loss.backward()

        # Gradients should exist each iteration
        assert model.fc1.weight.grad is not None
        assert model.fc2.weight.grad is not None

    dist.destroy_process_group()
    print("PASS: test_bucket_view_multiple_iterations")


if __name__ == '__main__':
    test_gradient_as_bucket_view()
    test_bucket_view_with_static_graph()
    test_bucket_view_multiple_iterations()
    print("\nAll gradient_as_bucket_view tests passed!")
