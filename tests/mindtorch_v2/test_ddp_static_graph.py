"""Test DDP static_graph feature."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist


class ModelWithUnused(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.unused = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def test_static_graph_caches_unused_params():
    """Verify static_graph caches unused param indices after first pass."""
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29510'

    dist.init_process_group(backend='gloo', rank=0, world_size=1)

    model = ModelWithUnused()
    ddp_model = nn.parallel.DistributedDataParallel(
        model,
        static_graph=True,
    )

    # Before first forward, cache should be empty
    assert ddp_model.reducer._cached_unused_param_indices is None

    # First forward/backward — detects unused params and caches
    x = torch.randn(4, 10)
    output = ddp_model(x)
    loss = output.sum()
    loss.backward()

    assert ddp_model.reducer._cached_unused_param_indices is not None
    assert len(ddp_model.reducer._cached_unused_param_indices) > 0

    # Unused param should have zero grad
    assert model.unused.weight.grad is not None
    assert torch.allclose(model.unused.weight.grad, torch.zeros_like(model.unused.weight.grad))

    # Used params should have non-zero grad
    assert model.fc1.weight.grad is not None
    assert not torch.allclose(model.fc1.weight.grad, torch.zeros_like(model.fc1.weight.grad))

    # Second forward/backward — uses cache (no graph traversal)
    cached = ddp_model.reducer._cached_unused_param_indices
    x2 = torch.randn(4, 10)
    output2 = ddp_model(x2)
    loss2 = output2.sum()
    loss2.backward()

    # Cache should be the same object (not rebuilt)
    assert ddp_model.reducer._cached_unused_param_indices is cached

    # Gradients should still be correct
    assert model.fc1.weight.grad is not None
    assert model.unused.weight.grad is not None

    dist.destroy_process_group()
    print("PASS: test_static_graph_caches_unused_params")


def test_static_graph_implies_find_unused():
    """static_graph=True should handle unused params even without find_unused_parameters."""
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29511'

    dist.init_process_group(backend='gloo', rank=0, world_size=1)

    model = ModelWithUnused()
    ddp_model = nn.parallel.DistributedDataParallel(
        model,
        static_graph=True,
        find_unused_parameters=False,
    )

    x = torch.randn(4, 10)
    output = ddp_model(x)
    loss = output.sum()
    loss.backward()

    # Should not hang — unused params handled by static_graph
    assert model.fc1.weight.grad is not None
    assert model.unused.weight.grad is not None

    dist.destroy_process_group()
    print("PASS: test_static_graph_implies_find_unused")


if __name__ == '__main__':
    test_static_graph_caches_unused_params()
    test_static_graph_implies_find_unused()
    print("\nAll static_graph tests passed!")
