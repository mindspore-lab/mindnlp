import mindtorch_v2 as torch
from mindtorch_v2._dispatch.keys import DispatchKey
from mindtorch_v2._dispatch.registry import registry


def test_registry_mutation_step1():
    def bad_add(a, b):
        return torch.tensor([99.0], device=a.device)

    registry.register_kernel("add", DispatchKey.CPU, bad_add)

    out = torch.add(torch.tensor([1.0]), torch.tensor([2.0]))
    assert out.storage().data.tolist() == [99.0]


def test_registry_mutation_step2_restores_baseline_add():
    out = torch.add(torch.tensor([1.0]), torch.tensor([2.0]))
    assert out.storage().data.tolist() == [3.0]
