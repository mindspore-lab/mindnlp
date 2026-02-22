import mindtorch_v2 as torch
from mindtorch_v2._dispatch.dispatcher import dispatch
from mindtorch_v2._dispatch.keys import DispatchKey
from mindtorch_v2._dispatch.registry import registry


def test_functionalize_writes_back_all_mutations():
    registry.register_schema("swap", "swap(Tensor a, Tensor b) -> (Tensor, Tensor)")
    registry.register_schema("swap_", "swap_(Tensor(a!) a, Tensor(b!) b) -> Tensor")

    def swap_kernel(a, b):
        out_a = torch.tensor(b._numpy_view().copy(), device=a.device)
        out_b = torch.tensor(a._numpy_view().copy(), device=a.device)
        return out_a, out_b

    registry.register_kernel("swap", DispatchKey.CPU, swap_kernel)
    registry.register_kernel("swap_", DispatchKey.CPU, swap_kernel)

    a = torch.tensor([1.0])
    b = torch.tensor([2.0])

    with torch.functionalize():
        dispatch("swap_", a.device.type, a, b)

    assert a.storage().data.tolist() == [2.0]
    assert b.storage().data.tolist() == [1.0]
