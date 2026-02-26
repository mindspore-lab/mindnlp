import numpy as np
import mindtorch_v2 as torch


def test_pipeline_records_functionalized_inplace():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    with torch.functionalize():
        with torch.pipeline():
            out = a.add_(b)
            assert out._pending is True
    assert out._pending is False



def test_pipeline_functionalize_inplace_returns_self_and_writebacks():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    with torch.functionalize():
        with torch.pipeline():
            out = a.add_(b)
            assert out is a
            assert out._pending is True
    assert out._pending is False
    assert a.storage().data.tolist() == [4.0, 6.0]


def test_pipeline_functionalize_multi_mutation_uses_return_alias_mapping():
    from mindtorch_v2._dispatch.dispatcher import dispatch
    from mindtorch_v2._dispatch.keys import DispatchKey
    from mindtorch_v2._dispatch.registry import registry

    registry.register_schema(
        "swap_alias",
        "swap_alias(Tensor a, Tensor b) -> (Tensor(b), Tensor(a))",
    )
    registry.register_schema(
        "swap_alias_",
        "swap_alias_(Tensor(a!) a, Tensor(b!) b) -> Tensor",
    )

    def _raw_cpu_tensor(values, ref):
        from mindtorch_v2._storage import typed_storage_from_numpy
        from mindtorch_v2._tensor import Tensor

        arr = np.array(values, dtype=np.float32)
        storage = typed_storage_from_numpy(arr, ref.dtype, device=ref.device)
        stride = tuple(np.array(arr.strides) // arr.itemsize)
        return Tensor(storage, arr.shape, stride)

    def swap_alias_kernel(a, b):
        # Build outputs without calling high-level dispatch to avoid nested pipeline enqueue.
        out_for_b = _raw_cpu_tensor([10.0], a)
        out_for_a = _raw_cpu_tensor([20.0], a)
        return out_for_b, out_for_a

    registry.register_kernel("swap_alias", DispatchKey.CPU, swap_alias_kernel)
    registry.register_kernel("swap_alias_", DispatchKey.CPU, swap_alias_kernel)

    from mindtorch_v2._backends.meta.infer import infer_binary

    def swap_alias_meta(a, b):
        return infer_binary(a, b)

    registry.register_kernel("swap_alias_", DispatchKey.Meta, swap_alias_meta)

    a = torch.tensor([1.0])
    b = torch.tensor([2.0])

    with torch.functionalize():
        with torch.pipeline():
            out = dispatch("swap_alias_", a.device.type, a, b)
            assert out is a
            assert out._pending is True

    assert out._pending is False
    assert a.storage().data.tolist() == [20.0]
    assert b.storage().data.tolist() == [10.0]
