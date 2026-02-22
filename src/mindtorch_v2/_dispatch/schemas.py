from .registry import registry


def register_schemas():
    registry.register_schema("add", "add(Tensor input, Tensor other, *, Scalar alpha=1) -> Tensor")
    registry.register_error_overrides(
        "add",
        {
            "missing": "{name}() received an invalid combination of arguments - got {got}, but expected (Tensor input, Tensor other, *, Number alpha = 1, Tensor out = None)",
        },
    )
    registry.register_schema("mul", "mul(Tensor input, Tensor other) -> Tensor")
    registry.register_schema("matmul", "matmul(Tensor input, Tensor other) -> Tensor")
    registry.register_schema("relu", "relu(Tensor input) -> Tensor")
    registry.register_schema("sum", "sum(Tensor input, int[]? dim=None, bool keepdim=False, Dtype? dtype=None) -> Tensor")
    registry.register_error_overrides(
        "sum",
        {
            "missing": "{name}() received an invalid combination of arguments - got {got}, but expected one of:\n * (Tensor input, *, torch.dtype dtype = None)\n * (Tensor input, tuple of ints dim, bool keepdim = False, *, torch.dtype dtype = None, Tensor out = None)\n * (Tensor input, tuple of names dim, bool keepdim = False, *, torch.dtype dtype = None, Tensor out = None)\n",
            "unexpected": "{name}() received an invalid combination of arguments - got {got}, but expected one of:\n * (Tensor input, *, torch.dtype dtype = None)\n * (Tensor input, tuple of ints dim, bool keepdim = False, *, torch.dtype dtype = None, Tensor out = None)\n * (Tensor input, tuple of names dim, bool keepdim = False, *, torch.dtype dtype = None, Tensor out = None)\n",
        },
    )
    registry.register_schema("reshape", "reshape(Tensor input, int[] shape) -> Tensor")
    registry.register_error_overrides(
        "reshape",
        {
            "missing": '{name}() missing 2 required positional argument: "input", "shape"',
        },
    )
    registry.register_schema("view", "view(Tensor input, int[] shape) -> Tensor")
    registry.register_error_overrides(
        "view",
        {
            "missing": "{name}() received an invalid combination of arguments - got {got}, but expected one of:\n * (torch.dtype dtype)\n * (tuple of ints size)\n",
        },
    )
    registry.register_schema("transpose", "transpose(Tensor input, int dim0, int dim1) -> Tensor")
    registry.register_error_overrides(
        "transpose",
        {
            "missing": "{name}() received an invalid combination of arguments - got {got}, but expected one of:\n * (Tensor input, int dim0, int dim1)\n * (Tensor input, name dim0, name dim1)\n",
        },
    )
    registry.register_schema("to", "to(Tensor input, Device? device=None, Dtype? dtype=None, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor")
    registry.register_schema("tensor", "tensor(Tensor data, *, Dtype? dtype=None, Device? device=None, bool requires_grad=False) -> Tensor")
    registry.register_schema("zeros", "zeros(int[] size, *, Dtype? dtype=None, Device? device=None, MemoryFormat? memory_format=None) -> Tensor")
    registry.register_schema("ones", "ones(int[] size, *, Dtype? dtype=None, Device? device=None, MemoryFormat? memory_format=None) -> Tensor")
    registry.register_schema("empty", "empty(int[] size, *, Dtype? dtype=None, Device? device=None, MemoryFormat? memory_format=None) -> Tensor")
    registry.register_schema("add_", "add_(Tensor(a!) self, Tensor other) -> Tensor")
    registry.register_schema("mul_", "mul_(Tensor(a!) self, Tensor other) -> Tensor")
    registry.register_schema("relu_", "relu_(Tensor(a!) self) -> Tensor")
    registry.register_schema("zero_", "zero_(Tensor(a!) self) -> Tensor")
