from .registry import registry


def register_schemas():
    registry.register_schema("add_", "add_(Tensor(a!) self, Tensor other) -> Tensor")
    registry.register_schema("mul_", "mul_(Tensor(a!) self, Tensor other) -> Tensor")
    registry.register_schema("relu_", "relu_(Tensor(a!) self) -> Tensor")
    registry.register_schema("zero_", "zero_(Tensor(a!) self) -> Tensor")
