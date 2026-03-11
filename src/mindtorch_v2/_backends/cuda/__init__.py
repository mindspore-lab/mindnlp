from ..common import convert as convert_backend
from ..._dispatch.registry import registry
from . import runtime
from . import storage
from .creation import empty_create, full_create, ones_create, tensor_create, zeros_create
from .ops import add


registry.register("add", "cuda", add)
registry.register("to", "cuda", convert_backend.to_device)
registry.register("tensor", "cuda", tensor_create)
registry.register("zeros", "cuda", zeros_create)
registry.register("ones", "cuda", ones_create)
registry.register("empty", "cuda", empty_create)
registry.register("full", "cuda", full_create)


__all__ = ["runtime", "storage"]
