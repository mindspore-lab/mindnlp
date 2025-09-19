# mypy: allow-untyped-defs
import mindtorch
from mindtorch.distributed._shard.sharded_tensor import _sharded_op_impl


# This is used by `_apply()` within module.py to set new
# parameters after apply a certain method, we should follow
# the future behavior of overwriting the existing tensor
# instead of doing in-place change using `.data = `.
# @_sharded_op_impl(mindtorch._has_compatible_shallow_copy_type)
def tensor_has_compatible_shallow_copy_type(types, args=(), kwargs=None, pg=None):
    return False
