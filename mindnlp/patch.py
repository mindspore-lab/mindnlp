import numpy as np
import mindspore
from mindspore import Tensor

def infer_value_for_BroadcastTo(x, shape):
    """Infer value for BroadcastTo op."""
    def none_in_tuple_or_list(x):
        return isinstance(x, (tuple, list)) and None in x
    if shape is None or none_in_tuple_or_list(shape) or x is None:
        return None

    shape = list(shape)
    for idx, s in enumerate(shape):
        if s == -1:
            shape[idx] = x.shape[idx]

    np_data = np.broadcast_to(x.asnumpy(), shape)
    return Tensor(np_data)

mindspore.ops.operations.manually_defined.ops_def.infer_value_for_BroadcastTo = infer_value_for_BroadcastTo
