from mindnlp import core
from .dispatcher import dispatcher

def execute(func_name, *args, **kwargs):
    out, device = dispatcher.dispatch(func_name, *args, **kwargs)
    if not isinstance(out, (tuple, list)):
        out._device = device
    else:
        for i in out:
            if isinstance(i, core.Tensor):
                i._device = device
    return out

