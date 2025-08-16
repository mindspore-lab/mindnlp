from typing import Any
from mindspore import Generator as msGenerator
import mindspore

from mindnlp import core
from . import _nn
from ..configs import DEVICE_TARGET

DEVICE_MAP = {
    'GPU': 'cuda',
    'Ascend': 'npu',
    'CPU': 'cpu'
}


def _jit_set_profiling_executor(mode):
    pass

def _jit_set_profiling_executor(mode):
    pass

def _jit_set_profiling_mode(mode):
    pass

def _jit_override_can_fuse_on_cpu(mode):
    pass

def _jit_override_can_fuse_on_gpu(mode):
    pass

def _jit_set_texpr_fuser_enabled(mode):
    pass

def _debug_set_autodiff_subgraph_inlining(mode):
    pass

Graph = None
Value = None

DisableTorchFunctionSubclass = None


class device():
    def __init__(self, type=None, index=None):
        if type is not None:
            if isinstance(type, str):
                if ':' in type:
                    if index is not None:
                        raise ValueError("`type` must not include an index because index was "
                                         f"passed explicitly: {type}")
                    _target, _id = type.split(':')
                    _id = int(_id)
                else:
                    _target = type
                    _id = None if _target == 'cpu' else 0
            elif isinstance(type, device):
                if index is not None:
                    raise ValueError("core.device(): When input is core.device, `index` can not be set.")
                _target = type.type
                _id = type.index
            elif isinstance(type, int):
                _id = type
                try:
                    device_target = mindspore.get_current_device().device_target
                except:
                    device_target = mindspore.get_context('device_target')
                _target = DEVICE_MAP[device_target]
            else:
                print(type)
                raise TypeError("core.device(): `type` must be type of 'str' or 'core.device'.")
        else:
            raise ValueError("core.device(): `type` can not be None")

        self.type = _target
        self.index = _id
        if DEVICE_TARGET == 'Ascned' and self.type == 'cuda':
            self.type = 'npu'

    def __repr__(self):
        if self.index is None:
            return f"device(type={self.type})"
        return f"device(type={self.type}, index={self.index})"

    def __eq__(self, __value):
        if not isinstance(__value, device):
            return False
        return hash(self) == hash(__value)

    def __hash__(self):
        return hash(self.type) ^ hash(self.index)

    def __gt__(self, other):
        if self.type == 'cpu':
            return False
        return True

    def __enter__(self):
        # self.prev_idx = torch.cuda._exchange_device(self.idx)
        core._bind.set_device_in_context(self)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # self.idx = torch.cuda._maybe_exchange_device(self.prev_idx)
        core._bind.set_device_in_context(None)
        return False

device_ = device

class Generator(msGenerator):
    def __init__(self, device='cpu'):
        super().__init__()
        if device == 'cuda' and DEVICE_TARGET == 'Ascend':
            device = 'npu'
        self._device = device_(device) if isinstance(device, str) else device

    @property
    def device(self):
        if hasattr(self, '_device'):
            return self._device
        return device('cpu')

default_generator = Generator()

class Tag: pass

def _log_api_usage_once(*args):
    pass

ScriptDict = dict
ScriptList = list