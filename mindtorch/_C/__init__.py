from typing import Any
from enum import Enum, IntEnum

import mindspore
from mindspore.ops.operations._inner_ops import Generator as GeneratorOp

import mindtorch
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
                    raise ValueError("mindtorch.device(): When input is mindtorch.device, `index` can not be set.")
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
                raise TypeError("mindtorch.device(): `type` must be type of 'str' or 'mindtorch.device'.")
        else:
            raise ValueError("mindtorch.device(): `type` can not be None")

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
        mindtorch._bind.set_device_in_context(self)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # self.idx = torch.cuda._maybe_exchange_device(self.prev_idx)
        mindtorch._bind.set_device_in_context(None)
        return False

device_ = device

STEP = 0
SEED = 1
GET_STATE = 2
SET_STATE = 3
MANUAL_SEED = 4
INITIAL_SEED = 5

class Generator:
    def __init__(self, device='cpu'):
        if device == 'cuda' and DEVICE_TARGET == 'Ascend':
            device = 'npu'
        self._device = device_(device) if isinstance(device, str) else device

        self._seed = mindspore.Tensor(0)
        self._offset = mindspore.Tensor(0)
        self._generator = GeneratorOp().set_device("CPU")
        self._generator.add_prim_attr("manual_seed", False)


    @property
    def device(self):
        if hasattr(self, '_device'):
            return self._device
        return device('cpu')

    def set_state(self, state):
        """
        Sets the generator state.

        Args:
            state (tensor): target state of the generator.
        """
        self._generator(SET_STATE, (self._seed, self._offset, state))

    def get_state(self):
        """
        Get the generator state.

        Returns:
            Tensor, generator state.
        """
        return self._generator(GET_STATE, (self._seed, self._offset))[2]

    def seed(self):  # pylint: disable=redefined-outer-name
        """
        Seed generator with random number.

        Returns:
            Randomly generated seeds, the type is int.
        """
        current_seed = self._generator(
            SEED, (self._seed, self._offset))[0]
        return current_seed.item()

    def manual_seed(self, seed):  # pylint: disable=redefined-outer-name
        """
        Set the generator seed.

        Args:
            seed (int): Set the generator seed.

        Returns:
            Generator, the generator instance.
        """
        if not isinstance(seed, int):
            raise TypeError("Seed must be an integer.")
        seed = mindspore.Tensor(seed, mindspore.int64)
        self._generator(MANUAL_SEED, (self._seed, self._offset, seed))
        self._generator.add_prim_attr("manual_seed", True)
        return self

    def initial_seed(self):
        """
        Return the initial seed of generator.

        Returns:
            The initial seed of generator.
        """
        current_seed = self._generator(
            INITIAL_SEED, (self._seed, self._offset))[0]
        return current_seed.item()


    def _step(self, step):
        """
        Return current seed and offset, and update offset for the next call.

        Args:
            step (Tensor): Update offset by step.

        Returns:
            Current seed and offset.
        """
        outs = self._generator(STEP, (self._seed, self._offset, step,))[:2]
        for o in outs:
            o._device = self.device
        return outs

default_generator = Generator()

class Tag: pass

def _log_api_usage_once(*args):
    pass

ScriptDict = dict
ScriptList = list

class _DistStoreError(RuntimeError): pass

def _get_accelerator():
    device_target = mindspore.get_context("device_target")
    return device_(DEVICE_MAP[device_target])

class DispatchKey(Enum):
    pass
