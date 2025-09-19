import os
from typing import Any

import mindspore
from mindspore._c_expression import _ms_memory_recycle
from mindspore import get_rng_state, set_rng_state, manual_seed
from mindspore.runtime import memory_reserved as ms_memory_reserved, \
    memory_allocated as ms_memory_allocated, StreamCtx as StreamContext, Stream, empty_cache as ms_empty_cache, \
    reset_peak_memory_stats, reset_max_memory_allocated, max_memory_allocated, synchronize, \
    current_stream
from mindspore.hal import get_device_properties

from mindspore.device_context.ascend import device_count as ms_device_count
from mindspore.communication import GlobalComm, get_group_size

import mindtorch
from mindtorch.executor import execute
from ..configs import SUPPORT_BF16, ON_A1
from . import random

FloatTensor = mindtorch.FloatTensor
HalfTensor = mindtorch.FloatTensor
BFloat16Tensor = mindtorch.BFloat16Tensor


class DefaultGenerators:
    def __getitem__(self, idx):
        return mindtorch.default_generator

    def __len__(self):
        return 1

default_generators = DefaultGenerators()

def set_compile_mode(*args, **kwargs):
    pass

def manual_seed_all(seed: int):
    manual_seed(seed)

def device_count():
    if not is_available():
        return 0
    if GlobalComm.INITED:
        return get_group_size()
    return 1

def current_device():
    return 0

def is_available():
    return mindspore.get_context('device_target') == 'Ascend'

def set_device(device):
    pass

def _lazy_call(callable, **kwargs):
    callable()

def is_bf16_supported():
    return SUPPORT_BF16

def memory_allocated(device=None):
    return ms_memory_allocated()

def memory_reserved(device=None):
    if os.environ.get("MS_ALLOC_CONF", None) is not None:
        # increase_size = 2GB
        out = ((ms_memory_allocated() // (1024 * 1024 * 2048)) + 1) * (1024 * 1024 * 2048)
        return out
    return ms_memory_reserved()

class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (mindtorch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = -1

    def __exit__(self, type: Any, value: Any, traceback: Any):
        return False

def _try_initial_ascend():
    x = mindtorch.tensor(1, device='npu')
    _ = x + 0

def mem_get_info(device=None):
    if not isinstance(device, int):
        device = mindspore.context.get_context("device_id")

    res = mindspore.hal.get_device_properties(device)
    if res.total_memory == 0:
        _try_initial_ascend()
        res = mindspore.hal.get_device_properties(device)

    return (res.free_memory, res.total_memory)

def get_device_capability(device=None):
    return 10, 0

def empty_cache():
    if not ON_A1:
        ms_empty_cache()
    _ms_memory_recycle()

def npu_rotary_mul(x, cos, sin):
    return execute('rotary_position_embedding', x, cos, sin, 0)

def npu_fusion_attention(query, key, value, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None,
                         scale=1., keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0,
                         drop_mask=None, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                         gen_mask_parallel=True, sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None):
    output = execute(
        'flash_attention_score',
        query, key, value, real_shift=pse, padding_mask=padding_mask, drop_mask=drop_mask,
        attn_mask=atten_mask, prefix=prefix, actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen, head_num=head_num, keep_prob=float(keep_prob),
        scale_value=scale, pre_tokens=pre_tockens, next_tokens=next_tockens,
        inner_precise=inner_precise, input_layout=input_layout, sparse_mode=sparse_mode
    )
    sfm_max, sfm_sum, sfm_out, atten_out = output

    return atten_out, sfm_max, sfm_sum

def is_initialized():
    True

def is_tf32_supported():
    return False
