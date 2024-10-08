"""utils"""
import random
import numpy as np
import mindspore
from mindnlp.configs import DEFAULT_DTYPE, GENERATOR_SEED

def set_default_dtype(dtype):
    """set default dtype"""
    global DEFAULT_DTYPE
    DEFAULT_DTYPE = dtype

def get_default_dtype():
    """get default dtype"""
    return DEFAULT_DTYPE

def manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)
    if GENERATOR_SEED:
        mindspore.manual_seed(seed)
