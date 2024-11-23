""" Activation Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from typing import Union, Callable, Type
from mindnlp.core import nn
from mindnlp.core.nn import functional as F
from mindnlp.common.activations import QuickGELU, GELUTanh, gelu_tanh, quick_gelu, hard_mish, HardMish

_ACT_FN_DEFAULT = dict(
    silu=F.silu,
    swish=F.silu,
    mish=F.mish,
    relu=F.relu,
    relu6=F.relu6,
    leaky_relu=F.leaky_relu,
    elu=F.elu,
    celu=F.celu,
    selu=F.selu,
    gelu=F.gelu,
    gelu_tanh=gelu_tanh,
    quick_gelu=quick_gelu,
    sigmoid=F.sigmoid,
    tanh=F.tanh,
    hard_sigmoid=F.hardsigmoid,
    hard_swish=F.hardswish,
    hard_mish=hard_mish,
)


_ACT_FNS = (_ACT_FN_DEFAULT,)
for a in _ACT_FNS:
    a.setdefault('hardsigmoid', a.get('hard_sigmoid'))
    a.setdefault('hardswish', a.get('hard_swish'))


_ACT_LAYER_DEFAULT = dict(
    silu=nn.SiLU,
    swish=nn.SiLU,
    mish=nn.Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    elu=nn.ELU,
    prelu=nn.PReLU,
    celu=nn.CELU,
    selu=nn.SELU,
    gelu=nn.GELU,
    gelu_tanh=GELUTanh,
    quick_gelu=QuickGELU,
    sigmoid=nn.Sigmoid,
    tanh=nn.Tanh,
    hard_sigmoid=nn.Hardsigmoid,
    hard_swish=nn.Hardswish,
    hard_mish=HardMish,
    identity=nn.Identity,
)

_ACT_LAYERS = (_ACT_LAYER_DEFAULT,)
for a in _ACT_LAYERS:
    a.setdefault('hardsigmoid', a.get('hard_sigmoid'))
    a.setdefault('hardswish', a.get('hard_swish'))


def get_act_fn(name: Union[Callable, str] = 'relu'):
    """ Activation Function Factory
    Fetching activation fns by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if not name:
        return None
    if isinstance(name, Callable):
        return name
    name = name.lower()
    return _ACT_FN_DEFAULT[name]


def get_act_layer(name: Union[Type[nn.Module], str] = 'relu'):
    """ Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if name is None:
        return None
    if not isinstance(name, str):
        # callable, module, etc
        return name
    if not name:
        return None
    name = name.lower()
    return _ACT_LAYER_DEFAULT[name]


def create_act_layer(name: Union[Type[nn.Module], str], inplace=None, **kwargs):
    act_layer = get_act_layer(name)
    if act_layer is None:
        return None
    if inplace is None:
        return act_layer(**kwargs)
    try:
        return act_layer(inplace=inplace, **kwargs)
    except TypeError:
        # recover if act layer doesn't have inplace arg
        return act_layer(**kwargs)
