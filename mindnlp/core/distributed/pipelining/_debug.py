# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from mindnlp import core


def friendly_debug_info(v):
    """
    Helper function to print out debug info in a friendly way.
    """
    if isinstance(v, core.Tensor):
        return f"Tensor({v.shape}, grad={v.requires_grad}, dtype={v.dtype})"
    else:
        return str(v)


def map_debug_info(a):
    """
    Helper function to apply `friendly_debug_info` to items in `a`.
    `a` may be a list, tuple, or dict.
    """
    return core.fx.node.map_aggregate(a, friendly_debug_info)
