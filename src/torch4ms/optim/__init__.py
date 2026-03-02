"""
torch4ms 优化器模块 - 仿照 torchax 的 optax 使用方式

这个模块提供了类似 optax 的函数式优化器接口，但底层使用 MindSpore 原生优化器。
支持链式组合梯度变换（gradient transformations），类似 optax.chain。

同时提供了 Torch4msOptimizer 适配器，使 PyTorch 优化器能够与 torch4ms.Tensor 一起工作。
"""
from .optimizer import (
    Optimizer,
    SGD,
    Adam,
    AdamW,
    RMSprop,
    chain,
    clip_by_global_norm,
    scale_by_learning_rate,
    add_decayed_weights,
    Torch4msOptimizer,
)

__all__ = [
    'Optimizer',
    'SGD',
    'Adam',
    'AdamW',
    'RMSprop',
    'chain',
    'clip_by_global_norm',
    'scale_by_learning_rate',
    'add_decayed_weights',
    'Torch4msOptimizer',
]
