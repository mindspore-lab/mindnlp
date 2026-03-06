"""Optimizers for mindtorch_v2.

This module provides optimizer implementations aligned with PyTorch's torch.optim.

Available optimizers:
- Optimizer: Base class for all optimizers
- SGD: Stochastic Gradient Descent with optional momentum
- Adam: Adaptive Moment Estimation
- AdamW: Adam with decoupled weight decay
- RMSprop: RMSprop optimizer
- Adagrad: Adagrad optimizer
- Adadelta: Adadelta optimizer
- Adamax: Adamax optimizer (Adam variant with infinity norm)
- NAdam: Nesterov Adam optimizer
- RAdam: Rectified Adam optimizer
- ASGD: Averaged SGD optimizer
- Rprop: Resilient backpropagation optimizer

Learning rate schedulers are available in the `lr_scheduler` submodule.
"""

from .optimizer import Optimizer
from .sgd import SGD
from .adam import Adam, AdamW
from .rmsprop import RMSprop
from .adagrad import Adagrad
from .adadelta import Adadelta
from .adamax import Adamax
from .nadam import NAdam
from .radam import RAdam
from .asgd import ASGD
from .rprop import Rprop
from .lbfgs import LBFGS
from .sparse_adam import SparseAdam
from . import lr_scheduler

__all__ = [
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    "RMSprop",
    "Adagrad",
    "Adadelta",
    "Adamax",
    "NAdam",
    "RAdam",
    "ASGD",
    "Rprop",
    "LBFGS",
    "SparseAdam",
    "lr_scheduler",
]
