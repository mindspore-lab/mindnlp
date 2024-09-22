"""accelerate optimizer"""
from mindspore.nn.optim.optimizer import Optimizer
from .state import AcceleratorState


class AcceleratedOptimizer(Optimizer):
    def __init__(self, optimizer):
        super().__init__(learning_rate=optimizer.learning_rate, parameters=optimizer.parameters)
        self.optimizer = optimizer
        self.accelerator_state = AcceleratorState()

    def step(self):
        """
        Performs a single optimization step.
        """
        self.optimizer.step()

    def zero_grad(self):
        """
        Clears the gradients of all optimized tensors.
        """
        self.optimizer.zero_grad()
