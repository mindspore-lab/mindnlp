from mindspore.nn.optim.optimizer import Optimizer
from .state import AcceleratorState

class AcceleratedOptimizer(Optimizer):
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.accelerator_state = AcceleratorState()

