import mindspore
from .state import AcceleratorState

class AcceleratedOptimizer(mindspore.nn.Optimizer):
    
    def __init__(self, optimizer, scaler):
        self.optimizer = optimizer
        self.accelerator_state = AcceleratorState()