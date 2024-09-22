"""accelerate scheduler."""


class AcceleratedScheduler:
    def __init__(self, scheduler, optimizers, step_with_optimizer: bool = True, split_batches: bool = False):
        self.scheduler = scheduler
        self.optimizers = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
        self.split_batches = split_batches
        self.step_with_optimizer = step_with_optimizer

    def step(self):
        """
        Performs a step of the scheduler.
        """
