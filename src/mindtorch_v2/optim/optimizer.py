"""Optimizer base class."""

from collections import defaultdict


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, params, defaults):
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []

        # Convert params to list if needed
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")

        # Check if params is a list of dicts (param groups) or list of tensors
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group):
        """Add a param group to the optimizer."""
        params = param_group['params']
        if isinstance(params, (list, tuple)):
            param_group['params'] = list(params)
        else:
            param_group['params'] = [params]

        for name, default in self.defaults.items():
            if name not in param_group:
                param_group[name] = default

        self.param_groups.append(param_group)

    def zero_grad(self, set_to_none=True):
        """Reset gradients of all parameters."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def step(self, closure=None):
        """Perform a single optimization step."""
        raise NotImplementedError

    def state_dict(self):
        """Return the state of the optimizer as a dict."""
        return {
            'state': dict(self.state),
            'param_groups': self.param_groups,
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.state = defaultdict(dict, state_dict['state'])
        self.param_groups = state_dict['param_groups']
