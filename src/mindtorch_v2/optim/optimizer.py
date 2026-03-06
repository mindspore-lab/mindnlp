"""
Base Optimizer class for mindtorch_v2.

Aligned with PyTorch's torch.optim.Optimizer API.
"""
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from .._tensor import Tensor


class Optimizer:
    """Base class for all optimizers.

    Args:
        params: Iterable of parameters or dicts defining parameter groups
        defaults: Dict containing default values for optimizer options
            (used when a parameter group doesn't specify them)

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(self, params: Iterable[Union[Tensor, Dict]], defaults: Dict[str, Any]):
        self.defaults = defaults
        self._hook_for_profile = False

        if params is None:
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got None")

        self.state: Dict[int, Dict[str, Any]] = {}
        self.param_groups: List[Dict[str, Any]] = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")

        if isinstance(param_groups[0], dict):
            # params is a list of param groups
            for param_group in param_groups:
                self.add_param_group(param_group)
        else:
            # params is an iterable of parameters
            self.add_param_group({"params": param_groups})

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "defaults": self.defaults,
            "state": self.state,
            "param_groups": self.param_groups,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += f'Parameter Group {i}\n'
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += f'    {key}: {group[key]}\n'
        format_string += ')'
        return format_string

    def _cuda_graph_capture_health_check(self) -> None:
        """Check for issues with CUDA graph capture."""
        # No-op for mindtorch_v2 (no CUDA graph support yet)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Sets the gradients of all optimized parameters to zero.

        Args:
            set_to_none: Instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can
                modestly improve performance. However, it changes certain
                behaviors. See the documentation for more details.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
                Optional for most optimizers.

        Returns:
            The loss value if closure is provided and evaluated, otherwise None.
        """
        raise NotImplementedError("step() must be implemented by subclass")

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer as a dict.

        It contains two entries:
        - state: a dict holding current optimization state.
        - param_groups: a list containing all parameter groups.
        """
        # Save the state for each parameter
        state_dict: Dict[str, Any] = {
            "state": {},
            "param_groups": [],
        }

        # Convert state keyed by id(param) to state keyed by param index
        for param_group in self.param_groups:
            for param in param_group["params"]:
                param_id = id(param)
                if param_id in self.state:
                    # Store state with string key (param index within group)
                    state_dict["state"][str(param_id)] = self.state[param_id]

        # Deep copy param_groups but replace params with their indices
        for param_group in self.param_groups:
            param_group_copy = {k: v for k, v in param_group.items() if k != "params"}
            # Store param ids for reconstruction
            param_group_copy["param_ids"] = [id(p) for p in param_group["params"]]
            state_dict["param_groups"].append(param_group_copy)

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the optimizer state.

        Args:
            state_dict: optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # Restore state
        self.state = {}
        for param_id_str, state in state_dict["state"].items():
            param_id = int(param_id_str)
            self.state[param_id] = state

        # Restore param_groups
        self.param_groups = []
        for param_group in state_dict["param_groups"]:
            param_ids = param_group.pop("param_ids", [])
            # We can't restore the actual params here, caller needs to
            # ensure params are the same objects
            self.param_groups.append(param_group)

        # Update param_groups to reference actual params
        # This requires the caller to have the same param objects
        # For now, we just restore the metadata

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Add a param group to the :class:`Optimizer` s `param_groups`.

        Args:
            param_group: Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        if not isinstance(param_group, dict):
            raise TypeError(f"param_group must be a dict, got {type(param_group)}")

        # Convert params to list if it's an iterable
        if "params" not in param_group:
            raise ValueError("param group must contain 'params' key")

        params = list(param_group["params"])
        param_group["params"] = params

        if len(params) == 0:
            raise ValueError("param group must have non-empty params")

        # Merge defaults with param_group options
        for key, value in self.defaults.items():
            if key not in param_group:
                param_group[key] = value

        # Verify all params are Tensors and require grad
        for param in params:
            if not isinstance(param, Tensor):
                raise TypeError(f"param must be a Tensor, got {type(param)}")
            if not param.requires_grad:
                raise ValueError("param must require gradients")

        self.param_groups.append(param_group)

    def _get_param_id_to_index(self) -> Dict[int, Tuple[int, int]]:
        """Build a mapping from param id to (group_index, param_index)."""
        id_to_idx = {}
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group["params"]):
                id_to_idx[id(param)] = (group_idx, param_idx)
        return id_to_idx
