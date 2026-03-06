"""
LBFGS (Limited-memory BFGS) optimizer for mindtorch_v2.

Aligned with PyTorch's torch.optim.LBFGS.
"""

import math
from typing import Any, Callable, Dict, Iterable, Optional, Union

import numpy as np

from .optimizer import Optimizer
from .._tensor import Tensor
from .._dispatch import dispatch
from .._functional import zeros_like


class LBFGS(Optimizer):
    """Implements L-BFGS algorithm.

    This optimizer requires a closure that re-evaluates the model and
    returns the loss. Only a single parameter group is supported.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 1).
        max_iter: Maximum number of iterations per optimization step (default: 20).
        max_eval: Maximum number of function evaluations per step.
            If None, set to max_iter * 5 / 4 (default: None).
        tolerance_grad: Termination tolerance on first order optimality (default: 1e-7).
        tolerance_change: Termination tolerance on function value/parameter
            changes (default: 1e-9).
        history_size: Update history size (default: 100).
        line_search_fn: Either 'strong_wolfe' or None (default: None).
    """

    def __init__(
        self,
        params: Iterable[Union[Tensor, Dict]],
        lr: float = 1,
        max_iter: int = 20,
        max_eval: Optional[int] = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: Optional[str] = None,
    ):
        if max_eval is None:
            max_eval = int(max_iter * 5 / 4)

        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]["params"]
        self.state.setdefault("func_evals", 0)
        self.state.setdefault("n_iter", 0)

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                views.append(np.zeros(np.prod(p._numpy_view().shape)))
            else:
                views.append(p.grad._numpy_view().ravel())
        return np.concatenate(views)

    def _gather_flat_params(self):
        views = []
        for p in self._params:
            views.append(p._numpy_view().ravel())
        return np.concatenate(views)

    def _set_flat_params(self, flat_params):
        offset = 0
        for p in self._params:
            shape = p._numpy_view().shape
            numel = int(np.prod(shape))
            p.storage()._data[:] = flat_params[offset:offset + numel].reshape(shape)
            offset += numel

    def _two_loop_recursion(self, flat_grad, old_dirs, old_stps, ro, H_diag):
        q = flat_grad.copy()
        num_old = len(old_dirs)
        if num_old == 0:
            return -q

        alphas = [0.0] * num_old

        for i in range(num_old - 1, -1, -1):
            alphas[i] = ro[i] * np.dot(old_stps[i], q)
            q -= alphas[i] * old_dirs[i]

        r = H_diag * q

        for i in range(num_old):
            beta = ro[i] * np.dot(old_dirs[i], r)
            r += (alphas[i] - beta) * old_stps[i]

        return -r

    def step(self, closure: Callable[[], float] = None) -> Optional[float]:
        """Perform a single optimization step.

        Args:
            closure: A callable that re-evaluates the model and returns the loss.
                Required for LBFGS.
        """
        if closure is None:
            raise RuntimeError("LBFGS requires a closure")

        self._call_step_pre_hooks()

        group = self.param_groups[0]
        lr = group["lr"]
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        history_size = group["history_size"]
        line_search_fn = group["line_search_fn"]

        state = self.state
        state.setdefault("old_dirs", [])
        state.setdefault("old_stps", [])
        state.setdefault("ro", [])
        state.setdefault("H_diag", 1.0)

        old_dirs = state["old_dirs"]
        old_stps = state["old_stps"]
        ro = state["ro"]

        # Evaluate initial loss and gradient
        orig_params = self._gather_flat_params().copy()
        loss = float(closure().detach().numpy())
        flat_grad = self._gather_flat_grad()
        n_eval = 1

        current_loss = loss
        current_params = orig_params.copy()

        state["n_iter"] += 1

        for _ in range(max_iter):
            abs_grad_max = np.max(np.abs(flat_grad))
            if abs_grad_max <= tolerance_grad:
                break

            d = self._two_loop_recursion(flat_grad, old_dirs, old_stps, ro, state["H_diag"])

            if line_search_fn == "strong_wolfe":
                step_size = self._line_search(closure, current_params, d, current_loss, flat_grad)
                n_eval += 1
            else:
                step_size = lr

            prev_flat_grad = flat_grad.copy()
            step = step_size * d
            current_params = current_params + step
            self._set_flat_params(current_params)

            prev_loss = current_loss
            current_loss = float(closure().detach().numpy())
            flat_grad = self._gather_flat_grad()
            n_eval += 1

            # Update LBFGS history
            y = flat_grad - prev_flat_grad
            ys = np.dot(y, step)
            if ys > 1e-10:
                if len(old_dirs) >= history_size:
                    old_dirs.pop(0)
                    old_stps.pop(0)
                    ro.pop(0)
                old_dirs.append(y)
                old_stps.append(step)
                ro.append(1.0 / ys)
                state["H_diag"] = ys / np.dot(y, y)

            if abs(current_loss - prev_loss) < tolerance_change:
                break

            if n_eval >= max_eval:
                break

        state["func_evals"] += n_eval
        self._call_step_post_hooks()
        return current_loss

    def _line_search(self, closure, x0, d, f0, g0, c1=1e-4, max_ls=25):
        """Backtracking line search with Armijo condition."""
        dg0 = np.dot(g0, d)
        step_size = 1.0
        for _ in range(max_ls):
            x_new = x0 + step_size * d
            self._set_flat_params(x_new)
            f_new = float(closure().detach().numpy())
            if f_new <= f0 + c1 * step_size * dg0:
                return step_size
            step_size *= 0.5
        return step_size
