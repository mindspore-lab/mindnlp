"""Loss functions for neural networks."""

import numpy as np
from ..module import Module
from ..._tensor import Tensor
from ..._dispatch import dispatch


class _Loss(Module):
    """Base class for loss functions."""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def _reduce(self, loss):
        """Apply reduction to loss tensor."""
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class MSELoss(_Loss):
    """Mean Squared Error loss."""

    def __init__(self, reduction='mean'):
        super().__init__(reduction)

    def forward(self, input, target):
        diff = input - target
        loss = diff * diff
        return self._reduce(loss)


class CrossEntropyLoss(_Loss):
    """Cross Entropy loss combining log_softmax and nll_loss."""

    def __init__(self, weight=None, ignore_index=-100, reduction='mean',
                 label_smoothing=0.0):
        super().__init__(reduction)
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        """
        Args:
            input: (N, C) logits
            target: (N,) class indices or (N, C) probabilities
        """
        input_np = input.numpy()
        target_np = target.numpy()

        # Log softmax
        max_val = np.max(input_np, axis=-1, keepdims=True)
        log_sum_exp = max_val + np.log(np.sum(np.exp(input_np - max_val), axis=-1, keepdims=True))
        log_probs = input_np - log_sum_exp

        # Handle class indices
        if target_np.ndim == 1:
            # Gather log probabilities at target indices
            batch_size = input_np.shape[0]
            loss_np = np.zeros(batch_size, dtype=np.float32)

            for i in range(batch_size):
                idx = int(target_np[i])
                if idx == self.ignore_index:
                    loss_np[i] = 0.0
                else:
                    loss_np[i] = -log_probs[i, idx]

            loss = Tensor(loss_np)
        else:
            # Soft targets
            loss_np = -np.sum(target_np * log_probs, axis=-1)
            loss = Tensor(loss_np)

        return self._reduce(loss)


class BCEWithLogitsLoss(_Loss):
    """Binary Cross Entropy with Logits loss."""

    def __init__(self, weight=None, reduction='mean', pos_weight=None):
        super().__init__(reduction)
        self.weight = weight
        self.pos_weight = pos_weight

    def forward(self, input, target):
        input_np = input.numpy()
        target_np = target.numpy()

        # Stable BCE: max(x, 0) - x * z + log(1 + exp(-abs(x)))
        max_val = np.maximum(input_np, 0)
        loss_np = max_val - input_np * target_np + np.log(1 + np.exp(-np.abs(input_np)))

        if self.pos_weight is not None:
            pw = self.pos_weight.numpy() if isinstance(self.pos_weight, Tensor) else self.pos_weight
            loss_np = loss_np * (1 + (pw - 1) * target_np)

        loss = Tensor(loss_np.astype(np.float32))
        return self._reduce(loss)


class NLLLoss(_Loss):
    """Negative Log Likelihood loss."""

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(reduction)
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        Args:
            input: (N, C) log probabilities
            target: (N,) class indices
        """
        input_np = input.numpy()
        target_np = target.numpy().astype(np.int64)

        batch_size = input_np.shape[0]
        loss_np = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            idx = target_np[i]
            if idx == self.ignore_index:
                loss_np[i] = 0.0
            else:
                loss_np[i] = -input_np[i, idx]

        loss = Tensor(loss_np)
        return self._reduce(loss)
