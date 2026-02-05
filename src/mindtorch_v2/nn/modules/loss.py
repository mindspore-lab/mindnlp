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


class L1Loss(_Loss):
    """L1 (Mean Absolute Error) loss."""

    def __init__(self, reduction='mean'):
        super().__init__(reduction)

    def forward(self, input, target):
        diff = input - target
        # Absolute value
        loss = diff.abs() if hasattr(diff, 'abs') else Tensor(np.abs(diff.numpy()))
        return self._reduce(loss)


class SmoothL1Loss(_Loss):
    """Smooth L1 loss (Huber loss)."""

    def __init__(self, reduction='mean', beta=1.0):
        super().__init__(reduction)
        self.beta = beta

    def forward(self, input, target):
        input_np = input.numpy()
        target_np = target.numpy()
        diff = np.abs(input_np - target_np)

        # Smooth L1: 0.5 * x^2 / beta if |x| < beta, else |x| - 0.5 * beta
        loss_np = np.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        loss = Tensor(loss_np.astype(np.float32))
        return self._reduce(loss)


class HuberLoss(_Loss):
    """Huber loss (alias for SmoothL1Loss with delta parameter)."""

    def __init__(self, reduction='mean', delta=1.0):
        super().__init__(reduction)
        self.delta = delta

    def forward(self, input, target):
        input_np = input.numpy()
        target_np = target.numpy()
        diff = np.abs(input_np - target_np)

        loss_np = np.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        loss = Tensor(loss_np.astype(np.float32))
        return self._reduce(loss)


class KLDivLoss(_Loss):
    """Kullback-Leibler divergence loss."""

    def __init__(self, reduction='mean', log_target=False):
        super().__init__(reduction)
        self.log_target = log_target

    def forward(self, input, target):
        """
        Args:
            input: Log probabilities
            target: Target probabilities (or log probabilities if log_target=True)
        """
        input_np = input.numpy()
        target_np = target.numpy()

        if self.log_target:
            loss_np = np.exp(target_np) * (target_np - input_np)
        else:
            # Avoid log(0) by adding small epsilon
            loss_np = target_np * (np.log(target_np + 1e-10) - input_np)

        loss = Tensor(loss_np.astype(np.float32))
        return self._reduce(loss)


class BCELoss(_Loss):
    """Binary Cross Entropy loss (expects probabilities as input)."""

    def __init__(self, weight=None, reduction='mean'):
        super().__init__(reduction)
        self.weight = weight

    def forward(self, input, target):
        input_np = input.numpy()
        target_np = target.numpy()

        # Clamp for numerical stability
        eps = 1e-7
        input_np = np.clip(input_np, eps, 1 - eps)

        loss_np = -(target_np * np.log(input_np) + (1 - target_np) * np.log(1 - input_np))

        if self.weight is not None:
            loss_np = loss_np * self.weight.numpy()

        loss = Tensor(loss_np.astype(np.float32))
        return self._reduce(loss)


class CosineEmbeddingLoss(_Loss):
    """Cosine embedding loss."""

    def __init__(self, margin=0.0, reduction='mean'):
        super().__init__(reduction)
        self.margin = margin

    def forward(self, input1, input2, target):
        """
        Args:
            input1, input2: Tensors of shape (N, D)
            target: 1 or -1 for each pair
        """
        input1_np = input1.numpy()
        input2_np = input2.numpy()
        target_np = target.numpy()

        # Cosine similarity
        dot = np.sum(input1_np * input2_np, axis=-1)
        norm1 = np.sqrt(np.sum(input1_np ** 2, axis=-1))
        norm2 = np.sqrt(np.sum(input2_np ** 2, axis=-1))
        cos_sim = dot / (norm1 * norm2 + 1e-8)

        # Loss: for y=1, 1-cos; for y=-1, max(0, cos-margin)
        loss_np = np.where(
            target_np == 1,
            1 - cos_sim,
            np.maximum(0, cos_sim - self.margin)
        )

        loss = Tensor(loss_np.astype(np.float32))
        return self._reduce(loss)
