"""PyTorch nn.utils compatibility module."""

# Parametrizations stub
class parametrizations:
    """Parametrizations for weight normalization etc."""

    @staticmethod
    def weight_norm(module, name='weight', dim=0):
        """Apply weight normalization to a module."""
        return module

    @staticmethod
    def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
        """Apply spectral normalization to a module."""
        return module


def clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False):
    """Clip gradient norm of parameters."""
    import numpy as np
    from ..._tensor import Tensor

    if isinstance(parameters, Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return Tensor([0.0])

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if norm_type == float('inf'):
        total_norm = max(p.grad.abs().max().item() for p in parameters)
    else:
        total_norm = 0.0
        for p in parameters:
            param_norm = p.grad.numpy()
            total_norm += np.sum(np.abs(param_norm) ** norm_type)
        total_norm = total_norm ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad = p.grad * clip_coef

    return Tensor([total_norm])


def clip_grad_value_(parameters, clip_value):
    """Clip gradient values of parameters."""
    import numpy as np
    from ..._tensor import Tensor

    if isinstance(parameters, Tensor):
        parameters = [parameters]

    clip_value = float(clip_value)

    for p in parameters:
        if p.grad is not None:
            grad_np = p.grad.numpy()
            grad_np = np.clip(grad_np, -clip_value, clip_value)
            p.grad = Tensor(grad_np)


# RNN utilities
def pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    """Pack a padded sequence for RNN. Returns input unchanged (stub)."""
    return input


def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
    """Unpack a packed sequence. Returns input unchanged (stub)."""
    return sequence, None


class PackedSequence:
    """Placeholder for packed sequence."""
    pass
