"""Spectral normalization (Miyato et al., 2018).

Reparameterizes a weight tensor by dividing it by its spectral norm
(largest singular value), estimated via power iteration.
"""

from ..parameter import Parameter
from ..._functional import matmul, div
from ..._dispatch import dispatch
from ..._creation import randn


def _reshape_weight_to_matrix(weight, dim=0):
    """Reshape weight to 2D matrix for spectral norm computation."""
    shape = weight.shape
    if dim != 0:
        # Permute so that dim becomes the first dimension
        perm = [dim] + [i for i in range(len(shape)) if i != dim]
        weight = dispatch("permute", weight.device.type, weight, perm)
        shape = weight.shape
    # Flatten all dims except first to get (shape[0], rest)
    rest = 1
    for i in range(1, len(shape)):
        rest *= shape[i]
    return weight.reshape((shape[0], rest))


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    """Apply spectral normalization to a parameter in the given module.

    Args:
        module: The module containing the parameter.
        name: Name of the weight parameter. Default: 'weight'.
        n_power_iterations: Number of power iterations for estimating
            spectral norm. Default: 1.
        eps: Epsilon for numerical stability. Default: 1e-12.
        dim: Dimension over which to compute spectral norm.
            Default: 0 for all params except ConvTranspose (dim=1).

    Returns:
        The module with spectral normalization applied.
    """
    weight = getattr(module, name)
    if dim is None:
        if isinstance(module, (type(None),)):
            dim = 1
        else:
            dim = 0

    # Remove the original weight parameter
    del module._parameters[name]

    # Register weight_orig
    module.register_parameter(name + '_orig', Parameter(weight.data))

    # Initialize u and v vectors
    w_mat = _reshape_weight_to_matrix(weight.data, dim)
    h = w_mat.shape[0]

    u = randn(h, device=weight.device)
    # Normalize u
    u_norm = dispatch("norm", u.device.type, u, 2.0, None, False)
    u_norm = dispatch("clamp_min", u.device.type, u_norm, eps)
    u = div(u, u_norm)

    # Register u as buffer (non-parameter)
    module.register_buffer(name + '_u', u)

    # Build and register the pre-hook
    def _spectral_norm_hook(mod, inputs, _name=name, _dim=dim,
                            _n_power_iterations=n_power_iterations, _eps=eps):
        weight_orig = getattr(mod, _name + '_orig')
        u = getattr(mod, _name + '_u')

        w_mat = _reshape_weight_to_matrix(weight_orig, _dim)

        for _ in range(_n_power_iterations):
            # v = W^T u / ||W^T u||
            v = matmul(w_mat.t(), u)
            v_norm = dispatch("norm", v.device.type, v, 2.0, None, False)
            v_norm = dispatch("clamp_min", v.device.type, v_norm, _eps)
            v = div(v, v_norm)

            # u = W v / ||W v||
            u_new = matmul(w_mat, v)
            u_norm = dispatch("norm", u_new.device.type, u_new, 2.0, None, False)
            u_norm = dispatch("clamp_min", u_new.device.type, u_norm, _eps)
            u = div(u_new, u_norm)

        # sigma = u^T W v
        sigma = matmul(u.unsqueeze(0), matmul(w_mat, v.unsqueeze(1)))
        sigma = sigma.squeeze()
        sigma = dispatch("clamp_min", sigma.device.type, sigma, _eps)

        # Update u buffer
        setattr(mod, _name + '_u', u)

        # Set normalized weight
        setattr(mod, _name, div(weight_orig, sigma))

    module.register_forward_pre_hook(_spectral_norm_hook)

    # Compute the initial weight value
    _spectral_norm_hook(module, None)

    return module


def remove_spectral_norm(module, name='weight'):
    """Remove spectral normalization from a module.

    Args:
        module: The module to remove spectral norm from.
        name: Name of the weight parameter. Default: 'weight'.

    Returns:
        The module with spectral normalization removed.
    """
    # Grab the currently-computed weight value
    weight_val = getattr(module, name)

    # Remove the reparameterization parameters and buffers
    del module._parameters[name + '_orig']
    if name + '_u' in module._buffers:
        del module._buffers[name + '_u']

    # Store the weight as a normal parameter
    module.register_parameter(name, Parameter(weight_val.data))

    # Remove forward pre-hooks
    keys_to_remove = []
    for key, hook in module._forward_pre_hooks.items():
        if hasattr(hook, '__closure__') and hook.__closure__:
            for cell in hook.__closure__:
                try:
                    if cell.cell_contents == name:
                        keys_to_remove.append(key)
                        break
                except ValueError:
                    pass
    for key in keys_to_remove:
        del module._forward_pre_hooks[key]

    return module
