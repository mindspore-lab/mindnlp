"""Weight normalization (Salimans & Kingma, 2016).

Reparameterizes a weight tensor as weight = g * (v / ||v||),
where g is a learned magnitude scalar and v is the direction vector.
"""

from ..parameter import Parameter
from ..._functional import norm as _norm, clamp_min


def weight_norm(module, name='weight', dim=0):
    """Apply weight normalization to a parameter in the given module.

    Reparameterizes ``weight`` as::

        weight = g * (v / ||v||)

    where ``g`` is a scalar (magnitude) and ``v`` is the direction.

    Adds ``name + '_g'`` and ``name + '_v'`` as new parameters, and
    registers a forward_pre_hook that recomputes the weight before
    every forward pass.

    Args:
        module (Module): The module containing the parameter.
        name (str): Name of the weight parameter. Default: ``'weight'``.
        dim (int): Dimension over which to compute the norm.
            Default: ``0``.

    Returns:
        The module with weight normalization applied.
    """
    weight = getattr(module, name)

    # Remove the original weight parameter
    del module._parameters[name]

    # Register v (direction) — same data as original weight
    module.register_parameter(name + '_v', Parameter(weight.data))

    # Compute g = ||weight|| reduced over all dims except ``dim``
    ndim = weight.dim()
    dims_to_reduce = [i for i in range(ndim) if i != dim]
    if dims_to_reduce:
        g_val = _norm(weight.data, p=2, dim=dims_to_reduce, keepdim=True)
    else:
        g_val = weight.data.abs()
    # Reshape g so that it has size weight.shape[dim] along ``dim`` and 1 elsewhere
    g_shape = [1] * ndim
    g_shape[dim] = weight.shape[dim]
    g_val = g_val.reshape(g_shape)
    module.register_parameter(name + '_g', Parameter(g_val))

    # Build and register the pre-hook
    def _weight_norm_hook(mod, inputs, _name=name, _dim=dim):
        v = getattr(mod, _name + '_v')
        g = getattr(mod, _name + '_g')
        dims = [i for i in range(v.dim()) if i != _dim]
        if dims:
            v_norm = _norm(v, p=2, dim=dims, keepdim=True)
        else:
            v_norm = v.abs()
        v_norm = clamp_min(v_norm, 1e-12)
        setattr(mod, _name, g * (v / v_norm))

    module.register_forward_pre_hook(_weight_norm_hook)

    # Compute the initial weight value
    _weight_norm_hook(module, None)

    return module


def remove_weight_norm(module, name='weight'):
    """Remove weight normalization reparameterization from a module.

    Restores the original weight parameter using the current values of
    ``name + '_g'`` and ``name + '_v'``.

    Args:
        module (Module): The module to remove weight norm from.
        name (str): Name of the weight parameter. Default: ``'weight'``.

    Returns:
        The module with weight normalization removed.
    """
    # Grab the currently-computed weight value
    weight_val = getattr(module, name)

    # Remove the reparameterization parameters
    del module._parameters[name + '_v']
    del module._parameters[name + '_g']

    # Store the weight as a normal parameter
    module.register_parameter(name, Parameter(weight_val.data))

    # Remove forward pre-hooks that were added by weight_norm.
    # We identify them by checking for the closure variable ``_name``.
    keys_to_remove = []
    for key, hook in module._forward_pre_hooks.items():
        # Check if this hook is a weight_norm hook by inspecting its closure
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
