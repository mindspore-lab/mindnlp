import warnings

warnings.warn(
    "The usage 'from mindhf.diffusers import xx' is deprecated. "
    "Please use 'import mindhf; from diffusers import xxx' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Setup backward compatibility: apply patches
from .patch.diffusers import setup_diffusers_module
setup_diffusers_module()
