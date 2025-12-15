import warnings

warnings.warn(
    "The usage 'from mindnlp.diffusers import xx' is deprecated. "
    "Please use 'import mindnlp; from diffusers import xxx' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Setup backward compatibility: apply patches
# pylint: disable=wrong-import-position
from .patch.diffusers import setup_diffusers_module
setup_diffusers_module()
