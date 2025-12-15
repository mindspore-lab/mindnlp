import warnings

warnings.warn(
    "The usage 'from mindnlp.transformers import xx' is deprecated. "
    "Please use 'import mindnlp; from transformers import xxx' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Setup backward compatibility: redirect to patched transformers
from .patch.transformers import setup_transformers_module
setup_transformers_module()
