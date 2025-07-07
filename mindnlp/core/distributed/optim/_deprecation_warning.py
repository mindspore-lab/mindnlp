import warnings

from mindnlp import core


@core.jit.ignore  # type: ignore[misc]
def _scripted_functional_optimizer_deprecation_warning(stacklevel: int = 0) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(
            "`TorchScript` support for functional optimizers is deprecated "
            "and will be removed in a future PyTorch release. "
            "Consider using the `core.compile` optimizer instead.",
            DeprecationWarning,
            stacklevel=stacklevel + 2,
        )
