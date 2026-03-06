"""torch.nn.utils.skip_init — instantiate a module without running parameter initialization."""


def skip_init(module_cls, *args, **kwargs):
    """Instantiate a module, nominally skipping default parameter initialization.

    In PyTorch this works by creating the module on the ``meta`` device and
    then moving it to a real device with ``Module.to_empty()``.  Our Module
    class does not yet support the meta device, so this implementation simply
    creates the module normally.  This is still useful as an API-compatibility
    shim so that callers (e.g. HuggingFace ``from_pretrained()``) do not crash
    when they invoke ``torch.nn.utils.skip_init``.

    Args:
        module_cls: The :class:`~torch.nn.Module` subclass to instantiate
            (e.g. ``nn.Linear``).
        *args: Positional arguments forwarded to ``module_cls.__init__``.
        **kwargs: Keyword arguments forwarded to ``module_cls.__init__``.

    Returns:
        An instance of *module_cls*.
    """
    return module_cls(*args, **kwargs)
