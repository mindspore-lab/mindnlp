from .registry import registry


def dispatch(name, device, *args, **kwargs):
    fn = registry.get(name, device)
    return fn(*args, **kwargs)
