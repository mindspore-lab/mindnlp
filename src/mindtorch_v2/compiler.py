def disable(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

