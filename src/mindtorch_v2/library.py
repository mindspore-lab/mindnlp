class Library:
    def __init__(self, *_args, **_kwargs):
        pass

    def impl(self, *_args, **_kwargs):
        return None

def register_fake(_name):
    def decorator(_fn):
        return None
    return decorator

