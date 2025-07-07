def register_fake(*args, **kwargs):
    def register(func):
        return func
    return register