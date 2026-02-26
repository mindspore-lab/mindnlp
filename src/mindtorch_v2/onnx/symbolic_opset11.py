def __getattr__(_name):
    return lambda *args, **kwargs: None
