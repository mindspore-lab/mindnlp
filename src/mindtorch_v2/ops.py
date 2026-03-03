class _Op:
    def __init__(self, return_value=None):
        self._return_value = return_value

    def __call__(self, *args, **kwargs):
        return self._return_value

    @property
    def default(self):
        return self


class _Namespace:
    def __getattr__(self, _name):
        return _Op(None)


class _TorchVisionNamespace(_Namespace):
    def __getattr__(self, name):
        if name == "_cuda_version":
            return _Op(0)
        return super().__getattr__(name)


class _Ops:
    def __init__(self):
        self.torchvision = _TorchVisionNamespace()

    def load_library(self, *_args, **_kwargs):
        return None

    def __getattr__(self, _name):
        return _Namespace()


ops = _Ops()
