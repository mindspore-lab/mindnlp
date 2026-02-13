class OpRegistry:
    def __init__(self):
        self._ops = {}

    def register(self, name, device, fn, meta=None):
        self._ops[(name, device)] = {"impl": fn, "meta": meta}

    def get(self, name, device):
        return self._ops[(name, device)]


registry = OpRegistry()
