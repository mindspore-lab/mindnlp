class OpRegistry:
    def __init__(self):
        self._ops = {}

    def register(self, name, device, fn):
        self._ops[(name, device)] = fn

    def get(self, name, device):
        return self._ops[(name, device)]


registry = OpRegistry()
