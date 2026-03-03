class device:
    def __init__(self, dev, index=None):
        if isinstance(dev, device):
            self.type = dev.type
            self.index = dev.index if index is None else int(index)
            return
        if isinstance(dev, str) and ":" in dev:
            dev, idx = dev.split(":", 1)
            index = int(idx)
        self.type = str(dev)
        if self.type == "npu" and index is None:
            index = 0
        self.index = None if index is None else int(index)

    def __repr__(self):
        if self.index is None:
            return f"device(type='{self.type}')"
        return f"device(type='{self.type}', index={self.index})"

    def __str__(self):
        if self.index is None:
            return f"{self.type}"
        return f"{self.type}:{self.index}"

    def __eq__(self, other):
        if not isinstance(other, device):
            return NotImplemented
        return self.type == other.type and self.index == other.index

    def __hash__(self):
        return hash((self.type, self.index))


_default_device = device("cpu")


def get_default_device():
    return _default_device


def set_default_device(dev):
    global _default_device
    _default_device = device(dev)
    return _default_device
