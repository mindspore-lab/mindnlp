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
        self.index = None if index is None else int(index)

    def __repr__(self):
        if self.index is None:
            return f"{self.type}"
        return f"{self.type}:{self.index}"


_default_device = device("cpu")
