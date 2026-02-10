class device:
    def __init__(self, dev):
        self.type = dev

    def __repr__(self):
        return f"{self.type}"


_default_device = device("cpu")
