class VersionCounter:
    def __init__(self, value=0):
        self.value = int(value)

    def bump(self):
        self.value += 1
