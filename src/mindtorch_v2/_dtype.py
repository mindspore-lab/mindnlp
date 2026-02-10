class DType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"torch.{self.name}"


float32 = DType("float32")
float16 = DType("float16")
int64 = DType("int64")
