"""PyTorch-compatible device representation."""

# Avoid conflict with Python builtins
import builtins as _builtins
_int = _builtins.int


class device:
    """Represents the device on which a tensor is or will be allocated.

    Matches torch.device API:
        device("cpu")
        device("cuda", 0)
        device("cuda:1")
    """

    __slots__ = ("type", "index")

    def __init__(self, type_or_str, index=None):
        if isinstance(type_or_str, device):
            self.type = type_or_str.type
            self.index = type_or_str.index
            return

        if isinstance(type_or_str, str):
            if ":" in type_or_str:
                parts = type_or_str.split(":", 1)
                self.type = parts[0]
                self.index = _int(parts[1])
            else:
                self.type = type_or_str
                self.index = index
        else:
            raise ValueError(f"Expected string or device, got {type(type_or_str)}")

    def __repr__(self):
        if self.index is not None:
            return f"device(type='{self.type}', index={self.index})"
        return f"device(type='{self.type}')"

    def __str__(self):
        if self.index is not None:
            return f"{self.type}:{self.index}"
        return self.type

    def __eq__(self, other):
        if isinstance(other, device):
            return self.type == other.type and self.index == other.index
        if isinstance(other, str):
            return self == device(other)
        return NotImplemented

    def __hash__(self):
        return hash((self.type, self.index))

    def __reduce__(self):
        """Support pickling."""
        return (device, (self.type, self.index))
