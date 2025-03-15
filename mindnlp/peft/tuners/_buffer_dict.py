import collections
from collections import OrderedDict
import mindspore
from mindspore import Tensor
import mindspore.context
from mindnlp.core import nn
class BufferDict(nn.Module):
    r"""
    Holds buffers in a dictionary.

    BufferDict can be indexed like a regular Python dictionary, but buffers it contains are properly registered, and
    will be visible by all Cell methods. `mindspore.nn.BufferDict` is an **ordered** dictionary that respects

    * the order of insertion, and
    * in `mindspore.nn.BufferDict.update`, the order of the merged `OrderedDict`
      or another `mindspore.nn.BufferDict` (the argument to
      :meth:`~mindspore.nn.BufferDict.update`).

    Note that :meth:`~mindspore.nn.BufferDict.update` with other unordered mapping
    types (e.g., Python's plain `dict`) does not preserve the order of the
    merged mapping.

    Args:
        buffers (iterable, optional):
            a mapping (dictionary) of (string : :class:`~mindspore.Tensor`) or an iterable of key-value pairs
            of type (string, :class:`~mindspore.Tensor`)

    Example::

        class MyCell(Cell):
            def __init__(self):
                super(MyCell, self).__init__()
                self.buffers = BufferDict({
                        'left': Tensor(shape=(5, 10), dtype=mindspore.float32),
                        'right': Tensor(shape=(5, 10), dtype=mindspore.float32)
                })

            def construct(self, x, choice):
                x = self.buffers[choice].matmul(x)
                return x
    """

    def __init__(self, buffers=None, persistent: bool = False):
        r"""
        Args:
            buffers (`dict`):
                A mapping (dictionary) from string to :class:`~mindspore.Tensor`, or an iterable of key-value pairs
                of type (string, :class:`~mindspore.Tensor`).
        """
        super(BufferDict, self).__init__()
        if buffers is not None:
            self.update(buffers)

        self.persistent = persistent

    def __getitem__(self, key):
        return self._buffers[key]

    def __setitem__(self, key, buffer):
        self._buffers[key] = buffer

    def __delitem__(self, key):
        del self._buffers[key]

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.keys())

    def __contains__(self, key):
        return key in self._buffers

    def clear(self):
        """Remove all items from the BufferDict."""
        self._buffers.clear()

    def pop(self, key):
        r"""Remove key from the BufferDict and return its buffer.

        Args:
            key (`str`):
                Key to pop from the BufferDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        r"""Return an iterable of the BufferDict keys."""
        return self._buffers.keys()

    def items(self):
        r"""Return an iterable of the BufferDict key/value pairs."""
        return self._buffers.items()

    def values(self):
        r"""Return an iterable of the BufferDict values."""
        return self._buffers.values()

    def update(self, buffers):
        r"""
        Update the `mindspore.nn.BufferDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        Note:
            If `buffers` is an `OrderedDict`, a `mindspore.nn.BufferDict`,
            or an iterable of key-value pairs, the order of new elements in it is
            preserved.

        Args:
            buffers (iterable):
                a mapping (dictionary) from string to :class:`~mindspore.Tensor`,
                or an iterable of key-value pairs of type (string, :class:`~mindspore.Tensor`)
        """
        if not isinstance(buffers, collections.abc.Iterable):
            raise TypeError(
                "BuffersDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(buffers).__name__
            )

        if isinstance(buffers, collections.abc.Mapping):
            if isinstance(buffers, (OrderedDict, BufferDict)):
                for key, buffer in buffers.items():
                    self[key] = buffer
            else:
                for key, buffer in sorted(buffers.items()):
                    self[key] = buffer
        else:
            for j, p in enumerate(buffers):
                if not isinstance(p, collections.abc.Iterable):
                    raise TypeError(
                        "BufferDict update sequence element " + str(j) + " should be Iterable; is" + type(p).__name__
                    )
                if not len(p) == 2:
                    raise ValueError(
                        "BufferDict update sequence element " + str(j) + " has length " + str(len(p)) + "; 2 is required"
                    )
                self[p[0]] = p[1]

    def extra_repr(self):
        child_lines = []
        for k, p in self._buffers.items():
            size_str = "x".join(str(size) for size in p.shape)
            parastr = f"Buffer containing: [{type(p)} of size {size_str}]"
            child_lines.append("  (" + k + "): " + parastr)
        tmpstr = "\n".join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError("BufferDict should not be called.")