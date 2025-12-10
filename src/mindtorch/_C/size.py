import operator
from functools import reduce


def _get_tuple_numel(input):
    if input == ():
        return 1
    return reduce(operator.mul, list(input))


class Size(tuple):
    def __new__(cls, shape=()):
        _shape = shape
        if not isinstance(_shape, (tuple, list)):
            raise TypeError("{} object is not supportted.".format(type(shape)))
        return tuple.__new__(Size, _shape)

    def numel(self):
        return _get_tuple_numel(self)

    def __repr__(self):
        return "mindtorch.Size(" + str(list(self)) + ")"

    def __getitem__(self, slice):
        out = super().__getitem__(slice)
        if isinstance(out, tuple):
            return Size(out)
        return out
