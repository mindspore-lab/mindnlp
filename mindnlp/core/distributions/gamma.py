"""gamma"""
# mypy: allow-untyped-defs
from numbers import Number

from .. import ops
from . import constraints
from .exp_family import ExponentialFamily
from .utils import broadcast_all


__all__ = ["Gamma"]


def _standard_gamma(concentration):
    return ops.gamma(concentration)


class Gamma(ExponentialFamily):
    r"""
    Creates a Gamma distribution parameterized by shape :attr:`concentration` and :attr:`rate`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # Gamma distributed with concentration=1 and rate=1
        tensor([ 0.1046])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate = 1 / scale of the distribution
            (often referred to as beta)
    """
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def mode(self):
        return ((self.concentration - 1) / self.rate).clamp(min=0)

    @property
    def variance(self):
        return self.concentration / self.rate.pow(2)

    def __init__(self, concentration, rate, validate_args=None):
        self.concentration, self.rate = broadcast_all(concentration, rate)
        if isinstance(concentration, Number) and isinstance(rate, Number):
            batch_shape = ()
        else:
            batch_shape = self.concentration.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Gamma, _instance)
        new.concentration = self.concentration.expand(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Gamma, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=()):
        shape = self._extended_shape(sample_shape)
        value = _standard_gamma(self.concentration.expand(shape)) / self.rate.expand(
            shape
        )
        value = value.clamp(
            min=ops.finfo(value.dtype).tiny
        )  # do not record in autograd graph
        return value

    def log_prob(self, value):
        value = ops.as_tensor(value, dtype=self.rate.dtype)
        if self._validate_args:
            self._validate_sample(value)
        return (
            ops.xlogy(self.concentration, self.rate)
            + ops.xlogy(self.concentration - 1, value)
            - self.rate * value
            - ops.lgamma(self.concentration)
        )

    def entropy(self):
        return (
            self.concentration
            - ops.log(self.rate)
            + ops.lgamma(self.concentration)
            + (1.0 - self.concentration) * ops.digamma(self.concentration)
        )

    @property
    def _natural_params(self):
        return (self.concentration - 1, -self.rate)

    def _log_normalizer(self, x, y):
        return ops.lgamma(x + 1) + (x + 1) * ops.log(-y.reciprocal())

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ops.gammainc(self.concentration, self.rate * value)
