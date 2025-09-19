# mypy: allow-untyped-defs
from typing import Optional, Union

import mindtorch
from mindtorch import Tensor
from mindtorch.distributions import constraints
from mindtorch.distributions.exp_family import ExponentialFamily
from mindtorch.distributions.utils import broadcast_all
from mindtorch.types import _Number, _size


__all__ = ["Gamma"]


def _standard_gamma(concentration):
    return mindtorch._standard_gamma(concentration)


class Gamma(ExponentialFamily):
    r"""
    Creates a Gamma distribution parameterized by shape :attr:`concentration` and :attr:`rate`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Gamma(mindtorch.tensor([1.0]), mindtorch.tensor([1.0]))
        >>> m.sample()  # Gamma distributed with concentration=1 and rate=1
        tensor([ 0.1046])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate parameter of the distribution
            (often referred to as beta), rate = 1 / scale
    """

    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self) -> Tensor:
        return self.concentration / self.rate

    @property
    def mode(self) -> Tensor:
        return ((self.concentration - 1) / self.rate).clamp(min=0)

    @property
    def variance(self) -> Tensor:
        return self.concentration / self.rate.pow(2)

    def __init__(
        self,
        concentration: Union[Tensor, float],
        rate: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ) -> None:
        self.concentration, self.rate = broadcast_all(concentration, rate)
        if isinstance(concentration, _Number) and isinstance(rate, _Number):
            batch_shape = mindtorch.Size()
        else:
            batch_shape = self.concentration.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Gamma, _instance)
        batch_shape = mindtorch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Gamma, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = mindtorch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        if shape == (): # pylint: disable=use-implicit-booleaness-not-comparison
            sample_shape = (1,)
        else:
            sample_shape = shape
        value = mindtorch.gamma(sample_shape, self.concentration, self.rate)

        if shape == (): # pylint: disable=use-implicit-booleaness-not-comparison
            value = mindtorch.squeeze(value)

        value.detach().clamp_(
            min=mindtorch.finfo(value.dtype).tiny
        )  # do not record in autograd graph
        return value

    def log_prob(self, value):
        value = mindtorch.as_tensor(value, dtype=self.rate.dtype, device=self.rate.device)
        if self._validate_args:
            self._validate_sample(value)
        return (
            mindtorch.xlogy(self.concentration, self.rate)
            + mindtorch.xlogy(self.concentration - 1, value)
            - self.rate * value
            - mindtorch.lgamma(self.concentration)
        )

    def entropy(self):
        return (
            self.concentration
            - mindtorch.log(self.rate)
            + mindtorch.lgamma(self.concentration)
            + (1.0 - self.concentration) * mindtorch.digamma(self.concentration)
        )

    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]:
        return (self.concentration - 1, -self.rate)

    def _log_normalizer(self, x, y):
        return mindtorch.lgamma(x + 1) + (x + 1) * mindtorch.log(-y.reciprocal())

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return mindtorch.special.gammainc(self.concentration, self.rate * value)