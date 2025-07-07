"""negative binomial"""
# mypy: allow-untyped-defs
# pylint: disable=method-hidden
from .. import ops
from ..autograd import no_grad
from ..nn import functional as F
from . import constraints
from .distribution import Distribution
from .gamma import Gamma
from .utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)


__all__ = ["NegativeBinomial"]


class NegativeBinomial(Distribution):
    r"""
    Creates a Negative Binomial distribution, i.e. distribution
    of the number of successful independent and identical Bernoulli trials
    before :attr:`total_count` failures are achieved. The probability
    of success of each Bernoulli trial is :attr:`probs`.

    Args:
        total_count (float or Tensor): non-negative number of negative Bernoulli
            trials to stop, although the distribution is still valid for real
            valued count
        probs (Tensor): Event probabilities of success in the half open interval [0, 1)
        logits (Tensor): Event log-odds for probabilities of success
    """
    arg_constraints = {
        "total_count": constraints.greater_than_eq(0),
        "probs": constraints.half_open_interval(0.0, 1.0),
        "logits": constraints.real,
    }
    support = constraints.nonnegative_integer

    def __init__(self, total_count, probs=None, logits=None, validate_args=None):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            (
                self.total_count,
                self.probs,
            ) = broadcast_all(total_count, probs)
            self.total_count = self.total_count.type_as(self.probs)
        else:
            (
                self.total_count,
                self.logits,
            ) = broadcast_all(total_count, logits)
            self.total_count = self.total_count.type_as(self.logits)

        self._param = self.probs if probs is not None else self.logits
        batch_shape = self._param.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NegativeBinomial, _instance)
        new.total_count = self.total_count.expand(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(NegativeBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @property
    def mean(self):
        return self.total_count * ops.exp(self.logits)

    @property
    def mode(self):
        return ((self.total_count - 1) * self.logits.exp()).floor().clamp(min=0.0)

    @property
    def variance(self):
        return self.mean / ops.sigmoid(-self.logits)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self):
        return self._param.shape

    @lazy_property
    def _gamma(self):
        # Note we avoid validating because self.total_count can be zero.
        return Gamma(
            concentration=self.total_count,
            rate=ops.exp(-self.logits),
            validate_args=False,
        )

    def sample(self, sample_shape=()):
        with no_grad():
            rate = self._gamma.sample(sample_shape=sample_shape)
            return ops.poisson(rate)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        log_unnormalized_prob = self.total_count * F.logsigmoid(
            -self.logits
        ) + value * F.logsigmoid(self.logits)

        log_normalization = (
            -ops.lgamma(self.total_count + value)
            + ops.lgamma(1.0 + value)
            + ops.lgamma(self.total_count)
        )
        # The case self.total_count == 0 and value == 0 has probability 1 but
        # lgamma(0) is infinite. Handle this case separately using a function
        # that does not modify tensors in place to allow Jit compilation.
        log_normalization = log_normalization.masked_fill(
            self.total_count + value == 0.0, 0.0
        )

        return log_unnormalized_prob - log_normalization
