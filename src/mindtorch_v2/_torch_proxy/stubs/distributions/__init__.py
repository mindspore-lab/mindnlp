"""PyTorch distributions stub for mindtorch_v2.

This provides basic distribution classes needed for time series models.
"""

import numpy as np

# Import submodules
from . import multivariate_normal


class Distribution:
    """Base distribution class."""

    has_rsample = True
    has_enumerate_support = False
    arg_constraints = {}

    def __init__(self, batch_shape=(), event_shape=(), validate_args=None):
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self._validate_args = validate_args

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError

    def sample(self, sample_shape=()):
        raise NotImplementedError

    def rsample(self, sample_shape=()):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError


class TransformedDistribution(Distribution):
    """Distribution with bijective transforms applied."""

    def __init__(self, base_distribution, transforms, validate_args=None):
        self.base_dist = base_distribution
        if isinstance(transforms, Transform):
            self.transforms = [transforms]
        else:
            self.transforms = list(transforms)
        super().__init__(
            batch_shape=base_distribution.batch_shape,
            event_shape=base_distribution.event_shape,
            validate_args=validate_args
        )

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    def sample(self, sample_shape=()):
        x = self.base_dist.sample(sample_shape)
        for t in self.transforms:
            x = t(x)
        return x

    def rsample(self, sample_shape=()):
        x = self.base_dist.rsample(sample_shape)
        for t in self.transforms:
            x = t(x)
        return x

    def log_prob(self, value):
        return self.base_dist.log_prob(value)


class Normal(Distribution):
    """Normal (Gaussian) distribution."""

    arg_constraints = {}
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc
        self.scale = scale
        batch_shape = np.broadcast_shapes(
            np.asarray(loc).shape,
            np.asarray(scale).shape
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale ** 2

    @property
    def stddev(self):
        return self.scale

    def sample(self, sample_shape=()):
        shape = sample_shape + self._batch_shape
        return np.random.normal(self.loc, self.scale, shape)

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        var = self.scale ** 2
        return -0.5 * (np.log(2 * np.pi * var) + (value - self.loc) ** 2 / var)


class StudentT(Distribution):
    """Student's t-distribution."""

    arg_constraints = {}
    has_rsample = True

    def __init__(self, df, loc=0.0, scale=1.0, validate_args=None):
        self.df = df
        self.loc = loc
        self.scale = scale
        batch_shape = np.broadcast_shapes(
            np.asarray(df).shape,
            np.asarray(loc).shape,
            np.asarray(scale).shape
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale ** 2 * self.df / (self.df - 2)

    def sample(self, sample_shape=()):
        shape = sample_shape + self._batch_shape
        return np.random.standard_t(self.df, shape) * self.scale + self.loc

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)


class NegativeBinomial(Distribution):
    """Negative binomial distribution."""

    arg_constraints = {}
    has_rsample = False

    def __init__(self, total_count, probs=None, logits=None, validate_args=None):
        self.total_count = total_count
        if probs is not None:
            self.probs = probs
            self.logits = np.log(probs / (1 - probs))
        elif logits is not None:
            self.logits = logits
            self.probs = 1 / (1 + np.exp(-logits))
        else:
            raise ValueError("Either probs or logits must be specified")
        batch_shape = np.broadcast_shapes(
            np.asarray(total_count).shape,
            np.asarray(self.probs).shape
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.total_count * self.probs / (1 - self.probs)

    @property
    def variance(self):
        return self.mean / (1 - self.probs)

    def sample(self, sample_shape=()):
        shape = sample_shape + self._batch_shape
        return np.random.negative_binomial(self.total_count, 1 - self.probs, shape)


class Poisson(Distribution):
    """Poisson distribution."""

    arg_constraints = {}
    has_rsample = False

    def __init__(self, rate, validate_args=None):
        self.rate = rate
        batch_shape = np.asarray(rate).shape
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.rate

    @property
    def variance(self):
        return self.rate

    def sample(self, sample_shape=()):
        shape = sample_shape + self._batch_shape
        return np.random.poisson(self.rate, shape)


class Transform:
    """Base class for transforms."""

    bijective = True
    domain = None
    codomain = None

    def __call__(self, x):
        return self._call(x)

    def _call(self, x):
        raise NotImplementedError

    def _inverse(self, y):
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y):
        raise NotImplementedError


class AffineTransform(Transform):
    """Affine transform: y = loc + scale * x"""

    def __init__(self, loc, scale, event_dim=0):
        self.loc = loc
        self.scale = scale
        self.event_dim = event_dim

    def _call(self, x):
        return self.loc + self.scale * x

    def _inverse(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x, y):
        return np.log(np.abs(self.scale))


class ExpTransform(Transform):
    """Exponential transform: y = exp(x)"""

    def _call(self, x):
        return np.exp(x)

    def _inverse(self, y):
        return np.log(y)

    def log_abs_det_jacobian(self, x, y):
        return x


class SigmoidTransform(Transform):
    """Sigmoid transform: y = 1 / (1 + exp(-x))"""

    def _call(self, x):
        return 1 / (1 + np.exp(-x))

    def _inverse(self, y):
        return np.log(y / (1 - y))

    def log_abs_det_jacobian(self, x, y):
        return -np.log(y) - np.log(1 - y)


class SoftplusTransform(Transform):
    """Softplus transform: y = log(1 + exp(x))"""

    def _call(self, x):
        return np.log1p(np.exp(x))

    def _inverse(self, y):
        return np.log(np.exp(y) - 1)


# Constraint classes
class Constraint:
    """Base constraint class."""

    def check(self, value):
        """Check if value satisfies constraint.

        Returns a tensor-like result that supports .all() method.
        """
        raise NotImplementedError


class _Real(Constraint):
    def check(self, value):
        # Import here to avoid circular dependency
        from mindtorch_v2 import tensor
        # Always return True as a tensor
        if hasattr(value, 'shape'):
            import numpy as np
            return tensor(np.ones(value.shape, dtype=bool))
        return tensor(True)


class _Positive(Constraint):
    def check(self, value):
        from mindtorch_v2 import tensor
        result = value > 0
        # Ensure result is always a tensor with .all() support
        if not hasattr(result, 'all'):
            return tensor(result)
        return result


class _GreaterThan(Constraint):
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def check(self, value):
        from mindtorch_v2 import tensor
        result = value > self.lower_bound
        if not hasattr(result, 'all'):
            return tensor(result)
        return result


class _Interval(Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, value):
        from mindtorch_v2 import tensor
        result = (value > self.lower_bound) & (value < self.upper_bound)
        if not hasattr(result, 'all'):
            return tensor(result)
        return result


# Constraint singletons
real = _Real()
positive = _Positive()
unit_interval = _Interval(0, 1)


def greater_than(lower_bound):
    return _GreaterThan(lower_bound)


def interval(lower_bound, upper_bound):
    return _Interval(lower_bound, upper_bound)


# Constraint module for compatibility
class constraints:
    real = real
    positive = positive
    unit_interval = unit_interval

    @staticmethod
    def greater_than(lower_bound):
        return _GreaterThan(lower_bound)

    @staticmethod
    def interval(lower_bound, upper_bound):
        return _Interval(lower_bound, upper_bound)

    # Additional constraints for compatibility
    positive_definite = _Real()
    lower_cholesky = _Real()
    simplex = _Real()
    boolean = _Real()
    nonnegative = _Positive()
    nonnegative_integer = _Real()

    @staticmethod
    def stack(constraints_list, dim=0):
        return _Real()

    @staticmethod
    def integer_interval(lower_bound, upper_bound):
        return _Interval(lower_bound, upper_bound)

    @staticmethod
    def half_open_interval(lower_bound, upper_bound):
        return _Interval(lower_bound, upper_bound)


# Import for compatibility
Independent = Distribution  # Simplified
