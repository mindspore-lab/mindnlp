"""Implementation of torch.special functions via dispatch."""

from .._dispatch.dispatcher import dispatch


def digamma(input):
    """Computes the logarithmic derivative of the gamma function (psi function)."""
    return dispatch("special_digamma", input.device.type, input)


def entr(input):
    """Computes the entropy: -x * ln(x), with 0 for x=0 and -inf for x<0."""
    return dispatch("special_entr", input.device.type, input)


def erf(input):
    """Computes the error function."""
    return dispatch("erf", input.device.type, input)


def erfc(input):
    """Computes the complementary error function."""
    return dispatch("erfc", input.device.type, input)


def erfcx(input):
    """Computes the scaled complementary error function: exp(x^2) * erfc(x)."""
    return dispatch("special_erfcx", input.device.type, input)


def erfinv(input):
    """Computes the inverse error function."""
    return dispatch("special_erfinv", input.device.type, input)


def exp2(input):
    """Computes 2^x element-wise."""
    return dispatch("exp2", input.device.type, input)


def expit(input):
    """Computes the expit (logistic sigmoid) function: 1 / (1 + exp(-x))."""
    return dispatch("sigmoid", input.device.type, input)


def expm1(input):
    """Computes exp(x) - 1 element-wise."""
    return dispatch("expm1", input.device.type, input)


def gammainc(input, other):
    """Computes the regularized lower incomplete gamma function."""
    return dispatch("special_gammainc", input.device.type, input, other)


def gammaincc(input, other):
    """Computes the regularized upper incomplete gamma function."""
    return dispatch("special_gammaincc", input.device.type, input, other)


def gammaln(input):
    """Computes the natural log of the absolute value of the gamma function."""
    return dispatch("special_gammaln", input.device.type, input)


def i0(input):
    """Computes the zeroth order modified Bessel function of the first kind."""
    return dispatch("special_i0", input.device.type, input)


def i0e(input):
    """Computes the exponentially scaled zeroth order modified Bessel function."""
    return dispatch("special_i0e", input.device.type, input)


def i1(input):
    """Computes the first order modified Bessel function of the first kind."""
    return dispatch("special_i1", input.device.type, input)


def i1e(input):
    """Computes the exponentially scaled first order modified Bessel function."""
    return dispatch("special_i1e", input.device.type, input)


def log1p(input):
    """Computes log(1 + x) accurately for small x."""
    return dispatch("log1p", input.device.type, input)


def log_ndtr(input):
    """Computes log of the area under the standard Gaussian PDF from -inf to x."""
    return dispatch("special_log_ndtr", input.device.type, input)


def log_softmax(input, dim=-1, *, dtype=None):
    """Computes log of softmax."""
    return dispatch("log_softmax", input.device.type, input, dim)


def logit(input, eps=None):
    """Computes the logit: log(x / (1 - x))."""
    return dispatch("special_logit", input.device.type, input, eps)


def logsumexp(input, dim, keepdim=False):
    """Computes log(sum(exp(x))) in a numerically stable way."""
    return dispatch("logsumexp", input.device.type, input, dim, keepdim)


def multigammaln(input, p):
    """Computes the multivariate log-gamma function with dimension p."""
    return dispatch("special_multigammaln", input.device.type, input, p)


def ndtr(input):
    """Computes the area under the standard Gaussian PDF from -inf to x."""
    return dispatch("special_ndtr", input.device.type, input)


def ndtri(input):
    """Computes the quantile function (inverse of ndtr) of the standard normal."""
    return dispatch("special_ndtri", input.device.type, input)


def polygamma(n, input):
    """Computes the n-th derivative of the digamma function."""
    return dispatch("special_polygamma", input.device.type, n, input)


def psi(input):
    """Alias for digamma."""
    return dispatch("special_digamma", input.device.type, input)


def round(input):
    """Rounds to the nearest integer."""
    return dispatch("round", input.device.type, input)


def sinc(input):
    """Computes the normalized sinc function: sin(pi*x) / (pi*x)."""
    return dispatch("special_sinc", input.device.type, input)


def softmax(input, dim=-1, *, dtype=None):
    """Computes softmax."""
    return dispatch("softmax", input.device.type, input, dim)


def xlog1py(input, other):
    """Computes x * log1p(y), with 0 when x=0."""
    return dispatch("special_xlog1py", input.device.type, input, other)


def xlogy(input, other):
    """Computes x * log(y), with 0 when x=0."""
    return dispatch("special_xlogy", input.device.type, input, other)


def zeta(input, other):
    """Computes the Hurwitz zeta function."""
    return dispatch("special_zeta", input.device.type, input, other)
