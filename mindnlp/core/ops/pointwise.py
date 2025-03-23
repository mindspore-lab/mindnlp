"""pointwise op"""
import mindspore
from mindspore import ops
from mindnlp.configs import use_pyboost

# abs
has_abs = hasattr(mindspore.mint, 'abs')
def abs(input):
    if use_pyboost() and has_abs:
        return mindspore.mint.abs(input)
    return ops.abs(input)

# absolute
def absolute(input):
    return abs(input)

# acos
has_acos = hasattr(mindspore.mint, 'acos')
def acos(input):
    if use_pyboost() and has_acos:
        return mindspore.mint.acos(input)
    return ops.acos(input)

# arccos
def arrcos(input):
    return acos(input)

# acosh
has_acosh = hasattr(mindspore.mint, 'acosh')
def acosh(input):
    if use_pyboost and has_acosh:
        return mindspore.mint.acosh(input)
    return ops.acosh(input)

# arccosh
def arccosh(input):
    return acosh(input)

# add
has_add = hasattr(mindspore.mint, 'add')
def add(input, other, *, alpha=1):
    if use_pyboost() and has_add:
        return mindspore.mint.add(input, other, alpha=alpha)
    if alpha != 1:
        other = mul(alpha, other)
    return ops.add(input, other)

# addcdiv
def addcdiv(input, tensor1, tensor2, *, value=1):
    return ops.addcdiv(input, tensor1, tensor2, value)

# addcmul
def addcmul(input, tensor1, tensor2, *, value=1):
    return ops.addcmul(input, tensor1, tensor2, value)

# angle
def angle(input):
    return ops.angle(input)

# asin
has_asin = hasattr(mindspore.mint, 'asin')
def asin(input):
    if use_pyboost and has_asin:
        return mindspore.mint.asin(input)
    return ops.asin(input)

# arcsin
def arcsin(input):
    return asin(input)

# asinh
has_asinh = hasattr(mindspore.mint, 'asinh')
def asinh(input):
    if use_pyboost and has_asinh:
        return mindspore.mint.asinh(input)
    return ops.asinh(input)

# arcsinh
def arcsinh(input):
    return asinh(input)

# atan
has_atan = hasattr(mindspore.mint, 'atan')
def atan(input):
    if use_pyboost and has_atan:
        return mindspore.mint.atan(input)
    return ops.atan(input)

# arctan
def arctan(input):
    return atan(input)

# atanh
has_atanh = hasattr(mindspore.mint, 'atanh')
def atanh(input):
    if use_pyboost and has_atanh:
        return mindspore.mint.atanh(input)
    return ops.atanh(input)

# arctanh
def arctanh(input):
    return atanh(input)

# atan2
has_atan2 = hasattr(mindspore.mint, 'atan2')
def atan2(input, other):
    if use_pyboost() and has_atan2:
        return mindspore.mint.atan2(input, other)
    return ops.atan2(input, other)

# arctan2
def arctan2(input, other):
    return atan2(input, other)

# bitwise_not

# bitwise_and
has_bitwise_and = hasattr(mindspore.mint, 'bitwise_and')
def bitwise_and(input, other):
    if use_pyboost() and has_bitwise_and:
        return mindspore.mint.bitwise_and(input, other)
    return ops.bitwise_and(input, other)

# bitwise_or
has_bitwise_or = hasattr(mindspore.mint, 'bitwise_or')
def bitwise_or(input, other):
    if use_pyboost() and has_bitwise_or:
        return mindspore.mint.bitwise_or(input, other)
    return ops.bitwise_or(input, other)

# bitwise_xor
has_bitwise_xor = hasattr(mindspore.mint, 'bitwise_xor')
def bitwise_xor(input, other):
    if use_pyboost() and has_bitwise_xor:
        return mindspore.mint.bitwise_xor(input, other)
    return ops.bitwise_xor(input, other)

# bitwise_left_shift
def bitwise_left_shift(input, other):
    return ops.bitwise_left_shift(input, other)

# bitwise_right_shift
def bitwise_right_shift(input, other):
    return ops.bitwise_right_shift(input, other)

# ceil
has_ceil = hasattr(mindspore.mint, 'ceil')
def ceil(input):
    if use_pyboost() and has_ceil:
        return mindspore.mint.ceil(input)
    return ops.ceil(input)

# clamp
has_clamp = hasattr(mindspore.mint, 'clamp')
def clamp(input, min=None, max=None):
    if use_pyboost() and has_clamp:
        return mindspore.mint.clamp(input, min, max)
    return ops.clamp(input, min, max)

# clip
def clip(input, min=None, max=None):
    return clamp(input, min, max)

# conj_physical


# copysign


# cos
has_cos = hasattr(mindspore.mint, 'cos')
def cos(input):
    if use_pyboost() and has_cos:
        return mindspore.mint.cos(input)
    return ops.cos(input)

# cosh
has_cosh = hasattr(mindspore.mint, 'cosh')
def cosh(input):
    if use_pyboost() and has_cosh:
        return mindspore.mint.cosh(input)
    return ops.cosh(input)

# deg2rad
def deg2rad(input):
    return ops.deg2rad(input)

# div
has_div = hasattr(mindspore.mint, 'div')
def div(input, other, *, rounding_mode=None):
    if use_pyboost() and has_div:
        return mindspore.mint.div(input, other, rounding_mode=rounding_mode)
    return ops.div(input, other, rounding_mode=rounding_mode)

# divide
has_divide = hasattr(mindspore.mint, 'divide')
def divide(input, other):
    return div(input, other)

# digamma
def digamma(input):
    return ops.digamma(input)

# erf
has_erf = hasattr(mindspore.mint, 'erf')
def erf(input):
    if use_pyboost() and has_erf:
        return mindspore.mint.erf(input)
    return ops.erf(input)

# erfc
has_erfc = hasattr(mindspore.mint, 'erfc')
def erfc(input):
    return ops.erfc(input)

# erfinv
has_erfinv = hasattr(mindspore.mint, 'erfinv')
def erfinv(input):
    if use_pyboost() and has_erfinv:
        return mindspore.mint.erfinv(input)
    return ops.erfinv(input)

# exp
has_exp = hasattr(mindspore.mint, 'exp')
def exp(input):
    if use_pyboost() and has_exp:
        return mindspore.mint.exp(input)
    return ops.exp(input)

# exp2
def exp2(input):
    return pow(2, input)

# expm1
has_expm1 = hasattr(mindspore.mint, 'expm1')
def expm1(input):
    if use_pyboost() and has_expm1:
        return mindspore.mint.expm1(input)
    return ops.expm1(input)

# fake_quantize_per_channel_affine


# fake_quantize_per_tensor_affine


# fix


# float_power
def float_power(input, exponent):
    return ops.float_power(input, exponent)

# floor
has_floor = hasattr(mindspore.mint, 'floor')
def floor(input):
    if use_pyboost() and has_floor:
        return mindspore.mint.floor(input)
    return ops.floor(input)

# floor_divide
def floor_divide(input, other):
    return ops.floor_divide(input, other)

# fmod
def fmod(input, other):
    return ops.fmod(input, other)

# frac
def frac(input):
    return fmod(input, 1)

# frexp


# imag
def imag(input):
    return ops.imag(input)

# ldexp


# lerp
def lerp(input, end, weight):
    return ops.lerp(input, end, weight)

# lgamma
def lgamma(input):
    return ops.lgamma(input)

# log
has_log = hasattr(mindspore.mint, 'log')
def log(input):
    if use_pyboost() and has_log:
        return mindspore.mint.log(input)
    return ops.log(input)

# log10

# log1p
has_log1p = hasattr(mindspore.mint, 'log1p')
def log1p(input):
    if use_pyboost() and has_log1p:
        return mindspore.mint.log1p(input)
    return ops.log1p(input)

# log2
def log2(input):
    return ops.log2(input)

# logaddexp


# logaddexp2


# logical_and
has_logical_and = hasattr(mindspore.mint, 'logical_and')
def logical_and(input, other):
    if use_pyboost() and has_logical_and:
        return mindspore.mint.logical_and(input, other)
    return ops.logical_and(input, other)

# logical_not
has_logical_not = hasattr(mindspore.mint, 'logical_not')
def logical_not(input):
    if use_pyboost() and has_logical_not:
        return mindspore.mint.logical_not(input)
    return ops.logical_not(input)

# logical_or
has_logical_or = hasattr(mindspore.mint, 'logical_or')
def logical_or(input, other):
    if use_pyboost() and has_logical_or:
        return mindspore.mint.logical_or(input, other)
    return ops.logical_or(input, other)

# logical_xor
has_logical_xor = hasattr(mindspore.mint, 'logical_xor')
def logical_xor(input, other):
    if use_pyboost() and has_logical_xor:
        return mindspore.mint.logical_xor(input, other)
    return ops.logical_xor(input, other)

# logit
def logit(input, eps=None):
    return ops.logit(input, eps)

# hypot
def hypot(input, other):
    return ops.hypot(input, other)

# i0

# igamma
def igamma(input, other):
    return ops.igamma(input, other)

# igammac
def igammac(input, other):
    return ops.igammac(input, other)

# mul
has_mul = hasattr(mindspore.mint, 'mul')
def mul(input, other):
    if use_pyboost() and has_mul:
        return mindspore.mint.mul(input, other)
    return ops.mul(input, other)

# multiply
def multiply(input, other):
    return mul(input, other)

# mvlgamma
def mvlgamma(input, p):
    return ops.mvlgamma(input, p)

# nan_to_num
has_nan_to_num = hasattr(mindspore.mint, 'nan_to_num')
def nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    if use_pyboost() and has_nan_to_num:
        return mindspore.mint.nan_to_num(input, nan, posinf, neginf)
    return ops.nan_to_num(input, nan, posinf, neginf)

# neg
has_neg = hasattr(mindspore.mint, 'neg')
def neg(input):
    if use_pyboost() and has_neg:
        return mindspore.mint.neg(input)
    return ops.neg(input)

# negative
has_negative = hasattr(mindspore.mint, 'negative')
def negative(input):
    return neg(input)

# nextafter
def nextafter(input, other):
    return ops.nextafter(input, other)

# polygamma
def polygamma(n, input):
    return ops.polygamma(n, input)

# positive
def positive(input):
    return input

# pow
has_pow = hasattr(mindspore.mint, 'pow')
def pow(input, exponent):
    if use_pyboost() and has_pow:
        return mindspore.mint.pow(input, exponent)
    return ops.pow(input, exponent)

# quantized_batch_norm


# quantized_max_pool1d


# quantized_max_pool2d


# rad2deg
def rad2deg(input):
    return ops.rad2deg(input)

# real
def real(input):
    return ops.real(input)

# reciprocal
has_reciprocal = hasattr(mindspore.mint, 'reciprocal')
def reciprocal(input):
    if use_pyboost() and has_reciprocal:
        return mindspore.mint.reciprocal(input)
    return ops.reciprocal(input)

# remainder
has_remainder = hasattr(mindspore.mint, 'remainder')
def remainder(input, other):
    if use_pyboost() and has_remainder:
        return mindspore.mint.remainder(input, other)
    return ops.remainder(input, other)

# round
has_round = hasattr(mindspore.mint, 'round')
def round(input):
    if use_pyboost() and has_round:
        return mindspore.mint.round(input)
    return ops.round(input)

# rsqrt
has_rsqrt = hasattr(mindspore.mint, 'rsqrt')
def rsqrt(input):
    if use_pyboost() and has_rsqrt:
        return mindspore.mint.rsqrt(input)
    return ops.rsqrt(input)

# sigmoid
has_sigmoid = hasattr(mindspore.mint, 'sigmoid')
def sigmoid(input):
    if use_pyboost() and has_sigmoid:
        return mindspore.mint.sigmoid(input)
    return ops.sigmoid(input)

# sign
has_sign = hasattr(mindspore.mint, 'sign')
def sign(input):
    if use_pyboost() and has_sign:
        return mindspore.mint.sign(input)
    return ops.sign(input)

# sgn

# signbit

# sin
has_sin = hasattr(mindspore.mint, 'sin')
def sin(input):
    if use_pyboost() and has_sin:
        return mindspore.mint.sin(input)
    return ops.sin(input)

# sinc
has_sinc = hasattr(mindspore.mint, 'sinc')
def sinc(input):
    if use_pyboost() and has_sinc:
        return mindspore.mint.sinc(input)
    return ops.sinc(input)

# sinh
has_sinh = hasattr(mindspore.mint, 'sinh')
def sinh(input):
    if use_pyboost() and has_sinh:
        return mindspore.mint.sinh(input)
    return ops.sinh(input)

# softmax
def softmax(input, dim=-1, *, dtype=None):
    if use_pyboost():
        return mindspore.mint.nn.functional.softmax(input, dim, dtype=dtype)
    return ops.softmax(input, dim, dtype=dtype)

# sqrt
has_sqrt = hasattr(mindspore.mint, 'sqrt')
def sqrt(input):
    if use_pyboost() and has_sqrt:
        return mindspore.mint.sqrt(input)
    return ops.sqrt(input)

# square
has_square = hasattr(mindspore.mint, 'square')
def square(input):
    if use_pyboost() and has_square:
        return mindspore.mint.square(input)
    return ops.square(input)

# sub
has_sub = hasattr(mindspore.mint, 'sub')
def sub(input, other):
    if use_pyboost() and has_sub:
        return mindspore.mint.sub(input, other)
    return ops.sub(input, other)

# subtract
def subtract(input, other):
    return sub(input, other)

# tan
has_tan = hasattr(mindspore.mint, 'tan')
def tan(input):
    if use_pyboost() and has_tan:
        return mindspore.mint.tan(input)
    return ops.tan(input)

# tanh
has_tanh = hasattr(mindspore.mint, 'tanh')
def tanh(input):
    if use_pyboost() and has_tanh:
        return mindspore.mint.tanh(input)
    return ops.tanh(input)

# true_divide
def true_divide(input, other):
    return div(input, other)

# trunc
has_trunc = hasattr(mindspore.mint, 'trunc')
def trunc(input):
    if use_pyboost() and has_trunc:
        return mindspore.mint.trunc(input)
    return ops.trunc(input)

# xlogy
has_xlogy = hasattr(mindspore.mint, 'xlogy')
def xlogy(input, other):
    if use_pyboost() and has_xlogy:
        return mindspore.mint.xlogy(input, other)
    return ops.xlogy(input, other)
