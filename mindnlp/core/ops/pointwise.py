"""pointwise op"""
import mindspore
from mindspore import ops
from mindnlp.configs import USE_PYBOOST

# abs
def abs(input):
    if USE_PYBOOST:
        return mindspore.mint.abs(input)
    return ops.abs(input)

# absolute
def absolute(input):
    return abs(input)

# acos
def acos(input):
    return ops.acos(input)

# arccos
def arrcos(input):
    return acos(input)

# acosh
def acosh(input):
    return ops.acosh(input)

# arccosh
def arccosh(input):
    return acosh(input)

# add
def add(input, other, *, alpha=1):
    if USE_PYBOOST:
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
def asin(input):
    return ops.asin(input)

# arcsin
def arcsin(input):
    return asin(input)

# asinh
def asinh(input):
    return ops.asinh(input)

# arcsinh
def arcsinh(input):
    return asinh(input)

# atan
def atan(input):
    return ops.atan(input)

# arctan
def arctan(input):
    return atan(input)

# atanh
def atanh(input):
    return ops.atanh(input)

# arctanh
def arctanh(input):
    return atanh(input)

# atan2
def atan2(input, other):
    if USE_PYBOOST:
        return mindspore.mint.atan2(input, other)
    return ops.atan2(input, other)

# arctan2
def arctan2(input, other):
    return atan2(input, other)

# bitwise_not

# bitwise_and
def bitwise_and(input, other):
    return ops.bitwise_and(input, other)

# bitwise_or
def bitwise_or(input, other):
    return ops.bitwise_or(input, other)

# bitwise_xor
def bitwise_xor(input, other):
    return ops.bitwise_xor(input, other)

# bitwise_left_shift
def bitwise_left_shift(input, other):
    return ops.bitwise_left_shift(input, other)

# bitwise_right_shift
def bitwise_right_shift(input, other):
    return ops.bitwise_right_shift(input, other)

# ceil
def ceil(input):
    if USE_PYBOOST:
        return mindspore.mint.ceil(input)
    return ops.ceil(input)

# clamp
def clamp(input, min=None, max=None):
    if USE_PYBOOST:
        return mindspore.mint.clamp(input, min, max)
    return ops.clamp(input, min, max)

# clip
def clip(input, min=None, max=None):
    return clamp(input, min, max)

# conj_physical


# copysign


# cos
def cos(input):
    if USE_PYBOOST:
        return mindspore.mint.cos(input)
    return ops.cos(input)

# cosh
def cosh(input):
    return ops.cosh(input)

# deg2rad
def deg2rad(input):
    return ops.deg2rad(input)

# div
def div(input, other, *, rounding_mode=None):
    if USE_PYBOOST:
        return mindspore.mint.div(input, other, rounding_mode=rounding_mode)
    return ops.div(input, other, rounding_mode=rounding_mode)

# divide
def divide(input, other):
    return div(input, other)

# digamma
def digamma(input):
    return ops.digamma(input)

# erf
def erf(input):
    if USE_PYBOOST:
        return mindspore.mint.erf(input)
    return ops.erf(input)

# erfc
def erfc(input):
    return ops.erfc(input)

# erfinv
def erfinv(input):
    if USE_PYBOOST:
        return mindspore.mint.erfinv(input)
    return ops.erfinv(input)

# exp
def exp(input):
    if USE_PYBOOST:
        return mindspore.mint.exp(input)
    return ops.exp(input)

# exp2
def exp2(input):
    return pow(2, input)

# expm1
def expm1(input):
    return ops.expm1(input)

# fake_quantize_per_channel_affine


# fake_quantize_per_tensor_affine


# fix


# float_power
def float_power(input, exponent):
    return ops.float_power(input, exponent)

# floor
def floor(input):
    if USE_PYBOOST:
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
def log(input):
    if USE_PYBOOST:
        return mindspore.mint.log(input)
    return ops.log(input)

# log10

# log1p
def log1p(input):
    return ops.log1p(input)

# log2
def log2(input):
    return ops.log2(input)

# logaddexp


# logaddexp2


# logical_and
def logical_and(input, other):
    if USE_PYBOOST:
        return mindspore.mint.logical_and(input, other)
    return ops.logical_and(input, other)

# logical_not
def logical_not(input):
    if USE_PYBOOST:
        return mindspore.mint.logical_not(input)
    return ops.logical_not(input)

# logical_or
def logical_or(input, other):
    if USE_PYBOOST:
        return mindspore.mint.logical_or(input, other)
    return ops.logical_or(input, other)

# logical_xor
def logical_xor(input, other):
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
def mul(input, other):
    if USE_PYBOOST:
        return mindspore.mint.mul(input, other)
    return ops.mul(input, other)

# multiply
def multiply(input, other):
    return mul(input, other)

# mvlgamma
def mvlgamma(input, p):
    return ops.mvlgamma(input, p)

# nan_to_num
def nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    return ops.nan_to_num(input, nan, posinf, neginf)

# neg
def neg(input):
    if USE_PYBOOST:
        return mindspore.mint.neg(input)
    return ops.neg(input)

# negative
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
def pow(input, exponent):
    if USE_PYBOOST:
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
def reciprocal(input):
    if USE_PYBOOST:
        return mindspore.mint.reciprocal(input)
    return ops.reciprocal(input)

# remainder
def remainder(input, other):
    return ops.remainder(input, other)

# round
def round(input):
    return ops.round(input)

# rsqrt
def rsqrt(input):
    if USE_PYBOOST:
        return mindspore.mint.rsqrt(input)
    return ops.rsqrt(input)

# sigmoid
def sigmoid(input):
    if USE_PYBOOST:
        return mindspore.mint.sigmoid(input)
    return ops.sigmoid(input)

# sign
def sign(input):
    return ops.sign(input)

# sgn

# signbit

# sin
def sin(input):
    if USE_PYBOOST:
        return mindspore.mint.sin(input)
    return ops.sin(input)

# sinc
def sinc(input):
    return ops.sinc(input)

# sinh
def sinh(input):
    return ops.sinh(input)

# softmax
def softmax(input, dim=-1, *, dtype=None):
    if USE_PYBOOST:
        return mindspore.mint.softmax(input, dim, dtype=dtype)
    return ops.softmax(input, dim, dtype=dtype)

# sqrt
def sqrt(input):
    if USE_PYBOOST:
        return mindspore.mint.sqrt(input)
    return ops.sqrt(input)

# square
def square(input):
    if USE_PYBOOST:
        return mindspore.mint.square(input)
    return ops.square(input)

# sub
def sub(input, other):
    if USE_PYBOOST:
        return mindspore.mint.sub(input, other)
    return ops.sub(input, other)

# subtract
def subtract(input, other):
    return sub(input, other)

# tan
def tan(input):
    return ops.tan(input)

# tanh
def tanh(input):
    if USE_PYBOOST:
        return mindspore.mint.tanh(input)
    return ops.tanh(input)

# true_divide
def true_divide(input, other):
    return div(input, other)

# trunc
def trunc(input):
    return ops.trunc(input)

# xlogy
def xlogy(input, other):
    return ops.xlogy(input, other)
