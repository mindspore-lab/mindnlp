"""pointwise op"""
import numpy as np
import mindspore
from mindspore import ops
from ..configs import use_pyboost, ON_A1, ON_ORANGE_PI
from ._inner import call_ms_func

from mindnlp import core

# abs
has_abs = hasattr(mindspore.mint, "abs")


def abs(input, *, out=None):
    if use_pyboost() and has_abs:
        return call_ms_func(mindspore.mint.abs, input, out=out)
    return call_ms_func(ops.abs, input, out=out)


# absolute
def absolute(input, *, out=None):
    return abs(input, out=out)


# acos
has_acos = hasattr(mindspore.mint, "acos")


def acos(input, *, out=None):
    if use_pyboost() and has_acos:
        return call_ms_func(mindspore.mint.acos, input, out=out)
    return call_ms_func(ops.acos, input, out=out)


# arccos
def arrcos(input, out=None):
    return acos(input, out=out)


# acosh
has_acosh = hasattr(mindspore.mint, "acosh")


def acosh(input, *, out=None):
    if use_pyboost and has_acosh:
        return call_ms_func(mindspore.mint.acosh, input, out=out)
    return call_ms_func(ops.acosh, input, out=out)


# arccosh
has_arccosh = hasattr(mindspore.mint, "arccosh")


def arccosh(input):
    return acosh(input)


# add
has_add = hasattr(mindspore.mint, "add")


def add(input, other, *, alpha=1, out=None):
    if use_pyboost() and has_add and not ON_ORANGE_PI:
        return call_ms_func(mindspore.mint.add, input, other, alpha=alpha, out=out)
    if alpha != 1:
        other = mul(alpha, other)
    if input.dtype == mindspore.bool_:
        return ops.add(input.int(), other.int()).bool()
    return call_ms_func(ops.add, input, other, out=out)


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
has_asin = hasattr(mindspore.mint, "asin")


def asin(input, *, out=None):
    if use_pyboost and has_asin:
        return call_ms_func(mindspore.mint.asin, input, out=out)
    return call_ms_func(ops.asin, input, out=out)


# arcsin
has_arcsin = hasattr(mindspore.mint, "arcsin")


def arcsin(input, *, out=None):
    return asin(input, out=out)


# asinh
has_asinh = hasattr(mindspore.mint, "asinh")


def asinh(input, *, out=None):
    if use_pyboost and has_asinh:
        return call_ms_func(mindspore.mint.asinh, input, out=out)
    return call_ms_func(ops.asinh, input, out=out)


# arcsinh
has_arcsinh = hasattr(mindspore.mint, "arcsinh")


def arcsinh(input, *, out=None):
    return asinh(input, out=out)


# atan
has_atan = hasattr(mindspore.mint, "atan")


def atan(input, *, out=None):
    if use_pyboost and has_atan:
        return call_ms_func(mindspore.mint.atan, input, out=out)
    return call_ms_func(ops.atan, input, out=out)


# arctan
has_arctan = hasattr(mindspore.mint, "arctan")


def arctan(input, *, out=None):
    return atan(input, out=out)


# atanh
has_atanh = hasattr(mindspore.mint, "atanh")


def atanh(input, *, out=None):
    if use_pyboost and has_atanh:
        return call_ms_func(mindspore.mint.atanh, input, out=out)
    return call_ms_func(ops.atanh, input, out=out)


# arctanh
has_arctanh = hasattr(mindspore.mint, "arctanh")


def arctanh(input, *, out=None):
    return atanh(input, out=out)


# atan2
has_atan2 = hasattr(mindspore.mint, "atan2")


def atan2(input, other, *, out=None):
    if use_pyboost() and has_atan2:
        return call_ms_func(mindspore.mint.atan2, input, other, out=out)
    return call_ms_func(ops.atan2, input, other, out=out)


# arctan2
has_arctan2 = hasattr(mindspore.mint, "arctan2")


def arctan2(input, other, out=None):
    return atan2(input, other, out=out)


# bitwise_not

# bitwise_and
has_bitwise_and = hasattr(mindspore.mint, "bitwise_and")


def bitwise_and(input, other, *, out=None):
    if use_pyboost() and has_bitwise_and:
        return call_ms_func(mindspore.mint.bitwise_and, input, other, out=out)
    return call_ms_func(ops.bitwise_and, input, other, out=out)


# bitwise_or
has_bitwise_or = hasattr(mindspore.mint, "bitwise_or")


def bitwise_or(input, other, *, out=None):
    if use_pyboost() and has_bitwise_or:
        return call_ms_func(mindspore.mint.bitwise_or, input, other, out=out)
    return call_ms_func(ops.bitwise_or, input, other, out=out)


# bitwise_xor
has_bitwise_xor = hasattr(mindspore.mint, "bitwise_xor")


def bitwise_xor(input, other, *, out=None):
    if use_pyboost() and has_bitwise_xor:
        return call_ms_func(mindspore.mint.bitwise_xor, input, other, out=out)
    return call_ms_func(ops.bitwise_xor, input, other, out=out)


# bitwise_left_shift
def bitwise_left_shift(input, other):
    return ops.bitwise_left_shift(input, other)


# bitwise_right_shift
def bitwise_right_shift(input, other):
    return ops.bitwise_right_shift(input, other)


# ceil
has_ceil = hasattr(mindspore.mint, "ceil")


def ceil(input, *, out=None):
    if use_pyboost() and has_ceil:
        return call_ms_func(mindspore.mint.ceil, input, out=out)
    return call_ms_func(ops.ceil, input, out=out)


# clamp
has_clamp = hasattr(mindspore.mint, "clamp")


def clamp(input, min=None, max=None, *, out=None):
    if use_pyboost() and has_clamp:
        return call_ms_func(mindspore.mint.clamp, input, min, max, out=out)
    return call_ms_func(ops.clamp, input, min, max, out=out)


def clamp_min(input, min):
    return clamp(input, min)

# clip
has_clip = hasattr(mindspore.mint, "clip")


def clip(input, min=None, max=None):
    return clamp(input, min, max)


# conj_physical


# copysign


# cos
has_cos = hasattr(mindspore.mint, "cos")


def cos(input, *, out=None):
    if use_pyboost() and has_cos:
        return call_ms_func(mindspore.mint.cos, input, out=out)
    return call_ms_func(ops.cos, input, out=out)


# cosh
has_cosh = hasattr(mindspore.mint, "cosh")


def cosh(input, *, out=None):
    if use_pyboost() and has_cosh:
        return call_ms_func(mindspore.mint.cosh, input, out=out)
    return call_ms_func(ops.cosh, input, out=out)


# deg2rad
def deg2rad(input):
    return ops.deg2rad(input)


# div
has_div = hasattr(mindspore.mint, "div")


def div(input, other, *, rounding_mode=None, out=None):
    if isinstance(other, mindspore.Tensor):
        other = other.to(input.dtype)

    if isinstance(other, np.number):
        other = other.item()

    if use_pyboost() and has_div:
        return call_ms_func(
            mindspore.mint.div, input, other, rounding_mode=rounding_mode, out=out
        )
    return call_ms_func(ops.div, input, other, rounding_mode=rounding_mode, out=out)


# divide
has_divide = hasattr(mindspore.mint, "divide")


def divide(input, other, rounding_mode=None):
    return div(input, other, rounding_mode=rounding_mode)


# digamma
def digamma(input):
    return ops.digamma(input)


# erf
has_erf = hasattr(mindspore.mint, "erf")


def erf(input, *, out=None):
    if use_pyboost() and has_erf:
        return call_ms_func(mindspore.mint.erf, input, out=out)
    return call_ms_func(ops.erf, input, out=out)


# erfc
has_erfc = hasattr(mindspore.mint, "erfc")


def erfc(input, *, out=None):
    if use_pyboost() and has_erfc:
        return call_ms_func(mindspore.mint.erfc, input, out=out)
    return call_ms_func(ops.erfc, input, out=out)


# erfinv
has_erfinv = hasattr(mindspore.mint, "erfinv")


def erfinv(input, *, out=None):
    if ON_ORANGE_PI:
        return erfinv_torch(input)
    if use_pyboost() and has_erfinv:
        return call_ms_func(mindspore.mint.erfinv, input, out=out)
    return call_ms_func(ops.erfinv, input, out=out)

def erfinv_torch(x):
    """
    使用有理函数近似实现erfinv，适用于PyTorch张量
    """
    # # 检查输入范围
    # if core.any((x < -1) | (x > 1)):
    #     raise ValueError("erfinv(x) is only defined for x in [-1, 1]")
    
    # 处理边界情况
    sign = core.where(x > 0, 1.0, -1.0)
    x = core.abs(x)
    
    # Cody的有理函数近似
    mask = x <= 0.7
    x_sq = x * x
    
    # 对于x <= 0.7的情况
    p1 = 0.426170613044 + x_sq * (-0.304570194263 + x_sq * 0.152645863430)
    q1 = 1.0 + x_sq * (-0.733058978416 + x_sq * 0.546875000000)
    result1 = x * (p1 / q1)
    
    # 对于x > 0.7的情况
    t = core.sqrt(-core.log((1.0 - x)/2.0))
    p2 = -0.322232431088 + t * (-1.00002368368 + t * (-0.342242088547 + 
         t * (-0.0204231210245 + t * (-0.0000453642210148))))
    q2 = 0.460398842078 + t * (0.588581570495 + t * (0.531103462366 + 
         t * (0.103537752850 + t * 0.0038560700634)))
    result2 = p2 / q2
    
    # 合并结果
    result = core.where(mask, result1, result2)
    
    return sign * result

# exp
has_exp = hasattr(mindspore.mint, "exp")
has_inplace_exp = hasattr(mindspore.Tensor, "exp_")


def exp(input, out=None):
    if has_inplace_exp:
        return inplace_exp(input, out)

    if use_pyboost() and has_exp:
        output = mindspore.mint.exp(input)
    else:
        output = ops.exp(input)
    if out is not None:
        # out.data = output
        out.assign_value(output)
    else:
        return output


def inplace_exp(input, out=None):
    if out is None:
        if use_pyboost() and has_exp:
            output = mindspore.mint.exp(input)
        else:
            output = ops.exp(input)
        return output

    if out is input:
        return out.exp_()
    else:
        out.copy_(input)
        return out.exp_()


# exp2
has_exp2 = hasattr(mindspore.mint, "exp2")


def exp2(input):
    if use_pyboost() and has_exp2:
        return mindspore.mint.exp2(input)
    return pow(2, input)


# expm1
has_expm1 = hasattr(mindspore.mint, "expm1")


def expm1(input, *, out=None):
    if input.dtype == mindspore.float64:
        return expm1(input.float(), out=out).double()
    if use_pyboost() and has_expm1:
        return call_ms_func(mindspore.mint.expm1, input, out=out)
    return call_ms_func(ops.expm1, input, out=out)


# fake_quantize_per_channel_affine


# fake_quantize_per_tensor_affine


# fix


# float_power
has_float_power = hasattr(mindspore.mint, "float_power")


def float_power(input, exponent):
    if use_pyboost() and has_float_power:
        return mindspore.mint.float_power(input, exponent)
    return ops.float_power(input, exponent)


# floor
has_floor = hasattr(mindspore.mint, "floor")


def floor(input, *, out=None):
    if use_pyboost() and has_floor:
        return call_ms_func(mindspore.mint.floor, input, out=out)
    return call_ms_func(ops.floor, input, out=out)


# floor_divide
def floor_divide(input, other):
    return ops.floor_divide(input, other)


# fmod
has_fmod = hasattr(mindspore.mint, "fmod")


def fmod(input, other):
    if use_pyboost() and has_fmod:
        return mindspore.mint.fmod(input, other)
    return ops.fmod(input, other)


# frac
has_frac = hasattr(mindspore.mint, "frac")


def frac(input):
    if use_pyboost() and has_frac:
        return mindspore.mint.frac(input)
    return fmod(input, 1)


# frexp


# imag
def imag(input):
    return ops.imag(input)


# ldexp


# lerp
has_lerp = hasattr(mindspore.mint, "lerp")


def lerp(input, end, weight):
    if use_pyboost() and has_lerp:
        return mindspore.mint.lerp(input, end, weight)
    return ops.lerp(input, end, weight)


# lgamma
def lgamma(input):
    return ops.lgamma(input)


# log
has_log = hasattr(mindspore.mint, "log")


def log(input, *, out=None):
    if use_pyboost() and has_log:
        return call_ms_func(mindspore.mint.log, input, out=out)
    return call_ms_func(ops.log, input, out=out)


# log10

# log1p
has_log1p = hasattr(mindspore.mint, "log1p")


def log1p(input, *, out=None):
    if use_pyboost() and has_log1p:
        return call_ms_func(mindspore.mint.log1p, input, out=out)
    return call_ms_func(ops.log1p, input, out=out)


# log2
has_log2 = hasattr(mindspore.mint, "log2")


def log2(input):
    if use_pyboost() and has_log2:
        return mindspore.mint.log2(input)
    return ops.log2(input)


# logaddexp


# logaddexp2


# logical_and
has_logical_and = hasattr(mindspore.mint, "logical_and")


def logical_and(input, other, *, out=None):
    if use_pyboost() and has_logical_and:
        return call_ms_func(mindspore.mint.logical_and, input, other, out=out)
    return call_ms_func(ops.logical_and, input, other, out=out)


# logical_not
has_logical_not = hasattr(mindspore.mint, "logical_not")


def logical_not(input, *, out=None):
    if use_pyboost() and has_logical_not:
        return call_ms_func(mindspore.mint.logical_not, input, out=out)
    return call_ms_func(ops.logical_not, input, out=out)


# logical_or
has_logical_or = hasattr(mindspore.mint, "logical_or")


def logical_or(input, other, *, out=None):
    if use_pyboost() and has_logical_or:
        return call_ms_func(mindspore.mint.logical_or, input, other, out=out)
    return call_ms_func(ops.logical_or, input, other, out=out)


# logical_xor
has_logical_xor = hasattr(mindspore.mint, "logical_xor")


def logical_xor(input, other, *, out=None):
    if use_pyboost() and has_logical_xor:
        return call_ms_func(mindspore.mint.logical_xor, input, other, out=out)
    return call_ms_func(ops.logical_xor, input, other, out=out)


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
has_mul = hasattr(mindspore.mint, "mul")


def mul(input, other, *, out=None):
    if use_pyboost() and has_mul and not ON_ORANGE_PI:
        out = mindspore.mint.mul(input, other)
    else:
        if input.dtype == mindspore.bool_:
            if isinstance(other, bool):
                if ON_ORANGE_PI:
                    out = ops.bitwise_and(input.int(), other).bool()
                else:
                    out = ops.bitwise_and(input, other)
            else:
                out = ops.mul(input.int(), other)
        else:
            out = ops.mul(input, other)
        return out

    if isinstance(other, mindspore.Tensor):
        out_dtype = min(input.dtype, other.dtype)
        return out.to(out_dtype)
    return out

# multiply
def multiply(input, other):
    return mul(input, other)


# mvlgamma
def mvlgamma(input, p):
    return ops.mvlgamma(input, p)


# nan_to_num
has_nan_to_num = hasattr(mindspore.mint, "nan_to_num")


def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None):
    if use_pyboost() and has_nan_to_num and not ON_A1:
        return call_ms_func(
            mindspore.mint.nan_to_num, input, nan, posinf, neginf, out=out
        )

    # 创建输入张量的副本
    output = input.clone()
    # 获取数据类型信息
    if output.is_floating_point():
        dtype = output.dtype
        # 获取默认替换值
        f_info = core.finfo(dtype)
        default_posinf = f_info.max if posinf is None else posinf
        default_neginf = f_info.min if neginf is None else neginf
    else:
        # 对于整数类型，使用给定值或默认值
        default_posinf = core.iinfo(dtype).max if posinf is None else posinf
        default_neginf = core.iinfo(dtype).min if neginf is None else neginf

    # 替换 NaN
    if core.isnan(output).any():
        output = core.where(
            core.isnan(output),
            core.tensor(nan, dtype=output.dtype, device=output.device),
            output,
        )

    # 替换正无穷大
    if core.isinf(output).any() and (posinf is not None or output.is_floating_point()):
        output = core.where(
            (output == float("inf")) & core.isinf(output),
            core.tensor(default_posinf, dtype=output.dtype, device=output.device),
            output,
        )

    # 替换负无穷大
    if core.isinf(output).any() and (neginf is not None or output.is_floating_point()):
        output = core.where(
            (output == float("-inf")) & core.isinf(output),
            core.tensor(default_neginf, dtype=output.dtype, device=output.device),
            output,
        )

    return output


# neg
has_neg = hasattr(mindspore.mint, "neg")


def neg(input, *, out=None):
    if use_pyboost() and has_neg:
        return call_ms_func(mindspore.mint.neg, input, out=out)
    return call_ms_func(ops.neg, input, out=out)


# negative
has_negative = hasattr(mindspore.mint, "negative")


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
has_pow = hasattr(mindspore.mint, "pow")


def pow(input, exponent, *, out=None):
    if use_pyboost() and has_pow:
        return call_ms_func(mindspore.mint.pow, input, exponent, out=out)
    return call_ms_func(ops.pow, input, exponent, out=out)


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
has_reciprocal = hasattr(mindspore.mint, "reciprocal")


def reciprocal(input, *, out=None):
    if use_pyboost() and has_reciprocal:
        return call_ms_func(mindspore.mint.reciprocal, input, out=out)
    return call_ms_func(ops.reciprocal, input, out=out)


# remainder
has_remainder = hasattr(mindspore.mint, "remainder")


def remainder(input, other, *, out=None):
    if use_pyboost() and has_remainder:
        return call_ms_func(mindspore.mint.remainder, input, other, out=out)
    return call_ms_func(ops.remainder, input, other, out=out)


# round
has_round = hasattr(mindspore.mint, "round")


def round(input, *, decimals=0):
    if use_pyboost() and has_round:
        return mindspore.mint.round(input, decimals=decimals)
    return ops.round(input, decimals=decimals)


# rsqrt
has_rsqrt = hasattr(mindspore.mint, "rsqrt")


def rsqrt(input, *, out=None):
    if use_pyboost() and has_rsqrt:
        return call_ms_func(mindspore.mint.rsqrt, input, out=out)
    return call_ms_func(ops.rsqrt, input, out=out)


# sigmoid
has_sigmoid = hasattr(mindspore.mint, "sigmoid")


def sigmoid(input, *, out=None):
    if use_pyboost() and has_sigmoid:
        return call_ms_func(mindspore.mint.sigmoid, input, out=out)
    return call_ms_func(ops.sigmoid, input, out=out)


# sign
has_sign = hasattr(mindspore.mint, "sign")


def sign(input, *, out=None):
    if use_pyboost() and has_sign:
        return call_ms_func(mindspore.mint.sign, input, out=out)
    return call_ms_func(ops.sign, input, out=out)


# sgn

# signbit

# sin
has_sin = hasattr(mindspore.mint, "sin")


def sin(input, *, out=None):
    if use_pyboost() and has_sin:
        return call_ms_func(mindspore.mint.sin, input, out=out)
    return call_ms_func(ops.sin, input, out=out)


# sinc
has_sinc = hasattr(mindspore.mint, "sinc")


def sinc(input, *, out=None):
    if use_pyboost() and has_sinc:
        return call_ms_func(mindspore.mint.sinc, input, out=out)
    return call_ms_func(ops.sinc, input, out=out)


# sinh
has_sinh = hasattr(mindspore.mint, "sinh")


def sinh(input, *, out=None):
    if use_pyboost() and has_sinh:
        return call_ms_func(mindspore.mint.sinh, input, out=out)
    return call_ms_func(ops.sinh, input, out=out)


# softmax
def softmax(input, dim, *, dtype=None):
    if use_pyboost():
        return mindspore.mint.nn.functional.softmax(input, dim, dtype=dtype)
    return ops.softmax(input, dim, dtype=dtype)


def log_softmax(input, dim=None, dtype=None):
    return core.nn.functional.log_softmax(input, dim, dtype)


# sqrt
has_sqrt = hasattr(mindspore.mint, "sqrt")


def sqrt(input, *, out=None):
    if use_pyboost() and has_sqrt:
        return call_ms_func(mindspore.mint.sqrt, input, out=out)
    return call_ms_func(ops.sqrt, input, out=out)


# square
has_square = hasattr(mindspore.mint, "square")


def square(input, *, out=None):
    if use_pyboost() and has_square:
        return call_ms_func(mindspore.mint.square, input, out=out)
    return call_ms_func(ops.square, input, out=out)


# sub
has_sub = hasattr(mindspore.mint, "sub")


def sub(input, other, *, alpha=1, out=None):
    if isinstance(other, mindspore.Tensor):
        other = other.to(input.dtype)
    if use_pyboost() and has_sub:
        return call_ms_func(mindspore.mint.sub, input, other, alpha=alpha, out=out)
    return call_ms_func(ops.sub, input, other, out=out)


# subtract
def subtract(input, other):
    return sub(input, other)


# tan
has_tan = hasattr(mindspore.mint, "tan")


def tan(input, *, out=None):
    if use_pyboost() and has_tan:
        return call_ms_func(mindspore.mint.tan, input, out=out)
    return call_ms_func(ops.tan, input, out=out)


# tanh
has_tanh = hasattr(mindspore.mint, "tanh")


def tanh(input, *, out=None):
    if use_pyboost() and has_tanh:
        return call_ms_func(mindspore.mint.tanh, input, out=out)
    return call_ms_func(ops.tanh, input, out=out)


# true_divide
def true_divide(input, other):
    return div(input, other)


# trunc
has_trunc = hasattr(mindspore.mint, "trunc")


def trunc(input, *, out=None):
    if use_pyboost() and has_trunc:
        return call_ms_func(mindspore.mint.trunc, input, out=out)
    return call_ms_func(ops.trunc, input, out=out)


# xlogy
has_xlogy = hasattr(mindspore.mint, "xlogy")


def xlogy(input, other, *, out=None):
    if use_pyboost() and has_xlogy:
        return call_ms_func(mindspore.mint.xlogy, input, other, out=out)
    return call_ms_func(ops.xlogy, input, other, out=out)


# relu
def relu(input):
    if use_pyboost():
        return mindspore.mint.nn.functional.relu(input)
    return ops.relu(input)


__all__ = [
    "abs",
    "absolute",
    "acos",
    "acosh",
    "add",
    "addcdiv",
    "addcmul",
    "angle",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "arrcos",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "clamp",
    "clamp_min",
    "clip",
    "cos",
    "cosh",
    "deg2rad",
    "digamma",
    "div",
    "divide",
    "erf",
    "erfc",
    "erfinv",
    "exp",
    "exp2",
    "expm1",
    "float_power",
    "floor",
    "floor_divide",
    "fmod",
    "frac",
    "hypot",
    "igamma",
    "igammac",
    "imag",
    "lerp",
    "lgamma",
    "log",
    "log1p",
    "log2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logit",
    "log_softmax",
    "mul",
    "multiply",
    "mvlgamma",
    "nan_to_num",
    "neg",
    "negative",
    "nextafter",
    "polygamma",
    "positive",
    "pow",
    "rad2deg",
    "real",
    "reciprocal",
    "remainder",
    "round",
    "rsqrt",
    "sigmoid",
    "sign",
    "sin",
    "sinc",
    "sinh",
    "softmax",
    "sqrt",
    "square",
    "sub",
    "subtract",
    "tan",
    "tanh",
    "true_divide",
    "trunc",
    "xlogy",
    "relu",
]
