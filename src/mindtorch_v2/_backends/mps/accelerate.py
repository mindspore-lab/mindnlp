"""Accelerate framework bindings via ctypes.

Provides fast vectorized math (vDSP), transcendentals (vForce), and BLAS
for float32/float64 contiguous arrays on macOS.
"""
import ctypes
import sys

_accel = None
_loaded = False


def _ensure_loaded():
    global _accel, _loaded
    if _loaded:
        return
    _loaded = True
    if sys.platform != "darwin":
        return
    try:
        _accel = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/Accelerate.framework/Accelerate"
        )
    except OSError:
        _accel = None


def available():
    _ensure_loaded()
    return _accel is not None


# ---------------------------------------------------------------------------
# vDSP — vectorised arithmetic (float32)
# ---------------------------------------------------------------------------

def vdsp_vadd(a_ptr, b_ptr, out_ptr, n):
    """out = a + b  (float32, stride-1)"""
    _ensure_loaded()
    _accel.vDSP_vadd.restype = None
    _accel.vDSP_vadd.argtypes = [
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_ulong,
    ]
    _accel.vDSP_vadd(a_ptr, 1, b_ptr, 1, out_ptr, 1, n)


def vdsp_vsub(a_ptr, b_ptr, out_ptr, n):
    """out = b - a  (vDSP convention: C = B - A)"""
    _ensure_loaded()
    _accel.vDSP_vsub.restype = None
    _accel.vDSP_vsub.argtypes = [
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_ulong,
    ]
    _accel.vDSP_vsub(a_ptr, 1, b_ptr, 1, out_ptr, 1, n)


def vdsp_vmul(a_ptr, b_ptr, out_ptr, n):
    """out = a * b  (float32, stride-1)"""
    _ensure_loaded()
    _accel.vDSP_vmul.restype = None
    _accel.vDSP_vmul.argtypes = [
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_ulong,
    ]
    _accel.vDSP_vmul(a_ptr, 1, b_ptr, 1, out_ptr, 1, n)


def vdsp_vdiv(a_ptr, b_ptr, out_ptr, n):
    """out = b / a  (vDSP convention: C = B / A)"""
    _ensure_loaded()
    _accel.vDSP_vdiv.restype = None
    _accel.vDSP_vdiv.argtypes = [
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_ulong,
    ]
    _accel.vDSP_vdiv(a_ptr, 1, b_ptr, 1, out_ptr, 1, n)


def vdsp_vneg(a_ptr, out_ptr, n):
    """out = -a  (float32, stride-1)"""
    _ensure_loaded()
    _accel.vDSP_vneg.restype = None
    _accel.vDSP_vneg.argtypes = [
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_ulong,
    ]
    _accel.vDSP_vneg(a_ptr, 1, out_ptr, 1, n)


def vdsp_vabs(a_ptr, out_ptr, n):
    """out = |a|  (float32, stride-1)"""
    _ensure_loaded()
    _accel.vDSP_vabs.restype = None
    _accel.vDSP_vabs.argtypes = [
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_ulong,
    ]
    _accel.vDSP_vabs(a_ptr, 1, out_ptr, 1, n)


def vdsp_vsq(a_ptr, out_ptr, n):
    """out = a*a  (float32, stride-1)"""
    _ensure_loaded()
    _accel.vDSP_vsq.restype = None
    _accel.vDSP_vsq.argtypes = [
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_ulong,
    ]
    _accel.vDSP_vsq(a_ptr, 1, out_ptr, 1, n)


def vdsp_vclip(a_ptr, low, high, out_ptr, n):
    """out = clip(a, low, high)  (float32, stride-1)"""
    _ensure_loaded()
    _accel.vDSP_vclip.restype = None
    _accel.vDSP_vclip.argtypes = [
        ctypes.c_void_p, ctypes.c_long,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p, ctypes.c_long,
        ctypes.c_ulong,
    ]
    lo = ctypes.c_float(low)
    hi = ctypes.c_float(high)
    _accel.vDSP_vclip(a_ptr, 1, ctypes.byref(lo), ctypes.byref(hi), out_ptr, 1, n)


# ---------------------------------------------------------------------------
# vForce — transcendental functions (float32)
# ---------------------------------------------------------------------------

def _vforce_unary(func_name, in_ptr, out_ptr, n):
    """Generic vForce unary: out[i] = f(in[i])."""
    _ensure_loaded()
    fn = getattr(_accel, func_name)
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    count = ctypes.c_int(n)
    fn(out_ptr, in_ptr, ctypes.byref(count))


def vvexpf(in_ptr, out_ptr, n):
    _vforce_unary("vvexpf", in_ptr, out_ptr, n)


def vvlogf(in_ptr, out_ptr, n):
    _vforce_unary("vvlogf", in_ptr, out_ptr, n)


def vvlog2f(in_ptr, out_ptr, n):
    _vforce_unary("vvlog2f", in_ptr, out_ptr, n)


def vvlog10f(in_ptr, out_ptr, n):
    _vforce_unary("vvlog10f", in_ptr, out_ptr, n)


def vvsqrtf(in_ptr, out_ptr, n):
    _vforce_unary("vvsqrtf", in_ptr, out_ptr, n)


def vvrsqrtf(in_ptr, out_ptr, n):
    _vforce_unary("vvrsqrtf", in_ptr, out_ptr, n)


def vvtanhf(in_ptr, out_ptr, n):
    _vforce_unary("vvtanhf", in_ptr, out_ptr, n)


def vvsinf(in_ptr, out_ptr, n):
    _vforce_unary("vvsinf", in_ptr, out_ptr, n)


def vvcosf(in_ptr, out_ptr, n):
    _vforce_unary("vvcosf", in_ptr, out_ptr, n)


def vvtanf(in_ptr, out_ptr, n):
    _vforce_unary("vvtanf", in_ptr, out_ptr, n)


def vvasinf(in_ptr, out_ptr, n):
    _vforce_unary("vvasinf", in_ptr, out_ptr, n)


def vvacosf(in_ptr, out_ptr, n):
    _vforce_unary("vvacosf", in_ptr, out_ptr, n)


def vvatanf(in_ptr, out_ptr, n):
    _vforce_unary("vvatanf", in_ptr, out_ptr, n)


def vvsinhf(in_ptr, out_ptr, n):
    _vforce_unary("vvsinhf", in_ptr, out_ptr, n)


def vvcoshf(in_ptr, out_ptr, n):
    _vforce_unary("vvcoshf", in_ptr, out_ptr, n)


def vvasinhf(in_ptr, out_ptr, n):
    _vforce_unary("vvasinhf", in_ptr, out_ptr, n)


def vvacoshf(in_ptr, out_ptr, n):
    _vforce_unary("vvacoshf", in_ptr, out_ptr, n)


def vvatanhf(in_ptr, out_ptr, n):
    _vforce_unary("vvatanhf", in_ptr, out_ptr, n)


def vvfloorf(in_ptr, out_ptr, n):
    _vforce_unary("vvfloorf", in_ptr, out_ptr, n)


def vvceilf(in_ptr, out_ptr, n):
    _vforce_unary("vvceilf", in_ptr, out_ptr, n)


def vvfabsf(in_ptr, out_ptr, n):
    _vforce_unary("vvfabsf", in_ptr, out_ptr, n)


def vvrecipf(in_ptr, out_ptr, n):
    _vforce_unary("vvrecipf", in_ptr, out_ptr, n)


def vvexp2f(in_ptr, out_ptr, n):
    _vforce_unary("vvexp2f", in_ptr, out_ptr, n)


def vvexpm1f(in_ptr, out_ptr, n):
    _vforce_unary("vvexpm1f", in_ptr, out_ptr, n)


def vvlog1pf(in_ptr, out_ptr, n):
    _vforce_unary("vvlog1pf", in_ptr, out_ptr, n)


# ---------------------------------------------------------------------------
# vForce — binary (float32)
# ---------------------------------------------------------------------------

def _vforce_binary(func_name, a_ptr, b_ptr, out_ptr, n):
    _ensure_loaded()
    fn = getattr(_accel, func_name)
    fn.restype = None
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    count = ctypes.c_int(n)
    fn(out_ptr, a_ptr, b_ptr, ctypes.byref(count))


def vvpowf(a_ptr, b_ptr, out_ptr, n):
    """out = a^b"""
    _vforce_binary("vvpowf", a_ptr, b_ptr, out_ptr, n)


def vvatan2f(a_ptr, b_ptr, out_ptr, n):
    """out = atan2(a, b)"""
    _vforce_binary("vvatan2f", a_ptr, b_ptr, out_ptr, n)


def vvcopysignf(a_ptr, b_ptr, out_ptr, n):
    """out = copysign(a, b)"""
    _vforce_binary("vvcopysignf", a_ptr, b_ptr, out_ptr, n)


# ---------------------------------------------------------------------------
# BLAS — matrix operations
# ---------------------------------------------------------------------------

# CblasRowMajor = 101, CblasNoTrans = 111, CblasTrans = 112

def cblas_sgemm(transA, transB, M, N, K, alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc):
    """Single-precision general matrix multiply: C = alpha*op(A)*op(B) + beta*C."""
    _ensure_loaded()
    _accel.cblas_sgemm.restype = None
    _accel.cblas_sgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p, ctypes.c_int,
    ]
    _accel.cblas_sgemm(
        101, transA, transB, M, N, K,
        alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc,
    )


def cblas_dgemm(transA, transB, M, N, K, alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc):
    """Double-precision general matrix multiply."""
    _ensure_loaded()
    _accel.cblas_dgemm.restype = None
    _accel.cblas_dgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_double,
        ctypes.c_void_p, ctypes.c_int,
    ]
    _accel.cblas_dgemm(
        101, transA, transB, M, N, K,
        alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc,
    )


def cblas_sgemv(transA, M, N, alpha, A_ptr, lda, x_ptr, incx, beta, y_ptr, incy):
    """Single-precision matrix-vector multiply: y = alpha*op(A)*x + beta*y."""
    _ensure_loaded()
    _accel.cblas_sgemv.restype = None
    _accel.cblas_sgemv.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p, ctypes.c_int,
    ]
    _accel.cblas_sgemv(101, transA, M, N, alpha, A_ptr, lda, x_ptr, incx, beta, y_ptr, incy)


def cblas_sdot(n, x_ptr, incx, y_ptr, incy):
    """Single-precision dot product."""
    _ensure_loaded()
    _accel.cblas_sdot.restype = ctypes.c_float
    _accel.cblas_sdot.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
    ]
    return float(_accel.cblas_sdot(n, x_ptr, incx, y_ptr, incy))


def cblas_ddot(n, x_ptr, incx, y_ptr, incy):
    """Double-precision dot product."""
    _ensure_loaded()
    _accel.cblas_ddot.restype = ctypes.c_double
    _accel.cblas_ddot.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
    ]
    return float(_accel.cblas_ddot(n, x_ptr, incx, y_ptr, incy))
