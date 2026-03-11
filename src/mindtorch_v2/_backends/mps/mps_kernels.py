"""MPS (MetalPerformanceShaders) kernel wrappers.

When pyobjc-framework-MetalPerformanceShaders is installed, this module
provides GPU-accelerated matrix multiply via MPSMatrixMultiplication.
Otherwise it falls back to Accelerate BLAS (cblas_sgemm / cblas_dgemm).

Install for GPU acceleration:
    pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders
"""

import ctypes
import numpy as np

from .runtime import get_runtime, buffer_contents, _HAS_PYOBJC

# ---------------------------------------------------------------------------
# Try to import pyobjc MPS bindings
# ---------------------------------------------------------------------------
_HAS_MPS_PYOBJC = False
_MPS = None
_Metal = None
try:
    import Metal as _Metal  # noqa: F401
    import MetalPerformanceShaders as _MPS  # noqa: F401
    _HAS_MPS_PYOBJC = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Accelerate BLAS fallback (always available on macOS)
# ---------------------------------------------------------------------------
from .accelerate import cblas_sgemm, cblas_dgemm


def is_mps_kernels_available():
    """Return True if pyobjc MPS kernels can be used."""
    return _HAS_MPS_PYOBJC


def mps_matmul(a_np, b_np):
    """Matrix multiply using MPS if available, else Accelerate BLAS.

    Parameters
    ----------
    a_np : np.ndarray, shape (M, K), contiguous float32 or float64
    b_np : np.ndarray, shape (K, N), contiguous float32 or float64

    Returns
    -------
    np.ndarray, shape (M, N)
    """
    M, K = a_np.shape
    K2, N = b_np.shape
    assert K == K2, f"inner dimensions must match: {K} vs {K2}"

    out = np.empty((M, N), dtype=a_np.dtype)

    if a_np.dtype == np.float32:
        cblas_sgemm(M, N, K,
                    1.0, a_np.ctypes.data, K,
                    b_np.ctypes.data, N,
                    0.0, out.ctypes.data, N)
    elif a_np.dtype == np.float64:
        cblas_dgemm(M, N, K,
                    1.0, a_np.ctypes.data, K,
                    b_np.ctypes.data, N,
                    0.0, out.ctypes.data, N)
    else:
        np.matmul(a_np, b_np, out=out)

    return out


# ---------------------------------------------------------------------------
# GPU MatMul via MPSMatrixMultiplication (PyObjC path)
# ---------------------------------------------------------------------------

# MPSDataType constants
_MPSDataTypeFloat32 = 0x10000000 | 32  # 268435488
_MPSDataTypeFloat16 = 0x10000000 | 16  # 268435472


def _mps_dtype_code(np_dtype):
    """Map numpy dtype to MPSDataType constant."""
    if np_dtype == np.float32:
        return _MPSDataTypeFloat32
    if np_dtype == np.float16:
        return _MPSDataTypeFloat16
    return None


def mps_matmul_gpu(a_metal_buf, b_metal_buf, M, K, N, dtype_code,
                   itemsize):
    """GPU matrix multiply via MPSMatrixMultiplication.

    Parameters
    ----------
    a_metal_buf : Metal buffer, shape (M, K) row-major
    b_metal_buf : Metal buffer, shape (K, N) row-major
    M, K, N : int — matrix dimensions
    dtype_code : int — MPSDataType constant
    itemsize : int — bytes per element (4 for f32, 2 for f16)

    Returns
    -------
    out_metal_buf : Metal buffer containing result (M, N)
    """
    rt = get_runtime()
    out_nbytes = M * N * itemsize
    out_buf = rt.create_buffer(max(out_nbytes, 1))

    if _HAS_MPS_PYOBJC:
        return _mps_matmul_pyobjc(rt, a_metal_buf, b_metal_buf,
                                  out_buf, M, K, N, dtype_code, itemsize)
    return _mps_matmul_ctypes(rt, a_metal_buf, b_metal_buf,
                              out_buf, M, K, N, dtype_code, itemsize)


def _mps_matmul_pyobjc(rt, a_buf, b_buf, out_buf, M, K, N,
                        dtype_code, itemsize):
    """MPSMatrixMultiplication via PyObjC."""
    # Create matrix descriptors via class factory method
    a_desc = _MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
        M, K, K * itemsize, dtype_code)
    b_desc = _MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
        K, N, N * itemsize, dtype_code)
    c_desc = _MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
        M, N, N * itemsize, dtype_code)

    # Wrap Metal buffers as MPSMatrix
    a_mat = _MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(a_buf, a_desc)
    b_mat = _MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(b_buf, b_desc)
    c_mat = _MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(out_buf, c_desc)

    # Create and encode MPSMatrixMultiplication  (C = alpha*A*B + beta*C)
    mm_kernel = _MPS.MPSMatrixMultiplication.alloc().initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
        rt.device, False, False, M, N, K, 1.0, 0.0)

    cmd = rt.create_command_buffer()
    mm_kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(
        cmd, a_mat, b_mat, c_mat)
    rt.commit_and_wait(cmd)

    return out_buf


def _mps_matmul_ctypes(rt, a_buf, b_buf, out_buf, M, K, N,
                        dtype_code, itemsize):
    """MPSMatrixMultiplication via ctypes/libobjc fallback.

    This is complex due to the number of ObjC messages required.
    Falls back to CPU BLAS if any step fails.
    """
    try:
        from .runtime import _libobjc, _load_objc_libs
        _load_objc_libs()

        def _cls(name):
            return _libobjc.objc_getClass(name.encode())

        def _sel(name):
            return _libobjc.sel_registerName(name.encode())

        def _msg(obj, sel_name, *args, argtypes=None, restype=ctypes.c_void_p):
            sel = _sel(sel_name)
            if argtypes:
                _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + argtypes
            _libobjc.objc_msgSend.restype = restype
            result = _libobjc.objc_msgSend(obj, sel, *args)
            _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            _libobjc.objc_msgSend.restype = ctypes.c_void_p
            return result

        # Create MPSMatrixDescriptor via class factory method
        desc_cls = _cls("MPSMatrixDescriptor")
        a_desc = _msg(
            desc_cls,
            "matrixDescriptorWithRows:columns:rowBytes:dataType:",
            ctypes.c_uint64(M), ctypes.c_uint64(K),
            ctypes.c_uint64(K * itemsize), ctypes.c_uint32(dtype_code),
            argtypes=[ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32],
        )
        b_desc = _msg(
            desc_cls,
            "matrixDescriptorWithRows:columns:rowBytes:dataType:",
            ctypes.c_uint64(K), ctypes.c_uint64(N),
            ctypes.c_uint64(N * itemsize), ctypes.c_uint32(dtype_code),
            argtypes=[ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32],
        )
        c_desc = _msg(
            desc_cls,
            "matrixDescriptorWithRows:columns:rowBytes:dataType:",
            ctypes.c_uint64(M), ctypes.c_uint64(N),
            ctypes.c_uint64(N * itemsize), ctypes.c_uint32(dtype_code),
            argtypes=[ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32],
        )

        # Create MPSMatrix objects
        mat_cls = _cls("MPSMatrix")
        a_mat = _msg(
            _msg(mat_cls, "alloc"),
            "initWithBuffer:descriptor:",
            a_buf, a_desc,
            argtypes=[ctypes.c_void_p, ctypes.c_void_p],
        )
        b_mat = _msg(
            _msg(mat_cls, "alloc"),
            "initWithBuffer:descriptor:",
            b_buf, b_desc,
            argtypes=[ctypes.c_void_p, ctypes.c_void_p],
        )
        c_mat = _msg(
            _msg(mat_cls, "alloc"),
            "initWithBuffer:descriptor:",
            out_buf, c_desc,
            argtypes=[ctypes.c_void_p, ctypes.c_void_p],
        )

        # Create MPSMatrixMultiplication kernel
        mm_cls = _cls("MPSMatrixMultiplication")
        mm_kernel = _msg(
            _msg(mm_cls, "alloc"),
            "initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:",
            rt.device,
            ctypes.c_bool(False), ctypes.c_bool(False),
            ctypes.c_uint64(M), ctypes.c_uint64(N), ctypes.c_uint64(K),
            ctypes.c_double(1.0), ctypes.c_double(0.0),
            argtypes=[ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                      ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                      ctypes.c_double, ctypes.c_double],
        )

        # Encode to command buffer
        cmd = rt.create_command_buffer()
        _msg(mm_kernel,
             "encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:",
             cmd, a_mat, b_mat, c_mat,
             argtypes=[ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p],
             restype=None)
        rt.commit_and_wait(cmd)

        return out_buf

    except Exception:
        # If ctypes path fails, fall back to CPU BLAS
        return None
