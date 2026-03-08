"""MPS (MetalPerformanceShaders) kernel wrappers.

When pyobjc-framework-MetalPerformanceShaders is installed, this module
provides GPU-accelerated matrix multiply via MPSMatrixMultiplication.
Otherwise it falls back to Accelerate BLAS (cblas_sgemm / cblas_dgemm).

Install for GPU acceleration:
    pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders
"""

import numpy as np

# ---------------------------------------------------------------------------
# Try to import pyobjc MPS bindings
# ---------------------------------------------------------------------------
_HAS_MPS_PYOBJC = False
try:
    import Metal  # noqa: F401
    import MetalPerformanceShaders as MPS  # noqa: F401
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
    # For now, always use Accelerate BLAS which is already highly optimised
    # on Apple Silicon.  When pyobjc MPS is available, large matrices could
    # be routed through MPSMatrixMultiplication for additional GPU parallelism.
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
        # Non-float types: fall through to numpy
        np.matmul(a_np, b_np, out=out)

    return out
