"""Stub for torch.backends - Backend configurations.

Tier 2 stub: provides configuration flags but no actual backend functionality.
"""


class cuda:
    """CUDA backend configuration."""

    @staticmethod
    def is_built():
        """Check if CUDA backend is built."""
        return False

    # Matrix multiplication settings
    matmul = type('matmul', (), {
        'allow_tf32': False,
        'allow_fp16_reduced_precision_reduction': True,
        'allow_bf16_reduced_precision_reduction': True,
    })()

    # Flash attention settings
    flash_sdp_enabled = lambda: False
    enable_flash_sdp = lambda enable: None
    math_sdp_enabled = lambda: True
    enable_math_sdp = lambda enable: None
    mem_efficient_sdp_enabled = lambda: False
    enable_mem_efficient_sdp = lambda enable: None

    # cuBLAS settings
    cublaslt_enabled = False

    # Prefer channels last
    preferred_linalg_library = lambda lib=None: None

    @staticmethod
    def sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        """Context manager to control which SDPA kernels are enabled.

        This is a no-op stub since we don't have CUDA support.
        Returns a context manager that does nothing.
        """
        class _SDPKernelContext:
            def __init__(self, enable_flash, enable_math, enable_mem_efficient):
                self.enable_flash = enable_flash
                self.enable_math = enable_math
                self.enable_mem_efficient = enable_mem_efficient

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        return _SDPKernelContext(enable_flash, enable_math, enable_mem_efficient)


class cudnn:
    """cuDNN backend configuration."""

    enabled = False
    benchmark = False
    deterministic = True
    allow_tf32 = False
    version = lambda: None

    @staticmethod
    def is_available():
        """Check if cuDNN is available."""
        return False


class mkl:
    """MKL backend configuration."""

    @staticmethod
    def is_available():
        """Check if MKL is available."""
        return False

    verbose = False


class mkldnn:
    """MKL-DNN (oneDNN) backend configuration."""

    @staticmethod
    def is_available():
        """Check if MKL-DNN is available."""
        return False

    enabled = False


class openmp:
    """OpenMP backend configuration."""

    @staticmethod
    def is_available():
        """Check if OpenMP is available."""
        return False


class opt_einsum:
    """opt_einsum backend configuration."""

    enabled = True
    strategy = 'auto'

    @staticmethod
    def is_available():
        """Check if opt_einsum is available."""
        try:
            import opt_einsum
            return True
        except ImportError:
            return False

    @staticmethod
    def get_opt_einsum():
        """Get opt_einsum module."""
        try:
            import opt_einsum
            return opt_einsum
        except ImportError:
            return None


class mps:
    """MPS (Apple Silicon) backend configuration."""

    @staticmethod
    def is_available():
        """Check if MPS is available."""
        return False

    @staticmethod
    def is_built():
        """Check if MPS backend is built."""
        return False


class quantized:
    """Quantized backend configuration."""

    engine = 'none'
    supported_engines = ['none']

    @staticmethod
    def set_engine(engine):
        """Set quantization engine."""
        pass


class xpu:
    """XPU (Intel GPU) backend configuration."""

    @staticmethod
    def is_available():
        """Check if XPU is available."""
        return False

    @staticmethod
    def is_built():
        """Check if XPU backend is built."""
        return False


# Flags class for managing backend flags
class __flags:
    """Backend flags."""

    def __init__(self):
        self._allow_tf32 = False

    @property
    def allow_tf32(self):
        return self._allow_tf32

    @allow_tf32.setter
    def allow_tf32(self, value):
        self._allow_tf32 = value


flags = __flags()


# CPU backend
class cpu:
    """CPU backend configuration."""

    @staticmethod
    def get_cpu_capability():
        """Get CPU capability string."""
        return "default"
