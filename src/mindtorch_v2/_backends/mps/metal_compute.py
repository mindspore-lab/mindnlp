"""Metal GPU compute dispatch engine.

Lazily compiles MSL kernels, caches compute pipeline states, and dispatches
element-wise, reduction, and in-place kernels on Metal GPU.
"""
import ctypes
import struct
import threading

from .runtime import get_runtime, buffer_contents, _HAS_PYOBJC

_MTLSize = None
if _HAS_PYOBJC:
    from Metal import MTLSizeMake as _MTLSizeMake  # pylint: disable=import-error,no-name-in-module

    def _MTLSize(w, h, d):
        return _MTLSizeMake(w, h, d)
else:
    def _MTLSize(w, h, d):  # noqa: E302
        return None

# ---------------------------------------------------------------------------
# Singleton dispatcher
# ---------------------------------------------------------------------------
_dispatcher = None
_dispatcher_lock = threading.Lock()


def get_dispatcher():
    """Return the singleton MetalKernelDispatcher (lazy init)."""
    global _dispatcher
    if _dispatcher is not None:
        return _dispatcher
    with _dispatcher_lock:
        if _dispatcher is None:
            _dispatcher = MetalKernelDispatcher()
        return _dispatcher


class MetalKernelDispatcher:
    """Compiles MSL kernels lazily, caches pipelines, dispatches compute work."""

    def __init__(self):
        self._library = None
        self._pipeline_cache = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def ensure_compiled(self):
        """Compile all MSL source on first use."""
        if self._library is not None:
            return
        with self._lock:
            if self._library is not None:
                return
            from .metal_shaders import MSL_SOURCE
            rt = get_runtime()
            self._library = rt.compile_library(MSL_SOURCE)

    def _get_pipeline(self, kernel_name):
        """Get or create a cached compute pipeline for *kernel_name*."""
        if kernel_name in self._pipeline_cache:
            return self._pipeline_cache[kernel_name]
        self.ensure_compiled()
        rt = get_runtime()
        if _HAS_PYOBJC:
            fn = self._library.newFunctionWithName_(kernel_name)
        else:
            fn = _library_get_function_ctypes(self._library, kernel_name)
        if fn is None or (isinstance(fn, int) and fn == 0):
            raise RuntimeError(f"Metal kernel '{kernel_name}' not found in library")
        pipeline = rt.make_compute_pipeline(fn)
        self._pipeline_cache[kernel_name] = pipeline
        return pipeline

    # ------------------------------------------------------------------
    # Thread dispatch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _threads_per_group(pipeline):
        if _HAS_PYOBJC:
            return min(256, int(pipeline.maxTotalThreadsPerThreadgroup()))
        return 256  # safe default for Apple Silicon

    # ------------------------------------------------------------------
    # Dispatch: unary  (a → out)
    # ------------------------------------------------------------------

    def dispatch_unary(self, kernel_name, a_metal_buf, out_metal_buf, numel):
        """Encode and execute a unary element-wise kernel."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_metal_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(out_metal_buf, 0, 1)
            n_bytes = struct.pack("I", numel)
            enc.setBytes_length_atIndex_(n_bytes, 4, 2)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_unary_ctypes(enc, pipeline, a_metal_buf, out_metal_buf,
                                 numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: binary  (a, b → out)
    # ------------------------------------------------------------------

    def dispatch_binary(self, kernel_name, a_buf, b_buf, out_buf, numel):
        """Encode and execute a binary element-wise kernel."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(b_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
            n_bytes = struct.pack("I", numel)
            enc.setBytes_length_atIndex_(n_bytes, 4, 3)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_binary_ctypes(enc, pipeline, a_buf, b_buf, out_buf,
                                  numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: binary with scalar  (a, scalar → out)
    # ------------------------------------------------------------------

    def dispatch_binary_scalar(self, kernel_name, a_buf, scalar, out_buf,
                               numel, scalar_fmt="f"):
        """Encode and execute a binary-scalar kernel (scalar embedded via setBytes)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        scalar_bytes = struct.pack(scalar_fmt, scalar)
        scalar_size = len(scalar_bytes)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBytes_length_atIndex_(scalar_bytes, scalar_size, 1)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
            n_bytes = struct.pack("I", numel)
            enc.setBytes_length_atIndex_(n_bytes, 4, 3)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_binary_scalar_ctypes(enc, pipeline, a_buf,
                                         scalar_bytes, scalar_size,
                                         out_buf, numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: reduction  (input → scalar output, two-pass)
    # ------------------------------------------------------------------

    def dispatch_reduction(self, partial_kernel, final_kernel, a_buf, out_buf,
                           numel):
        """Two-pass parallel reduction: per-threadgroup partials → final."""
        rt = get_runtime()
        pipeline_p = self._get_pipeline(partial_kernel)
        pipeline_f = self._get_pipeline(final_kernel)
        tpg = self._threads_per_group(pipeline_p)
        num_groups = (numel + tpg - 1) // tpg

        # Allocate partial-results buffer (one float per group)
        partials_buf = rt.create_buffer(num_groups * 4)

        # --- Pass 1: per-threadgroup partial ---
        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)
        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline_p)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(partials_buf, 0, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 2)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(num_groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_unary_ctypes(enc, pipeline_p, a_buf, partials_buf,
                                 numel, num_groups, tpg)
        rt.commit_and_wait(cmd)

        # --- Pass 2: reduce partials → single output ---
        final_tpg = self._threads_per_group(pipeline_f)
        # Final pass uses a single threadgroup
        final_tpg = max(final_tpg, num_groups)
        # Round up to next power-of-2 for proper tree reduction
        final_tpg = 1
        while final_tpg < num_groups:
            final_tpg *= 2
        final_tpg = min(final_tpg, 256)

        cmd2 = rt.create_command_buffer()
        enc2 = rt.get_compute_encoder(cmd2)
        if _HAS_PYOBJC:
            enc2.setComputePipelineState_(pipeline_f)
            enc2.setBuffer_offset_atIndex_(partials_buf, 0, 0)
            enc2.setBuffer_offset_atIndex_(out_buf, 0, 1)
            enc2.setBytes_length_atIndex_(struct.pack("I", num_groups), 4, 2)
            enc2.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(1, 1, 1), _MTLSize(final_tpg, 1, 1))
            enc2.endEncoding()
        else:
            _encode_unary_ctypes(enc2, pipeline_f, partials_buf, out_buf,
                                 num_groups, 1, final_tpg)
        rt.commit_and_wait(cmd2)

    # ------------------------------------------------------------------
    # Dispatch: argmax/argmin reduction  (two-pass with value+index)
    # ------------------------------------------------------------------

    def dispatch_arg_reduction(self, partial_kernel, final_kernel,
                               a_buf, out_buf, numel):
        """Two-pass argmax/argmin: partials carry (value, index) pairs."""
        rt = get_runtime()
        pipeline_p = self._get_pipeline(partial_kernel)
        pipeline_f = self._get_pipeline(final_kernel)
        tpg = self._threads_per_group(pipeline_p)
        num_groups = (numel + tpg - 1) // tpg

        partial_vals_buf = rt.create_buffer(num_groups * 4)   # float per group
        partial_idxs_buf = rt.create_buffer(num_groups * 4)   # uint per group

        # --- Pass 1 ---
        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)
        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline_p)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(partial_vals_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(partial_idxs_buf, 0, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 3)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(num_groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_arg_partial_ctypes(enc, pipeline_p, a_buf,
                                       partial_vals_buf, partial_idxs_buf,
                                       numel, num_groups, tpg)
        rt.commit_and_wait(cmd)

        # --- Pass 2 ---
        final_tpg = 1
        while final_tpg < num_groups:
            final_tpg *= 2
        final_tpg = min(final_tpg, 256)

        cmd2 = rt.create_command_buffer()
        enc2 = rt.get_compute_encoder(cmd2)
        if _HAS_PYOBJC:
            enc2.setComputePipelineState_(pipeline_f)
            enc2.setBuffer_offset_atIndex_(partial_vals_buf, 0, 0)
            enc2.setBuffer_offset_atIndex_(partial_idxs_buf, 0, 1)
            enc2.setBuffer_offset_atIndex_(out_buf, 0, 2)
            enc2.setBytes_length_atIndex_(struct.pack("I", num_groups), 4, 3)
            enc2.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(1, 1, 1), _MTLSize(final_tpg, 1, 1))
            enc2.endEncoding()
        else:
            _encode_arg_final_ctypes(enc2, pipeline_f, partial_vals_buf,
                                     partial_idxs_buf, out_buf,
                                     num_groups, final_tpg)
        rt.commit_and_wait(cmd2)

    # ------------------------------------------------------------------
    # Dispatch: in-place unary  (a → a)
    # ------------------------------------------------------------------

    def dispatch_inplace_unary(self, kernel_name, a_buf, numel):
        """In-place unary: writes output back to input buffer."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 1)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_inplace_unary_ctypes(enc, pipeline, a_buf, numel,
                                         groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: in-place binary  (a, b → a)
    # ------------------------------------------------------------------

    def dispatch_inplace_binary(self, kernel_name, a_buf, b_buf, numel):
        """In-place binary: a[i] op= b[i]."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(b_buf, 0, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 2)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_unary_ctypes(enc, pipeline, a_buf, b_buf, numel,
                                 groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: in-place binary scalar  (a, scalar → a)
    # ------------------------------------------------------------------

    def dispatch_inplace_binary_scalar(self, kernel_name, a_buf, scalar,
                                       numel, scalar_fmt="f"):
        """In-place binary-scalar: a[i] op= scalar."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        scalar_bytes = struct.pack(scalar_fmt, scalar)
        scalar_size = len(scalar_bytes)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBytes_length_atIndex_(scalar_bytes, scalar_size, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 2)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_inplace_scalar_ctypes(enc, pipeline, a_buf,
                                          scalar_bytes, scalar_size,
                                          numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: fill  (scalar → buffer)
    # ------------------------------------------------------------------

    def dispatch_fill(self, kernel_name, out_buf, scalar, numel,
                      scalar_fmt="f"):
        """Fill buffer with a scalar value."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        scalar_bytes = struct.pack(scalar_fmt, scalar)
        scalar_size = len(scalar_bytes)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 0)
            enc.setBytes_length_atIndex_(scalar_bytes, scalar_size, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 2)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_inplace_scalar_ctypes(enc, pipeline, out_buf,
                                          scalar_bytes, scalar_size,
                                          numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: copy  (src → dst)
    # ------------------------------------------------------------------

    def dispatch_copy(self, kernel_name, src_buf, dst_buf, numel):
        """Copy src buffer to dst buffer."""
        self.dispatch_binary(kernel_name, src_buf, dst_buf, numel)

    # ------------------------------------------------------------------
    # Dispatch: softmax 2D  (input → output, with rows/cols)
    # ------------------------------------------------------------------

    def dispatch_softmax_2d(self, kernel_name, a_buf, out_buf, rows, cols):
        """Dispatch softmax over last dim of a 2D tensor."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", rows), 4, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", cols), 4, 3)
            tpg_x = min(32, cols)
            tpg_y = min(8, rows)
            groups_x = (cols + tpg_x - 1) // tpg_x
            groups_y = (rows + tpg_y - 1) // tpg_y
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups_x, groups_y, 1), _MTLSize(tpg_x, tpg_y, 1))
            enc.endEncoding()
        else:
            _encode_softmax_ctypes(enc, pipeline, a_buf, out_buf,
                                   rows, cols)

        rt.commit_and_wait(cmd)


# ---------------------------------------------------------------------------
# ctypes encoding helpers (fallback when no PyObjC)
# ---------------------------------------------------------------------------

def _library_get_function_ctypes(library, name):
    """Get a MTLFunction from a MTLLibrary by name (ctypes path)."""
    from .runtime import _libobjc, _load_objc_libs
    _load_objc_libs()
    # Create NSString for function name
    ns_string_class = _libobjc.objc_getClass(b"NSString")
    sel_alloc = _libobjc.sel_registerName(b"alloc")
    sel_init = _libobjc.sel_registerName(b"initWithUTF8String:")
    ns_str = _libobjc.objc_msgSend(ns_string_class, sel_alloc)
    name_bytes = name.encode("utf-8")
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
    _libobjc.objc_msgSend.restype = ctypes.c_void_p
    ns_str = _libobjc.objc_msgSend(ns_str, sel_init, name_bytes)
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    sel = _libobjc.sel_registerName(b"newFunctionWithName:")
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _libobjc.objc_msgSend.restype = ctypes.c_void_p
    fn = _libobjc.objc_msgSend(library, sel, ns_str)
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return fn


def _ctypes_set_buffer(enc, buf, offset, index):
    """setBuffer:offset:atIndex: via ctypes."""
    from .runtime import _libobjc, _load_objc_libs
    _load_objc_libs()
    sel = _libobjc.sel_registerName(b"setBuffer:offset:atIndex:")
    _libobjc.objc_msgSend.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
    ]
    _libobjc.objc_msgSend.restype = None
    _libobjc.objc_msgSend(enc, sel, buf, ctypes.c_uint64(offset),
                           ctypes.c_uint64(index))
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _libobjc.objc_msgSend.restype = ctypes.c_void_p


def _ctypes_set_bytes(enc, data_bytes, length, index):
    """setBytes:length:atIndex: via ctypes."""
    from .runtime import _libobjc, _load_objc_libs
    _load_objc_libs()
    sel = _libobjc.sel_registerName(b"setBytes:length:atIndex:")
    buf = ctypes.create_string_buffer(data_bytes)
    _libobjc.objc_msgSend.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
    ]
    _libobjc.objc_msgSend.restype = None
    _libobjc.objc_msgSend(enc, sel, buf, ctypes.c_uint64(length),
                           ctypes.c_uint64(index))
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _libobjc.objc_msgSend.restype = ctypes.c_void_p


def _ctypes_set_pipeline(enc, pipeline):
    """setComputePipelineState: via ctypes."""
    from .runtime import _libobjc, _load_objc_libs
    _load_objc_libs()
    sel = _libobjc.sel_registerName(b"setComputePipelineState:")
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _libobjc.objc_msgSend.restype = None
    _libobjc.objc_msgSend(enc, sel, pipeline)
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _libobjc.objc_msgSend.restype = ctypes.c_void_p


def _ctypes_dispatch_threadgroups(enc, groups, tpg):
    """dispatchThreadgroups:threadsPerThreadgroup: via ctypes.

    MTLSize is a struct {uint64, uint64, uint64}.
    """
    from .runtime import _libobjc, _load_objc_libs
    _load_objc_libs()

    class MTLSize(ctypes.Structure):
        _fields_ = [("width", ctypes.c_uint64),
                     ("height", ctypes.c_uint64),
                     ("depth", ctypes.c_uint64)]

    sel = _libobjc.sel_registerName(b"dispatchThreadgroups:threadsPerThreadgroup:")
    grid = MTLSize(groups, 1, 1)
    tpg_s = MTLSize(tpg, 1, 1)
    _libobjc.objc_msgSend.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, MTLSize, MTLSize,
    ]
    _libobjc.objc_msgSend.restype = None
    _libobjc.objc_msgSend(enc, sel, grid, tpg_s)
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _libobjc.objc_msgSend.restype = ctypes.c_void_p


def _ctypes_end_encoding(enc):
    """endEncoding via ctypes."""
    from .runtime import _libobjc, _load_objc_libs
    _load_objc_libs()
    sel = _libobjc.sel_registerName(b"endEncoding")
    _libobjc.objc_msgSend.restype = None
    _libobjc.objc_msgSend(enc, sel)
    _libobjc.objc_msgSend.restype = ctypes.c_void_p


def _encode_unary_ctypes(enc, pipeline, a_buf, out_buf, numel, groups, tpg):
    """Encode a unary kernel dispatch via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_buffer(enc, out_buf, 0, 1)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 2)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_binary_ctypes(enc, pipeline, a_buf, b_buf, out_buf,
                           numel, groups, tpg):
    """Encode a binary kernel dispatch via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_buffer(enc, b_buf, 0, 1)
    _ctypes_set_buffer(enc, out_buf, 0, 2)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 3)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_binary_scalar_ctypes(enc, pipeline, a_buf, scalar_bytes,
                                  scalar_size, out_buf, numel, groups, tpg):
    """Encode a binary-scalar kernel dispatch via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_bytes(enc, scalar_bytes, scalar_size, 1)
    _ctypes_set_buffer(enc, out_buf, 0, 2)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 3)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_arg_partial_ctypes(enc, pipeline, a_buf, vals_buf, idxs_buf,
                                numel, groups, tpg):
    """Encode argmax/argmin partial pass via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_buffer(enc, vals_buf, 0, 1)
    _ctypes_set_buffer(enc, idxs_buf, 0, 2)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 3)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_arg_final_ctypes(enc, pipeline, vals_buf, idxs_buf, out_buf,
                              num_groups, tpg):
    """Encode argmax/argmin final pass via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, vals_buf, 0, 0)
    _ctypes_set_buffer(enc, idxs_buf, 0, 1)
    _ctypes_set_buffer(enc, out_buf, 0, 2)
    _ctypes_set_bytes(enc, struct.pack("I", num_groups), 4, 3)
    _ctypes_dispatch_threadgroups(enc, 1, tpg)
    _ctypes_end_encoding(enc)


def _encode_inplace_unary_ctypes(enc, pipeline, a_buf, numel, groups, tpg):
    """Encode an in-place unary kernel via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 1)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_inplace_scalar_ctypes(enc, pipeline, a_buf, scalar_bytes,
                                   scalar_size, numel, groups, tpg):
    """Encode an in-place binary-scalar kernel via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_bytes(enc, scalar_bytes, scalar_size, 1)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 2)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_softmax_ctypes(enc, pipeline, a_buf, out_buf, rows, cols):
    """Encode softmax 2D kernel via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_buffer(enc, out_buf, 0, 1)
    _ctypes_set_bytes(enc, struct.pack("I", rows), 4, 2)
    _ctypes_set_bytes(enc, struct.pack("I", cols), 4, 3)
    tpg_x = min(32, cols)
    tpg_y = min(8, rows)
    groups_x = (cols + tpg_x - 1) // tpg_x
    groups_y = (rows + tpg_y - 1) // tpg_y

    from .runtime import _libobjc, _load_objc_libs
    _load_objc_libs()

    class MTLSize(ctypes.Structure):
        _fields_ = [("width", ctypes.c_uint64),
                     ("height", ctypes.c_uint64),
                     ("depth", ctypes.c_uint64)]

    sel = _libobjc.sel_registerName(b"dispatchThreadgroups:threadsPerThreadgroup:")
    grid = MTLSize(groups_x, groups_y, 1)
    tpg_s = MTLSize(tpg_x, tpg_y, 1)
    _libobjc.objc_msgSend.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, MTLSize, MTLSize,
    ]
    _libobjc.objc_msgSend.restype = None
    _libobjc.objc_msgSend(enc, sel, grid, tpg_s)
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _libobjc.objc_msgSend.restype = ctypes.c_void_p
    _ctypes_end_encoding(enc)
