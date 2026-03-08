"""Metal runtime for MPS backend.

Provides Metal device creation, command queue management, buffer allocation,
and synchronization using pyobjc-framework-Metal (with ctypes fallback).
"""
import ctypes
import threading

_HAS_PYOBJC = False
_Metal = None

try:
    import Metal as _Metal
    _HAS_PYOBJC = True
except ImportError:
    pass


class MetalRuntime:
    """Manages a Metal device and command queue."""

    def __init__(self):
        self._device = None
        self._command_queue = None
        self._lock = threading.Lock()

    def _ensure_init(self):
        if self._device is not None:
            return
        with self._lock:
            if self._device is not None:
                return
            if _HAS_PYOBJC:
                self._device = _Metal.MTLCreateSystemDefaultDevice()
            else:
                self._device = _create_device_ctypes()
            if self._device is None:
                raise RuntimeError("Metal is not available on this system")
            if _HAS_PYOBJC:
                self._command_queue = self._device.newCommandQueue()
            else:
                self._command_queue = _create_command_queue_ctypes(self._device)

    @property
    def device(self):
        self._ensure_init()
        return self._device

    @property
    def command_queue(self):
        self._ensure_init()
        return self._command_queue

    def create_buffer(self, nbytes):
        """Create a shared-memory Metal buffer."""
        self._ensure_init()
        nbytes = max(nbytes, 1)
        if _HAS_PYOBJC:
            # MTLResourceStorageModeShared = 0
            buf = self._device.newBufferWithLength_options_(nbytes, 0)
        else:
            buf = _create_buffer_ctypes(self._device, nbytes)
        if buf is None:
            raise RuntimeError(f"Failed to allocate Metal buffer of {nbytes} bytes")
        return buf

    def create_buffer_from_bytes(self, data, nbytes):
        """Create a shared-memory Metal buffer initialized with data."""
        self._ensure_init()
        nbytes = max(nbytes, 1)
        if _HAS_PYOBJC:
            buf = self._device.newBufferWithBytes_length_options_(data, nbytes, 0)
        else:
            buf = _create_buffer_with_bytes_ctypes(self._device, data, nbytes)
        if buf is None:
            raise RuntimeError(f"Failed to allocate Metal buffer of {nbytes} bytes")
        return buf

    def create_command_buffer(self):
        """Create a new command buffer."""
        self._ensure_init()
        if _HAS_PYOBJC:
            return self._command_queue.commandBuffer()
        return _create_command_buffer_ctypes(self._command_queue)

    def commit_and_wait(self, cmd_buffer):
        """Commit a command buffer and wait for completion."""
        if _HAS_PYOBJC:
            cmd_buffer.commit()
            cmd_buffer.waitUntilCompleted()
        else:
            _commit_and_wait_ctypes(cmd_buffer)

    def synchronize(self):
        """Submit an empty command buffer and wait, ensuring all prior work completes."""
        self._ensure_init()
        cb = self.create_command_buffer()
        self.commit_and_wait(cb)

    def device_name(self):
        """Return the Metal device name."""
        self._ensure_init()
        if _HAS_PYOBJC:
            return str(self._device.name())
        return "Apple Metal GPU"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
_runtime = None
_runtime_lock = threading.Lock()


def get_runtime():
    """Get or create the singleton MetalRuntime."""
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = MetalRuntime()
        return _runtime


def is_available():
    """Check if Metal/MPS is available on this system."""
    try:
        rt = get_runtime()
        rt._ensure_init()
        return rt._device is not None
    except Exception:
        return False


def device_count():
    """Number of MPS devices (always 0 or 1)."""
    return 1 if is_available() else 0


# ---------------------------------------------------------------------------
# ctypes fallback for systems without pyobjc
# ---------------------------------------------------------------------------
_libobjc = None
_metal_framework = None


def _load_objc_libs():
    global _libobjc, _metal_framework
    if _libobjc is not None:
        return
    _libobjc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
    _libobjc.objc_getClass.restype = ctypes.c_void_p
    _libobjc.objc_getClass.argtypes = [ctypes.c_char_p]
    _libobjc.sel_registerName.restype = ctypes.c_void_p
    _libobjc.sel_registerName.argtypes = [ctypes.c_char_p]
    _libobjc.objc_msgSend.restype = ctypes.c_void_p
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _metal_framework = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/Metal.framework/Metal"
    )


def _objc_msg(obj, sel_name, *args):
    """Send an ObjC message via libobjc."""
    _load_objc_libs()
    sel = _libobjc.sel_registerName(sel_name.encode())
    return _libobjc.objc_msgSend(obj, sel, *args)


def _create_device_ctypes():
    _load_objc_libs()
    func = _metal_framework.MTLCreateSystemDefaultDevice
    func.restype = ctypes.c_void_p
    return func()


def _create_command_queue_ctypes(device):
    return _objc_msg(device, "newCommandQueue")


def _create_buffer_ctypes(device, nbytes):
    _load_objc_libs()
    _libobjc.objc_msgSend.restype = ctypes.c_void_p
    _libobjc.objc_msgSend.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
    ]
    sel = _libobjc.sel_registerName(b"newBufferWithLength:options:")
    buf = _libobjc.objc_msgSend(device, sel, ctypes.c_uint64(nbytes), ctypes.c_uint64(0))
    # Reset argtypes
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return buf


def _create_buffer_with_bytes_ctypes(device, data, nbytes):
    _load_objc_libs()
    _libobjc.objc_msgSend.restype = ctypes.c_void_p
    _libobjc.objc_msgSend.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
    ]
    sel = _libobjc.sel_registerName(b"newBufferWithBytes:length:options:")
    if isinstance(data, int):
        ptr = ctypes.c_void_p(data)
    else:
        ptr = ctypes.cast(ctypes.pointer(ctypes.c_char.from_buffer_copy(bytes(data)[:1])), ctypes.c_void_p)
        ptr = ctypes.c_void_p(data) if isinstance(data, int) else data
    buf = _libobjc.objc_msgSend(device, sel, ctypes.c_void_p(data), ctypes.c_uint64(nbytes), ctypes.c_uint64(0))
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return buf


def _get_buffer_contents_ctypes(metal_buffer):
    _load_objc_libs()
    return _objc_msg(metal_buffer, "contents")


def _create_command_buffer_ctypes(command_queue):
    return _objc_msg(command_queue, "commandBuffer")


def _commit_and_wait_ctypes(cmd_buffer):
    _objc_msg(cmd_buffer, "commit")
    _objc_msg(cmd_buffer, "waitUntilCompleted")


def buffer_contents(metal_buffer):
    """Get the CPU-accessible pointer from a shared Metal buffer."""
    if _HAS_PYOBJC:
        return int(metal_buffer.contents())
    return int(_get_buffer_contents_ctypes(metal_buffer))
