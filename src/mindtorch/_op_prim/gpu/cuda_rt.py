"""
CUDA Runtime Python bindings for direct GPU memory operations.
Similar to Ascend ACL's acl.rt API.
"""
import ctypes
import os
import sys

# CUDA error codes
CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 11
CUDA_ERROR_INVALID_DEVICE = 8

# cudaMemcpyKind
cudaMemcpyHostToHost = 0
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3
cudaMemcpyDefault = 4

_cuda_lib = None
_cuda_available = False


def _load_cuda_library():
    """Load CUDA runtime library."""
    global _cuda_lib, _cuda_available
    
    if _cuda_lib is not None:
        return _cuda_available
    
    # Try to find CUDA library
    cuda_lib_paths = [
        'libcudart.so',
        'libcudart.so.12.0',
        'libcudart.so.11.0',
        'libcudart.so.11.8',
        'libcudart.so.11.7',
        '/usr/local/cuda/lib64/libcudart.so',
        '/usr/local/cuda/lib64/libcudart.so.12.0',
        '/usr/local/cuda/lib64/libcudart.so.11.0',
        '/usr/local/cuda/lib64/libcudart.so.11.8',
        '/usr/local/cuda/lib64/libcudart.so.11.7',
    ]
    
    # Also check LD_LIBRARY_PATH
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ld_library_path:
        for path in ld_library_path.split(':'):
            if path:
                cuda_lib_paths.extend([
                    os.path.join(path, 'libcudart.so'),
                    os.path.join(path, 'libcudart.so.12.0'),
                    os.path.join(path, 'libcudart.so.11.0'),
                ])
    
    for lib_path in cuda_lib_paths:
        try:
            _cuda_lib = ctypes.CDLL(lib_path)
            break
        except (OSError, AttributeError):
            continue
    
    if _cuda_lib is None:
        _cuda_available = False
        return False
    
    # Define function signatures
    try:
        # cudaMemcpyAsync
        _cuda_lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,      # dst
            ctypes.c_void_p,      # src
            ctypes.c_size_t,      # count
            ctypes.c_int,         # kind
            ctypes.c_void_p       # stream
        ]
        _cuda_lib.cudaMemcpyAsync.restype = ctypes.c_int
        
        # cudaMemcpy (synchronous)
        _cuda_lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,      # dst
            ctypes.c_void_p,      # src
            ctypes.c_size_t,      # count
            ctypes.c_int          # kind
        ]
        _cuda_lib.cudaMemcpy.restype = ctypes.c_int
        
        # cudaGetLastError
        _cuda_lib.cudaGetLastError.argtypes = []
        _cuda_lib.cudaGetLastError.restype = ctypes.c_int
        
        # cudaDeviceSynchronize
        _cuda_lib.cudaDeviceSynchronize.argtypes = []
        _cuda_lib.cudaDeviceSynchronize.restype = ctypes.c_int
        
        # cudaStreamSynchronize
        _cuda_lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        _cuda_lib.cudaStreamSynchronize.restype = ctypes.c_int
        
        _cuda_available = True
        return True
    except AttributeError:
        _cuda_available = False
        return False


class CudaRTError(Exception):
    """CUDA runtime error."""
    pass


class rt:
    """CUDA Runtime API, similar to acl.rt"""
    
    @staticmethod
    def memcpy(dst, src, count, kind=cudaMemcpyDeviceToDevice, stream=None):
        """
        Copy memory on GPU (similar to acl.rt.memcpy).
        
        Args:
            dst: Destination device pointer (int or ctypes pointer)
            src: Source device pointer (int or ctypes pointer)
            count: Number of bytes to copy
            kind: Memory copy kind (default: cudaMemcpyDeviceToDevice)
            stream: CUDA stream (None for default stream, 0 for NULL stream)
            
        Returns:
            int: CUDA error code (0 for success)
            
        Raises:
            CudaRTError: If CUDA library is not available or operation fails
        """
        if not _load_cuda_library():
            raise CudaRTError("CUDA runtime library not available")
        
        # Validate inputs
        if count <= 0:
            raise CudaRTError(f"Invalid count: {count}, must be > 0")
        
        # Convert pointers to void*
        if isinstance(dst, int):
            if dst == 0:
                raise CudaRTError("Invalid destination pointer: NULL")
            dst_ptr = ctypes.c_void_p(dst)
        elif hasattr(dst, 'value'):
            # ctypes pointer with value attribute
            dst_ptr = ctypes.c_void_p(dst.value)
        elif hasattr(dst, 'contents'):
            dst_ptr = ctypes.c_void_p(ctypes.addressof(dst.contents))
        else:
            try:
                dst_ptr = ctypes.c_void_p(int(dst))
            except (TypeError, ValueError):
                raise CudaRTError(f"Invalid destination pointer type: {type(dst)}")
        
        if isinstance(src, int):
            if src == 0:
                raise CudaRTError("Invalid source pointer: NULL")
            src_ptr = ctypes.c_void_p(src)
        elif hasattr(src, 'value'):
            src_ptr = ctypes.c_void_p(src.value)
        elif hasattr(src, 'contents'):
            src_ptr = ctypes.c_void_p(ctypes.addressof(src.contents))
        else:
            try:
                src_ptr = ctypes.c_void_p(int(src))
            except (TypeError, ValueError):
                raise CudaRTError(f"Invalid source pointer type: {type(src)}")
        
        # Handle stream - use NULL (0) for default stream
        if stream is None:
            stream_ptr = ctypes.c_void_p(0)  # NULL stream
        elif isinstance(stream, int):
            stream_ptr = ctypes.c_void_p(stream)
        else:
            try:
                stream_ptr = ctypes.c_void_p(int(stream))
            except (TypeError, ValueError):
                raise CudaRTError(f"Invalid stream type: {type(stream)}")
        
        # Clear any previous CUDA errors
        _cuda_lib.cudaGetLastError()
        
        # Call cudaMemcpyAsync
        result = _cuda_lib.cudaMemcpyAsync(
            dst_ptr, src_ptr, ctypes.c_size_t(count), 
            ctypes.c_int(kind), stream_ptr
        )
        
        if result != CUDA_SUCCESS:
            # Get more detailed error info
            last_error = _cuda_lib.cudaGetLastError()
            raise CudaRTError(f"cudaMemcpyAsync failed with error code: {result}, last error: {last_error}")
        
        return result
    
    @staticmethod
    def memcpy_sync(dst, src, count, kind=cudaMemcpyDeviceToDevice):
        """
        Synchronous memory copy on GPU.
        
        Args:
            dst: Destination device pointer
            src: Source device pointer
            count: Number of bytes to copy
            kind: Memory copy kind
            
        Returns:
            int: CUDA error code (0 for success)
        """
        if not _load_cuda_library():
            raise CudaRTError("CUDA runtime library not available")
        
        # Validate inputs
        if count <= 0:
            raise CudaRTError(f"Invalid count: {count}, must be > 0")
        
        # Convert pointers to void*
        if isinstance(dst, int):
            if dst == 0:
                raise CudaRTError("Invalid destination pointer: NULL")
            dst_ptr = ctypes.c_void_p(dst)
        elif hasattr(dst, 'value'):
            dst_ptr = ctypes.c_void_p(dst.value)
        elif hasattr(dst, 'contents'):
            dst_ptr = ctypes.c_void_p(ctypes.addressof(dst.contents))
        else:
            try:
                dst_ptr = ctypes.c_void_p(int(dst))
            except (TypeError, ValueError):
                raise CudaRTError(f"Invalid destination pointer type: {type(dst)}")
        
        if isinstance(src, int):
            if src == 0:
                raise CudaRTError("Invalid source pointer: NULL")
            src_ptr = ctypes.c_void_p(src)
        elif hasattr(src, 'value'):
            src_ptr = ctypes.c_void_p(src.value)
        elif hasattr(src, 'contents'):
            src_ptr = ctypes.c_void_p(ctypes.addressof(src.contents))
        else:
            try:
                src_ptr = ctypes.c_void_p(int(src))
            except (TypeError, ValueError):
                raise CudaRTError(f"Invalid source pointer type: {type(src)}")
        
        # Clear any previous CUDA errors
        _cuda_lib.cudaGetLastError()
        
        result = _cuda_lib.cudaMemcpy(
            dst_ptr, src_ptr, ctypes.c_size_t(count), ctypes.c_int(kind)
        )
        
        if result != CUDA_SUCCESS:
            # Get more detailed error info
            last_error = _cuda_lib.cudaGetLastError()
            raise CudaRTError(f"cudaMemcpy failed with error code: {result}, last error: {last_error}")
        
        return result
    
    @staticmethod
    def synchronize(stream=None):
        """
        Synchronize CUDA operations.
        
        Args:
            stream: CUDA stream (None for device synchronization)
        """
        if not _load_cuda_library():
            raise CudaRTError("CUDA runtime library not available")
        
        if stream is None:
            result = _cuda_lib.cudaDeviceSynchronize()
        else:
            result = _cuda_lib.cudaStreamSynchronize(ctypes.c_void_p(stream))
        
        if result != CUDA_SUCCESS:
            raise CudaRTError(f"Synchronization failed with error code: {result}")
        
        return result
    
    @staticmethod
    def get_last_error():
        """Get the last CUDA error."""
        if not _load_cuda_library():
            return CUDA_ERROR_INVALID_DEVICE
        return _cuda_lib.cudaGetLastError()
    
    @staticmethod
    def is_available():
        """Check if CUDA runtime is available."""
        return _load_cuda_library()


# Export constants
__all__ = ['rt', 'CudaRTError', 'cudaMemcpyDeviceToDevice', 'cudaMemcpyHostToDevice', 
           'cudaMemcpyDeviceToHost', 'cudaMemcpyHostToHost', 'cudaMemcpyDefault']

