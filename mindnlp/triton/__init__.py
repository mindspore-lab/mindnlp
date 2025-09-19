"""triton adapter for mindspore"""
from functools import lru_cache
import mindspore
from mindtorch import ops
from mindnlp.utils import is_triton_available

if is_triton_available():
    from triton.backends.driver import DriverBase
    from triton.backends.nvidia.driver import CudaUtils, CudaLauncher
    from triton.backends.compiler import GPUTarget

class MSDriver(DriverBase):

    def __init__(self):
        self.utils = CudaUtils()  # TODO: make static
        self.launcher_cls = CudaLauncher
        super().__init__()

    def get_current_device(self):
        return 0

    def set_current_device(self):
        pass

    @lru_cache
    def get_current_stream(self, device=None):
        return mindspore.hal.current_stream().id

    @lru_cache
    def get_device_capability(self, device=0):
        return mindspore.hal.get_device_capability(0)

    @lru_cache
    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        warp_size = 32
        return GPUTarget("cuda", capability, warp_size)

    def get_device_interface(self):
        return mindspore.hal

    @staticmethod
    def is_active():
        return True

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return ops.empty(int(cache_size // 4), dtype=mindspore.int32, device='GPU')
