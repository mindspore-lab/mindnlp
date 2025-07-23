import contextlib
from typing_extensions import deprecated

class cuBLASModule:
    # def __getattr__(self, name):
    #     if name == "allow_tf32":
    #         return torch._C._get_cublas_allow_tf32()
    #     elif name == "allow_fp16_reduced_precision_reduction":
    #         return torch._C._get_cublas_allow_fp16_reduced_precision_reduction()
    #     elif name == "allow_bf16_reduced_precision_reduction":
    #         return torch._C._get_cublas_allow_bf16_reduced_precision_reduction()
    #     elif name == "allow_fp16_accumulation":
    #         return torch._C._get_cublas_allow_fp16_accumulation()
    #     elif name == "fp32_precision":
    #         return torch._C._get_fp32_precision_getter("cuda", "matmul")
    #     raise AttributeError("Unknown attribute " + name)

    # def __setattr__(self, name, value):
    #     if name == "allow_tf32":
    #         return torch._C._set_cublas_allow_tf32(value)
    #     elif name == "allow_fp16_reduced_precision_reduction":
    #         return torch._C._set_cublas_allow_fp16_reduced_precision_reduction(value)
    #     elif name == "allow_bf16_reduced_precision_reduction":
    #         return torch._C._set_cublas_allow_bf16_reduced_precision_reduction(value)
    #     elif name == "allow_fp16_accumulation":
    #         return torch._C._set_cublas_allow_fp16_accumulation(value)
    #     elif name == "fp32_precision":
    #         return torch._C._set_fp32_precision_setter("cuda", "matmul", value)
    #     raise AttributeError("Unknown attribute " + name)
    pass

matmul = cuBLASModule()

@contextlib.contextmanager
@deprecated(
    (
        "`torch.backends.cuda.sdp_kernel()` is deprecated. "
        "In the future, this context manager will be removed. "
        "Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, "
        "with updated signature."
    ),
    category=FutureWarning,
)
def sdp_kernel(
    enable_flash: bool = True,
    enable_math: bool = True,
    enable_mem_efficient: bool = True,
    enable_cudnn: bool = True,
):
    r"""
    .. warning:: This flag is beta and subject to change.

    This context manager can be used to temporarily enable or disable any of the three backends for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored.
    """
    # from torch.nn.attention import sdpa_kernel

    # backend_list = []
    # if enable_flash:
    #     backend_list.append(SDPBackend.FLASH_ATTENTION)
    # if enable_mem_efficient:
    #     backend_list.append(SDPBackend.EFFICIENT_ATTENTION)
    # if enable_math:
    #     backend_list.append(SDPBackend.MATH)
    # if enable_cudnn:
    #     backend_list.append(SDPBackend.CUDNN_ATTENTION)

    # with sdpa_kernel(backend_list) as context:
    #     try:
    #         yield context
    #     finally:
    #         pass

    pass
