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