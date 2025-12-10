import warnings
import mindspore
from mindtorch.configs import ON_A1


def dtype_wrapper(fn):
    def wrapper(*args, **kwargs):
        ms_dtype = kwargs.pop("ms_dtype", None)
        ms_dtype = kwargs.pop("mindspore_dtype", ms_dtype)
        if ON_A1 and ms_dtype == mindspore.bfloat16:
            warnings.warn("910A do not support bfloat16, use float16 for `ms_dtype`.")
            ms_dtype = mindspore.float16
        if ms_dtype is not None:
            kwargs["torch_dtype"] = ms_dtype
        return fn(*args, **kwargs)

    return wrapper


def patch_dtype_wrapper(cls, method_name, other_decorators=None):
    patch_wrappers(cls, method_name, [dtype_wrapper])


def patch_wrappers(cls, method_name, other_decorators=None):
    original_method = getattr(cls, method_name)
    wrapped_func = original_method.__func__

    if other_decorators is not None:
        for dec in other_decorators:
            wrapped_func = dec(wrapped_func)

    # 重新创建类方法并赋值回类
    setattr(cls, method_name, classmethod(wrapped_func))
