import contextlib
from typing import List, Dict, Any, Optional
import dataclasses
import mindspore as ms
import os
import torch
from torch.utils import _pytree as pytree
from torch4ms import tensor
from contextlib import contextmanager
from torch4ms.ms_config import set_ms_config, initialize_ms

__version__ = "0.0.8"
VERSION = __version__

# the "fast path" uses some sparse tensor thingies that currently we
# don't support
torch.backends.mha.set_fastpath_enabled(False)


# Base exports always available
__all__ = [
    "default_env",
    "extract_mindspore",
    "enable_globally",
    "set_ms_config",
    "initialize_ms",
    "compile",
    "t2ms",
]

# Try to import checkpoint functions, but handle case when Flax is not available
try:
    from .checkpoint import save_checkpoint, load_checkpoint
    # Only add to __all__ and namespace if import successful
    __all__.extend(["save_checkpoint", "load_checkpoint"])
except (ImportError, AttributeError):
    # Define dummy functions that raise helpful errors when called
    def save_checkpoint(*args, **kwargs):
        raise ImportError("Flax and JAX are required for save_checkpoint functionality")
    
    def load_checkpoint(*args, **kwargs):
        raise ImportError("Flax and JAX are required for load_checkpoint functionality")

# Import mapping functions from ops module
from .ops.mappings import t2ms

os.environ.setdefault("ENABLE_RUNTIME_UPTIME_TELEMETRY", "1")

# Initialize MindSpore environment
initialize_ms()

env = None


def default_env():
    global env

    if env is None:
        env = tensor.Environment()
    return env


def extract_mindspore(mod: torch.nn.Module, env=None):
    """Returns a pytree of mindspore.Tensor and a mindspore callable."""
    if env is None:
        env = default_env()
    states = dict(mod.named_buffers())
    states.update(mod.named_parameters())

    states = env.t2ms_copy(states)

    # @ms.jit (MindSpore's jit decorator)
    def mindspore_func(states, args, kwargs=None):
        (states, args, kwargs) = env.ms2t_iso((states, args, kwargs))
        with env:
            res = torch.func.functional_call(
                mod, states, args, kwargs, tie_weights=False
            )
        return env.t2ms_iso(res)

    return states, mindspore_func


def enable_globally(mode=None):
    """启用全局模式。
    
    Args:
        mode: 模式名称，目前支持的模式包括"mindspore"等
    """
    env = default_env().enable_torch_modes()
    # 暂时忽略mode参数，因为当前实现不使用它
    return env


def disable_globally():
    global env
    default_env().disable_torch_modes()


@contextlib.contextmanager
def disable_temporarily():
    prev = default_env().enabled
    if prev:
        disable_globally()
    yield ()
    if prev:
        enable_globally()


torch.utils.rename_privateuse1_backend("mindspore")
unsupported_dtype = [torch.quint8]

import torch4ms.device_module

torch._register_device_module("mindspore", torch4ms.device_module)


def enable_accuracy_mode():
    """Enable high precision mode for MindSpore."""
    set_ms_config(mode=ms.context.GRAPH_MODE)
    # In MindSpore, we can set precision_mode to control precision
    # For high accuracy, we use 32-bit floating point
    from mindspore import context
    context.set_context(precision_mode='fp32')
    default_env().config.internal_respect_torch_return_dtypes = True


def enable_performance_mode():
    """Enable performance optimization mode for MindSpore."""
    # For better performance, we can enable graph kernel optimization
    set_ms_config(enable_graph_kernel=True)
    # Use mixed precision when available
    from mindspore import context
    context.set_context(precision_mode='mixed_float16')
    default_env().config.internal_respect_torch_return_dtypes = False


@dataclasses.dataclass
class CompileOptions:
    # only valid if compiling nn.Module
    methods_to_compile: List[str] = dataclasses.field(
        default_factory=lambda: ["forward"]
    )
    ms_jit_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    mode: str = "mindspore"  # or dynamo or export


def compile(fn, options: Optional[CompileOptions] = None):
    """Compile a PyTorch function or module with MindSpore."""
    options = options or CompileOptions()
    if options.mode == "mindspore":
        from torch4ms import minterop

        if isinstance(fn, torch.nn.Module):
            module = minterop.JittableModule(
                fn, extra_jit_args=options.ms_jit_kwargs
            )
            for n in options.methods_to_compile:
                module.make_jitted(n)
            return module
        else:
            return minterop.ms_jit(fn)
    elif options.mode == "dynamo":
        raise RuntimeError("dynamo mode is not supported yet")
    elif options.mode == "export":
        raise RuntimeError("export mode is not supported yet")
    else:
        raise ValueError(f"Unknown compile mode: {options.mode}")
