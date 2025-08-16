from mindnlp import core
from .types import device as device_
from ._prims import ascend, cpu, numpy, meta, ascend_310b
from .configs import DEVICE_TARGET, CPU_USE_NUMPY_OP, SOC
from ._bind import is_autocast_enabled

device_map = {"cpu": "CPU", "npu": "Ascend", "cuda": "GPU"}

"""
__matmul__, addbmm, addmm, addmv, addr, baddbmm, bmm, chain_matmul, multi_dot,
conv1d, conv2d, conv3d, conv_transpose1d, conv_transpose2d, conv_transpose3d, GRUCell,
linear, LSTMCell, matmul, mm, mv, prelu, RNNCell
"""
AMP_AUTO_WHITE_LIST = [
    "dense",
    "matmul",
    "addbmm",
    "addmm",
    "addmv",
    "addr",
    "baddbmm",
    "bmm",
    "chain_matmul",
    "multi_dot",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "mm",
    "mv",
    "prelu",
]


"""
__pow__, __rdiv__, __rpow__, __rtruediv__, acos, asin, binary_cross_entropy_with_logits,
cosh, cosine_embedding_loss, cdist, cosine_similarity, cross_entropy,
cumprod, cumsum, dist, erfinv, exp, expm1, group_norm, hinge_embedding_loss,
kl_div, l1_loss, layer_norm, log, log_softmax, log10, log1p, log2, margin_ranking_loss, mse_loss,
multilabel_margin_loss, multi_margin_loss, nll_loss, norm, normalize, pdist, poisson_nll_loss,
pow, prod, reciprocal, rsqrt, sinh, smooth_l1_loss, soft_margin_loss, softmax, softmin, softplus,
sum, renorm, tan, triplet_margin_loss
"""

AMP_AUTO_BLACK_LIST = [
    'acos',
    'asin',
    'binary_cross_entropy_with_logits',
    'cosh',
    'cosine_embedding_loss',
    'cdist',
    'cosine_similarity',
    'cross_entropy',
    'cumprod',
    'cumsum',
    'dist',
    'erfinv',
    'exp',
    'expm1',
    'group_norm',
    'hinge_embedding_loss',
    'kl_div',
    'l1_loss',
    'layer_norm',
    'log',
    'log_softmax',
    'log10',
    'log1p',
    'log2',
    'margin_ranking_loss',
    'mse_loss',
    'multilabel_margin_loss',
    'multi_margin_loss',
    'nll_loss',
    'norm',
    'normalize',
    'pdist',
    'poisson_nll_loss',
    'pow',
    'prod',
    'reciprocal',
    'rsqrt',
    'sinh',
    'smooth_l1_loss',
    'soft_margin_loss',
    'softmax',
    'softmin',
    'softplus',
    'sum',
    'renorm',
    'tan',
    'triplet_margin_loss',
]


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Dispatcher(metaclass=SingletonMeta):
    def __init__(self):
        self._registry = {"cpu": {}, "npu": {}, "gpu": {}, "numpy": {}, "meta": {}}

    def register(self, func_name, device, func):
        self._registry[device][func_name] = func

    def dispatch(self, func_name, *args, **kwargs):
        device = kwargs.pop("device", None)
        if isinstance(device, str):
            device = device_(device)

        if device is None:
            tensors = (
                [arg for arg in args[0] if core.is_tensor(arg)]
                if isinstance(args[0], (tuple, list))
                else [arg for arg in args if core.is_tensor(arg)]
            )

            if len(tensors) == 1:
                device = tensors[0].device

            else:
                devices = {tensor.device for tensor in tensors}

                if len(devices) > 1:
                    raise ValueError("All tensor arguments must be on the same device.")

                device = next(iter(devices), device_("cpu"))

        if DEVICE_TARGET == "Ascend" and device.type == "cuda":
            device.type = "npu"

        device_type = device.type

        if CPU_USE_NUMPY_OP and device_type == "cpu":
            device_type = "numpy"

        # if is_autocast_enabled(device_type):
        #     if func_name in AMP_AUTO_WHITE_LIST or func_name.replace('_ext', '') in AMP_AUTO_WHITE_LIST:
        #         func_name = func_name + "_fp16"

        #     elif func_name in AMP_AUTO_BLACK_LIST or func_name.replace('_ext', '') in AMP_AUTO_BLACK_LIST:
        #         func_name = func_name + "_fp32"

        func = self._registry[device_type].get(func_name, None)
        if func is None:
            raise RuntimeError(
                f"No implementation for function: {func_name} on {device_type}."
            )
        return func(*args), device


dispatcher = Dispatcher()
if SOC == "ascend310b":
    for func_name in ascend_310b.__all__:
        dispatcher.register(func_name, "npu", getattr(ascend_310b, func_name))
else:
    for func_name in ascend.__all__:
        dispatcher.register(func_name, "npu", getattr(ascend, func_name))

for func_name in cpu.__all__:
    dispatcher.register(func_name, "cpu", getattr(cpu, func_name))

for func_name in numpy.__all__:
    dispatcher.register(func_name, "numpy", getattr(numpy, func_name))

for func_name in meta.__all__:
    dispatcher.register(func_name, "meta", getattr(meta, func_name))
