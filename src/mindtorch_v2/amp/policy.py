from .._dtype import float32
from .state import get_autocast_dtype, is_autocast_enabled


LOWER_PRECISION_FP = "lower_precision_fp"
FP32 = "fp32"
PROMOTE = "promote"
BUILTIN_PROMOTE = "builtin_promote"
BANNED = "banned"


# Full mapping sourced from torch/testing/_internal/autocast_test_lists.py
_TORCH_LISTS = {
    "torch_fp16": [
        "_convolution",
        "addbmm",
        "addmm",
        "addmv",
        "addr",
        "baddbmm",
        "bmm",
        "chain_matmul",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_tbc",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "convolution",
        "cudnn_convolution",
        "cudnn_convolution_transpose",
        "einsum",
        "gru_cell",
        "lstm_cell",
        "matmul",
        "mm",
        "mv",
        "prelu",
        "rnn_relu_cell",
        "rnn_tanh_cell",
    ],
    "torch_fp32": [
        "acos",
        "asin",
        "binary_cross_entropy_with_logits",
        "cdist",
        "cosh",
        "cosine_embedding_loss",
        "cosine_similarity",
        "cumprod",
        "cumsum",
        "dist",
        "erfinv",
        "exp",
        "expm1",
        "group_norm",
        "hinge_embedding_loss",
        "kl_div",
        "layer_norm",
        "log",
        "log10",
        "log1p",
        "log2",
        "log_softmax",
        "logsumexp",
        "margin_ranking_loss",
        "norm",
        "pdist",
        "poisson_nll_loss",
        "pow",
        "prod",
        "reciprocal",
        "renorm",
        "rsqrt",
        "sinh",
        "softmax",
        "sum",
        "tan",
        "triplet_margin_loss",
    ],
    "torch_need_autocast_promote": [
        "addcdiv",
        "addcmul",
        "atan2",
        "bilinear",
        "cross",
        "dot",
        "grid_sampler",
        "index_put",
        "scatter_add",
        "tensordot",
        "vdot",
    ],
    "torch_expect_builtin_promote": [
        "add",
        "cat",
        "div",
        "eq",
        "equal",
        "ge",
        "gt",
        "le",
        "lt",
        "mul",
        "ne",
        "stack",
    ],
    "nn_fp16": [
        "linear",
    ],
    "nn_fp32": [
        "l1_loss",
        "mse_loss",
        "multi_margin_loss",
        "multilabel_margin_loss",
        "nll_loss",
        "nll_loss2d",
        "smooth_l1_loss",
        "soft_margin_loss",
        "softplus",
    ],
    "linalg_fp16": [
        "linalg_multi_dot",
        "linalg_vecdot",
    ],
    "methods_fp16": [
        "__matmul__",
    ],
    "methods_fp32": [
        "__pow__",
    ],
    "banned": [
        "binary_cross_entropy",
    ],
}


_POLICY_MAP = {}

for name in _TORCH_LISTS["torch_fp16"] + _TORCH_LISTS["nn_fp16"] + _TORCH_LISTS["linalg_fp16"] + _TORCH_LISTS["methods_fp16"]:
    _POLICY_MAP[name] = LOWER_PRECISION_FP
for name in _TORCH_LISTS["torch_fp32"] + _TORCH_LISTS["nn_fp32"] + _TORCH_LISTS["methods_fp32"]:
    _POLICY_MAP[name] = FP32
for name in _TORCH_LISTS["torch_need_autocast_promote"]:
    _POLICY_MAP[name] = PROMOTE
for name in _TORCH_LISTS["torch_expect_builtin_promote"]:
    _POLICY_MAP[name] = BUILTIN_PROMOTE
for name in _TORCH_LISTS["banned"]:
    _POLICY_MAP[name] = BANNED


# Custom op autocast overrides registered via torch.library.register_autocast
_CUSTOM_AUTOCAST_RULES = {}


def register_custom_autocast_rule(op_name, device_type, cast_inputs):
    key = (op_name, device_type)
    if key in _CUSTOM_AUTOCAST_RULES:
        raise RuntimeError(
            "This is not allowed since there's already a kernel registered from python "
            f"overriding {op_name.split('::', 1)[1]}'s behavior for Autocast{device_type.upper()} dispatch key "
            f"and {op_name.split('::', 1)[0]} namespace."
        )
    _CUSTOM_AUTOCAST_RULES[key] = cast_inputs


def _normalize_name(name: str) -> str:
    if name.startswith("aten::"):
        return name.split("::", 1)[1]
    return name


def _cast_tensor(value, device_type, dtype):
    if hasattr(value, "dtype") and hasattr(value, "device"):
        if value.is_floating_point() and value.device.type == device_type:
            from .._functional import to as _to
            return _to(value, device=value.device, dtype=dtype)
    return value


def _cast_args(args, kwargs, device_type, dtype):
    casted_args = tuple(_cast_tensor(v, device_type, dtype) for v in args)
    casted_kwargs = {k: _cast_tensor(v, device_type, dtype) for k, v in kwargs.items()}
    return casted_args, casted_kwargs


def _promote_dtype(args):
    candidates = []
    for value in args:
        if hasattr(value, "dtype") and hasattr(value, "device") and value.is_floating_point():
            candidates.append(value.dtype)
    if not candidates:
        return None
    # widest by itemsize, break ties by float32 preference
    return max(candidates, key=lambda dt: (dt.itemsize, dt == float32))


def apply_autocast_policy(op_name, args, kwargs, device_type):
    if not is_autocast_enabled(device_type):
        return args, kwargs

    custom_cast_dtype = _CUSTOM_AUTOCAST_RULES.get((op_name, device_type))
    if custom_cast_dtype is not None:
        return _cast_args(args, kwargs, device_type, custom_cast_dtype)

    normalized_name = _normalize_name(op_name)
    # Keep torch-like mixed-dtype runtime checks for these CPU ops.
    # If we pre-promote here, backend kernels won't raise the same errors as torch.
    if device_type == "cpu" and normalized_name in {"dot", "tensordot", "cross"}:
        return args, kwargs

    policy = _POLICY_MAP.get(normalized_name)
    if policy is None or policy == BUILTIN_PROMOTE:
        return args, kwargs
    if policy == BANNED:
        raise RuntimeError(f"autocast is not supported for op {op_name}")
    if policy == FP32:
        return _cast_args(args, kwargs, device_type, float32)
    if policy == PROMOTE:
        dtype = _promote_dtype(args)
        if dtype is None:
            return args, kwargs
        return _cast_args(args, kwargs, device_type, dtype)

    dtype = get_autocast_dtype(device_type)
    return _cast_args(args, kwargs, device_type, dtype)


def policy_lists():
    return {k: list(v) for k, v in _TORCH_LISTS.items()}
