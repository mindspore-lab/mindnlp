"""Ascend 910A/B custom operators from monellz/ascend910a-extras.

These operators are compiled via opdev into OPP packages that export
ACLNN two-phase symbols (aclnnXxxGetWorkspaceSize + aclnnXxx).
This module wraps each op using ``@ascendc_op`` and ``AclnnCustomLauncher``
so they integrate transparently with the mindtorch_v2 dispatch system.

Prerequisites:
    - Build the OPP package from ascend910a-extras (opdev build)
    - The resulting ``libcust_opapi.so`` must be findable via:
      1. The ``CUST_OPAPI_PATH`` environment variable, OR
      2. A standard CANN OPP vendor path

Usage::

    PYTHONPATH=src python examples/mindtorch_v2/ascendc/ascend910a_extras/ops.py
"""

import os
import sys

from mindtorch_v2._tensor import Tensor
from mindtorch_v2._backends.npu.custom_kernel import (
    AclnnCustomLauncher,
    tensor_to_acl,
    destroy_acl_tensor,
    alloc_npu_tensor,
    ascendc_op,
)

# ---------------------------------------------------------------------------
# Lazy launcher singleton
# ---------------------------------------------------------------------------

_launcher = None
_AVAILABLE = None  # tri-state: None = unchecked, True, False

_CANN_SEARCH_PATHS = (
    # Standard OPP vendor install paths
    "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/ascend910a_extras/op_api/lib/linux/aarch64",
    "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/ascend910a_extras/op_api/lib/linux/x86_64",
    "/usr/local/Ascend/latest/opp/vendors/ascend910a_extras/op_api/lib/linux/aarch64",
    "/usr/local/Ascend/latest/opp/vendors/ascend910a_extras/op_api/lib/linux/x86_64",
    # opdev build output (common during development)
    "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib",
)


def _find_library():
    """Search for libcust_opapi.so."""
    # 1. Explicit env var
    env_path = os.environ.get("CUST_OPAPI_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Standard CANN OPP vendor paths
    for base in _CANN_SEARCH_PATHS:
        candidate = os.path.join(base, "libcust_opapi.so")
        if os.path.isfile(candidate):
            return candidate

    return None


def _get_launcher():
    global _launcher, _AVAILABLE
    if _AVAILABLE is True:
        return _launcher
    if _AVAILABLE is False:
        return None
    # First probe
    lib_path = _find_library()
    if lib_path is None:
        _AVAILABLE = False
        return None
    try:
        _launcher = AclnnCustomLauncher(lib_path)
        _AVAILABLE = True
        return _launcher
    except Exception:
        _AVAILABLE = False
        return None


def is_available():
    """Return True if the ascend910a-extras OPP library is loadable."""
    return _get_launcher() is not None


def _require_launcher():
    launcher = _get_launcher()
    if launcher is None:
        raise RuntimeError(
            "ascend910a-extras OPP library (libcust_opapi.so) not found. "
            "Set CUST_OPAPI_PATH or install the OPP package to a standard CANN path."
        )
    return launcher


# =========================================================================
# Op 1: RopeEx
# =========================================================================

@ascendc_op("ascend_extras::rope")
def rope(
    q: Tensor,
    k: Tensor,
    position_ids: Tensor,
    cos_cache: Tensor,
    sin_cache: Tensor,
) -> tuple:
    """Rotary position embedding (RoPE).

    Args:
        q: [bs, num_heads, head_dim] float16
        k: [bs, num_kv_heads, head_dim] float16
        position_ids: [bs] int32
        cos_cache: [max_seq_len, head_dim] float16
        sin_cache: [max_seq_len, head_dim] float16

    Returns:
        (out_q, out_k) — same shapes/dtypes as (q, k).
    """
    launcher = _require_launcher()
    out_q = alloc_npu_tensor(q.shape, q.dtype, q.device)
    out_k = alloc_npu_tensor(k.shape, k.dtype, k.device)

    acl_q, kq = tensor_to_acl(q)
    acl_k, kk = tensor_to_acl(k)
    acl_pos, kp = tensor_to_acl(position_ids)
    acl_cos, kc = tensor_to_acl(cos_cache)
    acl_sin, ks = tensor_to_acl(sin_cache)
    acl_oq, koq = tensor_to_acl(out_q)
    acl_ok, kok = tensor_to_acl(out_k)
    try:
        launcher.run("RopeEx", [acl_q, acl_k, acl_pos, acl_cos, acl_sin, acl_oq, acl_ok])
    finally:
        destroy_acl_tensor(acl_q)
        destroy_acl_tensor(acl_k)
        destroy_acl_tensor(acl_pos)
        destroy_acl_tensor(acl_cos)
        destroy_acl_tensor(acl_sin)
        destroy_acl_tensor(acl_oq)
        destroy_acl_tensor(acl_ok)
        _ = (kq, kk, kp, kc, ks, koq, kok)  # prevent GC until descriptors destroyed
    return out_q, out_k


@rope.register_fake
def _rope_fake(q, k, position_ids, cos_cache, sin_cache):
    from mindtorch_v2._functional import empty_like
    return empty_like(q), empty_like(k)


# =========================================================================
# Op 2: SwiGluEx
# =========================================================================

@ascendc_op("ascend_extras::swiglu")
def swiglu(x: Tensor) -> Tensor:
    """SwiGLU activation: swish(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2.

    Args:
        x: [num_tokens, 2*dim] float16. Last dim must be a multiple of 64.

    Returns:
        [num_tokens, dim] float16.
    """
    launcher = _require_launcher()
    y_shape = list(x.shape)
    y_shape[-1] = y_shape[-1] // 2
    out = alloc_npu_tensor(tuple(y_shape), x.dtype, x.device)

    acl_x, kx = tensor_to_acl(x)
    acl_y, ky = tensor_to_acl(out)
    try:
        launcher.run("SwiGluEx", [acl_x, acl_y])
    finally:
        destroy_acl_tensor(acl_x)
        destroy_acl_tensor(acl_y)
        _ = (kx, ky)
    return out


@swiglu.register_fake
def _swiglu_fake(x):
    from mindtorch_v2._functional import empty
    y_shape = list(x.shape)
    y_shape[-1] = y_shape[-1] // 2
    return empty(tuple(y_shape), dtype=x.dtype, device=x.device)


# =========================================================================
# Op 3: GroupedMatMulEx
# =========================================================================

@ascendc_op("ascend_extras::grouped_matmul")
def grouped_matmul(
    x: Tensor,
    w: Tensor,
    group_list: Tensor,
) -> Tensor:
    """Grouped matrix multiplication.

    Args:
        x: [num_tokens, dim] float16
        w: [num_experts, dim, inner_dim] float16, must be K-major
           (stride order: [0]=largest, [1]=1, [2]=dim)
        group_list: [num_experts] int64, cumulative token counts per expert

    Returns:
        [num_tokens, inner_dim] float16.

    Note:
        The ACLNN op expects w in transposed storage layout
        [num_experts, inner_dim, dim] with strides reflecting K-major order.
        The ACL tensor descriptor is created with swapped dims/strides
        to match the kernel expectation.
    """
    launcher = _require_launcher()
    num_tokens = x.shape[0]
    inner_dim = w.shape[2]
    out = alloc_npu_tensor((num_tokens, inner_dim), x.dtype, x.device)

    acl_x, kx = tensor_to_acl(x)
    # w needs transposed descriptor: logical [num_experts, inner_dim, dim]
    from mindtorch_v2._backends.npu.aclnn import get_bindings, _create_tensor
    bindings = get_bindings()
    w_strides = w.stride()
    w_transposed_shape = (w.shape[0], w.shape[2], w.shape[1])
    w_transposed_strides = (w_strides[0], w_strides[2], w_strides[1])
    acl_w, kw = _create_tensor(
        bindings, w_transposed_shape, w_transposed_strides,
        w.dtype, w.storage().data_ptr(),
    )
    acl_gl, kgl = tensor_to_acl(group_list)
    acl_y, ky = tensor_to_acl(out)
    try:
        launcher.run("GroupedMatMulEx", [acl_x, acl_w, acl_gl, acl_y])
    finally:
        destroy_acl_tensor(acl_x)
        destroy_acl_tensor(acl_w)
        destroy_acl_tensor(acl_gl)
        destroy_acl_tensor(acl_y)
        _ = (kx, kw, kgl, ky)
    return out


@grouped_matmul.register_fake
def _grouped_matmul_fake(x, w, group_list):
    from mindtorch_v2._functional import empty
    return empty((x.shape[0], w.shape[2]), dtype=x.dtype, device=x.device)


# =========================================================================
# Op 4: AddRMSNormEx
# =========================================================================

@ascendc_op("ascend_extras::add_rms_norm")
def add_rms_norm(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: Tensor,
) -> tuple:
    """Fused add + RMS normalization.

    Args:
        x: [num_tokens, dim] float16
        residual: [num_tokens, dim] float16
        weight: [dim] float16
        epsilon: scalar float32 tensor on device (e.g. ``torch.tensor([1e-5], dtype=float32, device="npu")``)

    Returns:
        (y, residual_output) — both [num_tokens, dim] float16.
    """
    launcher = _require_launcher()
    y = alloc_npu_tensor(x.shape, x.dtype, x.device)
    residual_out = alloc_npu_tensor(x.shape, x.dtype, x.device)

    acl_x, kx = tensor_to_acl(x)
    acl_res, kr = tensor_to_acl(residual)
    acl_w, kw = tensor_to_acl(weight)
    acl_eps, ke = tensor_to_acl(epsilon)
    acl_y, ky = tensor_to_acl(y)
    acl_ro, kro = tensor_to_acl(residual_out)
    try:
        launcher.run("AddRMSNormEx", [acl_x, acl_res, acl_w, acl_eps, acl_y, acl_ro])
    finally:
        destroy_acl_tensor(acl_x)
        destroy_acl_tensor(acl_res)
        destroy_acl_tensor(acl_w)
        destroy_acl_tensor(acl_eps)
        destroy_acl_tensor(acl_y)
        destroy_acl_tensor(acl_ro)
        _ = (kx, kr, kw, ke, ky, kro)
    return y, residual_out


@add_rms_norm.register_fake
def _add_rms_norm_fake(x, residual, weight, epsilon):
    from mindtorch_v2._functional import empty_like
    return empty_like(x), empty_like(x)


# =========================================================================
# Op 5: ReshapeAndCacheEx
# =========================================================================

@ascendc_op("ascend_extras::reshape_and_cache", mutates_args=("key_cache", "value_cache"))
def reshape_and_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_indices: Tensor,
) -> tuple:
    """Reshape key/value and write into paged KV cache in-place.

    Args:
        key: [num_tokens, num_kv_heads, head_size] float16
        value: [num_tokens, num_kv_heads, head_size] float16 (may be empty/dummy)
        key_cache: [num_blocks, block_size, nh16, h16] float16, mutated in-place
        value_cache: [num_blocks, block_size, nh16, h16] float16, mutated in-place (may be empty/dummy)
        slot_indices: [num_tokens] int32

    Returns:
        (key_cache, value_cache) — the same tensors, modified in-place.
    """
    launcher = _require_launcher()

    acl_key, kk = tensor_to_acl(key)
    acl_val = None
    kv = None
    if value.numel() > 0:
        acl_val, kv = tensor_to_acl(value)
    acl_kc, kkc = tensor_to_acl(key_cache)
    acl_vc = None
    kvc = None
    if value_cache.numel() > 0:
        acl_vc, kvc = tensor_to_acl(value_cache)
    acl_si, ksi = tensor_to_acl(slot_indices)
    try:
        launcher.run("ReshapeAndCacheEx", [acl_key, acl_val, acl_kc, acl_vc, acl_si])
    finally:
        destroy_acl_tensor(acl_key)
        destroy_acl_tensor(acl_val)
        destroy_acl_tensor(acl_kc)
        destroy_acl_tensor(acl_vc)
        destroy_acl_tensor(acl_si)
        _ = (kk, kv, kkc, kvc, ksi)
    return key_cache, value_cache


@reshape_and_cache.register_fake
def _reshape_and_cache_fake(key, value, key_cache, value_cache, slot_indices):
    return key_cache, value_cache


# =========================================================================
# Op 6: PagedAttentionEx
# =========================================================================

@ascendc_op("ascend_extras::paged_attention")
def paged_attention(
    q: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
) -> Tensor:
    """Paged attention (decode phase).

    Args:
        q: [bs, num_heads, head_dim] float16
        key_cache: [num_pages, num_kv_heads*head_dim//16, page_size, 16] float16
        value_cache: [num_pages, num_kv_heads*head_dim//16, page_size, 16] float16
        block_tables: [bs, max_page_num_per_seq] int32
        context_lens: [bs] int32

    Returns:
        [bs, num_heads, head_dim] float16 — same shape as q.
    """
    launcher = _require_launcher()
    out = alloc_npu_tensor(q.shape, q.dtype, q.device)

    acl_q, kq = tensor_to_acl(q)
    acl_kc, kkc = tensor_to_acl(key_cache)
    acl_vc, kvc = tensor_to_acl(value_cache)
    acl_bt, kbt = tensor_to_acl(block_tables)
    acl_cl, kcl = tensor_to_acl(context_lens)
    acl_o, ko = tensor_to_acl(out)
    try:
        launcher.run("PagedAttentionEx", [acl_q, acl_kc, acl_vc, acl_bt, acl_cl, acl_o])
    finally:
        destroy_acl_tensor(acl_q)
        destroy_acl_tensor(acl_kc)
        destroy_acl_tensor(acl_vc)
        destroy_acl_tensor(acl_bt)
        destroy_acl_tensor(acl_cl)
        destroy_acl_tensor(acl_o)
        _ = (kq, kkc, kvc, kbt, kcl, ko)
    return out


# paged_attention returns single Tensor → auto-fake from @ascendc_op works
