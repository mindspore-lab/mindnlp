"""End-to-end integration tests for ascend910a-extras custom operators.

Validates that all 6 operators load, dispatch, and produce correct
output shapes and dtypes.  The tests are skipped gracefully if
``libcust_opapi.so`` is not installed.

Usage::

    PYTHONPATH=src python examples/mindtorch_v2/ascendc/ascend910a_extras/test_ops.py

Hardware: Ascend 910A/B (NPU)
"""

import os
import sys
import traceback

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
_src_path = os.path.join(_project_root, "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2._backends.npu import runtime as npu_runtime

# Import op definitions — this registers them with the dispatch system
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ascend910a_extras import ops


def _sync():
    npu_runtime.get_runtime(0).synchronize()


def _to_numpy(t):
    _sync()
    return t.to("cpu").numpy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rope():
    print("\n=== Test: rope ===")
    bs, num_heads, head_dim = 4, 8, 64
    num_kv_heads = 4
    max_seq = 128

    q = torch.randn((bs, num_heads, head_dim), dtype=torch.float16, device="npu")
    k = torch.randn((bs, num_kv_heads, head_dim), dtype=torch.float16, device="npu")
    position_ids = torch.zeros((bs,), dtype=torch.int32, device="npu")
    cos_cache = torch.randn((max_seq, head_dim), dtype=torch.float16, device="npu")
    sin_cache = torch.randn((max_seq, head_dim), dtype=torch.float16, device="npu")

    out_q, out_k = ops.rope(q, k, position_ids, cos_cache, sin_cache)
    _sync()

    assert out_q.shape == q.shape, f"out_q shape mismatch: {out_q.shape} != {q.shape}"
    assert out_k.shape == k.shape, f"out_k shape mismatch: {out_k.shape} != {k.shape}"
    assert str(out_q.dtype) == str(q.dtype), f"out_q dtype mismatch: {out_q.dtype} != {q.dtype}"
    print(f"  out_q shape: {out_q.shape}, dtype: {out_q.dtype}")
    print(f"  out_k shape: {out_k.shape}, dtype: {out_k.dtype}")
    print("  PASSED")


def test_swiglu():
    print("\n=== Test: swiglu ===")
    num_tokens, dim = 8, 128  # dim must be multiple of 64

    x = torch.randn((num_tokens, dim * 2), dtype=torch.float16, device="npu")
    y = ops.swiglu(x)
    _sync()

    assert y.shape == (num_tokens, dim), f"shape mismatch: {y.shape} != {(num_tokens, dim)}"
    assert str(y.dtype) == str(x.dtype), f"dtype mismatch: {y.dtype} != {x.dtype}"
    print(f"  y shape: {y.shape}, dtype: {y.dtype}")
    print("  PASSED")


def test_grouped_matmul():
    print("\n=== Test: grouped_matmul ===")
    num_tokens = 8
    dim = 128       # must be multiple of 64
    num_experts = 2
    inner_dim = 64  # must be multiple of 64

    x = torch.randn((num_tokens, dim), dtype=torch.float16, device="npu")
    # w must be K-major: shape [num_experts, dim, inner_dim] with stride [dim*inner_dim, 1, dim]
    # We create with correct strides by transposing the inner dims
    w_np = np.random.randn(num_experts, inner_dim, dim).astype(np.float16)
    w_host = torch.tensor(w_np, dtype=torch.float16)
    w = w_host.to("npu")
    # w is [num_experts, inner_dim, dim], contiguous
    # We need logical shape [num_experts, dim, inner_dim] with strides [inner_dim*dim, 1, dim]
    # i.e., transposing dims 1 and 2
    w = w.transpose(1, 2)  # Now logical [num_experts, dim, inner_dim] with K-major strides

    # group_list: cumulative token counts
    group_list_np = np.array([4, 8], dtype=np.int64)  # expert 0 gets tokens 0-3, expert 1 gets 4-7
    group_list = torch.tensor(group_list_np, dtype=torch.int64).to("npu")

    y = ops.grouped_matmul(x, w, group_list)
    _sync()

    assert y.shape == (num_tokens, inner_dim), f"shape mismatch: {y.shape} != {(num_tokens, inner_dim)}"
    assert str(y.dtype) == str(x.dtype), f"dtype mismatch: {y.dtype} != {x.dtype}"
    print(f"  y shape: {y.shape}, dtype: {y.dtype}")
    print("  PASSED")


def test_add_rms_norm():
    print("\n=== Test: add_rms_norm ===")
    num_tokens = 8
    dim = 128  # must be multiple of 64

    x = torch.randn((num_tokens, dim), dtype=torch.float16, device="npu")
    residual = torch.randn((num_tokens, dim), dtype=torch.float16, device="npu")
    weight = torch.ones((dim,), dtype=torch.float16, device="npu")
    epsilon = torch.tensor([1e-5], dtype=torch.float32, device="npu")

    y, residual_out = ops.add_rms_norm(x, residual, weight, epsilon)
    _sync()

    assert y.shape == x.shape, f"y shape mismatch: {y.shape} != {x.shape}"
    assert residual_out.shape == x.shape, f"residual_out shape mismatch: {residual_out.shape} != {x.shape}"
    assert str(y.dtype) == str(x.dtype), f"y dtype mismatch: {y.dtype} != {x.dtype}"
    print(f"  y shape: {y.shape}, dtype: {y.dtype}")
    print(f"  residual_out shape: {residual_out.shape}, dtype: {residual_out.dtype}")
    print("  PASSED")


def test_reshape_and_cache():
    print("\n=== Test: reshape_and_cache ===")
    num_tokens = 4
    num_kv_heads = 4
    head_size = 64
    num_blocks = 8
    block_size = 16
    nh16 = num_kv_heads * head_size // 16
    h16 = 16

    key = torch.randn((num_tokens, num_kv_heads, head_size), dtype=torch.float16, device="npu")
    value = torch.randn((num_tokens, num_kv_heads, head_size), dtype=torch.float16, device="npu")
    key_cache = torch.zeros((num_blocks, block_size, nh16, h16), dtype=torch.float16, device="npu")
    value_cache = torch.zeros((num_blocks, block_size, nh16, h16), dtype=torch.float16, device="npu")
    # slot_indices: which slot each token maps to (must be < num_blocks * block_size)
    slot_np = np.array([0, 1, 2, 3], dtype=np.int32)
    slot_indices = torch.tensor(slot_np, dtype=torch.int32).to("npu")

    kc_out, vc_out = ops.reshape_and_cache(key, value, key_cache, value_cache, slot_indices)
    _sync()

    assert kc_out.shape == key_cache.shape, f"key_cache shape mismatch"
    assert vc_out.shape == value_cache.shape, f"value_cache shape mismatch"
    print(f"  key_cache shape: {kc_out.shape}")
    print(f"  value_cache shape: {vc_out.shape}")
    print("  PASSED")


def test_paged_attention():
    print("\n=== Test: paged_attention ===")
    bs = 2
    num_heads = 8
    head_dim = 64
    num_kv_heads = 4
    num_pages = 16
    page_size = 16
    max_page_num_per_seq = 4
    ctx_len = 16  # must be <= max_page_num_per_seq * page_size

    q = torch.randn((bs, num_heads, head_dim), dtype=torch.float16, device="npu")
    # kv cache: [num_pages, num_kv_heads * head_dim // 16, page_size, 16]
    kv_dim2 = num_kv_heads * head_dim // 16
    key_cache = torch.randn((num_pages, kv_dim2, page_size, 16), dtype=torch.float16, device="npu")
    value_cache = torch.randn((num_pages, kv_dim2, page_size, 16), dtype=torch.float16, device="npu")
    # block_tables: [bs, max_page_num_per_seq], page indices
    bt_np = np.zeros((bs, max_page_num_per_seq), dtype=np.int32)
    bt_np[0] = [0, 1, 2, 3]
    bt_np[1] = [4, 5, 6, 7]
    block_tables = torch.tensor(bt_np, dtype=torch.int32).to("npu")
    # context_lens: [bs]
    cl_np = np.array([ctx_len, ctx_len], dtype=np.int32)
    context_lens = torch.tensor(cl_np, dtype=torch.int32).to("npu")

    out = ops.paged_attention(q, key_cache, value_cache, block_tables, context_lens)
    _sync()

    assert out.shape == q.shape, f"shape mismatch: {out.shape} != {q.shape}"
    assert str(out.dtype) == str(q.dtype), f"dtype mismatch: {out.dtype} != {q.dtype}"
    print(f"  out shape: {out.shape}, dtype: {out.dtype}")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = [
    ("rope", test_rope),
    ("swiglu", test_swiglu),
    ("grouped_matmul", test_grouped_matmul),
    ("add_rms_norm", test_add_rms_norm),
    ("reshape_and_cache", test_reshape_and_cache),
    ("paged_attention", test_paged_attention),
]


if __name__ == "__main__":
    print("ascend910a-extras Integration Tests")
    print("=" * 50)

    if not ops.is_available():
        print("\nSKIPPED: libcust_opapi.so not found.")
        print("Set CUST_OPAPI_PATH or install the OPP package.")
        sys.exit(0)

    print(f"Library loaded successfully.")
    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  FAILED: {name}")
            traceback.print_exc()

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed > 0:
        sys.exit(1)
    print("ALL TESTS PASSED")
