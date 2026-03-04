"""Tests for P1 missing NPU ops.

NOTE: addmm and einsum use cubeMathType=1 (or matmul which does),
      causing CANN cross-op state contamination on 910B.
      These tests run in separate subprocesses to avoid contamination.
"""
import sys, os, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

def get_torch():
    from mindtorch_v2._backends.npu.runtime import is_available
    if not is_available():
        print("SKIP: NPU not available"); sys.exit(0)
    import mindtorch_v2 as torch
    torch.set_default_device("npu:0")
    return torch

torch = get_torch()
F = __import__('mindtorch_v2.nn.functional', fromlist=['functional'])

passed = 0
failed = 0

def check(name, cond, msg=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name}: {msg}")


def run_isolated(test_code, test_name):
    """Run a test in a separate subprocess to avoid CANN contamination."""
    global passed, failed
    script = f"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname('{__file__}'), '..', 'src'))
import numpy as np
import mindtorch_v2 as torch
torch.set_default_device('npu:0')
{test_code}
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), '..', 'src') + ':' + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, timeout=60,
    )
    output = result.stdout.strip()
    if result.returncode == 0 and output:
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith("PASS"):
                passed += 1
                print(f"  PASS {line[5:]}")
            elif line.startswith("FAIL"):
                failed += 1
                print(f"  FAIL {line[5:]}")
            elif line:
                print(f"  {line}")
    else:
        failed += 1
        err = result.stderr.strip().split('\n')[-1] if result.stderr else "unknown error"
        print(f"  FAIL {test_name}: {err}")


# ---- 1. torch.std ----
print("\n=== torch.std ===")
try:
    x = torch.randn(4, 8)
    s = torch.std(x)
    check("std_scalar", s.shape == (1,), f"shape={s.shape}")
    s2 = torch.std(x, dim=1, keepdim=True)
    check("std_dim", s2.shape == (4, 1), f"shape={s2.shape}")
except Exception as e:
    check("std", False, str(e))

# ---- 2. torch.reciprocal ----
print("\n=== torch.reciprocal ===")
try:
    x = torch.tensor([2.0, 4.0, 0.5], dtype=torch.float32)
    r = torch.reciprocal(x)
    r_np = r.cpu().numpy()
    expected = np.array([0.5, 0.25, 2.0])
    check("reciprocal_values", np.allclose(r_np, expected, atol=1e-5), f"got {r_np}")
except Exception as e:
    check("reciprocal", False, str(e))

# ---- 3. torch.randint ----
print("\n=== torch.randint ===")
try:
    ri = torch.randint(0, 10, (3, 4))
    check("randint_shape", ri.shape == (3, 4), f"shape={ri.shape}")
    ri_np = ri.cpu().numpy()
    check("randint_range", np.all(ri_np >= 0) and np.all(ri_np < 10), f"values={ri_np}")
except Exception as e:
    check("randint", False, str(e))

# ---- 4. torch.randperm ----
print("\n=== torch.randperm ===")
try:
    rp = torch.randperm(10)
    check("randperm_shape", rp.shape == (10,), f"shape={rp.shape}")
    rp_np = rp.cpu().numpy()
    check("randperm_unique", len(set(rp_np)) == 10 and set(rp_np) == set(range(10)),
          f"values={sorted(rp_np)}")
except Exception as e:
    check("randperm", False, str(e))

# ---- 5. F.interpolate ----
print("\n=== F.interpolate ===")
try:
    x = torch.randn(1, 3, 4, 4)
    out_nearest = F.interpolate(x, scale_factor=2, mode='nearest')
    check("interp_nearest_shape", out_nearest.shape == (1, 3, 8, 8),
          f"shape={out_nearest.shape}")
except Exception as e:
    check("interp_nearest", False, str(e))

try:
    x = torch.randn(1, 3, 4, 4)
    out_bilinear = F.interpolate(x, size=(8, 8), mode='bilinear')
    check("interp_bilinear_shape", out_bilinear.shape == (1, 3, 8, 8),
          f"shape={out_bilinear.shape}")
except Exception as e:
    check("interp_bilinear", False, str(e))

# ---- 6. F.normalize ----
print("\n=== F.normalize ===")
try:
    x = torch.randn(3, 4)
    n = F.normalize(x, dim=1)
    check("normalize_shape", n.shape == (3, 4), f"shape={n.shape}")
    n_np = n.cpu().numpy()
    norms = np.sqrt(np.sum(n_np ** 2, axis=1))
    check("normalize_unit", np.allclose(norms, 1.0, atol=1e-4), f"norms={norms}")
except Exception as e:
    check("normalize", False, str(e))

# ---- 7. F.one_hot ----
print("\n=== F.one_hot ===")
try:
    idx = torch.tensor([0, 1, 3, 2], dtype=torch.int64)
    oh = F.one_hot(idx, 5)
    check("one_hot_shape", oh.shape == (4, 5), f"shape={oh.shape}")
    oh_np = oh.cpu().numpy()
    expected = np.eye(5, dtype=np.float32)[[0, 1, 3, 2]]
    check("one_hot_values", np.allclose(oh_np, expected, atol=1e-5), f"got {oh_np}")
except Exception as e:
    check("one_hot", False, str(e))

# ---- 8. torch.from_numpy / torch.as_tensor ----
print("\n=== torch.from_numpy / torch.as_tensor ===")
try:
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = torch.from_numpy(arr)
    check("from_numpy_shape", t.shape == (3,), f"shape={t.shape}")
    t_np = t.cpu().numpy()
    check("from_numpy_values", np.allclose(t_np, arr), f"got {t_np}")
except Exception as e:
    check("from_numpy", False, str(e))

try:
    arr2 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    t2 = torch.as_tensor(arr2, dtype=torch.int64)
    check("as_tensor_shape", t2.shape == (2, 2), f"shape={t2.shape}")
except Exception as e:
    check("as_tensor", False, str(e))

# ---- 9. torch.addmm (ISOLATED: cubeMathType=1 contaminates CANN state) ----
print("\n=== torch.addmm (isolated subprocess) ===")
run_isolated("""
bias = torch.randn(5)
mat1 = torch.randn(3, 4)
mat2 = torch.randn(4, 5)
bias_np = bias.cpu().numpy()
m1_np = mat1.cpu().numpy()
m2_np = mat2.cpu().numpy()
result = torch.addmm(bias, mat1, mat2)
if result.shape != (3, 5):
    print(f"FAIL addmm_shape: shape={result.shape}")
else:
    print("PASS addmm_shape")
expected = bias_np + m1_np @ m2_np
diff = np.abs(result.cpu().numpy() - expected)
if np.allclose(diff, 0, atol=1e-2):
    print("PASS addmm_values")
else:
    print(f"FAIL addmm_values: max_diff={np.max(diff)}")
""", "addmm")

# ---- 10. torch.einsum (ISOLATED: uses matmul → cubeMathType=1) ----
print("\n=== torch.einsum (isolated subprocess) ===")
run_isolated("""
a = torch.randn(3, 4)
b = torch.randn(4, 5)
c = torch.einsum("ij,jk->ik", a, b)
if c.shape != (3, 5):
    print(f"FAIL einsum_shape: shape={c.shape}")
else:
    print("PASS einsum_shape")
a_np = a.cpu().numpy()
b_np = b.cpu().numpy()
c_np = c.cpu().numpy()
expected = a_np @ b_np
if np.allclose(c_np, expected, atol=1e-2):
    print("PASS einsum_matmul")
else:
    print(f"FAIL einsum_matmul: max_diff={np.max(np.abs(c_np - expected))}")
""", "einsum")

# ---- Summary ----
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
if failed:
    sys.exit(1)
