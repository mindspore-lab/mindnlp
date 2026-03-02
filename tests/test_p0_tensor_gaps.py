"""Verification test for P0 tensor API gaps."""
import sys
import os

# Ensure src is on the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import mindtorch_v2 as torch

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} {detail}")


def test_view_ops():
    print("\n=== View Ops ===")

    # squeeze
    x = torch.ones((1, 3, 1, 4))
    y = x.squeeze()
    check("squeeze() removes all size-1 dims", y.shape == (3, 4))
    y2 = x.squeeze(0)
    check("squeeze(0) removes dim 0", y2.shape == (3, 1, 4))
    y3 = x.squeeze(2)
    check("squeeze(2) removes dim 2", y3.shape == (1, 3, 4))
    y4 = x.squeeze(1)
    check("squeeze(1) no-op if size != 1", y4.shape == (1, 3, 1, 4))

    # unsqueeze
    x = torch.ones((3, 4))
    y = x.unsqueeze(0)
    check("unsqueeze(0)", y.shape == (1, 3, 4))
    y2 = x.unsqueeze(1)
    check("unsqueeze(1)", y2.shape == (3, 1, 4))
    y3 = x.unsqueeze(-1)
    check("unsqueeze(-1)", y3.shape == (3, 4, 1))

    # permute
    x = torch.ones((2, 3, 4))
    y = x.permute(2, 0, 1)
    check("permute(2,0,1)", y.shape == (4, 2, 3))
    y2 = x.permute((1, 2, 0))
    check("permute((1,2,0)) with tuple", y2.shape == (3, 4, 2))

    # Module-level
    y3 = torch.squeeze(torch.ones((1, 3, 1)))
    check("torch.squeeze()", y3.shape == (3,))
    y4 = torch.unsqueeze(torch.ones((3, 4)), 0)
    check("torch.unsqueeze()", y4.shape == (1, 3, 4))
    y5 = torch.permute(torch.ones((2, 3, 4)), (2, 0, 1))
    check("torch.permute()", y5.shape == (4, 2, 3))


def test_dunder_ops():
    print("\n=== Dunder Operators ===")

    x = torch.tensor([2.0, 3.0, 4.0])

    # pow
    y = x ** 2
    check("x ** 2", abs(y.tolist()[0] - 4.0) < 1e-5 and abs(y.tolist()[1] - 9.0) < 1e-5)

    # rpow
    y2 = 2 ** x
    check("2 ** x (rpow)", abs(y2.tolist()[0] - 4.0) < 1e-5 and abs(y2.tolist()[1] - 8.0) < 1e-5)

    # floordiv
    x2 = torch.tensor([7.0, 10.0, -3.0])
    y3 = x2 // 3
    check("x // 3", abs(y3.tolist()[0] - 2.0) < 1e-5 and abs(y3.tolist()[1] - 3.0) < 1e-5)

    # mod
    y4 = x2 % 3
    check("x % 3", abs(y4.tolist()[0] - 1.0) < 1e-5 and abs(y4.tolist()[1] - 1.0) < 1e-5)

    # iadd
    z = torch.tensor([1.0, 2.0, 3.0])
    z.requires_grad_(False)
    z += 1
    check("z += 1 (iadd)", abs(z.tolist()[0] - 2.0) < 1e-5)

    # isub
    z2 = torch.tensor([5.0, 6.0, 7.0])
    z2.requires_grad_(False)
    z2 -= 1
    check("z -= 1 (isub)", abs(z2.tolist()[0] - 4.0) < 1e-5)

    # imul
    z3 = torch.tensor([2.0, 3.0, 4.0])
    z3.requires_grad_(False)
    z3 *= 2
    check("z *= 2 (imul)", abs(z3.tolist()[0] - 4.0) < 1e-5)

    # itruediv
    z4 = torch.tensor([10.0, 20.0, 30.0])
    z4.requires_grad_(False)
    z4 /= 2
    check("z /= 2 (itruediv)", abs(z4.tolist()[0] - 5.0) < 1e-5)


def test_reduction_ops():
    print("\n=== Reduction Ops ===")

    # var
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    v = x.var()
    check("var() unbiased", abs(v.item() - 2.5) < 1e-4, f"got {v.item()}")
    v2 = x.var(unbiased=False)
    check("var(unbiased=False)", abs(v2.item() - 2.0) < 1e-4, f"got {v2.item()}")

    x2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    v3 = x2.var(dim=1, keepdim=True)
    check("var(dim=1, keepdim=True)", v3.shape == (2, 1))

    # norm
    x3 = torch.tensor([3.0, 4.0])
    n = x3.norm()
    check("norm() L2", abs(n.item() - 5.0) < 1e-4, f"got {n.item()}")
    n2 = x3.norm(p=1)
    check("norm(p=1)", abs(n2.item() - 7.0) < 1e-4, f"got {n2.item()}")

    # prod
    x4 = torch.tensor([2.0, 3.0, 4.0])
    p = x4.prod()
    check("prod()", abs(p.item() - 24.0) < 1e-4, f"got {p.item()}")

    x5 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    p2 = x5.prod(dim=1)
    check("prod(dim=1)", abs(p2.tolist()[0] - 2.0) < 1e-4 and abs(p2.tolist()[1] - 12.0) < 1e-4)

    # Module-level
    v4 = torch.var(x)
    check("torch.var()", abs(v4.item() - 2.5) < 1e-4)
    n3 = torch.norm(x3)
    check("torch.norm()", abs(n3.item() - 5.0) < 1e-4)
    p3 = torch.prod(x4)
    check("torch.prod()", abs(p3.item() - 24.0) < 1e-4)


def test_matmul_aliases():
    print("\n=== Matrix Multiplication Aliases ===")

    # mm
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a.mm(b)
    check("mm() 2D", c.shape == (2, 2) and abs(c.tolist()[0][0] - 19.0) < 1e-4)

    c2 = torch.mm(a, b)
    check("torch.mm()", abs(c2.tolist()[0][0] - 19.0) < 1e-4)

    # bmm
    a3 = torch.randn(2, 3, 4)
    b3 = torch.randn(2, 4, 5)
    c3 = a3.bmm(b3)
    check("bmm() 3D", c3.shape == (2, 3, 5))

    c4 = torch.bmm(a3, b3)
    check("torch.bmm()", c4.shape == (2, 3, 5))


def test_creation_like():
    print("\n=== Creation *_like Functions ===")

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    y = torch.ones_like(x)
    check("ones_like shape", y.shape == x.shape)
    check("ones_like values", abs(y.tolist()[0][0] - 1.0) < 1e-5)

    y2 = torch.empty_like(x)
    check("empty_like shape", y2.shape == x.shape)

    y3 = torch.full_like(x, 7.0)
    check("full_like shape", y3.shape == x.shape)
    check("full_like values", abs(y3.tolist()[0][0] - 7.0) < 1e-5)

    y4 = torch.randn_like(x)
    check("randn_like shape", y4.shape == x.shape)

    y5 = torch.rand_like(x)
    check("rand_like shape", y5.shape == x.shape)


def test_rmsnorm():
    print("\n=== RMSNorm ===")
    from mindtorch_v2.nn import RMSNorm

    rms = RMSNorm(4, eps=1e-6)
    x = torch.randn(2, 3, 4)
    y = rms(x)
    check("RMSNorm output shape", y.shape == (2, 3, 4))
    # RMSNorm should normalize: check output has reasonable magnitude
    check("RMSNorm output not all zeros", any(abs(v) > 1e-6 for row in y.tolist() for subrow in row for v in subrow))


def test_sdpa():
    print("\n=== Scaled Dot Product Attention ===")
    import mindtorch_v2.nn.functional as F

    B, H, L, S, D = 2, 4, 8, 8, 16
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)

    # Basic
    out = F.scaled_dot_product_attention(q, k, v)
    check("SDPA output shape", out.shape == (B, H, L, D))

    # Causal
    out2 = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    check("SDPA causal shape", out2.shape == (B, H, L, D))


def test_floor_divide():
    print("\n=== Floor Divide ===")

    a = torch.tensor([7.0, 10.0, -3.0])
    b = torch.tensor([3.0, 3.0, 2.0])
    c = torch.floor_divide(a, b)
    check("floor_divide tensor/tensor", abs(c.tolist()[0] - 2.0) < 1e-5 and abs(c.tolist()[1] - 3.0) < 1e-5)


if __name__ == "__main__":
    test_view_ops()
    test_dunder_ops()
    test_reduction_ops()
    test_matmul_aliases()
    test_creation_like()
    test_rmsnorm()
    test_sdpa()
    test_floor_divide()

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
