import mindtorch_v2 as torch


def test_autograd_add_mul():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    x.requires_grad = True
    y.requires_grad = True
    z = torch.mul(torch.add(x, y), x)
    z.sum().backward()
    assert x.grad is not None
    assert y.grad is not None


def test_autograd_mul_tensor_scalar_backward():
    x = torch.tensor([1.0, 2.0, 3.0])
    x.requires_grad = True
    y = x * 0.5
    y.sum().backward()
    assert x.grad is not None


def test_autograd_getitem_backward_and_retain_grad():
    x = torch.tensor([1.0, 2.0, 3.0])
    x.requires_grad = True

    y = x * 2.0
    y.retain_grad()
    z = y[0]
    z.backward()

    assert y.grad is not None
    assert y.grad.tolist() == [1.0, 0.0, 0.0]
    assert x.grad is not None
    assert x.grad.tolist() == [2.0, 0.0, 0.0]


def test_autograd_core_nn_ops_keep_graph():
    import mindtorch_v2.nn.functional as F

    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]])
    x.requires_grad = True

    y = F.layer_norm(x, (x.shape[-1],))
    y = F.gelu(y)
    y = F.softmax(y, dim=-1)
    y = F.dropout(y, p=0.1, training=True)

    y.retain_grad()
    y.flatten()[0].backward()

    assert y.grad is not None
    assert x.grad is not None


def test_autograd_batched_matmul_backward_shape_safe():
    a = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[0.5, -1.0, 2.0], [3.0, 1.5, -2.0]],
        ]
    )
    b = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]],
            [[2.0, 1.0], [1.0, 0.0], [0.0, 3.0]],
        ]
    )
    a.requires_grad = True
    b.requires_grad = True

    out = torch.matmul(a, b)
    out.flatten()[0].backward()

    assert a.grad is not None
    assert b.grad is not None
