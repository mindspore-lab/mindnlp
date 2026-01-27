# tests/mindtorch_v2/test_nn_functional.py
import numpy as np
import mindtorch_v2 as torch
import mindtorch_v2.nn.functional as F


def test_relu():
    """F.relu applies ReLU activation."""
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    y = F.relu(x)
    np.testing.assert_array_almost_equal(y.numpy(), [0.0, 0.0, 1.0, 2.0])


def test_gelu():
    """F.gelu applies GELU activation."""
    x = torch.tensor([0.0, 1.0, -1.0])
    y = F.gelu(x)
    # GELU(0) = 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
    assert abs(y.numpy()[0]) < 0.01
    assert abs(y.numpy()[1] - 0.841) < 0.01


def test_silu():
    """F.silu applies SiLU/Swish activation."""
    x = torch.tensor([0.0, 1.0, -1.0])
    y = F.silu(x)
    # SiLU(x) = x * sigmoid(x)
    expected = x.numpy() * (1 / (1 + np.exp(-x.numpy())))
    np.testing.assert_array_almost_equal(y.numpy(), expected)


def test_softmax():
    """F.softmax applies softmax."""
    x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    y = F.softmax(x, dim=1)
    # Each row should sum to 1
    np.testing.assert_array_almost_equal(y.sum(dim=1).numpy(), [1.0, 1.0])


def test_linear():
    """F.linear computes linear transformation."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    w = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 2x3
    b = torch.tensor([0.5, 0.5])
    y = F.linear(x, w, b)
    np.testing.assert_array_almost_equal(y.numpy(), [[1.5, 2.5]])


def test_linear_no_bias():
    """F.linear works without bias."""
    x = torch.tensor([[1.0, 2.0]])
    w = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])  # 3x2
    y = F.linear(x, w)
    np.testing.assert_array_almost_equal(y.numpy(), [[3.0, 6.0, 9.0]])


def test_embedding():
    """F.embedding looks up embeddings."""
    weight = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    indices = torch.tensor([0, 2, 1])
    y = F.embedding(indices, weight)
    np.testing.assert_array_almost_equal(y.numpy(), [[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]])


def test_dropout_eval():
    """F.dropout does nothing in eval mode."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = F.dropout(x, p=0.5, training=False)
    np.testing.assert_array_almost_equal(y.numpy(), x.numpy())


def test_layer_norm():
    """F.layer_norm normalizes."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = F.layer_norm(x, normalized_shape=(4,))
    # Should have mean≈0, std≈1
    assert abs(y.mean().item()) < 0.01
