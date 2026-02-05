# tests/mindtorch_v2/test_nn_integration.py
"""Integration tests for nn module."""

import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2 import nn


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def test_mlp_forward():
    """MLP forward pass works."""
    model = SimpleMLP(10, 20, 5)
    x = torch.randn(2, 10)
    y = model(x)
    assert y.shape == (2, 5)


def test_mlp_backward():
    """MLP backward pass works."""
    model = SimpleMLP(10, 20, 5)
    x = torch.randn(2, 10, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # All parameters should have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_mlp_parameter_count():
    """MLP has correct number of parameters."""
    model = SimpleMLP(10, 20, 5)
    params = list(model.parameters())
    # fc1: 10*20 + 20, fc2: 20*5 + 5 = 200+20+100+5 = 325
    total_params = sum(p.numel() for p in params)
    assert total_params == 325


def test_transformer_block_components():
    """Test components needed for transformer."""
    batch, seq_len, d_model = 2, 10, 64

    # Embedding
    vocab_size = 1000
    embed = nn.Embedding(vocab_size, d_model)
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
    x = embed(token_ids)
    assert x.shape == (batch, seq_len, d_model)

    # LayerNorm
    ln = nn.LayerNorm(d_model)
    x = ln(x)
    assert x.shape == (batch, seq_len, d_model)

    # Linear (for projection)
    proj = nn.Linear(d_model, d_model * 3)
    qkv = proj(x)
    assert qkv.shape == (batch, seq_len, d_model * 3)

    # Dropout
    dropout = nn.Dropout(0.1)
    dropout.eval()  # Use eval for deterministic test
    out = dropout(x)
    assert out.shape == (batch, seq_len, d_model)


def test_model_train_eval():
    """Model train/eval affects dropout."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.Dropout(0.5),
    )

    x = torch.ones(100, 10)

    # In train mode, dropout should zero some values
    model.train()
    y_train = model(x)

    # In eval mode, dropout should be identity
    model.eval()
    y_eval = model(x)

    # Eval output should have no zeros from dropout
    # (zeros could still come from Linear weights)
