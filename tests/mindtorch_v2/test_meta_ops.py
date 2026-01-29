"""Tests for meta device operations - shape inference without computation."""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import mindtorch_v2 as torch


class TestMetaTensorAdd:
    """Test add operation on meta tensors."""

    def test_meta_tensor_add_same_shape(self):
        """Test a + b on meta tensors with same shape returns meta tensor with correct shape."""
        with torch.device("meta"):
            a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

            # Both are meta tensors
            assert a.device.type == "meta"
            assert b.device.type == "meta"

            # Add should return meta tensor with correct shape
            c = a + b
            assert c.device.type == "meta"
            assert c.shape == (2, 2)
            assert c.dtype == torch.float32

    def test_meta_tensor_add_broadcast(self):
        """Test a + b on meta tensors with broadcast returns correct shape."""
        with torch.device("meta"):
            a = torch.tensor([[1.0, 2.0, 3.0]])  # shape (1, 3)
            b = torch.tensor([[1.0], [2.0], [3.0]])  # shape (3, 1)

            # Add with broadcast
            c = a + b
            assert c.device.type == "meta"
            assert c.shape == (3, 3)  # broadcast result


class TestMetaTensorMatmul:
    """Test matmul operation on meta tensors."""

    def test_meta_tensor_matmul_2d(self):
        """Test a @ b on meta tensors returns correct output shape for 2D matrices."""
        with torch.device("meta"):
            a = torch.tensor([[1.0] * 4] * 3)  # shape (3, 4)
            b = torch.tensor([[1.0] * 5] * 4)  # shape (4, 5)

            # Matmul: (3, 4) @ (4, 5) -> (3, 5)
            c = a @ b
            assert c.device.type == "meta"
            assert c.shape == (3, 5)

    def test_meta_tensor_matmul_batched(self):
        """Test batched matmul on meta tensors."""
        with torch.device("meta"):
            # Create 3D tensors for batched matmul
            a_data = [[[1.0] * 4] * 3] * 2  # shape (2, 3, 4)
            b_data = [[[1.0] * 5] * 4] * 2  # shape (2, 4, 5)
            a = torch.tensor(a_data)
            b = torch.tensor(b_data)

            # Batched matmul: (2, 3, 4) @ (2, 4, 5) -> (2, 3, 5)
            c = a @ b
            assert c.device.type == "meta"
            assert c.shape == (2, 3, 5)


class TestMetaTensorReshape:
    """Test reshape operation on meta tensors."""

    def test_meta_tensor_reshape(self):
        """Test reshape on meta tensor works."""
        with torch.device("meta"):
            a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape (2, 3)

            # Reshape to (3, 2)
            b = a.reshape(3, 2)
            assert b.device.type == "meta"
            assert b.shape == (3, 2)

    def test_meta_tensor_reshape_with_neg_one(self):
        """Test reshape with -1 on meta tensor."""
        with torch.device("meta"):
            a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape (2, 3), 6 elements

            # Reshape with inferred dimension
            b = a.reshape(-1)
            assert b.device.type == "meta"
            assert b.shape == (6,)

            c = a.reshape(6, -1)
            assert c.device.type == "meta"
            assert c.shape == (6, 1)


class TestMetaTensorTranspose:
    """Test transpose operation on meta tensors."""

    def test_meta_tensor_transpose(self):
        """Test transpose on meta tensor."""
        with torch.device("meta"):
            a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape (2, 3)

            # Transpose dims 0 and 1
            b = a.transpose(0, 1)
            assert b.device.type == "meta"
            assert b.shape == (3, 2)
