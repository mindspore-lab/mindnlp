"""Tests for torch.linalg module."""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import mindtorch_v2 as torch


class TestLinalgNorms:
    """Tests for linalg norm functions."""

    def test_vector_norm_l2(self):
        x = torch.tensor([3.0, 4.0])
        result = torch.linalg.vector_norm(x)
        assert abs(result.item() - 5.0) < 1e-5

    def test_vector_norm_l1(self):
        x = torch.tensor([3.0, -4.0])
        result = torch.linalg.vector_norm(x, ord=1)
        assert abs(result.item() - 7.0) < 1e-5

    def test_vector_norm_inf(self):
        x = torch.tensor([3.0, -4.0, 1.0])
        result = torch.linalg.vector_norm(x, ord=float('inf'))
        assert abs(result.item() - 4.0) < 1e-5

    def test_vector_norm_dim(self):
        x = torch.tensor([[3.0, 4.0], [5.0, 12.0]])
        result = torch.linalg.vector_norm(x, dim=1)
        np.testing.assert_allclose(result.numpy(), [5.0, 13.0], atol=1e-5)

    def test_norm(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = torch.linalg.norm(x)
        expected = np.linalg.norm(np.array([[1, 2], [3, 4]]))
        assert abs(result.item() - expected) < 1e-5

    def test_norm_with_dim(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = torch.linalg.norm(x, dim=1)
        expected = np.linalg.norm(np.array([[1, 2], [3, 4]]), axis=1)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    def test_matrix_norm_frobenius(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = torch.linalg.matrix_norm(x)
        expected = np.sqrt(1 + 4 + 9 + 16)
        assert abs(result.item() - expected) < 1e-5


class TestLinalgDecompositions:
    """Tests for matrix decompositions."""

    def test_qr(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Q, R = torch.linalg.qr(A)
        # Q @ R should reconstruct A
        reconstructed = Q.numpy() @ R.numpy()
        np.testing.assert_allclose(reconstructed, A.numpy(), atol=1e-5)

    def test_svd(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        U, S, Vh = torch.linalg.svd(A)
        # U @ diag(S) @ Vh should reconstruct A (need Sigma as m x n)
        m, n = A.shape
        Sigma = np.zeros((m, n))
        np.fill_diagonal(Sigma, S.numpy())
        reconstructed = U.numpy() @ Sigma @ Vh.numpy()
        np.testing.assert_allclose(reconstructed, A.numpy(), atol=1e-5)

    def test_svd_reduced(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        assert U.shape == (3, 2)
        assert S.shape == (2,)
        assert Vh.shape == (2, 2)

    def test_svdvals(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        S = torch.linalg.svdvals(A)
        assert S.shape == (2,)
        assert S.numpy()[0] > S.numpy()[1]  # Singular values in descending order

    def test_cholesky(self):
        # Create a positive definite matrix
        A = torch.tensor([[4.0, 2.0], [2.0, 3.0]])
        L = torch.linalg.cholesky(A)
        reconstructed = L.numpy() @ L.numpy().T
        np.testing.assert_allclose(reconstructed, A.numpy(), atol=1e-5)

    def test_cholesky_upper(self):
        A = torch.tensor([[4.0, 2.0], [2.0, 3.0]])
        U = torch.linalg.cholesky(A, upper=True)
        reconstructed = U.numpy().T @ U.numpy()
        np.testing.assert_allclose(reconstructed, A.numpy(), atol=1e-5)

    def test_eig(self):
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        eigenvalues, eigenvectors = torch.linalg.eig(A)
        # Verify A @ v = lambda * v for each eigenpair
        for i in range(2):
            lam = eigenvalues.numpy()[i]
            v = eigenvectors.numpy()[:, i]
            np.testing.assert_allclose(A.numpy() @ v, lam * v, atol=1e-5)

    def test_eigh(self):
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        # Eigenvalues should be real and sorted
        vals = eigenvalues.numpy()
        assert vals[0] <= vals[1]

    def test_eigvals(self):
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        vals = torch.linalg.eigvals(A)
        assert vals.shape == (2,)

    def test_eigvalsh(self):
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        vals = torch.linalg.eigvalsh(A)
        expected = np.linalg.eigvalsh(np.array([[2, 1], [1, 3]]))
        np.testing.assert_allclose(vals.numpy(), expected, atol=1e-5)


class TestLinalgSolvers:
    """Tests for linear system solvers."""

    def test_solve(self):
        A = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
        B = torch.tensor([[9.0], [8.0]])
        X = torch.linalg.solve(A, B)
        np.testing.assert_allclose(A.numpy() @ X.numpy(), B.numpy(), atol=1e-5)

    def test_inv(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        A_inv = torch.linalg.inv(A)
        identity = A.numpy() @ A_inv.numpy()
        np.testing.assert_allclose(identity, np.eye(2), atol=1e-5)

    def test_pinv(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        A_pinv = torch.linalg.pinv(A)
        # A @ pinv(A) @ A ≈ A
        reconstructed = A.numpy() @ A_pinv.numpy() @ A.numpy()
        np.testing.assert_allclose(reconstructed, A.numpy(), atol=1e-5)

    def test_lstsq(self):
        A = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        B = torch.tensor([[1.0], [2.0], [2.0]])
        result = torch.linalg.lstsq(A, B)
        assert result.solution.shape == (2, 1)


class TestLinalgProperties:
    """Tests for matrix properties."""

    def test_det(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        d = torch.linalg.det(A)
        assert abs(d.item() - (-2.0)) < 1e-5

    def test_slogdet(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        sign, logabsdet = torch.linalg.slogdet(A)
        assert abs(sign.item() - (-1.0)) < 1e-5
        assert abs(logabsdet.item() - math.log(2.0)) < 1e-5

    def test_matrix_rank(self):
        A = torch.tensor([[1.0, 2.0], [2.0, 4.0]])  # Rank 1
        rank = torch.linalg.matrix_rank(A)
        assert rank.item() == 1

    def test_matrix_rank_full(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # Rank 2
        rank = torch.linalg.matrix_rank(A)
        assert rank.item() == 2

    def test_cond(self):
        A = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # Identity matrix
        c = torch.linalg.cond(A)
        assert abs(c.item() - 1.0) < 1e-5

    def test_matrix_power(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        A2 = torch.linalg.matrix_power(A, 2)
        expected = A.numpy() @ A.numpy()
        np.testing.assert_allclose(A2.numpy(), expected, atol=1e-5)

    def test_matrix_power_zero(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        I = torch.linalg.matrix_power(A, 0)
        np.testing.assert_allclose(I.numpy(), np.eye(2), atol=1e-5)


class TestLinalgMisc:
    """Tests for misc linalg functions."""

    def test_cross(self):
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        c = torch.linalg.cross(a, b)
        np.testing.assert_allclose(c.numpy(), [0.0, 0.0, 1.0], atol=1e-5)

    def test_diagonal(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        d = torch.linalg.diagonal(A)
        np.testing.assert_allclose(d.numpy(), [1.0, 4.0], atol=1e-5)

    def test_multi_dot(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        C = torch.tensor([[1.0], [0.0]])
        result = torch.linalg.multi_dot([A, B, C])
        expected = A.numpy() @ B.numpy() @ C.numpy()
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    def test_vander(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        V = torch.linalg.vander(x)
        expected = np.vander([1, 2, 3], increasing=True)
        np.testing.assert_allclose(V.numpy(), expected, atol=1e-5)

    def test_vander_N(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        V = torch.linalg.vander(x, N=4)
        assert V.shape == (3, 4)

    def test_matmul(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result = torch.linalg.matmul(A, B)
        expected = A.numpy() @ B.numpy()
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)


class TestLinalgLU:
    """Tests for LU decomposition functions."""

    def test_lu(self):
        A = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
        try:
            P, L, U = torch.linalg.lu(A)
            reconstructed = P.numpy() @ L.numpy() @ U.numpy()
            np.testing.assert_allclose(reconstructed, A.numpy(), atol=1e-5)
        except ImportError:
            pytest.skip("scipy not available")

    def test_lu_factor(self):
        A = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
        try:
            result = torch.linalg.lu_factor(A)
            assert result.LU.shape == (2, 2)
        except ImportError:
            pytest.skip("scipy not available")


class TestLinalgMatrixExp:
    """Tests for matrix exponential."""

    def test_matrix_exp_identity(self):
        try:
            I = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
            result = torch.linalg.matrix_exp(I)
            np.testing.assert_allclose(result.numpy(), np.eye(2), atol=1e-5)
        except ImportError:
            pytest.skip("scipy not available")
