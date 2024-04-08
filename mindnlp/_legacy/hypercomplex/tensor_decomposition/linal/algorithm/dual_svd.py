from typing import List, Tuple
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.matrix import Matrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.algorithm.qr import QR
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar import Scalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_matrix import Matrix as HCMatrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.dual.dual_algebra_factory import DualAlgebraFactory
from abc import ABC


class DualSVD(ABC):
    TOLERANCE: float = 1.e-5

    @staticmethod
    def _decompose(matrix: Matrix, mult: int) -> Tuple[Matrix, Matrix, Matrix]:
        r"""Implementation of dual *-SVD with special restrictions."""
        height = matrix.get_height()
        width = matrix.get_width()
        np_matrix = matrix.as_array()
        if height < width:
            np_matrix = np.transpose(np_matrix, (1, 0, 2))
        m_real, m_dual = np_matrix[:, :, 0], np_matrix[:, :, 1]
        u_real, s_real, v_real = np.linalg.svd(m_real, compute_uv=True)
        rank = np.linalg.matrix_rank(np.diag(s_real), tol=DualSVD.TOLERANCE).item()
        k_dual = u_real.T @ m_dual @ v_real.T

        x_dual = np.zeros_like(u_real)
        y_dual = np.diag(k_dual)
        z_dual = np.zeros_like(v_real)
        StarSVD._compute_dual_parts(x_dual, s_real, k_dual, z_dual, mult, rank)

        u, sigma, v = DualSVD._get_dual_factors(u_real, s_real, v_real, x_dual, y_dual, z_dual)
        if height < width:
            u, sigma, v = DualSVD._swap_transpose_factors(u, sigma, v)
        u_matrix = HCMatrix(DualAlgebraFactory, items=u)
        sigma_matrix = HCMatrix(DualAlgebraFactory, items=sigma)
        v_matrix = HCMatrix(DualAlgebraFactory, items=v)

        return u_matrix, sigma_matrix, v_matrix

    @staticmethod
    def _get_dual_factors(u_real: np.ndarray,
                          s_real: np.ndarray,
                          v_real: np.ndarray,
                          x_dual: np.ndarray,
                          y_dual: np.ndarray,
                          z_dual: np.ndarray) -> Tuple[Matrix, Matrix, Matrix]:
        r"""Computes U,V, Sigma matrices for T-SVD and *-SVD after the preparatory calculations."""
        u = np.zeros((u_real.shape[0], u_real.shape[1], 2))
        sigma = np.zeros((u_real.shape[1], v_real.shape[0], 2))
        v = np.zeros((v_real.shape[0], v_real.shape[1], 2))
        u[:, :, 0] = u_real
        v[:, :, 0] = v_real
        u[:, :, 1] = u_real @ x_dual
        v[:, :, 1] = z_dual @ v_real
        sigma[:, :, 0] = np.eye(u_real.shape[1], v_real.shape[0]) @ np.diag(s_real)
        sigma[:, :, 1] = np.eye(u_real.shape[1], v_real.shape[0]) @ np.diag(y_dual)
        return u, sigma, v

    @staticmethod
    def _compute_dual_parts(x_dual: np.ndarray,
                            s_real: np.ndarray,
                            k_dual: np.ndarray,
                            z_dual: np.ndarray,
                            mult: int,
                            rank: int) -> None:
        r"""Fills matrices X and Z to get correct decomposition that looks like:
        matrix M = (U + Xe) * (S + Ye) * (V + Ze)"""
        n_big = np.max([len(x_dual), len(z_dual)])
        for i in range(rank, n_big):
            for j in range(rank):
                x_dual[i, j] = k_dual[i, j] / s_real[j]
                x_dual[j, i] = -x_dual[i, j]
        for i in range(rank):
            for j in range(i + 1, rank):
                b = np.array([k_dual[i, j], k_dual[j, i]])
                syst = np.array([[s_real[j], s_real[i]],
                                 mult * [s_real[i],  mult * s_real[j]]])
                sol = np.linalg.solve(syst, b)

                x_dual[j, i], x_dual[i, j] = mult * sol[0], sol[0]
                z_dual[j, i], z_dual[i, j] = mult * sol[1], sol[1]

    @staticmethod
    def _swap_transpose_factors(u: np.ndarray,
                                sigma: np.ndarray,
                                v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Swaps matrices U, V and transposes matrices U, V, Sigma for 'wide' matrices (H > W)"""
        temp = v
        sigma = np.transpose(sigma, (1, 0, 2))
        v = np.transpose(u, (1, 0, 2))
        u = np.transpose(temp, (1, 0, 2))
        return u, sigma, v
