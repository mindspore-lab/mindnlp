from typing import List, Tuple, Callable
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.matrix import Matrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.algorithm.qr import QR
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar import Scalar


class SVD:

    @staticmethod
    def decompose(matrix: Matrix,
                  iterations: int = 10,
                  qr_alg: Callable[[Matrix], Tuple[Matrix, Matrix]] = QR.decompose_householder
                  ) -> Tuple[Matrix, Matrix, Matrix]:
        height = matrix.get_height()
        width = matrix.get_width()
        matrix_t = matrix.transpose_conjugate()
        m = matrix @ matrix_t if height <= width else matrix_t @ matrix
        sigma, d = SVD._find_singular_values(m, iterations, qr_alg)

        if height <= width:
            u = d
            u_conj_t = u.transpose_conjugate()
            sigma, u_conj_t = SVD._sort_singular_values(sigma, u_conj_t)
            u = u_conj_t.transpose_conjugate()
            v_conj_t = u_conj_t @ matrix
            v_conj_t = SVD._divide_rows_by_singular_values(sigma, v_conj_t)
        else:
            v = d
            v_conj_t = v.transpose_conjugate()
            sigma, v_conj_t = SVD._sort_singular_values(sigma, v_conj_t)
            u_conj_t = v_conj_t @ matrix_t
            u_conj_t = SVD._divide_rows_by_singular_values(sigma, u_conj_t)
            u = u_conj_t.transpose_conjugate()

        sigma_items = np.zeros_like(m[:len(sigma), :len(sigma)].as_array())
        for i in range(v_conj_t.get_height()):
            sigma_items[i, i] = sigma[i].as_array()
        sigma = Matrix.full_like(m, sigma_items)
        return u, sigma, v_conj_t

    @staticmethod
    def _divide_rows_by_singular_values(sigma: List[Scalar],
                                        conj_transposed_u_or_v: Matrix) -> Matrix:
        rows = [
            conj_transposed_u_or_v[i] / sigma[i]
            for i in range(conj_transposed_u_or_v.get_height())
        ]
        return Matrix.full_like(conj_transposed_u_or_v, rows)

    @staticmethod
    def _sort_singular_values(sigma: List[Scalar],
                              conj_transposed_u_or_v: Matrix) -> Tuple[List[Scalar], Matrix]:
        perm = list(zip(sigma, conj_transposed_u_or_v))
        perm.sort(key=lambda item: item[0].get_real(), reverse=True)
        sigma, rows = list(zip(*perm))
        conj_transposed_u_or_v = Matrix.full_like(conj_transposed_u_or_v, list(rows))
        return list(sigma), conj_transposed_u_or_v

    @staticmethod
    def _remove_column(m: Matrix, idx: int) -> Matrix:
        return m[:, 1:] if idx == 0 \
            else m[:, :-1] if idx in [m.get_width() - 1, -1] \
            else m[:, :idx].concat_cols(m[:, (idx + 1):])

    @staticmethod
    def _find_singular_values(m: Matrix,
                              iterations: int,
                              qr_alg: Callable[[Matrix], Tuple[Matrix, Matrix]]) -> Tuple[List[Scalar], Matrix]:
        u_or_v = None
        for _ in range(iterations):
            q, r = qr_alg(m)
            u_or_v = q if u_or_v is None else u_or_v @ q
            m = r @ q
        sigma_items = []
        j = 0
        for i in range(m.get_height()):
            try:
                sigma = m[i, i].sqrt()
                sigma_val = sigma.as_array()
                if np.any(np.isnan(sigma_val)) or np.any(np.isinf(sigma_val)):
                    u_or_v = SVD._remove_column(u_or_v, j)
                else:
                    sigma_items.append(sigma)
                    j += 1
            except ValueError:
                u_or_v = SVD._remove_column(u_or_v, j)
        return sigma_items, u_or_v
