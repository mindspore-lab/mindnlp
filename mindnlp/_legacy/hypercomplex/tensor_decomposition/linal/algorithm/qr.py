from typing import Tuple
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_scalar import Scalar as HCScalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.matrix import Matrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.real.real_scalar import Scalar as RealScalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar import Scalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar_visitor import ScalarVisitor


class QR:

    class _HouseholderAlphaCalculator(ScalarVisitor):

        def visit_real(self, s: RealScalar, *args, **kwargs) -> None:
            output = kwargs.get('output')
            f = s.as_number()
            alpha = Scalar.one_like(s)
            if f < 0:
                alpha = -alpha
            output.append(alpha)

        def visit_complex(self, s: HCScalar, *args, **kwargs) -> None:
            output = kwargs.get('output')
            x, y = s.as_array()
            z = np.complex(x, y)
            angle = np.angle(z)
            alpha = Scalar.make_like(s, (np.cos(angle), np.sin(angle)))
            alpha = -alpha
            output.append(alpha)

        def visit_dual(self, s: HCScalar, *args, **kwargs) -> None:
            output = kwargs.get('output')
            x, y = s.as_array()
            y_ = y / np.abs(x)
            if np.isnan(y_):
                y_ = np.float64(0.)
            x_ = np.float64(1. if x < 0 else -1.)
            alpha = Scalar.make_like(s, (x_, y_))
            output.append(alpha)

        def calculate_alpha(self, x: Scalar, norm: Scalar) -> Scalar:
            output = []
            kwargs = {
                'output': output,
            }
            x.visit(self, **kwargs)
            alpha = output[0]
            alpha = alpha * norm
            return alpha

    @staticmethod
    def decompose_gram_schmidt(matrix: Matrix) -> Tuple[Matrix, Matrix]:
        if matrix.get_height() != matrix.get_width():
            raise ValueError(
                'Only square matrices can be QR decomposed, but got: '
                f'height={matrix.get_height()}, width={matrix.get_width()}'
            )
        q = QR._gram_schmidt(matrix)
        q_inv = q.transpose_conjugate()
        r = q_inv @ matrix
        return q, r

    @staticmethod
    def decompose_householder(matrix: Matrix) -> Tuple[Matrix, Matrix]:
        if matrix.get_height() != matrix.get_width():
            raise ValueError(
                'Only square matrices can be QR decomposed, but got: '
                f'height={matrix.get_height()}, width={matrix.get_width()}'
            )
        q, r = QR._householder(matrix)
        return q, r

    @staticmethod
    def _householder(m: Matrix):
        r = m
        q = identity = Matrix.identity_like(m)
        one = identity[0][0]
        two = one + one
        hh_alpha = QR._HouseholderAlphaCalculator()
        for k in range(m.get_width()):
            x = r[k:, k:(k + 1)]
            e = identity[k:, k:(k + 1)]
            x_conj_t = x.transpose_conjugate()
            x_norm_sqr = (x_conj_t @ x)[0][0]
            x_norm = x_norm_sqr.sqrt()
            alpha = hh_alpha.calculate_alpha(x[0][0], x_norm)
            u = x - e * alpha
            u_conj_t = u.transpose_conjugate()
            u_norm_sqr = (u_conj_t @ u)[0][0]
            u_norm = u_norm_sqr.sqrt()
            v = u / u_norm
            v_items = v.as_array()
            if np.any(np.isnan(v_items)) or np.any(np.isinf(v_items)):
                v = e
            v_conj_t = v.transpose_conjugate()
            supplementary = v @ v_conj_t
            q_k = identity[k:, k:] - supplementary * two
            if k > 0:
                q_k = identity[k:, :k].concat_cols(q_k)
                q_k = identity[:k].concat_rows(q_k)
            q = q @ q_k
            r = q_k @ r
        return q, r

    @staticmethod
    def _gram_schmidt(m: Matrix) -> Matrix:
        normalized = None
        projections = Matrix.ones_like(m[0:1])
        v_zero = Matrix.zeros_like(projections)
        for j in range(m.get_width()):
            v = m[:, j:(j + 1)]
            if j > 0:
                t = v.concat_cols(normalized)
                v = t @ projections[:, j:(j + 1)]
            v_conj = v.transpose_conjugate()
            v_sqr_len = (v_conj @ v)[0, 0]
            v_len = v_sqr_len.sqrt()
            v = v / v_len
            normalized = v if j == 0 else normalized.concat_cols(v)
            if j + 1 < m.get_width():
                v_conj = v.transpose_conjugate()
                dot_products = v_conj @ m[:, j + 1:]
                dot_products = -dot_products
                dot_products = v_zero[..., :(j + 1)].concat_cols(dot_products)
                projections = projections.concat_rows(dot_products)
        return normalized
