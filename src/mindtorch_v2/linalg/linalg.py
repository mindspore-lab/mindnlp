"""Implementation of torch.linalg functions via dispatch."""

from .._dispatch.dispatcher import dispatch


def _validate_matrix(input, fname):
    """Validate input is at least 2D with square last two dims."""
    if input.dim() < 2:
        raise RuntimeError(f"linalg.{fname}: Expected input to be at least 2-D, got {input.dim()}-D")
    if input.shape[-2] != input.shape[-1]:
        raise RuntimeError(
            f"linalg.{fname}: A must be batches of square matrices, "
            f"but they are {input.shape[-2]} by {input.shape[-1]} matrices"
        )


def cholesky(input, *, upper=False, out=None):
    """Computes the Cholesky decomposition of a symmetric positive-definite matrix."""
    return dispatch("linalg_cholesky", input.device.type, input, upper)


def cond(input, p=None):
    """Computes the condition number of a matrix."""
    return dispatch("linalg_cond", input.device.type, input, p)


def cross(input, other, *, dim=-1):
    """Computes the cross product of two 3D vectors."""
    return dispatch("cross", input.device.type, input, other, dim=dim)


def det(input):
    """Computes the determinant of a square matrix."""
    return dispatch("linalg_det", input.device.type, input)


def diagonal(input, *, offset=0, dim1=-2, dim2=-1):
    """Alias for torch.diagonal with default dims for matrices."""
    return dispatch("diagonal", input.device.type, input, offset, dim1, dim2)


def eig(input):
    """Computes the eigenvalue decomposition of a square matrix."""
    return dispatch("linalg_eig", input.device.type, input)


def eigh(input, UPLO='L'):
    """Computes the eigenvalue decomposition of a symmetric/Hermitian matrix."""
    return dispatch("linalg_eigh", input.device.type, input, UPLO)


def eigvals(input):
    """Computes the eigenvalues of a square matrix."""
    return dispatch("linalg_eigvals", input.device.type, input)


def eigvalsh(input, UPLO='L'):
    """Computes the eigenvalues of a symmetric/Hermitian matrix."""
    return dispatch("linalg_eigvalsh", input.device.type, input, UPLO)


def householder_product(input, tau):
    """Computes the first n columns of a product of Householder matrices."""
    return dispatch("linalg_householder_product", input.device.type, input, tau)


def inv(input):
    """Computes the inverse of a square matrix."""
    return dispatch("linalg_inv", input.device.type, input)


def lstsq(input, b, *, rcond=None, driver=None):
    """Computes the least squares solution to a system of linear equations."""
    return dispatch("linalg_lstsq", input.device.type, input, b, rcond, driver)


def lu(input, *, pivot=True, out=None):
    """Computes the LU decomposition with partial pivoting."""
    return dispatch("linalg_lu", input.device.type, input, pivot)


def lu_factor(input, *, pivot=True, out=None):
    """Computes a compact LU factorization."""
    return dispatch("linalg_lu_factor", input.device.type, input, pivot)


def lu_solve(LU, pivots, B, *, left=True, adjoint=False):
    """Solves a system using the LU factorization."""
    return dispatch("linalg_lu_solve", LU.device.type, LU, pivots, B, left, adjoint)


def matmul(input, other, *, out=None):
    """Matrix product. Alias for torch.matmul."""
    return dispatch("matmul", input.device.type, input, other)


def matrix_exp(input):
    """Computes the matrix exponential."""
    return dispatch("linalg_matrix_exp", input.device.type, input)


def matrix_norm(input, ord='fro', dim=(-2, -1), keepdim=False, *, dtype=None):
    """Computes a matrix norm."""
    return dispatch("linalg_matrix_norm", input.device.type, input, ord, dim, keepdim)


def matrix_power(input, n):
    """Computes the n-th power of a square matrix."""
    return dispatch("linalg_matrix_power", input.device.type, input, n)


def matrix_rank(input, *, atol=None, rtol=None, hermitian=False):
    """Computes the numerical rank of a matrix."""
    return dispatch("linalg_matrix_rank", input.device.type, input, atol, rtol, hermitian)


def multi_dot(tensors, *, out=None):
    """Efficiently multiplies two or more matrices."""
    device = tensors[0].device.type
    return dispatch("linalg_multi_dot", device, tensors)


def norm(input, ord=None, dim=None, keepdim=False, *, dtype=None):
    """Computes a vector or matrix norm."""
    return dispatch("linalg_norm", input.device.type, input, ord, dim, keepdim)


def pinv(input, *, atol=None, rtol=None, hermitian=False):
    """Computes the pseudoinverse (Moore-Penrose) of a matrix."""
    return dispatch("linalg_pinv", input.device.type, input, atol, rtol, hermitian)


def qr(input, mode='reduced'):
    """Computes the QR decomposition."""
    return dispatch("linalg_qr", input.device.type, input, mode)


def slogdet(input):
    """Computes the sign and log of the absolute value of the determinant."""
    return dispatch("linalg_slogdet", input.device.type, input)


def solve(input, B, *, left=True, out=None):
    """Solves a square system of linear equations."""
    return dispatch("linalg_solve", input.device.type, input, B, left)


def solve_triangular(input, B, *, upper, left=True, unitriangular=False, out=None):
    """Solves a triangular system of linear equations."""
    return dispatch("linalg_solve_triangular", input.device.type, input, B, upper, left, unitriangular)


def svd(input, full_matrices=True):
    """Computes the singular value decomposition."""
    return dispatch("linalg_svd", input.device.type, input, full_matrices)


def svdvals(input):
    """Computes the singular values."""
    return dispatch("linalg_svdvals", input.device.type, input)


def tensorinv(input, ind=2):
    """Computes the tensor inverse for tensors satisfying tensorsolve."""
    return dispatch("linalg_tensorinv", input.device.type, input, ind)


def tensorsolve(input, B, dims=None):
    """Solves a tensor equation."""
    return dispatch("linalg_tensorsolve", input.device.type, input, B, dims)


def vander(x, N=None):
    """Generates a Vandermonde matrix."""
    return dispatch("linalg_vander", x.device.type, x, N)


def vector_norm(input, ord=2, dim=None, keepdim=False, *, dtype=None):
    """Computes a vector norm."""
    return dispatch("linalg_vector_norm", input.device.type, input, ord, dim, keepdim)
