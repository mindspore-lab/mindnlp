# mypy: allow-untyped-defs
"""Various linear algebra utility methods for internal use."""

from typing import Optional

from mindnlp import core
from mindnlp.core import Tensor


def is_sparse(A):
    """Check if tensor A is a sparse tensor"""
    if isinstance(A, core.Tensor):
        return A.layout == core.sparse_coo

    error_str = "expected Tensor"
    if not core.jit.is_scripting():
        error_str += f" but got {type(A)}"
    raise TypeError(error_str)


def get_floating_dtype(A):
    """Return the floating point dtype of tensor A.

    Integer types map to float32.
    """
    dtype = A.dtype
    if dtype in (core.float16, core.float32, core.float64):
        return dtype
    return core.float32


def matmul(A: Optional[Tensor], B: Tensor) -> Tensor:
    """Multiply two matrices.

    If A is None, return B. A can be sparse or dense. B is always
    dense.
    """
    if A is None:
        return B
    if is_sparse(A):
        return core.sparse.mm(A, B)
    return core.matmul(A, B)


def bform(X: Tensor, A: Optional[Tensor], Y: Tensor) -> Tensor:
    """Return bilinear form of matrices: :math:`X^T A Y`."""
    return matmul(X.mT, matmul(A, Y))


def qform(A: Optional[Tensor], S: Tensor):
    """Return quadratic form :math:`S^T A S`."""
    return bform(S, A, S)


def basis(A):
    """Return orthogonal basis of A columns."""
    return core.linalg.qr(A).Q


def symeig(A: Tensor, largest: Optional[bool] = False) -> tuple[Tensor, Tensor]:
    """Return eigenpairs of A with specified ordering."""
    if largest is None:
        largest = False
    E, Z = core.linalg.eigh(A, UPLO="U")
    # assuming that E is ordered
    if largest:
        E = core.flip(E, dims=(-1,))
        Z = core.flip(Z, dims=(-1,))
    return E, Z


# These functions were deprecated and removed
# This nice error message can be removed in version 1.13+
def matrix_rank(input, tol=None, symmetric=False, *, out=None) -> Tensor:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed.\n"
        "Please use the `core.linalg.matrix_rank` function instead. "
        "The parameter 'symmetric' was renamed in `core.linalg.matrix_rank()` to 'hermitian'."
    )


def solve(input: Tensor, A: Tensor, *, out=None) -> tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`core.solve` is deprecated in favor of `core.linalg.solve`. "
        "`core.linalg.solve` has its arguments reversed and does not return the LU factorization.\n\n"
        "To get the LU factorization see `core.lu`, which can be used with `core.lu_solve` or `core.lu_unpack`.\n"
        "X = core.solve(B, A).solution "
        "should be replaced with:\n"
        "X = core.linalg.solve(A, B)"
    )


def lstsq(input: Tensor, A: Tensor, *, out=None) -> tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`core.lstsq` is deprecated in favor of `core.linalg.lstsq`.\n"
        "`core.linalg.lstsq` has reversed arguments and does not return the QR decomposition in "
        "the returned tuple (although it returns other information about the problem).\n\n"
        "To get the QR decomposition consider using `core.linalg.qr`.\n\n"
        "The returned solution in `core.lstsq` stored the residuals of the solution in the "
        "last m - n columns of the returned value whenever m > n. In core.linalg.lstsq, "
        "the residuals are in the field 'residuals' of the returned named tuple.\n\n"
        "The unpacking of the solution, as in\n"
        "X, _ = core.lstsq(B, A).solution[:A.size(1)]\n"
        "should be replaced with:\n"
        "X = core.linalg.lstsq(A, B).solution"
    )


def _symeig(
    input,
    eigenvectors=False,
    upper=True,
    *,
    out=None,
) -> tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "The default behavior has changed from using the upper triangular portion of the matrix by default "
        "to using the lower triangular portion.\n\n"
        "L, _ = core.symeig(A, upper=upper) "
        "should be replaced with:\n"
        "L = core.linalg.eigvalsh(A, UPLO='U' if upper else 'L')\n\n"
        "and\n\n"
        "L, V = core.symeig(A, eigenvectors=True) "
        "should be replaced with:\n"
        "L, V = core.linalg.eigh(A, UPLO='U' if upper else 'L')"
    )


def eig(
    self: Tensor,
    eigenvectors: bool = False,
    *,
    e=None,
    v=None,
) -> tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`core.linalg.eig` returns complex tensors of dtype `cfloat` or `cdouble` rather than real tensors "
        "mimicking complex tensors.\n\n"
        "L, _ = core.eig(A) "
        "should be replaced with:\n"
        "L_complex = core.linalg.eigvals(A)\n\n"
        "and\n\n"
        "L, V = core.eig(A, eigenvectors=True) "
        "should be replaced with:\n"
        "L_complex, V_complex = core.linalg.eig(A)"
    )