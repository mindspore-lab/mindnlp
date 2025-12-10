# mypy: allow-untyped-defs
"""Various linear algebra utility methods for internal use."""

from typing import Optional

import mindtorch
from mindtorch import Tensor


def is_sparse(A):
    """Check if tensor A is a sparse tensor"""
    if isinstance(A, mindtorch.Tensor):
        return A.layout == mindtorch.sparse_coo

    error_str = "expected Tensor"
    if not mindtorch.jit.is_scripting():
        error_str += f" but got {type(A)}"
    raise TypeError(error_str)


def get_floating_dtype(A):
    """Return the floating point dtype of tensor A.

    Integer types map to float32.
    """
    dtype = A.dtype
    if dtype in (mindtorch.float16, mindtorch.float32, mindtorch.float64):
        return dtype
    return mindtorch.float32


def matmul(A: Optional[Tensor], B: Tensor) -> Tensor:
    """Multiply two matrices.

    If A is None, return B. A can be sparse or dense. B is always
    dense.
    """
    if A is None:
        return B
    if is_sparse(A):
        return mindtorch.sparse.mm(A, B)
    return mindtorch.matmul(A, B)


def bform(X: Tensor, A: Optional[Tensor], Y: Tensor) -> Tensor:
    """Return bilinear form of matrices: :math:`X^T A Y`."""
    return matmul(X.mT, matmul(A, Y))


def qform(A: Optional[Tensor], S: Tensor):
    """Return quadratic form :math:`S^T A S`."""
    return bform(S, A, S)


def basis(A):
    """Return orthogonal basis of A columns."""
    return mindtorch.linalg.qr(A).Q


def symeig(A: Tensor, largest: Optional[bool] = False) -> tuple[Tensor, Tensor]:
    """Return eigenpairs of A with specified ordering."""
    if largest is None:
        largest = False
    E, Z = mindtorch.linalg.eigh(A, UPLO="U")
    # assuming that E is ordered
    if largest:
        E = mindtorch.flip(E, dims=(-1,))
        Z = mindtorch.flip(Z, dims=(-1,))
    return E, Z


# These functions were deprecated and removed
# This nice error message can be removed in version 1.13+
def matrix_rank(input, tol=None, symmetric=False, *, out=None) -> Tensor:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed.\n"
        "Please use the `mindtorch.linalg.matrix_rank` function instead. "
        "The parameter 'symmetric' was renamed in `mindtorch.linalg.matrix_rank()` to 'hermitian'."
    )


def solve(input: Tensor, A: Tensor, *, out=None) -> tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`mindtorch.solve` is deprecated in favor of `mindtorch.linalg.solve`. "
        "`mindtorch.linalg.solve` has its arguments reversed and does not return the LU factorization.\n\n"
        "To get the LU factorization see `mindtorch.lu`, which can be used with `mindtorch.lu_solve` or `mindtorch.lu_unpack`.\n"
        "X = mindtorch.solve(B, A).solution "
        "should be replaced with:\n"
        "X = mindtorch.linalg.solve(A, B)"
    )


def lstsq(input: Tensor, A: Tensor, *, out=None) -> tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`mindtorch.lstsq` is deprecated in favor of `mindtorch.linalg.lstsq`.\n"
        "`mindtorch.linalg.lstsq` has reversed arguments and does not return the QR decomposition in "
        "the returned tuple (although it returns other information about the problem).\n\n"
        "To get the QR decomposition consider using `mindtorch.linalg.qr`.\n\n"
        "The returned solution in `mindtorch.lstsq` stored the residuals of the solution in the "
        "last m - n columns of the returned value whenever m > n. In mindtorch.linalg.lstsq, "
        "the residuals are in the field 'residuals' of the returned named tuple.\n\n"
        "The unpacking of the solution, as in\n"
        "X, _ = mindtorch.lstsq(B, A).solution[:A.size(1)]\n"
        "should be replaced with:\n"
        "X = mindtorch.linalg.lstsq(A, B).solution"
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
        "L, _ = mindtorch.symeig(A, upper=upper) "
        "should be replaced with:\n"
        "L = mindtorch.linalg.eigvalsh(A, UPLO='U' if upper else 'L')\n\n"
        "and\n\n"
        "L, V = mindtorch.symeig(A, eigenvectors=True) "
        "should be replaced with:\n"
        "L, V = mindtorch.linalg.eigh(A, UPLO='U' if upper else 'L')"
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
        "`mindtorch.linalg.eig` returns complex tensors of dtype `cfloat` or `cdouble` rather than real tensors "
        "mimicking complex tensors.\n\n"
        "L, _ = mindtorch.eig(A) "
        "should be replaced with:\n"
        "L_complex = mindtorch.linalg.eigvals(A)\n\n"
        "and\n\n"
        "L, V = mindtorch.eig(A, eigenvectors=True) "
        "should be replaced with:\n"
        "L_complex, V_complex = mindtorch.linalg.eig(A)"
    )