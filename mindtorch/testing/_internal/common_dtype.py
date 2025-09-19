# mypy: ignore-errors


import mindtorch

# Functions and classes for describing the dtypes a function supports
# NOTE: these helpers should correspond to PyTorch's C++ dispatch macros


# Verifies each given dtype is a mindtorch.dtype
def _validate_dtypes(*dtypes):
    for dtype in dtypes:
        assert isinstance(dtype, mindtorch.dtype)
    return dtypes


# class for tuples corresponding to a PyTorch dispatch macro
class _dispatch_dtypes(tuple):
    __slots__ = ()

    def __add__(self, other):
        assert isinstance(other, tuple)
        return _dispatch_dtypes(tuple.__add__(self, other))


_empty_types = _dispatch_dtypes(())


def empty_types():
    return _empty_types


_floating_types = _dispatch_dtypes((mindtorch.float32, mindtorch.float64))


def floating_types():
    return _floating_types


_floating_types_and_half = _floating_types + (mindtorch.half,)


def floating_types_and_half():
    return _floating_types_and_half


def floating_types_and(*dtypes):
    return _floating_types + _validate_dtypes(*dtypes)


_floating_and_complex_types = _floating_types + (mindtorch.cfloat, mindtorch.cdouble)


def floating_and_complex_types():
    return _floating_and_complex_types


def floating_and_complex_types_and(*dtypes):
    return _floating_and_complex_types + _validate_dtypes(*dtypes)


_double_types = _dispatch_dtypes((mindtorch.float64, mindtorch.complex128))


def double_types():
    return _double_types


# NB: Does not contain uint16/uint32/uint64 for BC reasons
_integral_types = _dispatch_dtypes(
    (mindtorch.uint8, mindtorch.int8, mindtorch.int16, mindtorch.int32, mindtorch.int64)
)


def integral_types():
    return _integral_types


def integral_types_and(*dtypes):
    return _integral_types + _validate_dtypes(*dtypes)


_all_types = _floating_types + _integral_types


def all_types():
    return _all_types


def all_types_and(*dtypes):
    return _all_types + _validate_dtypes(*dtypes)


_complex_types = _dispatch_dtypes((mindtorch.cfloat, mindtorch.cdouble))


def complex_types():
    return _complex_types


def complex_types_and(*dtypes):
    return _complex_types + _validate_dtypes(*dtypes)


_all_types_and_complex = _all_types + _complex_types


def all_types_and_complex():
    return _all_types_and_complex


def all_types_and_complex_and(*dtypes):
    return _all_types_and_complex + _validate_dtypes(*dtypes)


_all_types_and_half = _all_types + (mindtorch.half,)


def all_types_and_half():
    return _all_types_and_half


_all_mps_types = (
    _dispatch_dtypes({mindtorch.float, mindtorch.half, mindtorch.bfloat16}) + _integral_types
)


def all_mps_types():
    return _all_mps_types


def all_mps_types_and(*dtypes):
    return _all_mps_types + _validate_dtypes(*dtypes)


_float8_types = _dispatch_dtypes(
    (
        mindtorch.float8_e4m3fn,
        mindtorch.float8_e4m3fnuz,
        mindtorch.float8_e5m2,
        mindtorch.float8_e5m2fnuz,
    )
)


def float8_types():
    return _float8_types


def float8_types_and(*dtypes):
    return _float8_types + _validate_dtypes(*dtypes)


def all_types_complex_float8_and(*dtypes):
    return _all_types + _complex_types + _float8_types + _validate_dtypes(*dtypes)


def custom_types(*dtypes):
    """Create a list of arbitrary dtypes"""
    return _empty_types + _validate_dtypes(*dtypes)


# The functions below are used for convenience in our test suite and thus have no corresponding C++ dispatch macro


# See AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS.
def get_all_dtypes(
    include_half=True,
    include_bfloat16=True,
    include_bool=True,
    include_complex=True,
    include_complex32=False,
    include_qint=False,
) -> list[mindtorch.dtype]:
    dtypes = get_all_int_dtypes() + get_all_fp_dtypes(
        include_half=include_half, include_bfloat16=include_bfloat16
    )
    if include_bool:
        dtypes.append(mindtorch.bool)
    if include_complex:
        dtypes += get_all_complex_dtypes(include_complex32)
    if include_qint:
        dtypes += get_all_qint_dtypes()
    return dtypes


def get_all_math_dtypes(device) -> list[mindtorch.dtype]:
    return (
        get_all_int_dtypes()
        + get_all_fp_dtypes(
            include_half=device.startswith("cuda"), include_bfloat16=False
        )
        + get_all_complex_dtypes()
    )


def get_all_complex_dtypes(include_complex32=False) -> list[mindtorch.dtype]:
    return (
        [mindtorch.complex32, mindtorch.complex64, mindtorch.complex128]
        if include_complex32
        else [mindtorch.complex64, mindtorch.complex128]
    )


def get_all_int_dtypes() -> list[mindtorch.dtype]:
    return [mindtorch.uint8, mindtorch.int8, mindtorch.int16, mindtorch.int32, mindtorch.int64]


def get_all_fp_dtypes(include_half=True, include_bfloat16=True) -> list[mindtorch.dtype]:
    dtypes = [mindtorch.float32, mindtorch.float64]
    if include_half:
        dtypes.append(mindtorch.float16)
    if include_bfloat16:
        dtypes.append(mindtorch.bfloat16)
    return dtypes


def get_all_qint_dtypes() -> list[mindtorch.dtype]:
    return [mindtorch.qint8, mindtorch.quint8, mindtorch.qint32, mindtorch.quint4x2, mindtorch.quint2x4]


float_to_corresponding_complex_type_map = {
    mindtorch.float16: mindtorch.complex32,
    mindtorch.float32: mindtorch.complex64,
    mindtorch.float64: mindtorch.complex128,
}