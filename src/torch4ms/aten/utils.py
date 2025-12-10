import mindspore
import torch

PT2MS_DTYPE_MAP = {
    torch.bool: mindspore.bool_,
    torch.int8: mindspore.int8,
    torch.int16: mindspore.int16,
    torch.int32: mindspore.int32,
    torch.int64: mindspore.int64,
    torch.long: mindspore.int64,
    torch.uint8: mindspore.uint8,
    torch.uint16: mindspore.uint16,
    torch.uint32: mindspore.uint32,
    torch.uint64: mindspore.uint64,
    torch.float8_e4m3fn: mindspore.float8_e4m3fn,
    torch.float8_e5m2: mindspore.float8_e5m2,
    torch.bfloat16: mindspore.bfloat16,
    torch.half: mindspore.float16,
    torch.float16: mindspore.float16,
    torch.float32: mindspore.float32,
    torch.float64: mindspore.float64,
    torch.double: mindspore.double,
    torch.complex64: mindspore.complex64,
    torch.complex128: mindspore.complex128,
    None: None,
}

MS2PT_DTYPE_MAP = {v: k for k, v in PT2MS_DTYPE_MAP.items()}

def ms2pt_dtype(ms_dtype):
    return MS2PT_DTYPE_MAP[ms_dtype]

def pt2ms_dtype(pt_dtype):
    return PT2MS_DTYPE_MAP[pt_dtype]
