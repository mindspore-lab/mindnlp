import mindspore as ms
import numpy as np
import torch
import torch.func
import torch.utils._mode_utils as mode_utils

NUMPY_UNSUPPORTED_DTYPES = {
    torch.bfloat16: ms.bfloat16,
    torch.float8_e4m3fn: ms.float8_e4m3fn,
    torch.float8_e5m2: ms.float8_e5m2,
    # 移除当前MindSpore版本不支持的类型
    # torch.float8_e4m3fnuz: ms.float8_e4m3fnuz,
    # torch.float8_e5m2fnuz: ms.float8_e5m2fnuz,
}

def t2ms(t, use_dlpack=True):
 is_bool = False
 if t.dtype == torch.bool:
  is_bool = True
  t = t.to(torch.int8)

 t = t.to_dense()

 if not t.is_contiguous():
  t = t.contiguous()

 # 处理特殊类型，包括那些不在NUMPY_UNSUPPORTED_DTYPES中的float8变体
 is_special_float8 = (t.dtype == torch.float8_e4m3fnuz or 
                     t.dtype == torch.float8_e5m2fnuz)

 # 直接使用numpy作为中间层，不使用dlpack
 if t.dtype in NUMPY_UNSUPPORTED_DTYPES or is_special_float8:
  # 对于不支持的float8类型，先转换为float32
  nparray = t.cpu().detach().to(torch.float32).numpy()
 else:
  nparray = t.cpu().detach().numpy()

 res = ms.Tensor(nparray)
 
 # 只对支持的类型进行转换
 if t.dtype in NUMPY_UNSUPPORTED_DTYPES:
  res = res.astype(NUMPY_UNSUPPORTED_DTYPES[t.dtype])
 # 对于不支持的float8类型，保持为float32

 if is_bool:
  res = res.astype(ms.bool_)
 return res

def ms2t(x, use_dlpack=True):
 with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
  orig_dtype = None
  if x.dtype == ms.bfloat16:
   orig_dtype = x.dtype
   x = x.astype(ms.float32)
  
  # 直接使用numpy作为中间层，不使用dlpack
  res = torch.from_numpy(np.asarray(x))

  if x.dtype == ms.bool_:
   res = res.to(torch.bool)

  if orig_dtype is not None:
   res = res.to(ms2t_dtype(orig_dtype))
  return res


TORCH_DTYPE_TO_MINDSPORE = {
 # NO_MAPPING        : ms.float0 (signless scalar int),
 torch.bool:
  ms.bool_,
 # 当前MindSpore版本不支持int4类型
 # NO_MAPPING        : ms.int4,
  torch.int8:
   ms.int8,
  torch.int16:
   ms.int16,
  torch.int32:
   ms.int32,
  torch.int64:
   ms.int64,
  torch.long:
   ms.int64,
  # 当前MindSpore版本不支持uint4类型
  # NO_MAPPING        : ms.uint4,
  torch.uint8:
   ms.uint8,
  torch.uint16:
   ms.uint16,
  torch.uint32:
   ms.uint32,
  torch.uint64:
   ms.uint64,
  # NO_MAPPING        : ms.bfloat16int,
  # NO_MAPPING        : ms.float8_e4m3b11fn,
  # NO_MAPPING        : ms.float8_e4m3b11fnuz,
  torch.float8_e4m3fn:
    ms.float8_e4m3fn,
   # 移除当前MindSpore版本不支持的类型
   # torch.float8_e4m3fnuz:
   #   ms.float8_e4m3fnuz,
    torch.float8_e5m2:
     ms.float8_e5m2,
   # 移除当前MindSpore版本不支持的类型
   # torch.float8_e5m2fnuz:
   #   ms.float8_e5m2fnuz,
    torch.bfloat16:
     ms.bfloat16,
  torch.half:
   ms.float16,
  torch.float16:
   ms.float16,
  torch.float32:
   ms.float32,
  torch.float64:
   ms.float64,
  torch.double:
   ms.float64,
  torch.complex64:
   ms.complex64,
  torch.complex128:
   ms.complex128,
  None:
   None,
}

MINDSPORE_DTYPE_TO_TORCH = {value: key for key, value in TORCH_DTYPE_TO_MINDSPORE.items()}
# Add imprecise mappings for some MindSpore dtypes which don't have torch analogues
# 当前MindSpore版本不支持int4和uint4类型，注释掉这些映射
# MINDSPORE_DTYPE_TO_TORCH[ms.int4] = torch.int8
# MINDSPORE_DTYPE_TO_TORCH[ms.uint4] = torch.uint8


def t2ms_dtype(dtype):
  if dtype not in TORCH_DTYPE_TO_MINDSPORE:
    raise RuntimeError(
        f'Attempting to convert unknown type: {dtype} to mindspore type,')
  return TORCH_DTYPE_TO_MINDSPORE[dtype]


def ms2t_dtype(dtype):
  if dtype not in MINDSPORE_DTYPE_TO_TORCH:
    raise RuntimeError(
        f'Attempting to convert unknown type: {dtype} to torch type,')
  return MINDSPORE_DTYPE_TO_TORCH[dtype]
