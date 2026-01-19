import dataclasses

@dataclasses.dataclass
class Configuration:
  debug_print_each_op: bool = False
  debug_accuracy_for_each_op: bool = False
  debug_mixed_tensor: bool = False
  debug_print_each_op_operands: bool = False

  use_int32_for_index: bool = False

  # normally, math between CPU torch.Tensor with torch4ms.Tensor is not
  # allowed. However, if that torch.Tensor happens to be scalar, then we
  # can use scalar * tensor math to handle it
  allow_mixed_math_with_scalar_tensor: bool = True

  # If true, we will convert Views into torch4ms.Tensors eagerly
  force_materialize_views: bool = False

  # Use DLPack for converting mindspore.Tensor <-> and torch.Tensor
  # Currently disabled as MindSpore dlpack module is not available
  use_dlpack_for_data_conversion: bool = False

  # device
  treat_cuda_as_mindspore_device: bool = True
  internal_respect_torch_return_dtypes: bool = False
