"""
Utilities for exporting a torch program to mindspore.
"""
import copy
from typing import Any, Dict, Tuple
import torch
from torch.utils import _pytree as pytree
import torch4ms
from torch4ms import tensor
from torch4ms import ops_registry, mappings
from torch4ms import decompositions
import mindspore as ms
import mindspore.numpy as mnp

DEBUG = False


class MsInterpreter(torch.fx.Interpreter):
  """
  Interpreter that runs a torch.fx.GraphModule using MindSpore operations.
  """

  def __init__(self, graph_module):
    super().__init__(graph_module)
    # Import MindSpore ops modules
    import torch4ms.ops.maten
  import torch4ms.ops.mtorch

  def call_function(self, target, args: Tuple, kwargs: Dict) -> Any:
    if not isinstance(target,
                      (torch._ops.OpOverloadPacket, torch._ops.OpOverload)):
      return super().call_function(target, args, kwargs)

    if DEBUG:
      print('Running ', target.name(), '--------')

    # Get the operator from registry
    op = ops_registry.all_aten_ops.get(target)
    if op is None:
      op = ops_registry.all_aten_ops.get(target.overloadpacket)
    
    assert op is not None, f"No operator found for {target}"
    
    # Use the function directly (we've already registered MindSpore implementations)
    return op.func(*args, **kwargs)

  def run_node(self, n) -> Any:
    res = super().run_node(n)
    if DEBUG:
      if n.op == 'call_function':
        if hasattr(res, 'shape'):
          print('Meta:', n.meta.get('val').shape, 'REAL: ', res.shape)
    return res


from torch._decomp import get_decompositions
import torch._refs

_extra_decomp = get_decompositions([torch.ops.aten.unfold])


def _extract_states_from_exported_program(exported_model):
  """
  Extract parameters and buffers from an exported program.
  
  Args:
    exported_model: Exported PyTorch model
    
  Returns:
    Tuple of (parameter/buffer names, parameter/buffer values)
  """
  # NOTE call convention: (parameters, buffers, user_inputs)
  param_and_buffer_keys = exported_model.graph_signature.parameters + exported_model.graph_signature.buffers
  state_dict = copy.copy(exported_model.state_dict)
  if (constants := getattr(exported_model, 'constants', None)) is not None:
    state_dict.update(constants)
  param_buffer_values = list(state_dict[key] for key in param_and_buffer_keys)

  if hasattr(exported_model.graph_signature, "lifted_tensor_constants"):
    for name in exported_model.graph_signature.lifted_tensor_constants:
      param_buffer_values.append(exported_model.tensor_constants[name])

  return param_and_buffer_keys, param_buffer_values


def exported_program_to_ms(exported_program, export_raw: bool = False):
  """
  Convert an exported PyTorch program to MindSpore.
  
  Args:
    exported_program: PyTorch exported program
    export_raw: Whether to export raw values or convert to MindSpore tensors
    
  Returns:
    If export_raw is True: (names, states, func)
    If export_raw is False: (states, func)
  """
  if torch.__version__ >= '2.2':
    # torch version 2.1 didn't expose this yet
    exported_program = exported_program.run_decompositions()
    exported_program = exported_program.run_decompositions(
        decompositions.DECOMPOSITIONS)
  if DEBUG:
    print(exported_program.graph_module.code)

  names, states = _extract_states_from_exported_program(exported_program)

  def _extract_args(args, kwargs):
    flat_args, received_spec = pytree.tree_flatten(
        (args, kwargs))  # type: ignore[possibly-undefined]
    return flat_args

  num_mutations = len(exported_program.graph_signature.buffers_to_mutate)

  def func(states, inputs):
    args = _extract_args(inputs, {})
    res = MsInterpreter(exported_program.graph_module).run(
        *states,
        *args,
        enable_io_processing=False,
    )
    res = res[num_mutations:]
    return res

  if export_raw:
    return names, states, func
  
  env = torch4ms.default_env()
  states = env.t2ms_copy(states)
  return states, func


def extract_tensor_types(exported):
  """
  Return MindSpore tensor types for all input parameters of the exported program.
  This supports dynamic batch dimensions.
  
  Args:
    exported: Exported PyTorch program
    
  Returns:
    List of MindSpore tensor type descriptions
  """

  def _to_tensor_type(arg_meta):
    """
    Convert from torch type to MindSpore tensor type description
    """
    val = arg_meta['val']
    is_scalar = isinstance(val, float) or isinstance(val, int) or isinstance(
        val, bool)
    if is_scalar:
      return {'shape': (), 'dtype': type(arg_meta['val'])}

    tensor_meta = arg_meta['tensor_meta']
    shape = list(tensor_meta.shape)
    dtype = mappings.t2ms_dtype(tensor_meta.dtype)
    
    return {'shape': shape, 'dtype': dtype}

  def _get_inputs(exported):
    """
    Return placeholders with input metadata
    """
    placeholders = [p for p in exported.graph.nodes if p.op == "placeholder"]
    input_placeholders = [
        p for p, s in zip(placeholders, exported.graph_signature.input_specs)
        if s.kind == torch.export.graph_signature.InputKind.USER_INPUT
    ]
    return input_placeholders

  args = _get_inputs(exported)

  if DEBUG:
    print('Inputs to tensor type:', args, '--------')
    for arg in args:
      print('Meta2TensorType', arg.meta, '--> ', _to_tensor_type(arg.meta))

  return [_to_tensor_type(arg.meta) for arg in args]


def create_ms_model_from_exported_program(exported_program):
  """
  Create a MindSpore model from an exported PyTorch program.
  
  Args:
    exported_program: PyTorch exported program
    
  Returns:
    Tuple of (weights, ms_model) where weights is a dictionary of parameters
    and ms_model is a callable MindSpore model
  """
  weights, func = exported_program_to_ms(exported_program)
  tensor_types = extract_tensor_types(exported_program)
  
  # Create a simple MindSpore model that wraps the function
  class WrappedModel:
    def __init__(self, weights, func):
      self.weights = weights
      self.func = func
    
    def __call__(self, *inputs):
      # Combine inputs into a tuple
      inputs_tuple = inputs if len(inputs) > 1 else inputs[0]
      return self.func(self.weights, inputs_tuple)
  
  ms_model = WrappedModel(weights, func)
  return weights, ms_model


def export_to_ms_script(exported_program, filename=None):
  """
  Export an exported PyTorch program to MindSpore script.
  
  Args:
    exported_program: PyTorch exported program
    filename: Optional filename to save the script to
    
  Returns:
    Generated MindSpore script as a string
  """
  # Extract information from the exported program
  names, states, func = exported_program_to_ms(exported_program, export_raw=True)
  tensor_types = extract_tensor_types(exported_program)
  
  # Generate script code
  script = [
      "import mindspore as ms",
      "import mindspore.numpy as mnp",
      "from mindspore import ops",
      "",
      "# Model generated from torch.export",
      ""
  ]
  
  # Add parameter definitions
  script.append("# Model parameters")
  for name, state in zip(names, states):
    if isinstance(state, torch.Tensor):
      # Convert PyTorch tensor to MindSpore tensor initialization code
      shape_str = str(tuple(state.shape))
      dtype_str = mappings.t2ms_dtype(state.dtype).__name__
      # For simplicity, we'll just create a tensor with zeros of the same shape and dtype
      # In a real implementation, you would save the actual weights
      script.append(f"{name} = mnp.zeros({shape_str}, dtype={dtype_str})")
  
  # Add model function
  script.append("\n# Model function")
  script.append("def model_function(parameters, inputs):")
  script.append("    # This is a placeholder for the actual model implementation")
  script.append("    # In a real implementation, this would contain the translated operations")
  script.append("    return inputs")  # Simple placeholder
  
  # Combine script lines
  script_str = '\n'.join(script)
  
  # Save to file if filename is provided
  if filename:
    with open(filename, 'w') as f:
      f.write(script_str)
    print(f"Model script saved to {filename}")
  
  return script_str