"""MindTorchV2Loader - Returns mindtorch_v2 modules for torch imports."""

import sys
import importlib
from types import ModuleType


class MindTorchV2Loader:
    """Module loader that returns mindtorch_v2 or stubs for torch.* imports.

    Tier 1 (Real): torch -> mindtorch_v2, torch.nn -> mindtorch_v2.nn, etc.
    Tier 2 (Functional stubs): torch.cuda, torch.jit, etc.
    Tier 3 (Import stubs): torch._C, torch._dynamo, etc.

    Also intercepts mindtorch imports and redirects to mindtorch_v2.
    """

    # Tier 1: Real implementations - these map to actual mindtorch_v2 modules
    REAL_MODULES = {
        'torch': 'mindtorch_v2',
        'torch.nn': 'mindtorch_v2.nn',
        'torch.nn.functional': 'mindtorch_v2.nn.functional',
        'torch.nn.modules': 'mindtorch_v2.nn.modules',
        'torch.nn.parameter': 'mindtorch_v2.nn.parameter',
        'torch.optim': 'mindtorch_v2.optim',
        'torch.optim.lr_scheduler': 'mindtorch_v2.optim.lr_scheduler',
        'torch.autograd': 'mindtorch_v2._autograd',
        'torch.autograd.function': 'mindtorch_v2._autograd',  # Function class is in _autograd
        # Mindtorch -> mindtorch_v2 redirects (for safetensors patch compatibility)
        'mindtorch': 'mindtorch_v2',
        'mindtorch.nn': 'mindtorch_v2.nn',
        'mindtorch.nn.functional': 'mindtorch_v2.nn.functional',
        'mindtorch.nn.modules': 'mindtorch_v2.nn.modules',
        'mindtorch.nn.parameter': 'mindtorch_v2.nn.parameter',
        'mindtorch.optim': 'mindtorch_v2.optim',
        'mindtorch.autograd': 'mindtorch_v2._autograd',
        'mindtorch.autograd.function': 'mindtorch_v2._autograd',  # Function class is in _autograd
    }

    # Tier 2 & 3: Stub modules - these return stub implementations
    STUB_MODULES = {
        'torch._C': 'mindtorch_v2._torch_proxy.stubs._C',
        'torch.cuda': 'mindtorch_v2._torch_proxy.stubs.cuda',
        'torch.jit': 'mindtorch_v2._torch_proxy.stubs.jit',
        'torch.jit.annotations': 'mindtorch_v2._torch_proxy.stubs.jit.annotations',
        'torch.hub': 'mindtorch_v2._torch_proxy.stubs.hub',
        'torch.ops': 'mindtorch_v2._torch_proxy.stubs.ops',
        'torch.library': 'mindtorch_v2._torch_proxy.stubs.library',
        'torch.onnx': 'mindtorch_v2._torch_proxy.stubs.onnx',
        'torch.onnx.symbolic_helper': 'mindtorch_v2._torch_proxy.stubs.onnx.symbolic_helper',
        'torch.onnx.symbolic_opset11': 'mindtorch_v2._torch_proxy.stubs.onnx',
        'torch.backends': 'mindtorch_v2._torch_proxy.stubs.backends',
        'torch.backends.cuda': 'mindtorch_v2._torch_proxy.stubs.backends',
        'torch.backends.cudnn': 'mindtorch_v2._torch_proxy.stubs.backends',
        'torch.distributed': 'mindtorch_v2._torch_proxy.stubs.distributed',
        'torch.distributed.tensor': 'mindtorch_v2._torch_proxy.stubs.distributed.tensor',
        'torch.distributed.algorithms': 'mindtorch_v2._torch_proxy.stubs.distributed.algorithms',
        'torch.distributed.algorithms.join': 'mindtorch_v2._torch_proxy.stubs.distributed.algorithms.join',
        'torch.distributions': 'mindtorch_v2._torch_proxy.stubs.distributions',
        'torch.utils': 'mindtorch_v2._torch_proxy.stubs.utils',
        'torch.utils._pytree': 'mindtorch_v2._torch_proxy.stubs.utils._pytree',
        'torch.utils.data': 'mindtorch_v2._torch_proxy.stubs.utils.data',
        'torch.utils.data.distributed': 'mindtorch_v2._torch_proxy.stubs.utils.data.distributed',
        'torch.amp': 'mindtorch_v2._torch_proxy.stubs.amp',
        'torch.compiler': 'mindtorch_v2._torch_proxy.stubs.compiler',
        'torch.fx': 'mindtorch_v2._torch_proxy.stubs.fx',
        'torch.fx._compatibility': 'mindtorch_v2._torch_proxy.stubs.fx._compatibility',
        'torch.fx.node': 'mindtorch_v2._torch_proxy.stubs.fx.node',
        'torch.version': 'mindtorch_v2._torch_proxy.stubs.version',
        'torch.profiler': 'mindtorch_v2._torch_proxy.stubs.profiler',
        'torch._dynamo': 'mindtorch_v2._dynamo',
        'torch._dynamo.eval_frame': 'mindtorch_v2._dynamo.eval_frame',
        'torch.nn.parallel': 'mindtorch_v2.nn.parallel',
    }

    def load_module(self, fullname):
        """Load a torch module, returning mindtorch_v2 equivalent or stub.

        Args:
            fullname: Full module name (e.g., 'torch.nn')

        Returns:
            The loaded module
        """
        # Check if already loaded
        if fullname in sys.modules:
            return sys.modules[fullname]

        # Tier 1: Real mindtorch_v2 modules
        if fullname in self.REAL_MODULES:
            real_name = self.REAL_MODULES[fullname]
            module = importlib.import_module(real_name)
            # Add torch-specific aliases if needed
            if fullname == 'torch':
                module = self._enhance_torch_module(module)
            sys.modules[fullname] = module
            return module

        # Tier 2 & 3: Stub modules
        if fullname in self.STUB_MODULES:
            stub_name = self.STUB_MODULES[fullname]
            try:
                module = importlib.import_module(stub_name)
            except ImportError:
                # Create empty stub if not found
                module = self._create_empty_stub(fullname)
            sys.modules[fullname] = module
            return module

        # Check if it's a submodule of a known module
        for prefix in self.REAL_MODULES:
            if fullname.startswith(prefix + '.'):
                # Try to import as submodule of real module
                real_prefix = self.REAL_MODULES[prefix]
                real_name = real_prefix + fullname[len(prefix):]
                try:
                    module = importlib.import_module(real_name)
                    sys.modules[fullname] = module
                    return module
                except ImportError:
                    pass

        # Default: create empty stub for unknown torch.* modules
        module = self._create_empty_stub(fullname)
        sys.modules[fullname] = module
        return module

    def create_module(self, spec):
        """Create module for ModuleSpec-based loading."""
        # Return the actual module directly instead of letting Python create empty one
        return self.load_module(spec.name)

    def exec_module(self, module):
        """Execute module for ModuleSpec-based loading."""
        # Module is already fully loaded by create_module, nothing to do
        pass

    def _enhance_torch_module(self, module):
        """Add torch-specific attributes to the main torch module."""
        import numpy as np

        # Add version info
        module.__version__ = "2.0.0"

        # Add submodules that should be accessible via torch.xxx
        # Import stubs and add them as attributes
        from .stubs import cuda, jit, backends, distributed, amp, version, profiler, hub, ops, library, onnx, compiler, fx, distributions
        from .stubs._C import _C  # Import the _C instance, not the module
        from .. import _dynamo
        module.cuda = cuda
        module.jit = jit
        module.backends = backends
        module.distributed = distributed
        module.distributions = distributions
        module.amp = amp
        module.version = version
        module.profiler = profiler
        module.hub = hub
        module.ops = ops
        module.library = library
        module.onnx = onnx
        module.compiler = compiler
        module.fx = fx
        module._dynamo = _dynamo
        module._C = _C

        # Also register in sys.modules so direct imports work
        import sys
        sys.modules['torch.cuda'] = cuda
        sys.modules['torch.jit'] = jit
        sys.modules['torch.backends'] = backends
        sys.modules['torch.distributed'] = distributed
        sys.modules['torch.distributions'] = distributions
        sys.modules['torch.amp'] = amp
        sys.modules['torch.version'] = version
        sys.modules['torch.profiler'] = profiler
        sys.modules['torch.hub'] = hub
        sys.modules['torch.ops'] = ops
        sys.modules['torch.library'] = library
        sys.modules['torch.onnx'] = onnx
        sys.modules['torch.compiler'] = compiler
        sys.modules['torch.fx'] = fx
        sys.modules['torch._C'] = _C  # Register _C for direct import access
        sys.modules['torch._dynamo'] = _dynamo
        sys.modules['torch._dynamo.eval_frame'] = _dynamo.eval_frame

        # Add FloatTensor / LongTensor type classes with isinstance support
        # These come from the mindtorch_v2 module which has the metaclass versions
        # The module already has BoolTensor, FloatTensor, etc. defined with _TensorTypeMeta
        # so we just need to ensure they're accessible (they already are)

        # Add from_numpy
        def from_numpy(arr):
            return module.Tensor(arr)
        module.from_numpy = from_numpy

        # Add einsum
        def einsum(equation, *operands):
            np_operands = [op.numpy() if hasattr(op, 'numpy') else np.asarray(op) for op in operands]
            result = np.einsum(equation, *np_operands)
            return module.Tensor(result.astype(np.float32))
        module.einsum = einsum

        # Add compile (no-op) - supports both @compile and @compile(options)
        def compile(fn=None, *args, **kwargs):
            if fn is not None:
                # Called as @compile without arguments
                return fn
            # Called as @compile(options) - return a decorator
            def decorator(f):
                return f
            return decorator
        module.compile = compile

        # Add Size as tuple alias
        module.Size = tuple

        # Add dtype as alias for DType (safetensors uses torch.dtype)
        module.dtype = module.DType

        # Add finfo/iinfo
        class finfo:
            def __init__(self, dtype):
                if dtype in (module.float32, getattr(module, 'float', None)):
                    self.min = np.finfo(np.float32).min
                    self.max = np.finfo(np.float32).max
                    self.eps = np.finfo(np.float32).eps
                    self.tiny = np.finfo(np.float32).tiny
                elif dtype in (module.float64, getattr(module, 'double', None)):
                    self.min = np.finfo(np.float64).min
                    self.max = np.finfo(np.float64).max
                    self.eps = np.finfo(np.float64).eps
                    self.tiny = np.finfo(np.float64).tiny
                elif dtype in (module.float16, getattr(module, 'half', None)):
                    self.min = np.finfo(np.float16).min
                    self.max = np.finfo(np.float16).max
                    self.eps = np.finfo(np.float16).eps
                    self.tiny = np.finfo(np.float16).tiny

        class iinfo:
            def __init__(self, dtype):
                if dtype in (module.int32, getattr(module, 'int', None)):
                    self.min = np.iinfo(np.int32).min
                    self.max = np.iinfo(np.int32).max
                elif dtype in (module.int64, getattr(module, 'long', None)):
                    self.min = np.iinfo(np.int64).min
                    self.max = np.iinfo(np.int64).max

        module.finfo = finfo
        module.iinfo = iinfo

        # Add device function
        if not hasattr(module, 'device'):
            module.device = module._device.device

        # Add manual_seed
        def manual_seed(seed):
            np.random.seed(seed)
        module.manual_seed = manual_seed

        # Add triu/tril/diag
        def triu(input, diagonal=0):
            return module.Tensor(np.triu(input.numpy(), k=diagonal))
        def tril(input, diagonal=0):
            return module.Tensor(np.tril(input.numpy(), k=diagonal))
        def diag(input, diagonal=0):
            arr = input.numpy()
            if arr.ndim == 1:
                return module.Tensor(np.diag(arr, k=diagonal))
            else:
                return module.Tensor(np.diag(arr, k=diagonal))

        module.triu = triu
        module.tril = tril
        module.diag = diag

        # Add is_tensor
        def is_tensor(obj):
            return isinstance(obj, module.Tensor)
        module.is_tensor = is_tensor

        # Add inference_mode (context manager, same as no_grad for now)
        module.inference_mode = module.no_grad

        # Add default device management
        from .._device import _get_default_device, _set_default_device

        def get_default_device():
            """Get the current default device.

            Returns the device from the device context manager if active,
            otherwise returns cpu.
            """
            ctx_device = _get_default_device()
            if ctx_device is not None:
                return ctx_device
            return module.device('cpu')

        def set_default_device(device):
            """Set the default device."""
            if device is None:
                _set_default_device(None)
            elif isinstance(device, str):
                _set_default_device(module.device(device))
            else:
                _set_default_device(device)

        module.get_default_device = get_default_device
        module.set_default_device = set_default_device

        # Add Generator class stub
        class Generator:
            """Random number generator stub."""
            def __init__(self, device='cpu'):
                self._device = device
                self._state = np.random.RandomState()

            def manual_seed(self, seed):
                self._state.seed(seed)
                return self

            def get_state(self):
                return self._state.get_state()

            def set_state(self, state):
                self._state.set_state(state)

            @property
            def device(self):
                return module.device(self._device)

        module.Generator = Generator

        # Add default_generator
        module.default_generator = Generator()

        # Add get_rng_state / set_rng_state
        def get_rng_state():
            return np.random.get_state()

        def set_rng_state(state):
            np.random.set_state(state)

        module.get_rng_state = get_rng_state
        module.set_rng_state = set_rng_state

        # Add save/load functions
        import pickle

        def save(obj, f, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True):
            """Save object to file."""
            if isinstance(f, str):
                with open(f, 'wb') as fp:
                    pickle_module.dump(obj, fp, protocol=pickle_protocol)
            else:
                pickle_module.dump(obj, f, protocol=pickle_protocol)

        def load(f, map_location=None, pickle_module=pickle, *, weights_only=False, **pickle_load_args):
            """Load object from file."""
            if isinstance(f, str):
                with open(f, 'rb') as fp:
                    return pickle_module.load(fp, **pickle_load_args)
            return pickle_module.load(f, **pickle_load_args)

        module.save = save
        module.load = load

        # Add serialization namespace
        class serialization:
            """Serialization utilities."""
            @staticmethod
            def default_restore_location(storage, location):
                return storage

            @staticmethod
            def register_package(priority, tagger, deserializer):
                pass

        module.serialization = serialization

        # Add is_floating_point
        def is_floating_point(input):
            """Check if tensor has floating point dtype."""
            if hasattr(input, 'dtype'):
                from mindtorch_v2 import _dtype as dt
                return input.dtype in (dt.float16, dt.float32, dt.float64, dt.bfloat16)
            return False
        module.is_floating_point = is_floating_point

        # Add allclose
        def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
            """Check if two tensors are element-wise close."""
            a = input.numpy() if hasattr(input, 'numpy') else np.asarray(input)
            b = other.numpy() if hasattr(other, 'numpy') else np.asarray(other)
            return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))
        module.allclose = allclose

        # Add testing module
        class _testing:
            """torch.testing compatibility module."""
            @staticmethod
            def assert_close(actual, expected, rtol=None, atol=None, **kwargs):
                a = actual.numpy() if hasattr(actual, 'numpy') else np.asarray(actual)
                b = expected.numpy() if hasattr(expected, 'numpy') else np.asarray(expected)
                if rtol is None:
                    rtol = 1e-5
                if atol is None:
                    atol = 1e-8
                if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                    max_diff = np.max(np.abs(a - b))
                    raise AssertionError(
                        f"Tensors are not close!\n"
                        f"Max absolute difference: {max_diff}\n"
                        f"Tolerances: rtol={rtol}, atol={atol}"
                    )

            @staticmethod
            def assert_allclose(actual, desired, rtol=1e-7, atol=0, **kwargs):
                a = actual.numpy() if hasattr(actual, 'numpy') else np.asarray(actual)
                b = desired.numpy() if hasattr(desired, 'numpy') else np.asarray(desired)
                np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

        module.testing = _testing
        sys.modules['torch.testing'] = _testing

        # Add max with out param and no-dim variant
        _orig_max = module.max
        def _enhanced_max(input, dim=None, keepdim=False, *, out=None, other=None):
            if other is not None:
                # torch.max(input, other) -> element-wise max
                a = input.numpy() if hasattr(input, 'numpy') else np.asarray(input)
                b = other.numpy() if hasattr(other, 'numpy') else np.asarray(other)
                return module.Tensor(np.maximum(a, b))
            # If dim is a Tensor, treat it as element-wise max: torch.max(a, b)
            if isinstance(dim, module.Tensor):
                a = input.numpy() if hasattr(input, 'numpy') else np.asarray(input)
                b = dim.numpy()
                return module.Tensor(np.maximum(a, b))
            return _orig_max(input, dim=dim, keepdim=keepdim)
        module.max = _enhanced_max

        # Add abs at module level (already in _functional, but ensure it's there)
        if not hasattr(module, 'abs') or module.abs is None:
            def _abs(input):
                return module.Tensor(np.abs(input.numpy()))
            module.abs = _abs

        # Add is_complex
        def is_complex(input):
            if hasattr(input, 'dtype'):
                from mindtorch_v2 import _dtype as dt
                return input.dtype in (dt.complex64, dt.complex128)
            return False
        module.is_complex = is_complex

        # Add is_nonzero
        def is_nonzero(input):
            return bool(input.item() != 0)
        module.is_nonzero = is_nonzero

        return module

    def _create_empty_stub(self, fullname):
        """Create an empty stub module for unknown torch.* imports."""
        module = ModuleType(fullname)
        module.__file__ = f"<mindtorch_v2 stub for {fullname}>"
        module.__loader__ = self
        module.__package__ = fullname.rsplit('.', 1)[0] if '.' in fullname else fullname

        # Make it return None for any attribute access
        class StubModule(ModuleType):
            def __getattr__(self, name):
                # Return a callable that returns None for most things
                if name.startswith('_'):
                    return None
                # Return another stub for submodule access
                subname = f"{fullname}.{name}"
                if subname not in sys.modules:
                    sys.modules[subname] = self.__class__(subname)
                return sys.modules[subname]

            def __call__(self, *args, **kwargs):
                return None

        return StubModule(fullname)
