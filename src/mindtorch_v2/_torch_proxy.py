"""Torch proxy that redirects `import torch` to mindtorch_v2.

Uses a MetaPathFinder so that `importlib.util.find_spec("torch")` succeeds
(required by transformers v5+ which probes torch availability that way).
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import sys
import types

_TORCH_VERSION = "2.7.1+dev"

# ─── submodule mapping ───────────────────────────────────────────────
# torch.xxx → mindtorch_v2.yyy   (only when names differ)
_SUBMODULE_ALIASES = {
    "torch.autograd": "mindtorch_v2._autograd",
    "torch.random": "mindtorch_v2._random",
}

# Submodules that are implemented as stub modules (created at install time)
_STUB_MODULES = set()  # populated by _create_stub_modules()


# ─── MetaPathFinder + Loader ─────────────────────────────────────────

class _MindTorchV2Loader(importlib.abc.Loader):
    """Loader that returns the already-imported mindtorch_v2 (sub)module."""

    def __init__(self, real_module):
        self._module = real_module

    def create_module(self, spec):
        return self._module

    def exec_module(self, module):
        pass  # module is already fully initialised


class _MindTorchV2Finder(importlib.abc.MetaPathFinder):
    """Intercepts ``import torch`` / ``import torch.*`` and resolves them
    to the corresponding mindtorch_v2 module."""

    def find_module(self, fullname, path=None):
        """Python 3.3 legacy hook — still consulted by some code."""
        if fullname == "torch" or fullname.startswith("torch."):
            return self
        return None

    def find_spec(self, fullname, path, target=None):
        if fullname == "torch":
            mod = sys.modules.get("torch")
            if mod is None:
                return None
            return importlib.machinery.ModuleSpec(
                "torch",
                _MindTorchV2Loader(mod),
                is_package=True,
            )
        if fullname.startswith("torch."):
            mod = self._resolve(fullname)
            if mod is not None:
                is_pkg = hasattr(mod, "__path__")
                spec = importlib.machinery.ModuleSpec(
                    fullname,
                    _MindTorchV2Loader(mod),
                    is_package=is_pkg,
                )
                return spec
            # Unknown submodule → auto-stub so ImportError is avoided
            stub = _make_stub(fullname)
            sys.modules[fullname] = stub
            # Set as attribute on parent so `parent.child` access works
            parts = fullname.split(".")
            if len(parts) >= 2:
                parent_name = ".".join(parts[:-1])
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    setattr(parent, parts[-1], stub)
            return importlib.machinery.ModuleSpec(
                fullname,
                _MindTorchV2Loader(stub),
                is_package=True,
            )
        return None

    def load_module(self, fullname):
        """Legacy loader interface."""
        if fullname in sys.modules:
            return sys.modules[fullname]
        spec = self.find_spec(fullname, None)
        if spec and spec.loader:
            mod = spec.loader.create_module(spec)
            sys.modules[fullname] = mod
            return mod
        raise ImportError(fullname)

    # ── internal ──

    @staticmethod
    def _resolve(torch_name):
        """Return the real module for *torch_name*, or None."""
        # Already in sys.modules?
        if torch_name in sys.modules:
            return sys.modules[torch_name]

        # Explicit alias?
        real_name = _SUBMODULE_ALIASES.get(torch_name)
        if real_name:
            mod = sys.modules.get(real_name)
            if mod is None:
                try:
                    mod = importlib.import_module(real_name)
                except ImportError:
                    return None
            sys.modules[torch_name] = mod
            return mod

        # Check if any alias is a prefix of torch_name
        # e.g. torch.autograd → mindtorch_v2._autograd means
        #      torch.autograd.function → mindtorch_v2._autograd.function
        for alias_from, alias_to in _SUBMODULE_ALIASES.items():
            if torch_name.startswith(alias_from + "."):
                remainder = torch_name[len(alias_from):]
                real_name = alias_to + remainder
                mod = sys.modules.get(real_name)
                if mod is not None:
                    sys.modules[torch_name] = mod
                    return mod
                try:
                    mod = importlib.import_module(real_name)
                    sys.modules[torch_name] = mod
                    return mod
                except ImportError:
                    pass

        # Default: torch.xxx → mindtorch_v2.xxx
        suffix = torch_name[len("torch"):]  # e.g. ".nn.functional"
        real_name = "mindtorch_v2" + suffix
        mod = sys.modules.get(real_name)
        if mod is not None:
            sys.modules[torch_name] = mod
            return mod
        try:
            mod = importlib.import_module(real_name)
            sys.modules[torch_name] = mod
            return mod
        except ImportError:
            return None


# ─── stub helper ──────────────────────────────────────────────────────

def _make_stub(name, attrs=None):
    """Create a minimal stub module so ``import <name>`` does not crash."""
    mod = types.ModuleType(name)
    mod.__package__ = name
    mod.__path__ = []
    mod.__file__ = __file__
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod


# ─── concrete stub modules ───────────────────────────────────────────

def _create_stub_modules():
    """Build all stub modules that transformers imports unconditionally."""

    stubs = {}

    # --- torch.version ---
    ver = _make_stub("torch.version", {
        "__version__": _TORCH_VERSION,
        "cuda": None,
        "hip": None,
        "debug": False,
        "git_version": "unknown",
    })
    stubs["torch.version"] = ver

    # --- torch.cuda ---
    _cuda_props = type("CudaDeviceProperties", (), {
        "total_memory": 0, "major": 0, "minor": 0, "name": "",
        "multi_processor_count": 0, "is_integrated": False, "is_multi_gpu_board": False,
    })()
    cuda = _make_stub("torch.cuda", {
        "is_available": lambda: False,
        "device_count": lambda: 0,
        "current_device": lambda: -1,
        "get_device_name": lambda d=0: "",
        "get_device_capability": lambda d=0: (0, 0),
        "get_device_properties": lambda d=0: _cuda_props,
        "set_device": lambda d: None,
        "synchronize": lambda d=None: None,
        "is_bf16_supported": lambda: False,
        "manual_seed": lambda seed: None,
        "manual_seed_all": lambda seed: None,
        "empty_cache": lambda: None,
        "reset_max_memory_allocated": lambda d=None: None,
        "max_memory_allocated": lambda d=None: 0,
        "reset_peak_memory_stats": lambda d=None: None,
        "memory_allocated": lambda d=None: 0,
        "amp": _make_stub("torch.cuda.amp"),
    })
    cuda.amp.autocast = lambda *a, **kw: (lambda fn: fn)  # identity decorator
    cuda.amp.GradScaler = type("GradScaler", (), {
        "__init__": lambda self, *a, **kw: None,
        "scale": lambda self, loss: loss,
        "step": lambda self, opt: opt.step(),
        "update": lambda self: None,
    })
    stubs["torch.cuda"] = cuda
    stubs["torch.cuda.amp"] = cuda.amp

    # --- torch.backends (+ .mps, .cuda, .cudnn) ---
    mps = _make_stub("torch.backends.mps", {
        "is_available": lambda: False,
        "is_built": lambda: False,
    })
    bcuda = _make_stub("torch.backends.cuda", {
        "is_built": lambda: True,
        "matmul": _make_stub("torch.backends.cuda.matmul", {
            "allow_tf32": False,
        }),
    })
    cudnn = _make_stub("torch.backends.cudnn", {
        "is_available": lambda: False,
        "enabled": False,
        "benchmark": False,
        "deterministic": False,
        "allow_tf32": False,
        "version": lambda: 0,
    })
    mkl = _make_stub("torch.backends.mkl", {
        "is_available": lambda: False,
    })
    mkldnn = _make_stub("torch.backends.mkldnn", {
        "is_available": lambda: False,
    })
    openmp = _make_stub("torch.backends.openmp", {
        "is_available": lambda: False,
    })
    opt_einsum = _make_stub("torch.backends.opt_einsum", {
        "is_available": lambda: False,
        "enabled": False,
    })
    backends = _make_stub("torch.backends", {
        "mps": mps,
        "cuda": bcuda,
        "cudnn": cudnn,
        "mkl": mkl,
        "mkldnn": mkldnn,
        "openmp": openmp,
        "opt_einsum": opt_einsum,
    })
    stubs["torch.backends"] = backends
    stubs["torch.backends.mps"] = mps
    stubs["torch.backends.cuda"] = bcuda
    stubs["torch.backends.cuda.matmul"] = bcuda.matmul
    stubs["torch.backends.cudnn"] = cudnn
    stubs["torch.backends.mkl"] = mkl
    stubs["torch.backends.mkldnn"] = mkldnn
    stubs["torch.backends.openmp"] = openmp
    stubs["torch.backends.opt_einsum"] = opt_einsum

    # --- torch._dynamo ---
    def _identity_decorator(fn=None, **kw):
        if fn is not None:
            return fn
        return lambda f: f

    dynamo = _make_stub("torch._dynamo", {
        "allow_in_graph": _identity_decorator,
        "is_compiling": lambda: False,
        "disable": _identity_decorator,
        "optimize": _identity_decorator,
        "reset": lambda: None,
    })
    dynamo_utils = _make_stub("torch._dynamo.utils", {
        "is_compiling": lambda: False,
    })
    dynamo.utils = dynamo_utils

    class _OptimizedModule:
        """Stub for torch._dynamo.eval_frame.OptimizedModule."""

    dynamo_eval_frame = _make_stub("torch._dynamo.eval_frame", {
        "OptimizedModule": _OptimizedModule,
    })
    dynamo.eval_frame = dynamo_eval_frame

    stubs["torch._dynamo"] = dynamo
    stubs["torch._dynamo.utils"] = dynamo_utils
    stubs["torch._dynamo.eval_frame"] = dynamo_eval_frame

    # --- torch.compiler (extend existing mindtorch_v2.compiler) ---
    # The real compiler module is already in mindtorch_v2.compiler,
    # but we need is_compiling/is_dynamo_compiling
    try:
        comp = importlib.import_module("mindtorch_v2.compiler")
        if not hasattr(comp, "is_compiling"):
            comp.is_compiling = lambda: False
        if not hasattr(comp, "is_dynamo_compiling"):
            comp.is_dynamo_compiling = lambda: False
    except ImportError:
        pass

    # --- torch.distributions (+ constraints) ---
    class _Constraint:
        """Stub constraint that always passes check."""
        def check(self, value):
            from mindtorch_v2._creation import tensor
            return tensor(True)

    constraints = _make_stub("torch.distributions.constraints")
    constraints.positive_definite = _Constraint()
    constraints.real = _Constraint()
    constraints.positive = _Constraint()
    distributions = _make_stub("torch.distributions", {
        "constraints": constraints,
    })
    stubs["torch.distributions"] = distributions
    stubs["torch.distributions.constraints"] = constraints

    # --- torch.fx ---
    class _FxProxy:
        """Stub Proxy class for isinstance checks."""

    _fx_node = _make_stub("torch.fx.node", {
        "Target": str,
        "Argument": object,
        "Node": type("Node", (), {}),
        "map_arg": lambda a, fn: a,
    })
    _fx_proxy = _make_stub("torch.fx.proxy", {
        "Proxy": _FxProxy,
    })
    _fx_graph = _make_stub("torch.fx.graph", {
        "Graph": type("Graph", (), {}),
    })
    _fx_graph_module = _make_stub("torch.fx.graph_module", {
        "GraphModule": type("GraphModule", (), {}),
        "_CodeOnlyModule": type("_CodeOnlyModule", (), {}),
        "_copy_attr": lambda *a: None,
        "_USER_PRESERVED_ATTRIBUTES_KEY": "_user_preserved_attributes",
    })
    _fx_passes = _make_stub("torch.fx.passes", {})

    fx = _make_stub("torch.fx", {
        "Proxy": _FxProxy,
        "Graph": type("Graph", (), {}),
        "GraphModule": type("GraphModule", (), {}),
        "Node": type("Node", (), {}),
        "Tracer": type("Tracer", (), {}),
        "wrap": _identity_decorator,
        "node": _fx_node,
        "proxy": _fx_proxy,
        "graph": _fx_graph,
        "graph_module": _fx_graph_module,
        "passes": _fx_passes,
    })
    stubs["torch.fx"] = fx
    stubs["torch.fx.node"] = _fx_node
    stubs["torch.fx.proxy"] = _fx_proxy
    stubs["torch.fx.graph"] = _fx_graph
    stubs["torch.fx.graph_module"] = _fx_graph_module
    stubs["torch.fx.passes"] = _fx_passes

    # --- torch._subclasses ---
    class _FakeTensor:
        """Stub FakeTensor for isinstance checks."""

    subclasses = _make_stub("torch._subclasses", {
        "FakeTensor": _FakeTensor,
    })
    fake_tensor_mod = _make_stub("torch._subclasses.fake_tensor", {
        "FakeTensor": _FakeTensor,
    })
    subclasses.fake_tensor = fake_tensor_mod
    stubs["torch._subclasses"] = subclasses
    stubs["torch._subclasses.fake_tensor"] = fake_tensor_mod

    # --- torch.utils._pytree ---
    pytree = _make_stub("torch.utils._pytree", {
        "register_pytree_node": lambda *a, **kw: None,
    })
    stubs["torch.utils._pytree"] = pytree

    # --- torch.export ---
    export = _make_stub("torch.export", {
        "export": lambda *a, **kw: None,
    })
    stubs["torch.export"] = export

    # --- torch.hub ---
    import os as _os
    _torch_home = _os.path.expanduser(
        _os.environ.get("TORCH_HOME",
                        _os.path.join(_os.environ.get("XDG_CACHE_HOME", "~/.cache"), "torch"))
    )
    hub = _make_stub("torch.hub", {
        "_get_torch_home": lambda: _torch_home,
        "get_dir": lambda: _os.path.join(_torch_home, "hub"),
        "set_dir": lambda d: None,
        "load_state_dict_from_url": lambda *a, **kw: {},
    })
    stubs["torch.hub"] = hub

    # --- torch.nn.attention / flex_attention ---
    class BlockMask:
        """Stub BlockMask for type annotations and isinstance checks."""

    def create_block_mask(*args, **kwargs):
        raise NotImplementedError("flex_attention not available in mindtorch_v2")

    flex_attn = _make_stub("torch.nn.attention.flex_attention", {
        "BlockMask": BlockMask,
        "create_block_mask": create_block_mask,
        "_DEFAULT_SPARSE_BLOCK_SIZE": 128,
    })
    nn_attention = _make_stub("torch.nn.attention", {
        "flex_attention": flex_attn,
    })
    stubs["torch.nn.attention"] = nn_attention
    stubs["torch.nn.attention.flex_attention"] = flex_attn

    # --- torch._C ---
    # Some code probes torch._C for internal attributes
    _c = _make_stub("torch._C", {
        "_get_tracing_state": lambda: None,
        "_jit_set_profiling_mode": lambda b: None,
        "_jit_set_profiling_executor": lambda b: None,
        "DisableTorchFunction": type("DisableTorchFunction", (), {
            "__enter__": lambda self: self,
            "__exit__": lambda self, *a: None,
        }),
        "_disabled_torch_function_impl": lambda *a, **kw: NotImplemented,
        "Graph": type("Graph", (), {}),
        "Value": type("Value", (), {}),
        "Node": type("Node", (), {}),
    })
    stubs["torch._C"] = _c

    _STUB_MODULES.update(stubs.keys())
    return stubs


# ─── importlib.metadata patching ──────────────────────────────────────

class _TorchDistribution:
    """Fake importlib.metadata.Distribution for torch."""
    @property
    def metadata(self):
        return {"Name": "torch", "Version": _TORCH_VERSION}

    @property
    def version(self):
        return _TORCH_VERSION


_original_distribution = importlib.metadata.distribution
_original_version = importlib.metadata.version


def _patched_distribution(name):
    if name == "torch":
        return _TorchDistribution()
    return _original_distribution(name)


def _patched_version(name):
    if name == "torch":
        return _TORCH_VERSION
    return _original_version(name)


# ─── install() ────────────────────────────────────────────────────────

_installed = False


def install():
    """Install the mindtorch_v2 → torch proxy.

    After calling this function every ``import torch`` in the process
    will resolve to mindtorch_v2.
    """
    global _installed
    if _installed:
        return
    _installed = True

    # 1. Make sure mindtorch_v2 itself is loaded
    import mindtorch_v2  # noqa: F401

    # 2. Patch importlib.metadata so version checks pass
    importlib.metadata.distribution = _patched_distribution
    importlib.metadata.version = _patched_version

    # 3. Register the MetaPathFinder (must come first so find_spec works)
    finder = _MindTorchV2Finder()
    sys.meta_path.insert(0, finder)

    # 4. Put mindtorch_v2 into sys.modules as "torch"
    sys.modules["torch"] = mindtorch_v2
    mindtorch_v2.__version__ = _TORCH_VERSION
    # Make torch look like a package at its original location
    if not hasattr(mindtorch_v2, "__path__"):
        mindtorch_v2.__path__ = []

    # 5. Wire up submodule aliases so torch.xxx → mindtorch_v2.xxx
    #    for submodules that are already imported (BEFORE stubs, so stub
    #    parent lookups find the real modules)
    for attr_name in list(dir(mindtorch_v2)):
        obj = getattr(mindtorch_v2, attr_name, None)
        if isinstance(obj, types.ModuleType):
            torch_name = "torch." + attr_name
            if torch_name not in sys.modules:
                sys.modules[torch_name] = obj

    # 6. Register specific alias overrides
    for torch_name, real_name in _SUBMODULE_ALIASES.items():
        if torch_name not in sys.modules:
            try:
                mod = importlib.import_module(real_name)
                sys.modules[torch_name] = mod
            except ImportError:
                pass

    # 7. Create and register all stub modules
    stubs = _create_stub_modules()
    for name, mod in stubs.items():
        sys.modules[name] = mod
        # Also set as attribute on parent module
        parts = name.split(".")
        if len(parts) >= 2:
            parent_name = ".".join(parts[:-1])
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, parts[-1], mod)
