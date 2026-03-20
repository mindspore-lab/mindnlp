"""Configure sys.path so wizard.merge modules can be imported independently
of the full mindnlp init (which requires mindtorch)."""

import importlib
import importlib.machinery
import os
import sys
import types
from pathlib import Path

# Disable torch device backend auto-loading (avoids torch_npu libhccl errors).
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

# Stub torch_npu before anything tries to import it (accelerate, transformers).
if "torch_npu" not in sys.modules:
    _fake_npu = types.ModuleType("torch_npu")
    _fake_npu.__spec__ = importlib.machinery.ModuleSpec("torch_npu", None)
    _fake_npu.__path__ = []
    for _sub in ("_C", "utils", "utils._error_code", "npu"):
        _full = f"torch_npu.{_sub}"
        _mod = types.ModuleType(_full)
        _mod.__spec__ = importlib.machinery.ModuleSpec(_full, None)
        sys.modules[_full] = _mod
    _fake_npu._C = sys.modules["torch_npu._C"]
    _fake_npu.utils = sys.modules["torch_npu.utils"]
    _fake_npu.npu = sys.modules["torch_npu.npu"]
    sys.modules["torch_npu.utils._error_code"].ErrCode = None
    sys.modules["torch_npu.utils._error_code"].pta_error = None
    sys.modules["torch_npu"] = _fake_npu

SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Replace the pip-installed mindnlp with a lightweight stub
# so that subpackage imports (mindnlp.wizard.merge.*) work
# without triggering the full mindnlp init (mindtorch dependency).
stub = types.ModuleType("mindnlp")
stub.__path__ = [str(Path(SRC_DIR) / "mindnlp")]
stub.__package__ = "mindnlp"
sys.modules["mindnlp"] = stub

# Also stub mindnlp.wizard
wizard_stub = types.ModuleType("mindnlp.wizard")
wizard_stub.__path__ = [str(Path(SRC_DIR) / "mindnlp" / "wizard")]
wizard_stub.__package__ = "mindnlp.wizard"
sys.modules["mindnlp.wizard"] = wizard_stub
