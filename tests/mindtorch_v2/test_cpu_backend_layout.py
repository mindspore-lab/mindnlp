from pathlib import Path

import mindtorch_v2._backends.cpu as cpu_pkg


def test_cpu_backend_has_no_meta_module():
    root = Path(cpu_pkg.__file__).parent
    assert not (root / "meta.py").exists()
