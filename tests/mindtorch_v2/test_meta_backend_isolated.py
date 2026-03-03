from pathlib import Path

import mindtorch_v2._backends.meta as meta_pkg


def test_meta_backend_does_not_import_cpu():
    root = Path(meta_pkg.__file__).parent
    for name in ("ops.py", "creation.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert "from ..cpu" not in src
        assert "from ..cpu." not in src
