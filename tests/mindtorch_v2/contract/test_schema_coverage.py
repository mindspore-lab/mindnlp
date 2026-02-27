import mindtorch_v2  # noqa: F401
from mindtorch_v2._dispatch.registry import registry


def test_registered_ops_have_schema():
    allowed_no_schema = {
        "getitem",
        "setitem",
    }

    missing = []
    for name, entry in registry._ops.items():
        if not entry.kernels:
            continue
        short = name.split("::", 1)[-1]
        if short in allowed_no_schema:
            continue
        if entry.schema_obj is None:
            missing.append(name)

    assert not missing, f"ops missing schema: {sorted(missing)}"
