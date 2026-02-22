import mindtorch_v2 as torch
from mindtorch_v2._dispatch.registry import registry


def test_inplace_schema_registered():
    entry = registry.get("aten::add_")
    assert entry.schema_obj is not None
    assert any(param.mutates for param in entry.schema_obj.params)
