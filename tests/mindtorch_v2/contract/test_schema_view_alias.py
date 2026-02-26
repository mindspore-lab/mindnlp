from mindtorch_v2._dispatch.registry import registry
from mindtorch_v2._dispatch.schema import OpSchema


def test_schema_parses_view_alias_on_input_and_return():
    schema = OpSchema("view(Tensor(a) input, int[] shape) -> Tensor(a)")
    assert schema.params[0].alias_set == "a"
    assert schema.returns[0].alias_set == "a"


def test_core_view_like_schemas_expose_alias_sets():
    for name in ("view", "reshape", "transpose"):
        entry = registry.get(f"aten::{name}")
        assert entry.schema_obj is not None
        assert entry.schema_obj.params[0].alias_set == "a"
        assert entry.schema_obj.returns[0].alias_set == "a"
