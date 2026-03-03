from mindtorch_v2._dispatch.schema import OpSchema


def test_schema_parses_single_return_alias_set():
    schema = OpSchema("foo(Tensor x) -> Tensor(a)")
    assert schema.returns[0].alias_set == "a"


def test_schema_parses_tuple_return_alias_set():
    schema = OpSchema("foo(Tensor x) -> (Tensor(a), Tensor(b), Tensor)")
    assert schema.returns[0].alias_set == "a"
    assert schema.returns[1].alias_set == "b"
    assert schema.returns[2].alias_set is None
