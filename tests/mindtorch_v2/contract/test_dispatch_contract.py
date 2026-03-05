import uuid

from tests.mindtorch_v2.contract.helpers import assert_torch_error


def test_pipeline_requires_meta_kernel_error():
    from mindtorch_v2._dispatch import pipeline
    from mindtorch_v2._dispatch.dispatcher import dispatch
    from mindtorch_v2._dispatch.keys import DispatchKey
    from mindtorch_v2._dispatch.registry import registry

    op_name = f"pipeline_no_meta_{uuid.uuid4().hex}"
    registry.register_schema(op_name, f"{op_name}() -> Any")
    registry.register_kernel(op_name, DispatchKey.CPU, lambda: None)

    def mt():
        with pipeline.pipeline_context():
            dispatch(op_name, "cpu")

    def th():
        raise RuntimeError(f"pipeline requires meta kernel for op {op_name}")

    assert_torch_error(mt, th)


def test_pipeline_last_error_structured_payload():
    from mindtorch_v2._dispatch import pipeline
    from mindtorch_v2._dispatch.dispatcher import dispatch
    from mindtorch_v2._dispatch.keys import DispatchKey
    from mindtorch_v2._dispatch.registry import registry

    op_name = f"pipeline_error_payload_{uuid.uuid4().hex}"
    registry.register_schema(op_name, f"{op_name}() -> Any")
    def _meta():
        from mindtorch_v2._backends.meta.infer import TensorSpec
        from mindtorch_v2._dtype import float32

        return TensorSpec(shape=(1,), stride=(1,), dtype=float32)

    registry.register_kernel(op_name, DispatchKey.Meta, _meta)

    def _boom():
        raise RuntimeError("boom-from-kernel")

    registry.register_kernel(op_name, DispatchKey.CPU, _boom)

    with pipeline.pipeline_context() as pipe:
        dispatch(op_name, "cpu")
        try:
            pipe.flush()
        except RuntimeError:
            pass

        err = pipe.last_error()
        assert err is not None
        payload = err.to_dict()
        assert payload["op_name"] == op_name
        assert payload["phase"] == "submit"
        assert payload["backend"] in {"cpu", "npu", "meta"}
        assert isinstance(payload["op_seq"], int)
        assert "callsite" in payload
        assert "read_set" in payload
        assert "write_set" in payload
        assert "alias_set" in payload
        assert "version_plan" in payload
        assert payload["read_set"] == []
        assert payload["write_set"] == []
        assert "dependency_edges" in payload
        assert "runtime_code" in payload
        assert "suppressed_errors" in payload


def test_pipeline_error_debug_interfaces():
    from mindtorch_v2._dispatch import pipeline
    from mindtorch_v2._dispatch.dispatcher import dispatch
    from mindtorch_v2._dispatch.keys import DispatchKey
    from mindtorch_v2._dispatch.registry import registry

    op_name = f"pipeline_error_debug_{uuid.uuid4().hex}"
    registry.register_schema(op_name, f"{op_name}() -> Any")
    def _meta():
        from mindtorch_v2._backends.meta.infer import TensorSpec
        from mindtorch_v2._dtype import float32

        return TensorSpec(shape=(1,), stride=(1,), dtype=float32)

    registry.register_kernel(op_name, DispatchKey.Meta, _meta)

    def _boom():
        raise RuntimeError("boom-for-debug")

    registry.register_kernel(op_name, DispatchKey.CPU, _boom)

    with pipeline.pipeline_context() as pipe:
        dispatch(op_name, "cpu")
        assert pipe.pending_count() == 1
        try:
            pipe.flush()
        except RuntimeError:
            pass

        short = pipe.format_error("short")
        full = pipe.format_error("full")
        dump = pipe.debug_dump(failed_only=True)

        assert "boom-for-debug" in short
        assert "op_name" in full
        assert dump["last_error"] is not None
        assert len(dump["entries"]) == 1


def test_pipeline_error_id_is_deterministic_for_same_site():
    from mindtorch_v2._dispatch import pipeline
    from mindtorch_v2._dispatch.dispatcher import dispatch
    from mindtorch_v2._dispatch.keys import DispatchKey
    from mindtorch_v2._dispatch.registry import registry

    op_name = f"pipeline_error_stable_id_{uuid.uuid4().hex}"
    registry.register_schema(op_name, f"{op_name}() -> Any")
    def _meta():
        from mindtorch_v2._backends.meta.infer import TensorSpec
        from mindtorch_v2._dtype import float32

        return TensorSpec(shape=(1,), stride=(1,), dtype=float32)

    registry.register_kernel(op_name, DispatchKey.Meta, _meta)

    def _boom():
        raise RuntimeError("stable-id-error")

    registry.register_kernel(op_name, DispatchKey.CPU, _boom)

    ids = []
    for _ in range(2):
        with pipeline.pipeline_context() as pipe:
            dispatch(op_name, "cpu")
            try:
                pipe.flush()
            except RuntimeError:
                pass
            ids.append(pipe.last_error().to_dict()["error_id"])

    assert ids[0] == ids[1]


def test_pipeline_error_payload_captures_mutating_alias_set():
    from mindtorch_v2._dispatch import pipeline
    from mindtorch_v2._dispatch.dispatcher import dispatch
    from mindtorch_v2._dispatch.keys import DispatchKey
    from mindtorch_v2._dispatch.registry import registry

    op_name = f"pipeline_error_mutate_{uuid.uuid4().hex}"
    registry.register_schema(op_name, f"{op_name}(Tensor(a!) self) -> Tensor")

    def _meta(x):
        from mindtorch_v2._backends.meta.infer import TensorSpec
        from mindtorch_v2._dtype import float32

        return TensorSpec(shape=(1,), stride=(1,), dtype=float32)

    registry.register_kernel(op_name, DispatchKey.Meta, _meta)

    def _boom(x):
        raise RuntimeError("mutate-boom")

    registry.register_kernel(op_name, DispatchKey.CPU, _boom)

    import mindtorch_v2 as torch

    x = torch.tensor([1.0])
    with pipeline.pipeline_context() as pipe:
        dispatch(op_name, "cpu", x)
        try:
            pipe.flush()
        except RuntimeError:
            pass

        payload = pipe.last_error().to_dict()
        assert payload["write_set"] == ["self"]
        assert payload["alias_set"] == "a"
        assert payload["version_plan"].get("a") == 1


def test_pipeline_version_plan_counts_multiple_mutations():
    from mindtorch_v2._dispatch import pipeline
    from mindtorch_v2._dispatch.dispatcher import dispatch
    from mindtorch_v2._dispatch.keys import DispatchKey
    from mindtorch_v2._dispatch.registry import registry
    import mindtorch_v2 as torch

    op_name = f"pipeline_error_multi_mutate_{uuid.uuid4().hex}"
    registry.register_schema(op_name, f"{op_name}(Tensor(a!) a, Tensor(a!) b) -> Tensor")

    def _meta(a, b):
        from mindtorch_v2._backends.meta.infer import TensorSpec
        from mindtorch_v2._dtype import float32

        return TensorSpec(shape=(1,), stride=(1,), dtype=float32)

    registry.register_kernel(op_name, DispatchKey.Meta, _meta)

    def _boom(a, b):
        raise RuntimeError("multi-mutate-boom")

    registry.register_kernel(op_name, DispatchKey.CPU, _boom)

    a = torch.tensor([1.0])
    b = torch.tensor([2.0])
    with pipeline.pipeline_context() as pipe:
        dispatch(op_name, "cpu", a, b)
        try:
            pipe.flush()
        except RuntimeError:
            pass

        payload = pipe.last_error().to_dict()
        assert payload["version_plan"].get("a") == 2


def test_pipeline_dependency_edges_capture_write_write():
    from mindtorch_v2._dispatch import pipeline
    from mindtorch_v2._dispatch.dispatcher import dispatch
    from mindtorch_v2._dispatch.keys import DispatchKey
    from mindtorch_v2._dispatch.registry import registry
    import mindtorch_v2 as torch

    op1 = f"pipeline_dep_ww_ok_{uuid.uuid4().hex}"
    op2 = f"pipeline_dep_ww_fail_{uuid.uuid4().hex}"

    registry.register_schema(op1, f"{op1}(Tensor! self) -> Tensor")
    registry.register_schema(op2, f"{op2}(Tensor! self) -> Tensor")

    def _meta(x):
        from mindtorch_v2._backends.meta.infer import TensorSpec
        from mindtorch_v2._dtype import float32

        return TensorSpec(shape=(1,), stride=(1,), dtype=float32)

    registry.register_kernel(op1, DispatchKey.Meta, _meta)
    registry.register_kernel(op2, DispatchKey.Meta, _meta)

    def _ok(x):
        return x

    def _boom(x):
        raise RuntimeError("ww-dep-boom")

    registry.register_kernel(op1, DispatchKey.CPU, _ok)
    registry.register_kernel(op2, DispatchKey.CPU, _boom)

    x = torch.tensor([1.0])
    with pipeline.pipeline_context() as pipe:
        dispatch(op1, "cpu", x)
        dispatch(op2, "cpu", x)
        try:
            pipe.flush()
        except RuntimeError:
            pass

        deps = pipe.last_error().to_dict()["dependency_edges"]
        assert any(
            edge["from"] == 0 and edge["to"] == 1 and "write->write" in edge["reason"]
            for edge in deps
        )


def test_pipeline_dependency_edges_capture_alias_writes():
    from mindtorch_v2._dispatch import pipeline
    from mindtorch_v2._dispatch.dispatcher import dispatch
    from mindtorch_v2._dispatch.keys import DispatchKey
    from mindtorch_v2._dispatch.registry import registry
    import mindtorch_v2 as torch

    op1 = f"pipeline_dep_write_ok_{uuid.uuid4().hex}"
    op2 = f"pipeline_dep_write_fail_{uuid.uuid4().hex}"

    registry.register_schema(op1, f"{op1}(Tensor(a!) self) -> Tensor")
    registry.register_schema(op2, f"{op2}(Tensor(a!) self) -> Tensor")

    def _meta(x):
        from mindtorch_v2._backends.meta.infer import TensorSpec
        from mindtorch_v2._dtype import float32

        return TensorSpec(shape=(1,), stride=(1,), dtype=float32)

    registry.register_kernel(op1, DispatchKey.Meta, _meta)
    registry.register_kernel(op2, DispatchKey.Meta, _meta)

    def _ok(x):
        return x

    def _boom(x):
        raise RuntimeError("dep-boom")

    registry.register_kernel(op1, DispatchKey.CPU, _ok)
    registry.register_kernel(op2, DispatchKey.CPU, _boom)

    x = torch.tensor([1.0])
    with pipeline.pipeline_context() as pipe:
        dispatch(op1, "cpu", x)
        dispatch(op2, "cpu", x)
        try:
            pipe.flush()
        except RuntimeError:
            pass

        payload = pipe.last_error().to_dict()
        deps = payload["dependency_edges"]
        assert deps
        assert deps[0]["from"] == 0
        assert deps[0]["to"] == 1
        assert "alias_set:a" in deps[0]["reason"]


def test_pipeline_dependency_edges_capture_tensor_rw():
    from mindtorch_v2._dispatch import pipeline
    from mindtorch_v2._dispatch.dispatcher import dispatch
    from mindtorch_v2._dispatch.keys import DispatchKey
    from mindtorch_v2._dispatch.registry import registry
    import mindtorch_v2 as torch

    op1 = f"pipeline_dep_rw_ok_{uuid.uuid4().hex}"
    op2 = f"pipeline_dep_rw_fail_{uuid.uuid4().hex}"

    registry.register_schema(op1, f"{op1}(Tensor! self) -> Tensor")
    registry.register_schema(op2, f"{op2}(Tensor self) -> Tensor")

    def _meta(x):
        from mindtorch_v2._backends.meta.infer import TensorSpec
        from mindtorch_v2._dtype import float32

        return TensorSpec(shape=(1,), stride=(1,), dtype=float32)

    registry.register_kernel(op1, DispatchKey.Meta, _meta)
    registry.register_kernel(op2, DispatchKey.Meta, _meta)

    def _ok(x):
        return x

    def _boom(x):
        raise RuntimeError("rw-dep-boom")

    registry.register_kernel(op1, DispatchKey.CPU, _ok)
    registry.register_kernel(op2, DispatchKey.CPU, _boom)

    x = torch.tensor([1.0])
    with pipeline.pipeline_context() as pipe:
        dispatch(op1, "cpu", x)
        dispatch(op2, "cpu", x)
        try:
            pipe.flush()
        except RuntimeError:
            pass

        deps = pipe.last_error().to_dict()["dependency_edges"]
        assert any(
            edge["from"] == 0 and edge["to"] == 1 and "write->read" in edge["reason"]
            for edge in deps
        )
