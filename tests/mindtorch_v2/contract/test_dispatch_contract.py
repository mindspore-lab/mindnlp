import uuid

from tests.mindtorch_v2.contract.helpers import assert_torch_error


def test_pipeline_requires_meta_kernel_error():
    from mindtorch_v2._dispatch import pipeline
    from mindtorch_v2._dispatch.dispatcher import dispatch
    from mindtorch_v2._dispatch.keys import DispatchKey
    from mindtorch_v2._dispatch.registry import registry

    op_name = f"pipeline_no_meta_{uuid.uuid4().hex}"
    registry.register_kernel(op_name, DispatchKey.CPU, lambda: None)

    def mt():
        with pipeline.pipeline_context():
            dispatch(op_name, "cpu")

    def th():
        raise RuntimeError(f"pipeline requires meta kernel for op {op_name}")

    assert_torch_error(mt, th)
