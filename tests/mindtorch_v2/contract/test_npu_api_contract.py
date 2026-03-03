import mindtorch_v2 as torch
import torch as pt
from tests.mindtorch_v2.contract.helpers import assert_torch_error


def test_npu_set_device_invalid_string_matches_torch():
    def mt():
        torch.npu.set_device("cpu")

    def th():
        pt.cuda.set_device("cpu")

    assert_torch_error(mt, th)


def test_npu_device_ctx_invalid_string_matches_torch():
    def mt():
        torch.npu.device("cpu").__enter__()

    def th():
        pt.cuda.device("cpu").__enter__()

    assert_torch_error(mt, th)


def test_npu_memory_summary_returns_str():
    assert isinstance(torch.npu.memory_summary(), str)


def test_npu_memory_snapshot_has_keys():
    snap = torch.npu.memory_snapshot()
    assert isinstance(snap, dict)
    assert "segments" in snap
    assert "device" in snap
