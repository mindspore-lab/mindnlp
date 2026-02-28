import inspect
import os
import json

import pytest

import mindtorch_v2 as torch
from mindtorch_v2 import nn, optim


def test_profiler_context_and_step_basics():
    x = torch.ones((2, 2))
    with torch.profiler.profile() as prof:
        y = x + x
        prof.step()
        z = y * y

    assert z is not None
    events = prof.events()
    assert len(events) >= 2
    assert {event["step"] for event in events} == {0, 1}


def test_profiler_captures_dispatch_ops_forward_backward():
    x = torch.ones((2, 2))
    x.requires_grad_(True)

    with torch.profiler.profile() as prof:
        y = (x * x).sum()
        y.backward()

    names = [event["name"] for event in prof.events() if event["kind"] == "op"]
    assert any("mul" in name for name in names)
    assert any("sum" in name for name in names)


def test_record_function_nesting_and_exception_safe():
    with torch.profiler.profile() as prof:
        with pytest.raises(RuntimeError):
            with torch.profiler.record_function("outer"):
                with torch.profiler.record_function("inner"):
                    _ = torch.ones((2, 2)) + 1
                raise RuntimeError("boom")

    scopes = [event for event in prof.events() if event["kind"] == "scope"]
    assert [scope["name"] for scope in scopes] == ["outer", "inner"]
    assert all(scope["duration_ns"] >= 0 for scope in scopes)


def test_key_averages_table_contains_expected_columns():
    x = torch.ones((4, 4))
    with torch.profiler.profile() as prof:
        _ = x + x
        _ = x * x

    table = prof.key_averages().table(sort_by="self_cpu_time_total")
    assert "Name" in table
    assert "Count" in table
    assert "Total" in table


def test_export_chrome_trace_json_valid(tmp_path):
    x = torch.ones((2, 2))
    out = tmp_path / "trace.json"

    with torch.profiler.profile() as prof:
        _ = x + x

    prof.export_chrome_trace(str(out))
    payload = json.loads(out.read_text())
    assert "traceEvents" in payload
    assert len(payload["traceEvents"]) > 0


def test_profiler_session_covers_optimizer_call():
    layer = nn.Linear(2, 1)
    opt = optim.SGD(layer.parameters(), lr=0.1)
    x = torch.tensor([[1.0, 2.0]])

    with torch.profiler.profile() as prof:
        y = layer(x)
        y.sum().backward()
        opt.step()

    op_names = [event["name"] for event in prof.events() if event["kind"] == "op"]
    assert len(op_names) > 0


def test_profiler_no_mindspore_dependency():
    import mindtorch_v2.profiler.profiler as profiler_impl

    source = inspect.getsource(profiler_impl)
    assert "mindspore" not in source


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_profiler_npu_event_device_type():
    x = torch.ones((2, 2), device="npu")

    with torch.profiler.profile() as prof:
        _ = x + x

    assert any(event["device_type"] == "NPU" for event in prof.events())


def test_profiler_rejects_unknown_activity():
    with pytest.raises(ValueError):
        torch.profiler.profile(activities=["TPU"])


def test_on_trace_ready_receives_profiler_instance():
    seen = []

    def callback(prof):
        seen.append(prof)

    with torch.profiler.profile(on_trace_ready=callback) as prof:
        _ = torch.ones((2, 2)) + 1

    assert seen == [prof]


def test_on_trace_ready_type_error_from_callback_is_not_swallowed():
    def callback(prof):
        raise TypeError("callback boom")

    with pytest.raises(TypeError, match="callback boom"):
        with torch.profiler.profile(on_trace_ready=callback):
            _ = torch.ones((2, 2)) + 1


def test_export_chrome_trace_required_fields_and_pid(tmp_path):
    out = tmp_path / "trace_with_pid.json"

    with torch.profiler.profile() as prof:
        _ = torch.ones((2, 2)) + 1

    prof.export_chrome_trace(str(out))
    payload = json.loads(out.read_text())
    event = payload["traceEvents"][0]

    for key in ("name", "ph", "ts", "dur", "pid", "tid"):
        assert key in event
    assert event["pid"] == os.getpid()


def test_record_function_is_noop_when_profiler_inactive():
    with torch.profiler.record_function("noop"):
        x = torch.ones((1,))
    assert x is not None


def test_profiler_step_requires_active_session():
    prof = torch.profiler.profile()
    with pytest.raises(RuntimeError):
        prof.step()
