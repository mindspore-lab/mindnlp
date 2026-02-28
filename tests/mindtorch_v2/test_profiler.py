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


def test_profile_rejects_non_callable_schedule():
    with pytest.raises(TypeError):
        torch.profiler.profile(schedule=123)


def test_profiler_schedule_filters_recorded_steps():
    sched = torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=0)

    with torch.profiler.profile(schedule=sched) as prof:
        _ = torch.ones((1,)) + 1
        prof.step()
        _ = torch.ones((1,)) + 2
        prof.step()
        _ = torch.ones((1,)) + 3

    steps = {event["step"] for event in prof.events() if event["kind"] == "op"}
    assert steps == {2}


def test_profiler_schedule_triggers_trace_ready_on_save_action():
    calls = []

    def on_trace_ready(prof):
        calls.append(len(prof.events()))

    sched = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0)

    with torch.profiler.profile(schedule=sched, on_trace_ready=on_trace_ready) as prof:
        _ = torch.ones((1,)) + 1
        prof.step()
        _ = torch.ones((1,)) + 2

    assert len(calls) >= 1


def test_profiler_schedule_invalid_config_raises():
    with pytest.raises(ValueError):
        torch.profiler.schedule(wait=-1, warmup=0, active=1)
    with pytest.raises(ValueError):
        torch.profiler.schedule(wait=0, warmup=0, active=0)


def test_profiler_record_shapes_captures_tensor_shapes():
    with torch.profiler.profile(record_shapes=True) as prof:
        x = torch.ones((2, 3))
        _ = x + x

    op_events = [event for event in prof.events() if event["kind"] == "op"]
    assert op_events
    assert any("input_shapes" in event for event in op_events)


def test_profiler_with_stack_captures_frame_metadata():
    with torch.profiler.profile(with_stack=True) as prof:
        x = torch.ones((2, 2))
        _ = x + x

    op_events = [event for event in prof.events() if event["kind"] == "op"]
    assert op_events
    assert any("stack" in event and len(event["stack"]) > 0 for event in op_events)


def test_profiler_default_flags_do_not_emit_shape_or_stack():
    with torch.profiler.profile() as prof:
        x = torch.ones((2, 2))
        _ = x + x

    op_events = [event for event in prof.events() if event["kind"] == "op"]
    assert op_events
    assert all("input_shapes" not in event for event in op_events)
    assert all("stack" not in event for event in op_events)


def test_export_chrome_trace_includes_shape_and_stack_args_when_enabled(tmp_path):
    out = tmp_path / "trace_shapes_stack.json"

    with torch.profiler.profile(record_shapes=True, with_stack=True) as prof:
        x = torch.ones((2, 4))
        _ = x + x

    prof.export_chrome_trace(str(out))
    payload = json.loads(out.read_text())
    events = payload["traceEvents"]
    assert events
    assert any("input_shapes" in event.get("args", {}) for event in events)
    assert any("stack" in event.get("args", {}) for event in events)


def test_key_averages_supports_group_by_input_shape():
    with torch.profiler.profile(record_shapes=True) as prof:
        x2 = torch.ones((2, 2))
        x4 = torch.ones((4, 4))
        _ = x2 + x2
        _ = x4 + x4

    default_rows = prof.key_averages()
    grouped_rows = prof.key_averages(group_by_input_shape=True)

    assert len(grouped_rows) >= len(default_rows)


def test_key_averages_supports_group_by_stack_n():
    with torch.profiler.profile(with_stack=True) as prof:
        x = torch.ones((2, 2))

        def call_site_one():
            return x + x

        def call_site_two():
            return x + x

        _ = call_site_one()
        _ = call_site_two()

    default_rows = prof.key_averages()
    grouped_rows = prof.key_averages(group_by_stack_n=1)

    assert len(grouped_rows) >= len(default_rows)


def test_key_averages_table_unknown_sort_key_raises():
    with torch.profiler.profile() as prof:
        x = torch.ones((2, 2))
        _ = x + x

    with pytest.raises(AttributeError):
        prof.key_averages().table(sort_by="foo")
