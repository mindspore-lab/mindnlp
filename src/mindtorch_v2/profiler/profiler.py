import json
import os
import threading
import time
import traceback
import tracemalloc
from dataclasses import dataclass

from .common import ProfilerActivity, ProfilerAction


_ACTIVE_SESSION = None
_ACTIVE_LOCK = threading.Lock()
_SCOPE_STATE = threading.local()


@dataclass
class _Event:
    name: str
    kind: str
    device_type: str
    start_ns: int
    end_ns: int
    duration_ns: int
    step: int
    thread_id: int
    metadata: dict = None

    def to_dict(self):
        payload = {
            "name": self.name,
            "kind": self.kind,
            "device_type": self.device_type,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.duration_ns,
            "step": self.step,
            "thread_id": self.thread_id,
        }
        if self.metadata:
            payload.update(self.metadata)
        return payload


class _ProfilerSession:
    def __init__(self, activities, record_shapes=False, with_stack=False, profile_memory=False):
        self.activities = set(activities)
        self.current_step = 0
        self._events = []
        self._lock = threading.Lock()
        self.is_recording = True
        self.record_shapes = bool(record_shapes)
        self.with_stack = bool(with_stack)
        self.profile_memory = bool(profile_memory)

    def make_op_token(self, name, device_type, metadata=None):
        return (
            self,
            name,
            device_type,
            time.perf_counter_ns(),
            self.current_step,
            threading.get_ident(),
            metadata,
        )

    def append_event(self, event):
        if event.device_type not in self.activities:
            return
        with self._lock:
            self._events.append(event)

    def snapshot(self):
        with self._lock:
            return list(self._events)


def _activity_name(activity):
    if isinstance(activity, ProfilerActivity):
        name = activity.value
    else:
        name = str(activity).split(".")[-1].upper()

    if name in ("CUDA", "GPU"):
        return "NPU"
    if name in ("CPU", "NPU"):
        return name
    raise ValueError(f"unsupported profiler activity: {activity}")


def _resolve_activities(activities):
    if activities is None:
        return {"CPU", "NPU"}
    return {_activity_name(item) for item in activities}


def _validate_schedule_value(name, value):
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def schedule(wait=0, warmup=0, active=0, repeat=0, skip_first=0):
    _validate_schedule_value("wait", wait)
    _validate_schedule_value("warmup", warmup)
    _validate_schedule_value("active", active)
    _validate_schedule_value("repeat", repeat)
    _validate_schedule_value("skip_first", skip_first)

    if active <= 0:
        raise ValueError("active must be > 0")

    cycle = wait + warmup + active

    def _schedule(step):
        if step < skip_first:
            return ProfilerAction.NONE

        local = step - skip_first
        cycle_idx = local // cycle
        if repeat > 0 and cycle_idx >= repeat:
            return ProfilerAction.NONE

        pos = local % cycle
        if pos < wait:
            return ProfilerAction.NONE
        if pos < wait + warmup:
            return ProfilerAction.WARMUP

        active_pos = pos - (wait + warmup)
        if active_pos == active - 1:
            return ProfilerAction.RECORD_AND_SAVE
        return ProfilerAction.RECORD

    return _schedule


def _normalize_device_type(device):
    if hasattr(device, "type"):
        dev = str(device.type)
    else:
        dev = str(device or "cpu")
    dev = dev.split(":", 1)[0].lower()
    if dev in ("cuda", "gpu", "npu"):
        return "NPU"
    if dev == "cpu":
        return "CPU"
    return dev.upper()


def _is_tensor_like(value):
    return hasattr(value, "shape") and hasattr(value, "device")


def _shape_payload(value, depth=0):
    if depth > 3:
        return None
    if _is_tensor_like(value):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            item_shape = _shape_payload(item, depth + 1)
            if item_shape is not None:
                items.append(item_shape)
        return items or None
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            item_shape = _shape_payload(item, depth + 1)
            if item_shape is not None:
                out[str(key)] = item_shape
        return out or None
    return None


def _capture_input_shapes(args, kwargs):
    arg_shapes = []
    for value in args:
        shape = _shape_payload(value)
        if shape is not None:
            arg_shapes.append(shape)

    kw_shapes = {}
    for key, value in kwargs.items():
        shape = _shape_payload(value)
        if shape is not None:
            kw_shapes[key] = shape

    if not arg_shapes and not kw_shapes:
        return None

    payload = {}
    if arg_shapes:
        payload["args"] = arg_shapes
    if kw_shapes:
        payload["kwargs"] = kw_shapes
    return payload


def _is_internal_stack_frame(filename):
    normalized = filename.replace("\\", "/")
    return (
        "/mindtorch_v2/profiler/profiler.py" in normalized
        or "/mindtorch_v2/_dispatch/dispatcher.py" in normalized
    )


def _capture_stack(limit=48):
    frames = traceback.extract_stack(limit=limit)
    entries = []
    for frame in frames:
        if _is_internal_stack_frame(frame.filename):
            continue
        entries.append(f"{frame.filename}:{frame.lineno}:{frame.name}")
    return entries[-32:] if entries else None


def _npu_memory_allocated_snapshot(device_type):
    if device_type != "NPU":
        return None

    from .. import npu

    if not npu.is_available():
        return None
    return int(npu.memory_allocated())


def _cpu_memory_allocated_snapshot(device_type):
    if device_type != "CPU":
        return None
    if not tracemalloc.is_tracing():
        return None
    current, _peak = tracemalloc.get_traced_memory()
    return int(current)


def _memory_prefix_for_device(device_type):
    if device_type == "NPU":
        return "npu_memory_allocated"
    if device_type == "CPU":
        return "cpu_memory_allocated"
    return None


def _memory_allocated_snapshot(device_type):
    if device_type == "NPU":
        return _npu_memory_allocated_snapshot(device_type)
    if device_type == "CPU":
        return _cpu_memory_allocated_snapshot(device_type)
    return None


def _active_session():
    return _ACTIVE_SESSION


def is_profiler_enabled():
    return _ACTIVE_SESSION is not None


def dispatch_op_enter(name, dispatch_device, args, kwargs):
    session = _active_session()
    if session is None:
        return None

    device_type = _normalize_device_type(dispatch_device)
    if device_type not in session.activities:
        return None
    if not session.is_recording:
        return None

    metadata = None
    if session.record_shapes or session.with_stack or session.profile_memory:
        metadata = {}
        if session.record_shapes:
            shapes = _capture_input_shapes(args, kwargs)
            if shapes is not None:
                metadata["input_shapes"] = shapes
        if session.with_stack:
            stack = _capture_stack()
            if stack:
                metadata["stack"] = stack
        if session.profile_memory:
            prefix = _memory_prefix_for_device(device_type)
            if prefix is not None:
                before = _memory_allocated_snapshot(device_type)
                if before is not None:
                    metadata[f"{prefix}_before"] = before
        if not metadata:
            metadata = None

    return session.make_op_token(name, device_type, metadata)


def dispatch_op_exit(token):
    if token is None:
        return

    session, name, device_type, start_ns, step, thread_id, metadata = token
    end_ns = time.perf_counter_ns()

    if metadata is not None:
        for prefix in ("npu_memory_allocated", "cpu_memory_allocated"):
            before_key = f"{prefix}_before"
            if before_key not in metadata:
                continue
            after = _memory_allocated_snapshot(device_type)
            if after is None:
                continue
            before = metadata[before_key]
            metadata[f"{prefix}_after"] = after
            metadata[f"{prefix}_delta"] = int(after - before)

    session.append_event(
        _Event(
            name=name,
            kind="op",
            device_type=device_type,
            start_ns=start_ns,
            end_ns=end_ns,
            duration_ns=end_ns - start_ns,
            step=step,
            thread_id=thread_id,
            metadata=metadata,
        )
    )


def _scope_stack():
    stack = getattr(_SCOPE_STATE, "stack", None)
    if stack is None:
        stack = []
        _SCOPE_STATE.stack = stack
    return stack


class _RecordFunction:
    def __init__(self, name):
        self.name = str(name)
        self._token = None

    def __enter__(self):
        session = _active_session()
        if session is None or not session.is_recording:
            return self

        metadata = None
        if session.with_stack:
            stack = _capture_stack()
            if stack:
                metadata = {"stack": stack}

        self._token = (
            session,
            self.name,
            time.perf_counter_ns(),
            session.current_step,
            threading.get_ident(),
            metadata,
        )
        _scope_stack().append(self._token)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is None:
            return False

        session, name, start_ns, step, thread_id, metadata = self._token
        end_ns = time.perf_counter_ns()
        stack = _scope_stack()
        if stack and stack[-1] is self._token:
            stack.pop()
        elif self._token in stack:
            stack.remove(self._token)

        session.append_event(
            _Event(
                name=name,
                kind="scope",
                device_type="CPU",
                start_ns=start_ns,
                end_ns=end_ns,
                duration_ns=end_ns - start_ns,
                step=step,
                thread_id=thread_id,
                metadata=metadata,
            )
        )
        return False


def record_function(name):
    return _RecordFunction(name)


def _event_self_time_map(events):
    indexed = list(enumerate(events))
    indexed.sort(key=lambda item: (item[1].thread_id, item[1].start_ns, -item[1].end_ns))

    self_time = {idx: max(0, ev.duration_ns) for idx, ev in indexed}
    stack = []

    for idx, event in indexed:
        # Pop intervals that no longer contain current event.
        while stack and event.start_ns >= stack[-1][1].end_ns:
            stack.pop()

        if stack:
            parent_idx, parent_event = stack[-1]
            # Child entirely inside parent on same thread contributes to parent's child coverage.
            if event.end_ns <= parent_event.end_ns:
                self_time[parent_idx] = max(0, self_time[parent_idx] - max(0, event.duration_ns))

        stack.append((idx, event))

    return self_time


class _KeyAverages:
    _SORT_MAP = {
        "self_cpu_time_total": "self_time_ns",
        "cpu_time_total": "total_time_ns",
        "count": "count",
    }

    def __init__(self, events, *, group_by_input_shape=False, group_by_stack_n=0):
        self._events = list(events)
        self._rows = None
        self._group_by_input_shape = bool(group_by_input_shape)
        self._group_by_stack_n = int(group_by_stack_n)

    def _group_key(self, event):
        key = [event.name, event.device_type]
        metadata = event.metadata or {}

        if self._group_by_input_shape:
            key.append(json.dumps(metadata.get("input_shapes", None), sort_keys=True))

        if self._group_by_stack_n > 0:
            stack = metadata.get("stack") or []
            key.append(tuple(stack[-self._group_by_stack_n :]))

        return tuple(key)

    def _build_rows(self):
        if self._rows is not None:
            return self._rows

        per_event_self_time = _event_self_time_map(self._events)

        grouped = {}
        for idx, event in enumerate(self._events):
            group_key = self._group_key(event)
            row = grouped.setdefault(
                group_key,
                {
                    "name": event.name,
                    "device_type": event.device_type,
                    "count": 0,
                    "total_time_ns": 0,
                    "self_time_ns": 0,
                },
            )
            metadata = event.metadata or {}
            if self._group_by_input_shape:
                row["input_shapes"] = metadata.get("input_shapes")
            if self._group_by_stack_n > 0:
                stack = metadata.get("stack") or []
                row["stack"] = stack[-self._group_by_stack_n :] if stack else []

            row["count"] += 1
            row["total_time_ns"] += event.duration_ns
            row["self_time_ns"] += per_event_self_time.get(idx, max(0, event.duration_ns))

        for row in grouped.values():
            row["avg_time_ns"] = row["total_time_ns"] // max(1, row["count"])

        self._rows = list(grouped.values())
        return self._rows

    def table(self, sort_by="self_cpu_time_total", row_limit=100):
        if sort_by not in self._SORT_MAP:
            raise AttributeError(f"'FunctionEventAvg' object has no attribute '{sort_by}'")

        rows = list(self._build_rows())
        sort_key = self._SORT_MAP[sort_by]
        rows.sort(key=lambda item: (item[sort_key], item["name"]), reverse=True)
        if row_limit is not None:
            rows = rows[: int(row_limit)]

        header = f"{'Name':<32} {'Device':<8} {'Count':>6} {'Total(us)':>12} {'Self(us)':>12} {'Avg(us)':>12}"
        lines = [header]
        for row in rows:
            total_us = row["total_time_ns"] / 1000.0
            self_us = row["self_time_ns"] / 1000.0
            avg_us = row["avg_time_ns"] / 1000.0
            lines.append(
                f"{row['name']:<32} {row['device_type']:<8} {row['count']:>6} {total_us:>12.3f} {self_us:>12.3f} {avg_us:>12.3f}"
            )
        return "\n".join(lines)

    def __len__(self):
        return len(self._build_rows())


class profile:
    def __init__(
        self,
        *,
        activities=None,
        schedule=None,
        on_trace_ready=None,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=None,
        use_cuda=None,
    ):
        del with_flops, with_modules, experimental_config, use_cuda

        self.activities = _resolve_activities(activities)
        self._session = _ProfilerSession(
            self.activities,
            record_shapes=record_shapes,
            with_stack=with_stack,
            profile_memory=profile_memory,
        )
        self._started = False
        self._stopped = False
        self._trace_ready = on_trace_ready[0] if isinstance(on_trace_ready, tuple) else on_trace_ready
        self._owns_tracemalloc = False

        if schedule is not None and not callable(schedule):
            raise TypeError("schedule must be callable")
        self._schedule = schedule
        self._current_action = ProfilerAction.RECORD if schedule is None else ProfilerAction.NONE

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    def _maybe_sync_npu(self):
        if "NPU" not in self.activities:
            return

        from .. import npu

        if npu.is_available():
            npu.synchronize()

    def _maybe_start_cpu_memory_tracer(self):
        if not self._session.profile_memory:
            return
        if "CPU" not in self.activities:
            return
        if tracemalloc.is_tracing():
            self._owns_tracemalloc = False
            return
        tracemalloc.start()
        self._owns_tracemalloc = True

    def _maybe_stop_cpu_memory_tracer(self):
        if not self._owns_tracemalloc:
            return
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        self._owns_tracemalloc = False

    def _action_for_step(self, step):
        if self._schedule is None:
            return ProfilerAction.RECORD
        return self._schedule(step)

    def _set_action_for_step(self, step):
        self._current_action = self._action_for_step(step)
        self._session.is_recording = self._current_action in (
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
        )

    def start(self):
        global _ACTIVE_SESSION
        if self._started and not self._stopped:
            return

        with _ACTIVE_LOCK:
            if _ACTIVE_SESSION is not None and _ACTIVE_SESSION is not self._session:
                raise RuntimeError("another profiler session is already active")
            _ACTIVE_SESSION = self._session

        self._started = True
        self._stopped = False
        self._maybe_start_cpu_memory_tracer()
        self._set_action_for_step(self._session.current_step)

    def stop(self):
        global _ACTIVE_SESSION
        if not self._started or self._stopped:
            return

        self._maybe_sync_npu()
        with _ACTIVE_LOCK:
            if _ACTIVE_SESSION is self._session:
                _ACTIVE_SESSION = None

        self._maybe_stop_cpu_memory_tracer()
        self._stopped = True
        if callable(self._trace_ready) and self._schedule is None:
            self._trace_ready(self)

    def step(self):
        if _active_session() is not self._session:
            raise RuntimeError("profiler session is not active")

        action = self._current_action
        self._maybe_sync_npu()
        if action == ProfilerAction.RECORD_AND_SAVE and callable(self._trace_ready):
            self._trace_ready(self)

        self._session.current_step += 1
        self._set_action_for_step(self._session.current_step)

    def events(self):
        events = self._session.snapshot()
        events.sort(key=lambda item: (item.start_ns, item.end_ns, item.name))
        return [event.to_dict() for event in events]

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0):
        return _KeyAverages(
            self._session.snapshot(),
            group_by_input_shape=group_by_input_shape,
            group_by_stack_n=group_by_stack_n,
        )

    def export_chrome_trace(self, path):
        self._maybe_sync_npu()
        trace_events = []
        for event in self._session.snapshot():
            args = {
                "device_type": event.device_type,
                "step": event.step,
            }
            if event.metadata:
                args.update(event.metadata)
            trace_events.append(
                {
                    "name": event.name,
                    "cat": event.kind,
                    "ph": "X",
                    "pid": os.getpid(),
                    "tid": event.thread_id,
                    "ts": event.start_ns / 1000.0,
                    "dur": event.duration_ns / 1000.0,
                    "args": args,
                }
            )

        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"traceEvents": trace_events, "displayTimeUnit": "ms"}, handle)
