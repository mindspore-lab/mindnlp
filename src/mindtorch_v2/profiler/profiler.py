import json
import os
import threading
import time
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

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "device_type": self.device_type,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.duration_ns,
            "step": self.step,
            "thread_id": self.thread_id,
        }


class _ProfilerSession:
    def __init__(self, activities):
        self.activities = set(activities)
        self.current_step = 0
        self._events = []
        self._lock = threading.Lock()
        self.is_recording = True

    def make_op_token(self, name, device_type):
        return (
            self,
            name,
            device_type,
            time.perf_counter_ns(),
            self.current_step,
            threading.get_ident(),
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


def _active_session():
    return _ACTIVE_SESSION


def is_profiler_enabled():
    return _ACTIVE_SESSION is not None


def dispatch_op_enter(name, dispatch_device):
    session = _active_session()
    if session is None:
        return None

    device_type = _normalize_device_type(dispatch_device)
    if device_type not in session.activities:
        return None
    if not session.is_recording:
        return None
    return session.make_op_token(name, device_type)


def dispatch_op_exit(token):
    if token is None:
        return

    session, name, device_type, start_ns, step, thread_id = token
    end_ns = time.perf_counter_ns()
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

        self._token = (
            session,
            self.name,
            time.perf_counter_ns(),
            session.current_step,
            threading.get_ident(),
        )
        _scope_stack().append(self._token)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is None:
            return False

        session, name, start_ns, step, thread_id = self._token
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
            )
        )
        return False


def record_function(name):
    return _RecordFunction(name)


class _KeyAverages:
    def __init__(self, events):
        self._events = list(events)
        self._rows = None

    def _build_rows(self):
        if self._rows is not None:
            return self._rows

        grouped = {}
        for event in self._events:
            key = (event.name, event.device_type)
            row = grouped.setdefault(
                key,
                {
                    "name": event.name,
                    "device_type": event.device_type,
                    "count": 0,
                    "total_time_ns": 0,
                    "self_time_ns": 0,
                },
            )
            row["count"] += 1
            row["total_time_ns"] += event.duration_ns
            row["self_time_ns"] += event.duration_ns

        for row in grouped.values():
            row["avg_time_ns"] = row["total_time_ns"] // max(1, row["count"])

        self._rows = list(grouped.values())
        return self._rows

    def table(self, sort_by="self_cpu_time_total", row_limit=100):
        rows = list(self._build_rows())
        sort_key = "self_time_ns" if "self" in str(sort_by) else "total_time_ns"
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
        del record_shapes, profile_memory, with_stack
        del with_flops, with_modules, experimental_config, use_cuda

        self.activities = _resolve_activities(activities)
        self._session = _ProfilerSession(self.activities)
        self._started = False
        self._stopped = False
        self._trace_ready = on_trace_ready[0] if isinstance(on_trace_ready, tuple) else on_trace_ready

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
        self._set_action_for_step(self._session.current_step)

    def stop(self):
        global _ACTIVE_SESSION
        if not self._started or self._stopped:
            return

        self._maybe_sync_npu()
        with _ACTIVE_LOCK:
            if _ACTIVE_SESSION is self._session:
                _ACTIVE_SESSION = None

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

    def key_averages(self):
        return _KeyAverages(self._session.snapshot())

    def export_chrome_trace(self, path):
        self._maybe_sync_npu()
        trace_events = []
        for event in self._session.snapshot():
            trace_events.append(
                {
                    "name": event.name,
                    "cat": event.kind,
                    "ph": "X",
                    "pid": os.getpid(),
                    "tid": event.thread_id,
                    "ts": event.start_ns / 1000.0,
                    "dur": event.duration_ns / 1000.0,
                    "args": {
                        "device_type": event.device_type,
                        "step": event.step,
                    },
                }
            )

        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"traceEvents": trace_events, "displayTimeUnit": "ms"}, handle)
