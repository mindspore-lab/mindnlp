import contextlib
import hashlib
import inspect
import json
import time
import threading
from dataclasses import asdict, dataclass


_TLS = threading.local()


def _get_last_window_ops():
    return getattr(_TLS, "last_window_ops", None)


def _set_last_window_ops(value):
    _TLS.last_window_ops = value


@dataclass
class PipelineConfig:
    max_ops: int | None = None
    max_pending_bytes: int | None = None
    max_wait_us: int | None = None
    debug_enabled: bool = False
    min_defer_ops: int | None = None
    adaptive_defer: bool = False
    adaptive_small_window_ops: int = 24
    adaptive_min_defer_ops: int = 4


@dataclass
class ErrorEnvelope:
    error_id: str
    flush_id: int
    op_seq: int
    op_name: str
    schema: str
    phase: str
    backend: str
    callsite: dict
    read_set: list
    write_set: list
    alias_set: str | None
    version_plan: dict
    dependency_edges: list
    runtime_code: str | None
    suppressed_errors: list
    message: str

    def to_dict(self):
        return asdict(self)


_GLOBAL_CONFIG = PipelineConfig()


def get_pipeline_config():
    return asdict(_GLOBAL_CONFIG)


def set_pipeline_config(**kwargs):
    reset_adaptive_state = False
    for key in (
        "max_ops",
        "max_pending_bytes",
        "max_wait_us",
        "debug_enabled",
        "min_defer_ops",
        "adaptive_defer",
        "adaptive_small_window_ops",
        "adaptive_min_defer_ops",
    ):
        if key in kwargs:
            setattr(_GLOBAL_CONFIG, key, kwargs[key])
            if key.startswith("adaptive_") or key == "min_defer_ops":
                reset_adaptive_state = True
    if reset_adaptive_state:
        _set_last_window_ops(None)
    return get_pipeline_config()


def _merged_config(overrides=None):
    cfg = PipelineConfig(**get_pipeline_config())
    if overrides:
        for key in (
            "max_ops",
            "max_pending_bytes",
            "max_wait_us",
            "debug_enabled",
            "min_defer_ops",
            "adaptive_defer",
            "adaptive_small_window_ops",
            "adaptive_min_defer_ops",
        ):
            if key in overrides and overrides[key] is not None:
                setattr(cfg, key, overrides[key])
    return cfg


def _estimate_pending_bytes(obj):
    if obj is None:
        return 0
    if isinstance(obj, (tuple, list)):
        return sum(_estimate_pending_bytes(item) for item in obj)
    numel = getattr(obj, "numel", None)
    element_size = getattr(obj, "element_size", None)
    if callable(numel) and callable(element_size):
        try:
            return int(numel() * element_size())
        except Exception:
            return 0
    return 0


def _infer_callsite():
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_back if frame and frame.f_back else None
        if caller is None:
            return {"file": None, "line": None, "func": None}
        return {
            "file": caller.f_code.co_filename,
            "line": caller.f_lineno,
            "func": caller.f_code.co_name,
        }
    finally:
        del frame


def _bind_schema_args(schema_obj, args, kwargs):
    if schema_obj is None:
        return {}
    if kwargs is None:
        kwargs = {}
    params = schema_obj.params
    positional = [p for p in params if not p.kw_only]
    bound = {}
    for idx, value in enumerate(args or []):
        if idx < len(positional):
            bound[positional[idx].name] = value
    for key, value in kwargs.items():
        bound[key] = value
    return bound


def _collect_alias_info(entry):
    schema_obj = getattr(entry, "schema_obj", None)
    args = getattr(entry, "args", None)
    kwargs = getattr(entry, "kwargs", None)
    if schema_obj is None:
        return [], [], None, {}
    bound = _bind_schema_args(schema_obj, args, kwargs)
    read_set = []
    write_set = []
    alias_set = None
    version_plan = {}
    for param in schema_obj.params:
        if param.name not in bound:
            continue
        value = bound[param.name]
        if hasattr(value, "device"):
            read_set.append(param.name)
        if param.mutates:
            write_set.append(param.name)
            if param.alias_set:
                alias_set = param.alias_set if alias_set is None else alias_set
                version_plan[param.alias_set] = version_plan.get(param.alias_set, 0) + 1
    if alias_set is None:
        for ret in getattr(schema_obj, "returns", []) or []:
            if ret.alias_set:
                alias_set = ret.alias_set
                break
    return read_set, write_set, alias_set, version_plan


def _collect_tensor_rw(entry):
    schema_obj = getattr(entry, "schema_obj", None)
    args = getattr(entry, "args", None)
    kwargs = getattr(entry, "kwargs", None)
    if schema_obj is None:
        return set(), set()
    bound = _bind_schema_args(schema_obj, args, kwargs)
    read = set()
    write = set()
    for param in schema_obj.params:
        if param.name not in bound:
            continue
        value = bound[param.name]
        if hasattr(value, "device"):
            read.add(id(value))
            if param.mutates:
                write.add(id(value))
    return read, write


def _build_dependency_edges(alias_sets_by_op):
    # Build simple dependency edges when consecutive ops share alias_set.
    edges = []
    last_by_alias = {}
    for op_seq in sorted(alias_sets_by_op.keys()):
        alias_set = alias_sets_by_op[op_seq]
        if not alias_set:
            continue
        prev = last_by_alias.get(alias_set)
        if prev is not None:
            edges.append({"from": prev, "to": op_seq, "reason": f"alias_set:{alias_set}"})
        last_by_alias[alias_set] = op_seq
    return edges


def _build_rw_edges(entries):
    edges = []
    last_write = {}
    for entry in entries:
        op_seq = getattr(entry, "_pipe_op_seq", -1)
        read, write = _collect_tensor_rw(entry)
        for rid in read:
            if rid in last_write:
                edges.append({"from": last_write[rid], "to": op_seq, "reason": "write->read"})
        for wid in write:
            if wid in last_write:
                edges.append({"from": last_write[wid], "to": op_seq, "reason": "write->write"})
            last_write[wid] = op_seq
    return edges


def _make_error_id(op_name, phase, callsite, message):
    payload = "|".join(
        [
            op_name or "",
            phase or "",
            str((callsite or {}).get("file")),
            str((callsite or {}).get("line")),
            str((callsite or {}).get("func")),
            message or "",
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


class Pipeline:
    def __init__(self, config=None):
        self.queue = []
        self.outputs = {}
        self.config = config or PipelineConfig()
        self.flush_id = 0
        self.last_flush_reason = None
        self._window_started_ns = None
        self._next_op_seq = 0
        self._last_error = None
        self._last_window_entries = []
        self._last_op_alias_sets = {}
        self._observed_ops = 0
        self._predicted_window_ops = _get_last_window_ops()

    def should_defer_next(self):
        self._observed_ops += 1
        min_defer_ops = self.config.min_defer_ops
        if min_defer_ops is None and self.config.adaptive_defer:
            predicted = self._predicted_window_ops
            if predicted is not None and predicted <= int(self.config.adaptive_small_window_ops):
                min_defer_ops = int(self.config.adaptive_min_defer_ops)
        if min_defer_ops is None:
            return True
        if int(min_defer_ops) <= 1:
            return True
        return self._observed_ops >= int(min_defer_ops)

    def record(self, entry, *, pending=None):
        entry._pipe_op_seq = self._next_op_seq
        self._next_op_seq += 1
        if self.config.debug_enabled and not hasattr(entry, "_pipe_callsite"):
            entry._pipe_callsite = _infer_callsite()
        alias_set = None
        if self.config.debug_enabled:
            _, _, alias_set, _ = _collect_alias_info(entry)
            self._last_op_alias_sets[entry._pipe_op_seq] = alias_set
        entry._pipe_alias_set = alias_set
        self.queue.append(entry)
        if self._window_started_ns is None:
            self._window_started_ns = time.monotonic_ns()
        if pending is not None:
            if isinstance(pending, (tuple, list)):
                for item in pending:
                    self.outputs[id(item)] = item
            else:
                self.outputs[id(pending)] = pending
            entry.out = pending
        if self._should_auto_flush(pending):
            self.flush(reason="ops")

    def _should_auto_flush(self, pending):
        if self.config.max_ops is not None and len(self.queue) >= int(self.config.max_ops):
            return True
        if self.config.max_pending_bytes is not None:
            total = _estimate_pending_bytes(pending)
            if total >= int(self.config.max_pending_bytes):
                return True
        if self.config.max_wait_us is not None and self._window_started_ns is not None:
            elapsed_us = (time.monotonic_ns() - self._window_started_ns) // 1000
            if elapsed_us >= int(self.config.max_wait_us):
                return True
        return False

    def flush(self, reason="manual"):
        pending = list(self.queue)
        self.queue.clear()
        self._last_window_entries = pending
        self._last_op_alias_sets = {}
        if self.config.debug_enabled:
            for entry in pending:
                self._last_op_alias_sets[getattr(entry, "_pipe_op_seq", -1)] = getattr(entry, "_pipe_alias_set", None)
        if not pending:
            self.last_flush_reason = reason
            return
        self._predicted_window_ops = len(pending)
        _set_last_window_ops(self._predicted_window_ops)
        self.flush_id += 1
        self.last_flush_reason = reason
        self._window_started_ns = None
        self._last_error = None
        active_ctx = None
        active_entry = None
        for entry in pending:
            try:
                ctx = (getattr(entry, "keyset", None), getattr(entry, "key", None))
                if ctx != active_ctx:
                    if active_ctx is not None and active_entry is not None:
                        active_entry.exit_dispatch_context()
                    if hasattr(entry, "enter_dispatch_context"):
                        entry.enter_dispatch_context()
                        active_ctx = ctx
                        active_entry = entry
                    else:
                        active_ctx = None
                        active_entry = None
                if active_ctx is not None and hasattr(entry, "execute_with_active_context"):
                    entry.execute_with_active_context()
                else:
                    entry.execute()
            except Exception as exc:  # noqa: BLE001
                if active_ctx is not None and active_entry is not None and hasattr(active_entry, "exit_dispatch_context"):
                    active_entry.exit_dispatch_context()
                    active_ctx = None
                    active_entry = None
                op_name = getattr(entry, "op_name", type(entry).__name__)
                backend = "cpu"
                if hasattr(entry, "args") and entry.args:
                    first = entry.args[0]
                    device = getattr(first, "device", None)
                    backend = getattr(device, "type", backend) if device is not None else backend
                callsite = {"file": None, "line": None, "func": None}
                read_set, write_set, alias_set, version_plan = [], [], None, {}
                dependency_edges = []
                if self.config.debug_enabled:
                    callsite = getattr(entry, "_pipe_callsite", callsite)
                    read_set, write_set, alias_set, version_plan = _collect_alias_info(entry)
                    dependency_edges = _build_dependency_edges(self._last_op_alias_sets)
                    dependency_edges.extend(_build_rw_edges(self._last_window_entries))
                self._last_error = ErrorEnvelope(
                    error_id=_make_error_id(op_name, "submit", callsite, str(exc)),
                    flush_id=self.flush_id,
                    op_seq=getattr(entry, "_pipe_op_seq", -1),
                    op_name=op_name,
                    schema=op_name,
                    phase="submit",
                    backend=backend,
                    callsite=callsite,
                    read_set=read_set,
                    write_set=write_set,
                    alias_set=alias_set,
                    version_plan=version_plan,
                    dependency_edges=dependency_edges,
                    runtime_code=None,
                    suppressed_errors=[],
                    message=str(exc),
                )
                self.outputs.clear()
                raise
        if active_ctx is not None and active_entry is not None:
            active_entry.exit_dispatch_context()
        self.outputs.clear()

    def last_error(self):
        return self._last_error

    def pending_count(self):
        return len(self.queue)

    def format_error(self, style="short"):
        if self._last_error is None:
            return ""
        if style == "short":
            return (
                f"pipeline flush {self._last_error.flush_id} failed at op "
                f"{self._last_error.op_name}#{self._last_error.op_seq}: {self._last_error.message}"
            )
        if style == "full":
            return json.dumps(self._last_error.to_dict(), ensure_ascii=False, sort_keys=True)
        raise ValueError("style must be 'short' or 'full'")

    def debug_dump(self, failed_only=False):
        items = []
        for entry in self._last_window_entries:
            items.append(
                {
                    "op_seq": getattr(entry, "_pipe_op_seq", -1),
                    "op_name": getattr(entry, "op_name", type(entry).__name__),
                    "callsite": getattr(entry, "_pipe_callsite", {"file": None, "line": None, "func": None}),
                }
            )
        if failed_only and self._last_error is not None:
            items = [item for item in items if item["op_seq"] == self._last_error.op_seq]
        return {
            "flush_id": self.flush_id,
            "reason": self.last_flush_reason,
            "pending_count": len(self.queue),
            "last_error": None if self._last_error is None else self._last_error.to_dict(),
            "entries": items,
        }


def _get_current():
    return getattr(_TLS, "current", None)


def _set_current(pipe):
    _TLS.current = pipe


@contextlib.contextmanager
def pipeline_context(**config_overrides):
    prev = _get_current()
    pipe = Pipeline(config=_merged_config(config_overrides))
    _set_current(pipe)
    try:
        yield pipe
    finally:
        pipe.flush(reason="context_exit")
        _set_current(prev)


def current_pipeline():
    return _get_current()
