import pytest
import mindtorch_v2 as torch


def test_memory_summary_contains_keys(monkeypatch):
    class DummyAlloc:
        def memory_stats(self):
            return {
                "allocated_bytes.all.current": 1,
                "allocated_bytes.all.peak": 2,
                "reserved_bytes.all.current": 3,
                "reserved_bytes.all.peak": 4,
                "active_bytes.all.current": 5,
                "active_bytes.all.peak": 6,
                "inactive_split_bytes.all.current": 7,
                "inactive_split_bytes.all.peak": 8,
            }

    monkeypatch.setattr(torch.npu, "_get_allocator", lambda device=None: DummyAlloc())
    text = torch.npu.memory_summary()
    assert "Allocated memory" in text
    assert "Reserved memory" in text


def test_memory_snapshot_schema(monkeypatch):
    class DummyAlloc:
        def snapshot(self):
            return {"segments": [], "device": 0, "allocator": "npu"}

    monkeypatch.setattr(torch.npu, "_get_allocator", lambda device=None: DummyAlloc(), raising=False)
    snap = torch.npu.memory_snapshot()
    assert "segments" in snap
    assert "device" in snap
