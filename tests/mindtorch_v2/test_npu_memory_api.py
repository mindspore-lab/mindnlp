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


def test_memory_fraction_enforced(monkeypatch):
    torch.npu._reset_memory_fraction_for_test()
    torch.npu.set_per_process_memory_fraction(0.5)

    monkeypatch.setattr(
        torch.npu,
        "_get_memory_stats",
        lambda device=None: {"total_reserved_memory": 100, "total_allocated_memory": 60},
        raising=False,
    )

    with pytest.raises(RuntimeError):
        torch.npu._enforce_memory_fraction(50)


def test_allocator_enforces_memory_fraction(monkeypatch):
    import mindtorch_v2._backends.npu.allocator as allocator

    torch.npu._reset_memory_fraction_for_test()
    torch.npu.set_per_process_memory_fraction(0.5)

    monkeypatch.setattr(
        torch.npu,
        "_get_memory_stats",
        lambda device=None: {"total_reserved_memory": 100, "total_allocated_memory": 60},
        raising=False,
    )

    alloc = allocator.NpuAllocator(device_id=0)
    with pytest.raises(RuntimeError):
        alloc.malloc(50)
