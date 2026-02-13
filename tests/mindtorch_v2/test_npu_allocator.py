import mindtorch_v2 as torch


def test_allocator_stats_defaults():
    from mindtorch_v2._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    stats = alloc.memory_stats()
    assert stats["allocated_bytes.all.current"] == 0
    assert stats["reserved_bytes.all.current"] == 0
    assert stats["active_bytes.all.current"] == 0
    assert stats["allocated.all.current"] == 0


def test_get_allocator_singleton():
    from mindtorch_v2._backends.npu import allocator

    a1 = allocator.get_allocator(0)
    a2 = allocator.get_allocator(0)
    assert a1 is a2


def test_allocator_allocates_and_tracks(monkeypatch):
    from mindtorch_v2._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    monkeypatch.setattr(alloc, "_raw_malloc", lambda size: (1234, size))

    ptr = alloc.malloc(512)
    stats = alloc.memory_stats()
    assert ptr == 1234
    assert stats["allocated_bytes.all.current"] >= 512
    assert stats["allocated.all.current"] == 1
    assert stats["segment.all.current"] == 1
