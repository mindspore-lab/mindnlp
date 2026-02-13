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


def test_allocator_free_uses_pending_events(monkeypatch):
    from mindtorch_v2._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    monkeypatch.setattr(alloc, "_raw_malloc", lambda size: (1234, size))
    fake_event = object()
    monkeypatch.setattr(alloc, "_record_event", lambda stream: fake_event)
    monkeypatch.setattr(alloc, "_event_complete", lambda event: True)
    monkeypatch.setattr(alloc, "_sync_device", lambda: None)

    ptr = alloc.malloc(512, stream="s0")
    alloc.free(ptr, stream="s0")
    alloc.synchronize()

    stats = alloc.memory_stats()
    assert stats["active_bytes.all.current"] == 0


def test_empty_cache_releases_cached(monkeypatch):
    from mindtorch_v2._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    monkeypatch.setattr(alloc, "_raw_malloc", lambda size: (1234, size))
    freed = []
    monkeypatch.setattr(alloc, "_raw_free", lambda ptr: freed.append(ptr))
    monkeypatch.setattr(alloc, "_record_event", lambda stream: object())
    monkeypatch.setattr(alloc, "_event_complete", lambda event: True)
    monkeypatch.setattr(alloc, "_sync_device", lambda: None)

    ptr = alloc.malloc(512)
    alloc.free(ptr, stream=None)
    alloc.synchronize()
    alloc.empty_cache()

    stats = alloc.memory_stats()
    assert stats["reserved_bytes.all.current"] == 0
    assert freed


def test_allocator_reuses_cached_block(monkeypatch):
    from mindtorch_v2._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    calls = []

    def fake_raw_malloc(size):
        calls.append(size)
        return (1000 + len(calls), size)

    monkeypatch.setattr(alloc, "_raw_malloc", fake_raw_malloc)
    monkeypatch.setattr(alloc, "_record_event", lambda stream: object())
    monkeypatch.setattr(alloc, "_event_complete", lambda event: True)
    monkeypatch.setattr(alloc, "_sync_device", lambda: None)

    ptr = alloc.malloc(512, stream="s0")
    alloc.free(ptr, stream="s0")
    alloc.synchronize()
    calls.clear()

    ptr2 = alloc.malloc(512, stream="s1")
    stats = alloc.memory_stats()

    assert ptr2 == ptr
    assert calls == []
    assert stats["reserved_bytes.all.current"] == 512


def test_allocator_splits_cached_block(monkeypatch):
    from mindtorch_v2._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    calls = []

    def fake_raw_malloc(size):
        calls.append(size)
        return (2000 + len(calls), size)

    monkeypatch.setattr(alloc, "_raw_malloc", fake_raw_malloc)
    monkeypatch.setattr(alloc, "_record_event", lambda stream: object())
    monkeypatch.setattr(alloc, "_event_complete", lambda event: True)
    monkeypatch.setattr(alloc, "_sync_device", lambda: None)

    ptr = alloc.malloc(2048, stream="s0")
    alloc.free(ptr, stream="s0")
    alloc.synchronize()
    calls.clear()

    ptr2 = alloc.malloc(512, stream="s1")
    stats = alloc.memory_stats()

    assert calls == []
    assert ptr2 == ptr
    assert stats["inactive_split_bytes.all.current"] == 1536
