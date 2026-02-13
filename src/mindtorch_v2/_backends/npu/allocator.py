SMALL_POOL_THRESHOLD = 1 << 20
ROUNDING_BYTES = 512
MAX_SPLIT_SIZE = 1 << 20


_ALL_STAT_KEYS = [
    "allocated.all.current",
    "allocated.all.peak",
    "allocated.all.allocated",
    "allocated.all.freed",
    "allocated.small_pool.current",
    "allocated.small_pool.peak",
    "allocated.small_pool.allocated",
    "allocated.small_pool.freed",
    "allocated.large_pool.current",
    "allocated.large_pool.peak",
    "allocated.large_pool.allocated",
    "allocated.large_pool.freed",
    "allocated_bytes.all.current",
    "allocated_bytes.all.peak",
    "allocated_bytes.all.allocated",
    "allocated_bytes.all.freed",
    "allocated_bytes.small_pool.current",
    "allocated_bytes.small_pool.peak",
    "allocated_bytes.small_pool.allocated",
    "allocated_bytes.small_pool.freed",
    "allocated_bytes.large_pool.current",
    "allocated_bytes.large_pool.peak",
    "allocated_bytes.large_pool.allocated",
    "allocated_bytes.large_pool.freed",
    "active.all.current",
    "active.all.peak",
    "active.all.allocated",
    "active.all.freed",
    "active.small_pool.current",
    "active.small_pool.peak",
    "active.small_pool.allocated",
    "active.small_pool.freed",
    "active.large_pool.current",
    "active.large_pool.peak",
    "active.large_pool.allocated",
    "active.large_pool.freed",
    "active_bytes.all.current",
    "active_bytes.all.peak",
    "active_bytes.all.allocated",
    "active_bytes.all.freed",
    "active_bytes.small_pool.current",
    "active_bytes.small_pool.peak",
    "active_bytes.small_pool.allocated",
    "active_bytes.small_pool.freed",
    "active_bytes.large_pool.current",
    "active_bytes.large_pool.peak",
    "active_bytes.large_pool.allocated",
    "active_bytes.large_pool.freed",
    "segment.all.current",
    "segment.all.peak",
    "segment.all.allocated",
    "segment.all.freed",
    "segment.small_pool.current",
    "segment.small_pool.peak",
    "segment.small_pool.allocated",
    "segment.small_pool.freed",
    "segment.large_pool.current",
    "segment.large_pool.peak",
    "segment.large_pool.allocated",
    "segment.large_pool.freed",
    "reserved_bytes.all.current",
    "reserved_bytes.all.peak",
    "reserved_bytes.all.allocated",
    "reserved_bytes.all.freed",
    "reserved_bytes.small_pool.current",
    "reserved_bytes.small_pool.peak",
    "reserved_bytes.small_pool.allocated",
    "reserved_bytes.small_pool.freed",
    "reserved_bytes.large_pool.current",
    "reserved_bytes.large_pool.peak",
    "reserved_bytes.large_pool.allocated",
    "reserved_bytes.large_pool.freed",
    "inactive_split.all.current",
    "inactive_split.all.peak",
    "inactive_split.all.allocated",
    "inactive_split.all.freed",
    "inactive_split.small_pool.current",
    "inactive_split.small_pool.peak",
    "inactive_split.small_pool.allocated",
    "inactive_split.small_pool.freed",
    "inactive_split.large_pool.current",
    "inactive_split.large_pool.peak",
    "inactive_split.large_pool.allocated",
    "inactive_split.large_pool.freed",
    "inactive_split_bytes.all.current",
    "inactive_split_bytes.all.peak",
    "inactive_split_bytes.all.allocated",
    "inactive_split_bytes.all.freed",
    "inactive_split_bytes.small_pool.current",
    "inactive_split_bytes.small_pool.peak",
    "inactive_split_bytes.small_pool.allocated",
    "inactive_split_bytes.small_pool.freed",
    "inactive_split_bytes.large_pool.current",
    "inactive_split_bytes.large_pool.peak",
    "inactive_split_bytes.large_pool.allocated",
    "inactive_split_bytes.large_pool.freed",
    "oversize_allocations.current",
    "oversize_allocations.peak",
    "oversize_allocations.allocated",
    "oversize_allocations.freed",
    "oversize_segments.current",
    "oversize_segments.peak",
    "oversize_segments.allocated",
    "oversize_segments.freed",
    "requested_bytes.all.current",
    "requested_bytes.all.peak",
    "requested_bytes.all.allocated",
    "requested_bytes.all.freed",
    "requested_bytes.small_pool.current",
    "requested_bytes.small_pool.peak",
    "requested_bytes.small_pool.allocated",
    "requested_bytes.small_pool.freed",
    "requested_bytes.large_pool.current",
    "requested_bytes.large_pool.peak",
    "requested_bytes.large_pool.allocated",
    "requested_bytes.large_pool.freed",
    "num_alloc_retries",
    "num_ooms",
    "num_sync_all_streams",
    "num_device_alloc",
    "num_device_free",
    "max_split_size",
]




class Block:
    def __init__(self, ptr, size, requested, pool, stream):
        self.ptr = int(ptr)
        self.size = int(size)
        self.requested = int(requested)
        self.pool = pool
        self.stream = stream
        self.event = None


def _round_size(size):
    size = int(size)
    return ((size + ROUNDING_BYTES - 1) // ROUNDING_BYTES) * ROUNDING_BYTES


class NpuAllocator:
    def __init__(self, device_id):
        self.device_id = int(device_id)
        self._stats = {}
        self._active = {}
        self._cached = {"small_pool": [], "large_pool": []}
        self._pending = []
        self._init_stats()

    def _pool_for_size(self, size):
        return "small_pool" if size < SMALL_POOL_THRESHOLD else "large_pool"

    def _bump(self, prefix, pool, current=0, allocated=0, freed=0):
        pools = (pool, "all") if pool != "all" else ("all",)
        for target in pools:
            if current:
                key = f"{prefix}.{target}.current"
                peak_key = f"{prefix}.{target}.peak"
                self._stats[key] += current
                if self._stats[key] > self._stats[peak_key]:
                    self._stats[peak_key] = self._stats[key]
            if allocated:
                self._stats[f"{prefix}.{target}.allocated"] += allocated
            if freed:
                self._stats[f"{prefix}.{target}.freed"] += freed

    def _track_alloc(self, requested, allocated, pool):
        self._bump("allocated_bytes", pool, current=allocated, allocated=allocated)
        self._bump("requested_bytes", pool, current=requested, allocated=requested)
        self._bump("active_bytes", pool, current=allocated, allocated=allocated)
        self._bump("allocated", pool, current=1, allocated=1)
        self._bump("active", pool, current=1, allocated=1)
        self._bump("segment", pool, current=1, allocated=1)
        self._bump("reserved_bytes", pool, current=allocated, allocated=allocated)
        self._stats["num_device_alloc"] += 1

    def _track_reuse(self, requested, allocated, pool):
        self._bump("allocated_bytes", pool, current=allocated, allocated=allocated)
        self._bump("requested_bytes", pool, current=requested, allocated=requested)
        self._bump("allocated", pool, current=1, allocated=1)
        self._bump("active_bytes", pool, current=allocated, allocated=allocated)
        self._bump("active", pool, current=1, allocated=1)

    def _track_free(self, block):
        self._bump("allocated_bytes", block.pool, current=-block.size, freed=block.size)
        self._bump("requested_bytes", block.pool, current=-block.requested, freed=block.requested)
        self._bump("allocated", block.pool, current=-1, freed=1)
        self._bump("active_bytes", block.pool, current=-block.size, freed=block.size)
        self._bump("active", block.pool, current=-1, freed=1)


    def _find_cached(self, size, pool):
        blocks = self._cached[pool]
        for idx, block in enumerate(blocks):
            if block.size >= size:
                return blocks.pop(idx)
        return None

    def _split_block(self, block, size):
        if block.size - size <= 0:
            return block, None
        if block.size > MAX_SPLIT_SIZE:
            return block, None
        remaining = block.size - size
        block.size = size
        remainder = Block(block.ptr + size, remaining, 0, block.pool, None)
        return block, remainder

    def _drain_pending(self):
        ready = []
        pending = []
        for block in self._pending:
            if self._event_complete(block.event):
                ready.append(block)
            else:
                pending.append(block)
        self._pending = pending
        for block in ready:
            self._cached[block.pool].append(block)
            self._bump("inactive_split_bytes", block.pool, current=block.size, allocated=block.size)
            self._bump("inactive_split", block.pool, current=1, allocated=1)

    def _record_event(self, stream):
        from . import runtime as npu_runtime
        from . import state as npu_state

        runtime = npu_runtime.get_runtime(self.device_id)
        if stream is None:
            stream = npu_state.current_stream(self.device_id).stream
        event = runtime.create_event(False, False, False)
        runtime.record_event(event, stream)
        return event

    def _event_complete(self, event):
        from . import runtime as npu_runtime
        runtime = npu_runtime.get_runtime(self.device_id)
        return runtime.query_event(event)

    def _sync_device(self):
        from . import runtime as npu_runtime
        runtime = npu_runtime.get_runtime(self.device_id)
        runtime.synchronize_device()

    def _raw_free(self, ptr):
        from . import runtime as npu_runtime

        runtime = npu_runtime.get_runtime(self.device_id)
        runtime.activate()
        ret = npu_runtime.acl.rt.free(ptr)
        if ret != 0:
            raise RuntimeError(f"acl.rt.free failed: {ret}")
        self._stats["num_device_free"] += 1

    def _raw_malloc(self, size):
        from . import runtime as npu_runtime

        runtime = npu_runtime.get_runtime(self.device_id)
        runtime.activate()
        ptr, ret = npu_runtime.acl.rt.malloc(size, npu_runtime.ACL_MEM_MALLOC_HUGE_FIRST)
        if ret != 0:
            self._stats["num_ooms"] += 1
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        return int(ptr), int(size)

    def malloc(self, size, stream=None):
        requested = int(size)
        allocated = _round_size(requested)
        pool = self._pool_for_size(allocated)
        self._drain_pending()
        block = self._find_cached(allocated, pool)
        if block is None:
            ptr, _ = self._raw_malloc(allocated)
            block = Block(ptr, allocated, requested, pool, stream)
            self._track_alloc(requested, allocated, pool)
        else:
            cached_size = block.size
            block, remainder = self._split_block(block, allocated)
            self._bump("inactive_split_bytes", pool, current=-cached_size, freed=cached_size)
            self._bump("inactive_split", pool, current=-1, freed=1)
            if remainder is not None:
                self._cached[pool].append(remainder)
                self._bump("inactive_split_bytes", pool, current=remainder.size, allocated=remainder.size)
                self._bump("inactive_split", pool, current=1, allocated=1)
            self._track_reuse(requested, block.size, pool)
        block.requested = requested
        block.stream = stream
        self._active[block.ptr] = block
        return int(block.ptr)

    def free(self, ptr, stream=None):
        block = self._active.pop(int(ptr), None)
        if block is None:
            return
        if stream is None:
            stream = block.stream
        block.event = self._record_event(stream)
        self._pending.append(block)
        self._track_free(block)

    def synchronize(self):
        self._sync_device()
        self._stats["num_sync_all_streams"] += 1
        self._drain_pending()

    def empty_cache(self):
        self.synchronize()
        for pool, blocks in self._cached.items():
            for block in blocks:
                self._raw_free(block.ptr)
                self._bump("reserved_bytes", block.pool, current=-block.size, freed=block.size)
                self._bump("segment", block.pool, current=-1, freed=1)
                self._bump("inactive_split_bytes", block.pool, current=-block.size, freed=block.size)
                self._bump("inactive_split", block.pool, current=-1, freed=1)
            blocks.clear()

    def reset_peak_memory_stats(self):
        for key in list(self._stats.keys()):
            if key.endswith('.peak'):
                self._stats[key] = self._stats[key.replace('.peak', '.current')]

    def reset_accumulated_memory_stats(self):
        for key in list(self._stats.keys()):
            if key.endswith('.allocated') or key.endswith('.freed'):
                self._stats[key] = 0
        self._stats["num_alloc_retries"] = 0
        self._stats["num_ooms"] = 0
        self._stats["num_device_alloc"] = 0
        self._stats["num_device_free"] = 0
        self._stats["num_sync_all_streams"] = 0

    def _init_stats(self):
        for key in _ALL_STAT_KEYS:
            self._stats[key] = 0
        self._stats["max_split_size"] = MAX_SPLIT_SIZE

    def memory_stats(self):
        return dict(self._stats)



_ALLOCATORS = {}


def get_allocator(device_id=0):
    device_id = int(device_id)
    alloc = _ALLOCATORS.get(device_id)
    if alloc is None:
        alloc = NpuAllocator(device_id)
        _ALLOCATORS[device_id] = alloc
    return alloc
