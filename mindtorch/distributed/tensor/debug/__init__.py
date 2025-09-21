# mypy: allow-untyped-defs
from mindtorch.distributed.tensor.debug._comm_mode import CommDebugMode
from mindtorch.distributed.tensor.debug._visualize_sharding import visualize_sharding


__all__ = ["CommDebugMode", "visualize_sharding"]


def _get_sharding_prop_cache_info():
    """
    Get the cache info for the sharding propagation cache, used for debugging purpose only.
    This would return a named tuple showing hits, misses, maxsize and cursize of the sharding
    propagator cache.
    """
    from mindtorch.distributed.tensor._api import DTensor

    return (
        DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding.cache_info()  # type:ignore[attr-defined]
    )


# Set namespace for exposed private names
CommDebugMode.__module__ = "mindtorch.distributed.tensor.debug"
visualize_sharding.__module__ = "mindtorch.distributed.tensor.debug"
