from typing import Callable
import mindspore

def vmap(
    func: Callable,
    in_dims = 0,
    out_dims = 0,
    randomness: str = "error",
    *,
    chunk_size=None,
) -> Callable:
    return mindspore.vmap(func, in_dims, out_dims)
