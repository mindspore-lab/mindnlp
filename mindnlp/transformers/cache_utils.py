from typing import Any, Optional
import mindtorch

def dynamic_layer_update(
    self,
    key_states: mindtorch.Tensor,
    value_states: mindtorch.Tensor,
    cache_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[mindtorch.Tensor, mindtorch.Tensor]:
    """
    Update the key and value caches in-place, and return the necessary keys and value states.

    Args:
        key_states (`mindtorch.Tensor`): The new key states to cache.
        value_states (`mindtorch.Tensor`): The new value states to cache.
        cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

    Returns:
        tuple[`mindtorch.Tensor`, `mindtorch.Tensor`]: The key and value states.
    """
    # Lazy initialization
    if not self.is_initialized:
        self.lazy_initialization(key_states)
        self.keys = key_states
        self.values = value_states
    else:
        self.keys = mindtorch.cat([self.keys, key_states], dim=-2)
        self.values = mindtorch.cat([self.values, value_states], dim=-2)
    return self.keys, self.values
