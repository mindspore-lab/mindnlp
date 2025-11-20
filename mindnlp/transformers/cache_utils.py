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

def dynamic_sliding_window_layer_update(
        self,
        key_states: mindtorch.Tensor,
        value_states: mindtorch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[mindtorch.Tensor, mindtorch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states)
            full_key_states = key_states
            full_value_states = value_states
        else:
            # Compute the full states
            full_key_states = mindtorch.cat([self.keys, key_states], dim=-2)
            full_value_states = mindtorch.cat([self.values, value_states], dim=-2)

        self.cumulative_length += key_states.shape[-2]

        # Only cache the last `self.sliding_window - 1` tokens (or all of them if lower than that)
        self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

        # Return the full states
        return full_key_states, full_value_states