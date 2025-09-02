from typing import Optional, Callable
import mindspore
from mindspore import ops, nn, mint
from mindnlp import core

def token_type_ids_mask_function(
    token_type_ids: Optional[core.Tensor],
    image_group_ids: Optional[core.Tensor],
    tokens_per_image: int,
) -> Optional[Callable]:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    """
    # Do not return an additional mask in this case
    if token_type_ids is None:
        return None

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # If it's 1 for both query and key/value, we are in an image block
        # NOTE: static cache shape goes beyond input seq length, while token_type_ids.shape[1] == input seq length
        # Since vmap doesn't support `if statement` we workaround it with `torch.where`
        safe_idx = core.where(kv_idx < token_type_ids.shape[1], kv_idx, 0)
        token_type_ids_at_kv_idx = token_type_ids[:, safe_idx]
        token_type_ids_at_kv_idx = core.where(kv_idx < token_type_ids.shape[1], token_type_ids_at_kv_idx, 0)

        image_group_ids_at_kv_idx = image_group_ids[:, safe_idx]
        image_group_ids_at_kv_idx = core.where(kv_idx < image_group_ids.shape[1], image_group_ids_at_kv_idx, -1)

        is_image_block = (token_type_ids[:, q_idx] == 1) & (token_type_ids_at_kv_idx == 1)
        same_image_block = image_group_ids[:, q_idx] == image_group_ids_at_kv_idx

        # This is bidirectional attention whenever we are dealing with image tokens
        return is_image_block & same_image_block

    return inner_mask