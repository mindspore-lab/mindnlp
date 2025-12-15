"""
Patches for transformers 4.55.x
"""

from ..registry import register_transformers_patch
from ...utils.decorators import patch_dtype_wrapper


@register_transformers_patch(">=4.55.0,<4.56.0", priority=10,
                            description="Patch PreTrainedModel for transformers 4.55.x")
def patch_pre_trained_model_v4_55():
    """针对 transformers 4.55.x 的补丁"""
    import transformers

    patch_dtype_wrapper(
        transformers.modeling_utils.PreTrainedModel,
        'from_pretrained',
        [transformers.modeling_utils.restore_default_torch_dtype]
    )

