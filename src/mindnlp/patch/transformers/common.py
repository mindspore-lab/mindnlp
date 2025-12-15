"""
Common patches for all transformers versions
"""

from ..registry import register_transformers_patch
from ...utils.decorators import dtype_wrapper, patch_dtype_wrapper, patch_wrappers
from ..utils import (
    _load_pretrained_model_wrapper,
    _get_resolved_checkpoint_files_wrapper,
    apply_chat_template_wrapper,
    create_causal_mask,
    create_sliding_window_causal_mask,
    create_masks_for_generate,
    training_step,
    dynamic_layer_update,
    dynamic_sliding_window_layer_update,
)


@register_transformers_patch(">=4.0.0", priority=5,
                            description="Common patches for all transformers versions")
def patch_common_features():
    """所有版本通用的补丁"""
    import transformers

    def not_supported():
        return False

    def empty_fn(*args, **kwargs):
        pass

    # Disable torch-specific features
    transformers.utils.import_utils._torch_fx_available = False
    transformers.utils.import_utils.is_torch_sdpa_available = not_supported

    # Patch AutoModel
    patch_dtype_wrapper(transformers.AutoModel, 'from_pretrained')

    # Patch _load_pretrained_model
    patch_wrappers(
        transformers.modeling_utils.PreTrainedModel,
        '_load_pretrained_model',
        [_load_pretrained_model_wrapper]
    )

    # Patch _get_resolved_checkpoint_files
    transformers.modeling_utils._get_resolved_checkpoint_files = _get_resolved_checkpoint_files_wrapper(
        transformers.modeling_utils._get_resolved_checkpoint_files
    )

    # Patch tokenization
    transformers.tokenization_utils_base.PreTrainedTokenizerBase.apply_chat_template = apply_chat_template_wrapper(
        transformers.tokenization_utils_base.PreTrainedTokenizerBase.apply_chat_template
    )
    transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__ = apply_chat_template_wrapper(
        transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__
    )

    # Patch pipeline
    transformers.pipelines.pipeline = dtype_wrapper(transformers.pipelines.pipeline)

    # Patch caching
    transformers.modeling_utils.caching_allocator_warmup = empty_fn

    # Patch masking utils
    transformers.masking_utils.create_causal_mask = create_causal_mask
    transformers.masking_utils.create_sliding_window_causal_mask = create_sliding_window_causal_mask
    transformers.masking_utils.create_masks_for_generate = create_masks_for_generate
    transformers.generation.utils.create_masks_for_generate = create_masks_for_generate

    # Patch trainer
    transformers.trainer.Trainer.training_step = training_step

    # Patch cache utils
    transformers.cache_utils.DynamicLayer.update = dynamic_layer_update
    transformers.cache_utils.DynamicSlidingWindowLayer.update = dynamic_sliding_window_layer_update

