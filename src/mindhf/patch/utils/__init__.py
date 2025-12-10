"""
Patch utility functions

These utilities are used by patches to modify HuggingFace libraries.
"""

from .masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
    create_masks_for_generate
)
from .modeling_utils import (
    construct_pipeline_parallel_model,
    _load_pretrained_model_wrapper,
    _get_resolved_checkpoint_files_wrapper
)
from .cache_utils import (
    dynamic_layer_update,
    dynamic_sliding_window_layer_update
)
from .tokenization_utils import apply_chat_template_wrapper
from .trainer import training_step

__all__ = [
    'create_causal_mask',
    'create_sliding_window_causal_mask',
    'create_masks_for_generate',
    'construct_pipeline_parallel_model',
    '_load_pretrained_model_wrapper',
    '_get_resolved_checkpoint_files_wrapper',
    'dynamic_layer_update',
    'dynamic_sliding_window_layer_update',
    'apply_chat_template_wrapper',
    'training_step',
]

