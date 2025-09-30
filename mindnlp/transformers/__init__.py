import sys
from packaging import version

import transformers

from mindnlp.utils.import_utils import *
from mindnlp.utils.import_utils import _LazyModule


from .masking_utils import create_causal_mask, create_sliding_window_causal_mask, create_masks_for_generate
from .modeling_utils import construct_pipeline_parallel_model, _load_pretrained_model_wrapper, \
    _get_resolved_checkpoint_files_wrapper
from .tokenization_utils import apply_chat_template_wrapper
from .trainer import training_step
from ..utils.decorators import dtype_wrapper, patch_dtype_wrapper, patch_wrappers

# redirect mindnlp.transformers to transformers
sys.modules[__name__] = _LazyModule(
    'transformers',
    transformers.__file__,
    transformers._import_structure,
    module_spec=__spec__,
    extra_objects={"__version__": transformers.__version__},
)


# patch transformers
def not_supported():
    return False

def empty_fn(*args, **kwargs):
    pass

transformers.utils.import_utils._torch_fx_available = False
transformers.utils.import_utils.is_torch_sdpa_available = not_supported



patch_dtype_wrapper(transformers.AutoModel, 'from_pretrained')
if version.parse(transformers.__version__) >= version.parse('4.56.0'):
    patch_dtype_wrapper(transformers.modeling_utils.PreTrainedModel, 'from_pretrained',
                        [transformers.modeling_utils.restore_default_dtype]
                        )
else:
    patch_dtype_wrapper(transformers.modeling_utils.PreTrainedModel, 'from_pretrained',
                        [transformers.modeling_utils.restore_default_torch_dtype]
                        )
patch_wrappers(transformers.modeling_utils.PreTrainedModel, '_load_pretrained_model',
                [_load_pretrained_model_wrapper])

transformers.modeling_utils._get_resolved_checkpoint_files = _get_resolved_checkpoint_files_wrapper(
    transformers.modeling_utils._get_resolved_checkpoint_files
)

transformers.tokenization_utils_base.PreTrainedTokenizerBase.apply_chat_template = apply_chat_template_wrapper(
    transformers.tokenization_utils_base.PreTrainedTokenizerBase.apply_chat_template
)
transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__ = apply_chat_template_wrapper(
    transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__
)

transformers.pipelines.pipeline = dtype_wrapper(transformers.pipelines.pipeline)
transformers.modeling_utils.caching_allocator_warmup = empty_fn
transformers.masking_utils.create_causal_mask = create_causal_mask
transformers.masking_utils.create_sliding_window_causal_mask = create_sliding_window_causal_mask
transformers.masking_utils.create_masks_for_generate = create_masks_for_generate
transformers.generation.utils.create_masks_for_generate = create_masks_for_generate

transformers.trainer.Trainer.training_step = training_step

# add mindnlp.transformers modules/attrs to lazymodule
# setattr(sys.modules[__name__], 'test_ms_model', test_ms_model)
