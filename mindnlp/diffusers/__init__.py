import diffusers
from diffusers import *

from ..utils.decorators import patch_dtype_wrapper

patch_dtype_wrapper(diffusers.pipelines.pipeline_utils.DiffusionPipeline, 'from_pretrained')
patch_dtype_wrapper(diffusers.models.modeling_utils.ModelMixin, 'from_pretrained')
