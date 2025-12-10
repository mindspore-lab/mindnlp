"""
Common patches for diffusers
"""

from ..registry import register_diffusers_patch
from ...utils.decorators import patch_dtype_wrapper


@register_diffusers_patch(">=0.20.0", priority=10,
                         description="Patch DiffusionPipeline and ModelMixin")
def patch_diffusers_common():
    """diffusers 通用补丁"""
    try:
        import diffusers
        
        patch_dtype_wrapper(
            diffusers.pipelines.pipeline_utils.DiffusionPipeline,
            'from_pretrained'
        )
        patch_dtype_wrapper(
            diffusers.models.modeling_utils.ModelMixin,
            'from_pretrained'
        )
    except ImportError:
        pass

