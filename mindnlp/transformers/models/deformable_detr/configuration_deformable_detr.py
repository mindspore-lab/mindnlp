# coding=utf-8
# Copyright 2022 SenseTime and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deformable DETR model configuration"""

from mindnlp.transformers.configuration_utils import PretrainedConfig
from mindnlp.utils import logging
#from ...utils.backbone_utils import verify_backbone_config_arguments
from mindnlp.transformers.models.auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)
'''
def verify_backbone_config_arguments(use_timm_backbone, use_pretrained_backbone, backbone, backbone_config, backbone_kwargs):
    """
    Verifies that the backbone configuration arguments are valid.

    Args:
        use_timm_backbone (bool): Whether to use timm's backbone models.
        use_pretrained_backbone (bool): Whether to use a pretrained backbone model.
        backbone (str): The name of the backbone model.
        backbone_config (dict): Configuration dictionary for the backbone.
        backbone_kwargs (dict): Additional keyword arguments for the backbone.

    Raises:
        ValueError: If the arguments are not consistent or valid.
    """
    if use_timm_backbone and backbone_config:
        raise ValueError("Cannot use both `use_timm_backbone` and `backbone_config`. Please choose one.")

    if not use_timm_backbone and not backbone_config:
        raise ValueError("You must provide either `use_timm_backbone` or `backbone_config`.")

    if use_timm_backbone and not backbone:
        raise ValueError("Backbone name must be provided when `use_timm_backbone` is True.")

    if use_timm_backbone and backbone_kwargs is None:
        raise ValueError("`backbone_kwargs` must be provided when `use_timm_backbone` is True.")

    if use_pretrained_backbone and not use_timm_backbone:
        raise ValueError("`use_pretrained_backbone` can only be True when `use_timm_backbone` is also True.")

    if backbone_config and not isinstance(backbone_config, dict):
        raise ValueError("`backbone_config` must be a dictionary.")

    if backbone_kwargs and not isinstance(backbone_kwargs, dict):
        raise ValueError("`backbone_kwargs` must be a dictionary.")

    print("Backbone configuration arguments are valid.")'''

class DeformableDetrConfig(PretrainedConfig):

    model_type = "deformable_detr"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    def __init__(
        self,
        use_timm_backbone=False,    # MindNLP defaults to false
        backbone_config=None,
        num_channels=3,
        num_queries=300,
        max_position_embeddings=1024,
        encoder_layers=6,
        encoder_ffn_dim=1024,
        encoder_attention_heads=8,
        decoder_layers=6,
        decoder_ffn_dim=1024,
        decoder_attention_heads=8,
        encoder_layerdrop=0.0,
        is_encoder_decoder=True,
        activation_function="relu",
        d_model=256,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        init_xavier_std=1.0,
        return_intermediate=True,
        auxiliary_loss=False,
        position_embedding_type="sine",
        backbone=None,
        use_pretrained_backbone=False,
        backbone_kwargs=None,
        dilation=False,
        num_feature_levels=4,
        encoder_num_points=4,
        decoder_num_points=4,
        two_stage=False,
        two_stage_num_proposals=300,
        with_box_refine=False,
        class_cost=1,
        bbox_cost=5,
        giou_cost=2,
        mask_loss_coefficient=1,
        dice_loss_coefficient=1,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        eos_coefficient=0.1,
        focal_alpha=0.25,
        disable_custom_kernels=False,
        **kwargs,
    ):
        if backbone_config is not None and backbone is not None:
            raise ValueError("You can't specify both `backbone` and `backbone_config`.")

        if backbone_config is not None and use_timm_backbone:
            raise ValueError("You can't specify both `backbone_config` and `use_timm_backbone`.")

        if backbone_kwargs is not None and backbone_kwargs and backbone_config is not None:
            raise ValueError("You can't specify both `backbone_kwargs` and `backbone_config`.")
        # We default to values which were previously hard-coded in the model. This enables configurability of the config
        # while keeping the default behavior the same.
        if use_timm_backbone and backbone_kwargs is None:
            backbone_kwargs = {}
            if dilation:
                backbone_kwargs["output_stride"] = 16
            backbone_kwargs["out_indices"] = [2, 3, 4] if num_feature_levels > 1 else [4]
            backbone_kwargs["in_chans"] = num_channels
        # Backwards compatibility
        elif not use_timm_backbone and backbone in (None, "resnet50"):
            # 在初始化配置时，确保backbone_config是字典
            if backbone_config is None:
                logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
                backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage4"])
            elif isinstance(backbone_config, dict):
                backbone_model_type = backbone_config.get("model_type")
                if backbone_model_type is None:
                    raise ValueError("`backbone_config` must include a valid `model_type` key.")
                config_class = CONFIG_MAPPING[backbone_model_type]
                backbone_config = config_class.from_dict(backbone_config)

        self.use_timm_backbone = use_timm_backbone
        self.backbone_config = backbone_config
        self.num_channels = num_channels
        self.num_queries = num_queries
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.encoder_layerdrop = encoder_layerdrop
        self.auxiliary_loss = auxiliary_loss
        self.position_embedding_type = position_embedding_type
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.backbone_kwargs = backbone_kwargs
        self.dilation = dilation
        # deformable attributes
        self.num_feature_levels = num_feature_levels
        self.encoder_num_points = encoder_num_points
        self.decoder_num_points = decoder_num_points
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.with_box_refine = with_box_refine
        if two_stage is True and with_box_refine is False:
            raise ValueError("If two_stage is True, with_box_refine must be True.")
        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Loss coefficients
        self.mask_loss_coefficient = mask_loss_coefficient
        self.dice_loss_coefficient = dice_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.eos_coefficient = eos_coefficient
        self.focal_alpha = focal_alpha
        self.disable_custom_kernels = disable_custom_kernels
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

    @classmethod
    def from_backbone_config(cls, backbone_config: PretrainedConfig, **kwargs):
        """Instantiate a [`DetrConfig`] (or a derived class) from a pre-trained backbone model configuration.

        Args:
            backbone_config ([`PretrainedConfig`]):
                The backbone configuration.
        Returns:
            [`DetrConfig`]: An instance of a configuration object
        """
        return cls(backbone_config=backbone_config, **kwargs)

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model
__all__ = ['DeformableDetrConfig']