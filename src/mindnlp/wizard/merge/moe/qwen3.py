# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

import logging
from typing import List, Optional

import mindspore  # pylint: disable=import-error
import tqdm

try:
    from mindnlp.transformers import PretrainedConfig
    from mindnlp.transformers.models.qwen3_moe import Qwen3MoeConfig
except ImportError:
    from transformers import PretrainedConfig
    from transformers.models.qwen3_moe import Qwen3MoeConfig

from ..architecture import NAME_TO_ARCH
from .arch import MoEOutputArchitecture
from .common import copy_tensor_out, initialize_io, select_dtype
from .config import MoEMergeConfig
from ..options import MergeOptions

QWEN3_INFO = NAME_TO_ARCH["Qwen3ForCausalLM"][0]


class Qwen3MoE(MoEOutputArchitecture):
    def name(self) -> str:
        return "Qwen3 MoE"

    def supports_config(
        self,
        config: MoEMergeConfig,
        explain: bool = False,
        trust_remote_code: bool = False,
    ) -> bool:
        if len(config.shared_experts or []) != 0:
            if explain:
                logging.warning("Qwen3 MoE merge does not support shared experts")
            return False

        for model_ref in (
            [config.base_model]
            + [e.source_model for e in config.experts]
            + [e.source_model for e in (config.shared_experts or [])]
        ):
            model_cfg = model_ref.config(trust_remote_code=trust_remote_code)
            if model_cfg.model_type != "qwen3":
                if explain:
                    logging.warning("Qwen3 MoE only supports Qwen3 input models")
                return False
        return True

    def _generate_config(
        self,
        base_config: PretrainedConfig,
        num_experts: int,
        experts_per_token: Optional[int] = None,
    ) -> Qwen3MoeConfig:
        out_cfg = Qwen3MoeConfig(**base_config.to_dict())
        out_cfg.architectures = ["Qwen3MoeForCausalLM"]
        out_cfg.num_experts = num_experts
        out_cfg.num_experts_per_tok = experts_per_token or 2
        out_cfg.decoder_sparse_step = 1
        out_cfg.norm_topk_prob = True
        out_cfg.moe_intermediate_size = out_cfg.intermediate_size

        if (out_cfg.num_experts & (out_cfg.num_experts - 1)) != 0:
            logging.warning(
                f"Your model has {out_cfg.num_experts} experts, which is "
                "not a power of two. The model will not be usable in llama.cpp."
            )
        return out_cfg

    def write_model(  # pylint: disable=too-many-positional-arguments
        self,
        out_path: str,
        config: MoEMergeConfig,
        merge_options: MergeOptions,
        router_weights: List[mindspore.Tensor],
        shared_router_weights: Optional[List[mindspore.Tensor]] = None,
    ):
        base_model = config.base_model
        base_cfg = base_model.config(trust_remote_code=merge_options.trust_remote_code)

        out_dtype = select_dtype(config, base_cfg)
        out_cfg = self._generate_config(
            base_cfg,
            len(config.experts),
            config.experts_per_token,
        )
        if out_dtype is not None:
            out_cfg.torch_dtype = out_dtype
        out_cfg.save_pretrained(out_path)

        loaders, base_loader, writer = initialize_io(config, out_path, merge_options)
        for weight_info in tqdm.tqdm(
            QWEN3_INFO.all_weights(base_cfg),
            desc="Weights",
        ):
            tensor_name = weight_info.name
            if ".mlp." in tensor_name:
                for expert_idx, expert in enumerate(config.experts):
                    expert_name = tensor_name.replace(
                        ".mlp.", f".mlp.experts.{expert_idx}."
                    )
                    expert_loader = loaders.get(expert.source_model)
                    copy_tensor_out(
                        weight_info,
                        expert_loader,
                        writer,
                        expert=expert,
                        is_residual="down_proj" in tensor_name,
                        output_name=expert_name,
                        out_dtype=out_dtype,
                        clone=merge_options.clone_tensors,
                    )
            else:
                tensor = base_loader.get_tensor(
                    tensor_name,
                    aliases=weight_info.aliases,
                    raise_on_missing=not weight_info.optional,
                )
                if tensor is None:
                    continue

                writer.save_tensor(
                    tensor_name,
                    tensor.astype(out_dtype) if out_dtype is not None else tensor,
                    clone=merge_options.clone_tensors,
                )

        for layer_idx, weight in enumerate(
            tqdm.tqdm(router_weights, desc="Router weights")
        ):
            tensor = weight.astype(out_dtype) if out_dtype is not None else weight
            writer.save_tensor(
                f"model.layers.{layer_idx}.mlp.gate.weight",
                tensor,
                clone=merge_options.clone_tensors,
            )

        writer.finalize()
