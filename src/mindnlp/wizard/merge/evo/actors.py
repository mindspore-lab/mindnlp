# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

import gc
import logging
import tempfile
from typing import Optional, Union

import lm_eval  # pylint: disable=import-error
import lm_eval.api.model  # pylint: disable=import-error
import lm_eval.models.huggingface  # pylint: disable=import-error
import lm_eval.tasks  # pylint: disable=import-error
import mindspore  # pylint: disable=import-error
import ray  # pylint: disable=import-error
import ray.util.queue  # pylint: disable=import-error
import ray.util.scheduling_strategies  # pylint: disable=import-error
import transformers
from transformers.utils import is_flash_attn_2_available

from ..architecture.base import ConfiguredModelArchitecture

try:
    import vllm
except ImportError:
    vllm = None

from ..architecture import arch_info_for_config
from ..common import get_accelerator_type
from ..config import MergeConfiguration
from .config import EvolMergeConfiguration
from .genome import InvalidGenotypeError, ModelGenome
from .helpers import _eval_model, evaluate_model, merge_model
from .monkeypatch import (
    NoInit,
    monkeypatch_lmeval_shuffle,
    monkeypatch_lmeval_vllm,
)
from ..graph import Executor
from ..io.tasks import LoaderCache, ReturnTensor
from ..merge import _model_out_config
from ..options import MergeOptions
from ..plan import MergePlanner

LOG = logging.getLogger(__name__)


class MergeActorBase:
    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        model_storage_path: Optional[str] = None,
        vllm: bool = False,
        batch_size: Optional[int] = None,
        task_manager: Optional[lm_eval.tasks.TaskManager] = None,
        quantization_config: Optional[transformers.BitsAndBytesConfig] = None,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.cache = LoaderCache()
        self.cache.setup(merge_options)
        self.model_storage_path = model_storage_path
        self.vllm = vllm
        self.batch_size = batch_size
        self.task_manager = task_manager
        self.quantization_config = quantization_config

        if config.shuffle:
            monkeypatch_lmeval_shuffle()

        monkeypatch_lmeval_vllm()


@ray.remote(num_cpus=1, num_gpus=1.0)
class OnDiskMergeEvaluator(MergeActorBase):
    """
    Merges models to disk then evaluates them in a separate process.

    Maximum compatibility and potential for parallelism, but higher overhead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_genotype(
        self,
        genotype: mindspore.Tensor,
    ) -> dict:
        gc.collect()
        accelerator_type = get_accelerator_type(self.merge_options.device)
        if accelerator_type == "Ascend":
            try:
                import acl  # pylint: disable=import-error
                acl.rt.reset_device(0)
            except Exception as exc:
                LOG.debug(
                    "Failed to reset Ascend device before evaluation (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
        LOG.info("Merging model")
        merged_path = merge_model(
            genotype, self.genome, self.model_storage_path, self.merge_options
        )
        if not merged_path:
            LOG.error("Model merge failed")
            return {"score": None, "results": None}

        model_kwargs = {}
        if self.quantization_config is not None:
            model_kwargs["quantization_config"] = self.quantization_config
        LOG.info(f"Model merged to {merged_path}")
        return evaluate_model(
            merged_path,
            self.config.tasks,
            num_fewshot=self.config.num_fewshot,
            limit=self.config.limit,
            vllm=self.vllm,
            batch_size=self.batch_size,
            task_manager=self.task_manager,
            apply_chat_template=self.config.apply_chat_template,
            fewshot_as_multiturn=self.config.fewshot_as_multiturn,
            model_kwargs=model_kwargs,
        )


@ray.remote(num_cpus=1, num_gpus=1)
class InMemoryMergeEvaluator(MergeActorBase):
    """
    Performs merges in memory, using a single model instance.

    This reduces overhead from disk I/O and model loading, but prevents
    parallelism and may be slower for large models.

    Implementation is dark sorcery tampering with the internals of lm-eval,
    transformers, and vLLM and may break at any time.
    """

    model: Union[
        lm_eval.models.huggingface.HFLM, lm_eval.models.vllm_causallms.VLLM, None
    ] = None
    arch_info: Optional[ConfiguredModelArchitecture] = None

    def __init__(
        self,
        *args,
        vllm: bool = False,
        **kwargs,
    ):
        super().__init__(*args, vllm=vllm, **kwargs)

    def _maybe_init_model(self, config: MergeConfiguration):
        ai = arch_info_for_config(self.genome._input_config_example)
        cfg_out = _model_out_config(
            config,
            ai,
            trust_remote_code=self.merge_options.trust_remote_code,
        )
        cfg_out.use_cache = True
        cfg_out.ms_dtype = mindspore.bfloat16

        if self.arch_info is not None:
            different = False
            for key in cfg_out.to_diff_dict():
                if key in ["architectures", "model_type"]:
                    continue
                if key in ["use_cache", "ms_dtype", "torch_dtype"]:
                    continue
                if key.endswith("_token_id"):
                    setattr(self.arch_info.config, key, getattr(cfg_out, key, None))
                    continue

                if getattr(cfg_out, key) != getattr(self.arch_info.config, key, None):
                    LOG.warning(f"Config key {key} changed, reinitializing model")
                    different = True
                    break

            if not different:
                return

        self.inner_model = None

        model_kwargs = {
            "trust_remote_code": self.merge_options.trust_remote_code,
            "torch_dtype": "bfloat16",
        }
        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        with NoInit():
            inner_model = transformers.AutoModelForCausalLM.from_config(
                cfg_out,
                **model_kwargs,
            )
            inner_model.eval()

        if self.vllm:
            with tempfile.TemporaryDirectory(
                dir=self.model_storage_path, prefix="vllm"
            ) as tempdir:
                inner_model.save_pretrained(
                    tempdir, safe_serialization=True, out_shard_size=1_000_000_000_000
                )
                del inner_model
                tokenizer_donor = self.genome.definition.base_model
                if tokenizer_donor is None:
                    LOG.warning(
                        "Base model not set, using tokenizer from first model in genome"
                    )
                    tokenizer_donor = self.genome.definition.models[0]
                tok = transformers.AutoTokenizer.from_pretrained(
                    tokenizer_donor.model.path, use_fast=True
                )
                tok.save_pretrained(tempdir)

                max_model_len = None
                if (
                    seq_len := getattr(cfg_out, "max_position_embeddings", None)
                ) is not None:
                    max_model_len = seq_len
                if (window_sz := getattr(cfg_out, "sliding_window", None)) is not None:
                    max_model_len = min(max_model_len or 1024, window_sz)
                if max_model_len and max_model_len > 8192:
                    max_model_len = 8192
                    LOG.warning(f"Clipping sequence length to {max_model_len}")

                accelerator_type = get_accelerator_type(self.merge_options.device)
                mem_util = (
                    0.7 if accelerator_type in ["Ascend", "cuda", "xpu"] else 0.9
                )
                self.model = lm_eval.models.vllm_causallms.VLLM(
                    pretrained=tempdir,
                    batch_size=self.batch_size or "auto",
                    max_model_len=max_model_len,
                    gpu_memory_utilization=mem_util,
                    dtype="bfloat16",
                    device=self.merge_options.device,
                    trust_remote_code=self.merge_options.trust_remote_code,
                )
        else:
            self.model = lm_eval.models.huggingface.HFLM(pretrained=inner_model)
        self.arch_info = (
            ConfiguredModelArchitecture(
                info=ai,
                config=cfg_out,
            )
            if ai
            else None
        )
        LOG.info("Model initialized")

    def evaluate(self, genotype: mindspore.Tensor) -> dict:
        try:
            config = self.genome.genotype_merge_config(genotype)
        except InvalidGenotypeError as e:
            LOG.error("Invalid genotype", exc_info=e)
            return {"score": None, "results": None}

        self._maybe_init_model(config)

        planner = MergePlanner(
            config,
            self.arch_info.info,
            self.merge_options,
            self.arch_info.config,
        )

        tasks = planner.plan_in_memory()

        model = self.model.model
        if vllm is not None and isinstance(model, vllm.LLM):
            assert (
                model.llm_engine.parallel_config.world_size == 1
            ), "Must be single GPU"
            engine = model.llm_engine
            if hasattr(engine, "model_executor"):
                worker = engine.model_executor.worker
            elif hasattr(engine, "driver_worker"):
                worker = engine.driver_worker
            else:
                raise ValueError("Unknown LLM engine type")
            model = worker.model_runner.model
        param_dict = dict(model.named_parameters())

        stacked_mapping = {
            ".q_proj.": (".qkv_proj.", "q"),
            ".k_proj.": (".qkv_proj.", "k"),
            ".v_proj.": (".qkv_proj.", "v"),
            ".gate_proj.": (".gate_up_proj.", 0),
            ".up_proj.": (".gate_up_proj.", 1),
        }

        accelerator_type = get_accelerator_type(self.merge_options.device)
        executor = Executor(
            tasks,
            math_device=(
                self.merge_options.device
                if accelerator_type in ["Ascend", "cuda", "xpu"]
                else "CPU"
            ),
            storage_device=(
                self.merge_options.device
                if accelerator_type in ["Ascend", "cuda", "xpu"]
                else "CPU"
            ),
        )
        for tensor_task, value in executor.run(quiet=True):
            assert isinstance(tensor_task, ReturnTensor)
            name = tensor_task.weight_info.name

            if name in param_dict:
                param_dict[name].set_data(value)
            elif self.vllm:
                stacked = False
                for needle, (replacement, shard_id) in stacked_mapping.items():
                    if needle in name:
                        target = name.replace(needle, replacement)
                        param = param_dict[target]
                        weight_loader = param.weight_loader
                        weight_loader(param, value, shard_id)
                        stacked = True
                        break

                if not stacked:
                    raise ValueError(f"Unknown parameter {name}")
            else:
                raise ValueError(f"Unknown parameter {name}")

            del value

        return _eval_model(
            self.model,
            self.config.tasks,
            num_fewshot=self.config.num_fewshot,
            limit=self.config.limit,
            task_manager=self.task_manager,
            batch_size=self.batch_size,
            apply_chat_template=self.config.apply_chat_template,
            fewshot_as_multiturn=self.config.fewshot_as_multiturn,
        )

    def evaluate_genotype(
        self,
        genotype: mindspore.Tensor,
    ) -> dict:
        return self.evaluate(genotype)
