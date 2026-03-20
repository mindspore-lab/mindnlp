# Originally from MergeKit (https://github.com/arcee-ai/mergekit)
# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.

"""Main merge entry point."""

import importlib
import importlib.resources
import json
import logging
import os
import shutil
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import tqdm
import transformers

from ._data import chat_templates
from .architecture import ModelArchitecture, get_architecture_info
from .card import generate_card
from .common import ModelReference, set_config_value
from .config import MergeConfiguration
from .graph import Executor
from .io.tasks import LoaderCache
from .multigpu_executor import MultiDeviceExecutor
from .options import MergeOptions
from .plan import MergePlanner
from .preflight import run_merge_preflight
from .tokenizer import TokenizerInfo

LOG = logging.getLogger(__name__)


def run_merge(
    merge_config: MergeConfiguration,
    out_path: str,
    options: MergeOptions,
    config_source: Optional[str] = None,
):
    if options.random_seed is not None:
        transformers.trainer_utils.set_seed(options.random_seed)

    if not merge_config.models and not merge_config.slices and not merge_config.modules:
        raise RuntimeError("No output requested")

    run_merge_preflight(merge_config=merge_config, options=options)

    arch_info = get_architecture_info(merge_config, options)
    loader_cache = LoaderCache()
    loader_cache.setup(options=options)

    cfg_out = _model_out_config(
        merge_config, arch_info, trust_remote_code=options.trust_remote_code
    )

    for model in (
        pbar := tqdm.tqdm(
            merge_config.referenced_models(),
            desc="Warmup loader cache",
            disable=options.quiet,
        )
    ):
        loader_cache.get(model)
    del pbar

    LOG.info("Planning operations")
    targets = MergePlanner(
        merge_config,
        arch_info,
        options=options,
        out_model_config=cfg_out,
    ).plan_to_disk(out_path=out_path)

    if options.multi_npu:
        exec = MultiDeviceExecutor(
            targets=targets,
            storage_device=None if options.low_cpu_memory else "CPU",
        )
    else:
        exec = Executor(
            targets=targets,
            math_device=options.device,
            storage_device=options.device if options.low_cpu_memory else "CPU",
        )

    tokenizer = None
    for _task, value in exec.run(quiet=options.quiet):
        if isinstance(value, TokenizerInfo):
            tokenizer = value.tokenizer

    if hasattr(exec, "metrics_snapshot"):
        _write_execution_report(
            out_path=out_path,
            metrics=exec.metrics_snapshot(),
            metadata={},
        )

    if tokenizer:
        pad_to_multiple_of = None
        if merge_config.tokenizer and merge_config.tokenizer.pad_to_multiple_of:
            pad_to_multiple_of = merge_config.tokenizer.pad_to_multiple_of
        _update_config_vocab(
            cfg_out, arch_info, tokenizer, pad_to_multiple_of=pad_to_multiple_of
        )

    LOG.info("Saving config")
    cfg_out.save_pretrained(out_path)

    if options.write_model_card:
        if not config_source:
            config_source = merge_config.to_yaml()

        card_md = generate_card(
            config=merge_config,
            config_yaml=config_source,
            name=os.path.basename(out_path),
        )
        with open(os.path.join(out_path, "README.md"), "w", encoding="utf-8") as fp:
            fp.write(card_md)

        with open(
            os.path.join(out_path, "wizard_config.yml"), "w", encoding="utf-8"
        ) as fp:
            fp.write(config_source)

    if tokenizer is not None:
        LOG.info("Saving tokenizer")
        _set_chat_template(tokenizer, merge_config)
        tokenizer.save_pretrained(out_path, safe_serialization=True)
    else:
        if options.copy_tokenizer:
            try:
                _copy_tokenizer(merge_config, out_path, options=options)
            except Exception as e:
                LOG.error(
                    "Failed to copy tokenizer. The merge was still successful, "
                    "just copy it from somewhere else.",
                    exc_info=e,
                )
        elif merge_config.chat_template:
            LOG.warning(
                "Chat template specified but no tokenizer found. "
                "Chat template will not be saved."
            )

    _copy_tagalong_files(
        merge_config,
        out_path,
        files=arch_info.tagalong_files or [],
        options=options,
    )

    if getattr(arch_info, "post_fill_parameters", False):
        logging.info(
            "Filling missing parameters from base model %s into new directory",
            arch_info.post_fill_parameters,
        )
        try:
            from .scripts.fill_missing_params import copy_and_fill_missing_params

            copy_and_fill_missing_params(
                base_model_repo_id=arch_info.post_fill_parameters,
                sub_model_dir=out_path,
            )
            logging.info("Deleting initial merge directory: %s", out_path)
            shutil.rmtree(out_path)
        except ImportError:
            LOG.warning(
                "fill_missing_params script not available — skipping post-fill step"
            )


def _set_chat_template(
    tokenizer: transformers.PreTrainedTokenizerBase,
    merge_config: MergeConfiguration,
    trust_remote_code: bool = False,
):
    chat_template = merge_config.chat_template
    if not chat_template:
        return

    if chat_template == "auto":
        model_templates = []
        for model in merge_config.referenced_models():
            try:
                tok = transformers.AutoTokenizer.from_pretrained(
                    model.model.path,
                    revision=model.model.revision,
                    trust_remote_code=trust_remote_code,
                )
                template = tok.chat_template
                if isinstance(template, dict):
                    template = template.get("default", None)
                if template:
                    model_templates.append(template.strip())
            except Exception as e:
                LOG.warning("Unable to load tokenizer for %s", model, exc_info=e)

        if not model_templates:
            return

        chat_template = Counter(model_templates).most_common(1)[0][0]
        LOG.info("Auto-selected chat template: %s", chat_template)

    elif (
        t := importlib.resources.files(chat_templates).joinpath(
            chat_template + ".jinja"
        )
    ).is_file():
        chat_template = t.read_text()

    elif len(chat_template) < 20 or "{" not in chat_template:
        raise RuntimeError(f"Invalid chat template: {chat_template}")

    tokenizer.chat_template = chat_template


def _get_donor_model(
    merge_config: MergeConfiguration,
    options: MergeOptions,
) -> Tuple[ModelReference, str]:
    donor_model = merge_config.base_model or (merge_config.referenced_models()[0])
    donor_local_path = donor_model.merged(
        cache_dir=options.lora_merge_cache,
        trust_remote_code=options.trust_remote_code,
        lora_merge_dtype=options.lora_merge_dtype,
    ).local_path(cache_dir=options.transformers_cache)
    if not donor_local_path:
        raise RuntimeError(f"Unable to find local path for {donor_model}")
    return donor_model, donor_local_path


def _copy_tagalong_files(
    merge_config: MergeConfiguration,
    out_path: str,
    files: List[str],
    options: MergeOptions,
):
    donor_model, donor_local_path = _get_donor_model(merge_config, options=options)

    for file_name in files:
        fp = os.path.join(donor_local_path, file_name)
        if os.path.exists(fp):
            LOG.info("Copying %s from %s", file_name, donor_model)
            shutil.copy(
                fp,
                os.path.join(out_path, file_name),
            )


def _copy_tokenizer(
    merge_config: MergeConfiguration, out_path: str, options: MergeOptions
):
    donor_model, donor_local_path = _get_donor_model(merge_config, options=options)

    # MergeKit-compatible warning: when base_model is used as tokenizer donor,
    # chat templates from instruct/chat models will not be inherited unless
    # users explicitly set `chat_template: auto` or `tokenizer_source`.
    if not merge_config.chat_template and merge_config.base_model and not merge_config.tokenizer_source:
        try:
            donor_tok = transformers.AutoTokenizer.from_pretrained(
                donor_model.model.path,
                revision=donor_model.model.revision,
                trust_remote_code=options.trust_remote_code,
            )
            donor_has_template = bool(getattr(donor_tok, "chat_template", None))
            other_has_template = False
            for model in merge_config.referenced_models():
                if model == donor_model:
                    continue
                try:
                    tok = transformers.AutoTokenizer.from_pretrained(
                        model.model.path,
                        revision=model.model.revision,
                        trust_remote_code=options.trust_remote_code,
                    )
                    if getattr(tok, "chat_template", None):
                        other_has_template = True
                        break
                except Exception:
                    continue
            if (not donor_has_template) and other_has_template:
                LOG.warning(
                    "Base model tokenizer is being used as donor and does not define a "
                    "chat template, while another input model does. This matches "
                    "MergeKit default behavior, but the merged output may lose the "
                    "instruct/chat template. Consider setting `chat_template: auto` "
                    "or `tokenizer_source` to the instruct/chat model."
                )
        except Exception:
            pass

    if (
        (not merge_config.chat_template)
        and os.path.exists(os.path.join(donor_local_path, "tokenizer_config.json"))
        and (
            os.path.exists(os.path.join(donor_local_path, "tokenizer.json"))
            or os.path.exists(os.path.join(donor_local_path, "tokenizer.model"))
        )
    ):
        LOG.info("Copying tokenizer from %s", donor_model)

        for file_name in [
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "added_tokens.json",
            "merges.txt",
            "chat_template.jinja",
            "generation_config.json",
        ]:
            if os.path.exists(os.path.join(donor_local_path, file_name)):
                shutil.copy(
                    os.path.join(donor_local_path, file_name),
                    os.path.join(out_path, file_name),
                )

        return

    LOG.info("Reserializing tokenizer from %s", donor_model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        donor_model.model.path,
        revision=donor_model.model.revision,
        trust_remote_code=options.trust_remote_code,
    )
    _set_chat_template(tokenizer, merge_config)
    tokenizer.save_pretrained(out_path, safe_serialization=True)


def _model_out_config(
    config: MergeConfiguration,
    arch_info: ModelArchitecture,
    trust_remote_code: bool = False,
) -> transformers.PretrainedConfig:
    """Return a configuration for the resulting model."""
    if config.base_model:
        res = config.base_model.config(trust_remote_code=trust_remote_code)
    else:
        res = config.referenced_models()[0].config(trust_remote_code=trust_remote_code)
    if config.out_dtype:
        res.torch_dtype = config.out_dtype
    elif config.dtype:
        res.torch_dtype = config.dtype

    module_layers = {}
    for module_name in arch_info.modules:
        if config.modules and module_name in config.modules:
            module_def = config.modules.get(module_name)
            if module_def and module_def.slices:
                module_layers[module_name] = sum(
                    [
                        s.sources[0].layer_range[1] - s.sources[0].layer_range[0]
                        for s in module_def.slices
                    ]
                )
        elif config.slices:
            module_layers[module_name] = sum(
                [
                    s.sources[0].layer_range[1] - s.sources[0].layer_range[0]
                    for s in config.slices
                ]
            )

    if module_layers:
        for module_name in module_layers:
            if module_name not in arch_info.modules:
                LOG.warning(
                    "Module %s in config but not in architecture info",
                    module_name,
                )
                continue
            module_info = arch_info.modules[module_name]
            cfg_key = module_info.architecture.num_layers_config_key()
            if not cfg_key:
                if module_layers[module_name] > 0:
                    LOG.warning(
                        "Module %s has no configuration key for number of layers, "
                        "but the number of layers is not zero.",
                        module_name,
                    )
                continue
            try:
                set_config_value(res, cfg_key, module_layers[module_name])
            except Exception as e:
                LOG.warning(
                    "Unable to set number of layers for module %s in output config "
                    "- you may need to manually correct it.",
                    module_name,
                    exc_info=e,
                )

    return res


def _update_config_vocab(
    config: transformers.PretrainedConfig,
    arch_info: ModelArchitecture,
    tokenizer: transformers.PreTrainedTokenizerBase,
    pad_to_multiple_of: Optional[int] = None,
):
    vocab_size = len(tokenizer.get_vocab())
    if pad_to_multiple_of and vocab_size % pad_to_multiple_of:
        vocab_size = vocab_size + pad_to_multiple_of - (vocab_size % pad_to_multiple_of)
    try:
        set_config_value(
            config, arch_info.vocab_size_config_key or "vocab_size", vocab_size
        )
    except Exception as e:
        LOG.warning(
            "Unable to set vocabulary size in output config "
            "- you may need to manually correct it.",
            exc_info=e,
        )


__all__ = ["MergeOptions", "run_merge"]


def _write_execution_report(
    out_path: str,
    metrics: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    os.makedirs(out_path, exist_ok=True)
    task_runs = [float(t.get("run_ms", 0.0)) for t in metrics.get("tasks", [])]
    queue_depth = [int(x) for x in metrics.get("queue_depth_samples", [])]
    summary = {
        "meta": metadata,
        "executor": metrics.get("executor"),
        "task_count": metrics.get("task_count", 0),
        "run_ms_avg": statistics.fmean(task_runs) if task_runs else 0.0,
        "run_ms_p95": _p95(task_runs) if task_runs else 0.0,
        "queue_depth_avg": statistics.fmean(queue_depth) if queue_depth else 0.0,
        "queue_depth_peak": max(queue_depth) if queue_depth else 0,
        "backpressure_trigger_count": int(
            metrics.get("backpressure_trigger_count", 0)
        ),
        "rss_peak_mb": float(metrics.get("rss_peak_mb", 0.0)),
        "npu_used_peak_mb": metrics.get("npu_used_peak_mb"),
        "island_assignment": metrics.get("island_assignment", []),
        "tasks": metrics.get("tasks", []),
    }

    json_path = os.path.join(out_path, "wizard_execution_report.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    md_path = os.path.join(out_path, "wizard_execution_report.md")
    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write("# Wizard Execution Report\n\n")
        fp.write(f"- executor: `{summary['executor']}`\n")
        fp.write(f"- task_count: `{summary['task_count']}`\n")
        fp.write(f"- run_ms_avg: `{summary['run_ms_avg']:.3f}`\n")
        fp.write(f"- run_ms_p95: `{summary['run_ms_p95']:.3f}`\n")
        fp.write(f"- queue_depth_avg: `{summary['queue_depth_avg']:.3f}`\n")
        fp.write(f"- queue_depth_peak: `{summary['queue_depth_peak']}`\n")
        fp.write(
            f"- backpressure_trigger_count: `{summary['backpressure_trigger_count']}`\n"
        )
        fp.write(f"- rss_peak_mb: `{summary['rss_peak_mb']:.3f}`\n")
        fp.write(f"- npu_used_peak_mb: `{summary['npu_used_peak_mb']}`\n")
        fp.write(
            f"- island_assignment_count: `{len(summary['island_assignment'])}`\n"
        )
        fp.write("\n## Schedule Metadata\n\n")
        for key, value in metadata.items():
            fp.write(f"- {key}: `{value}`\n")
        if summary["island_assignment"]:
            fp.write("\n## Island Assignment\n\n")
            for item in summary["island_assignment"]:
                fp.write(
                    "- device: `{device}`, task_count: `{task_count}`, dominant_locality_key: `{dominant}`\n".format(
                        device=item.get("device"),
                        task_count=item.get("task_count"),
                        dominant=item.get("dominant_locality_key"),
                    )
                )


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(x) for x in values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * 0.95))))
    return ordered[idx]
