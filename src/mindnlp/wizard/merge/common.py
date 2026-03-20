# Originally from MergeKit (https://github.com/arcee-ai/mergekit)
# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.

"""Common types shared across the merge package."""

import binascii
import logging
import os
import os.path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Union,
    get_args,
)

import huggingface_hub
import immutables
import mindspore
from pydantic import BaseModel, model_serializer, model_validator
from pydantic_core import core_schema
from transformers import AutoConfig, PretrainedConfig
from typing_extensions import TypeVar

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def set_config_value(config: PretrainedConfig, key: str, value: Any):
    parts = key.split(".")
    obj = config
    for idx, part in enumerate(parts[:-1]):
        if not hasattr(obj, part):
            raise RuntimeError(
                f"Config {config} has no attribute {'.'.join(parts[: idx + 1])}"
            )
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def get_config_value(config: PretrainedConfig, key: str) -> Any:
    parts = key.split(".")
    obj = config
    for idx, part in enumerate(parts):
        if not hasattr(obj, part):
            raise RuntimeError(
                f"Config {config} has no attribute {'.'.join(parts[: idx + 1])}"
            )
        obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# ModelPath / ModelReference
# ---------------------------------------------------------------------------

class ModelPath(BaseModel, frozen=True):
    path: str
    revision: Optional[str] = None

    @model_validator(mode="before")
    def validate_string(cls, value):
        if isinstance(value, str):
            at_ct = value.count("@")
            if at_ct > 1:
                raise RuntimeError(f"Invalid model path - multiple @: {value}")
            elif at_ct == 1:
                path, rev = value.split("@")
                return {"path": path, "revision": rev}
            else:
                return {"path": value}
        return value

    def __str__(self):
        if self.revision:
            return f"{self.path}@{self.revision}"
        return self.path

    def _unique_id(self):
        return (
            os.path.basename(self.path)
            + "_"
            + str(binascii.crc32(self.__str__().encode()))
        )


class ModelReference(BaseModel, frozen=True):
    """A reference to a language model (hub path or local).

    Optionally includes a LoRA adapter path.
    """

    model: ModelPath
    lora: Optional[ModelPath] = None
    override_architecture: Optional[str] = None

    def merged(
        self,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        lora_merge_dtype: Optional[str] = None,
    ) -> "ModelReference":
        """Merge the LoRA adapter (if any) and return a new reference."""
        if not self.lora:
            return self

        if not cache_dir:
            raise RuntimeError("Need to specify cache dir to merge adapters")

        out_path = os.path.join(
            cache_dir,
            self.model._unique_id() + "_" + self.lora._unique_id(),
        )

        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)

            config = self.config(trust_remote_code)
            auto_cls = get_auto_cls(config.architectures[0])

            logging.info("Loading %s for LoRA merge...", self.model)
            model = auto_cls.from_pretrained(
                self.model.path,
                revision=self.model.revision,
                ms_dtype=dtype_from_name(lora_merge_dtype),
                trust_remote_code=trust_remote_code,
            )

            # MindSpore PEFT equivalent (mindnlp / mindpet)
            try:
                from mindnlp.peft import PeftModel
            except ImportError:
                raise ImportError(
                    "mindnlp.peft is required for LoRA merging. "
                    "Install mindnlp with PEFT support."
                )
            model = PeftModel.from_pretrained(
                model,
                self.lora.path,
                is_trainable=False,
            )
            logging.info("Merging %s into %s", self.lora, self.model)
            model = model.merge_and_unload()
            model.save_pretrained(out_path, safe_serialization=True)
            del model

        return ModelReference(model=ModelPath(path=out_path))

    def config(self, trust_remote_code: bool = False) -> PretrainedConfig:
        res = AutoConfig.from_pretrained(
            self.model.path,
            revision=self.model.revision,
            trust_remote_code=trust_remote_code,
        )
        if self.override_architecture:
            res.architectures = [self.override_architecture]
        return res

    def local_path(
        self, cache_dir: Optional[str] = None, ignore_lora: bool = False
    ) -> str:
        if not ignore_lora:
            assert (
                self.lora is None
            ), "LoRA not merged - use .merged() to get a local path"

        path = self.model.path
        if not os.path.exists(path):
            has_safetensors = any(
                fn.lower().endswith(".safetensors")
                for fn in huggingface_hub.list_repo_files(
                    path, repo_type="model", revision=self.model.revision
                )
            )
            patterns = ["tokenizer.model", "*.json"]
            if has_safetensors:
                patterns.append("*.safetensors")
            else:
                patterns.append("*.bin")

            path = huggingface_hub.snapshot_download(
                path,
                revision=self.model.revision,
                cache_dir=cache_dir,
                allow_patterns=patterns,
            )
        return path

    def tensor_index(self, cache_dir: Optional[str] = None):
        from .io import ShardedTensorIndex

        return ShardedTensorIndex.from_disk(self.local_path(cache_dir))

    def lazy_loader(
        self, cache_dir: Optional[str] = None, lazy_loader: bool = True
    ):
        from .io import LazyTensorLoader

        return LazyTensorLoader(
            self.tensor_index(cache_dir),
            lazy_loader=lazy_loader,
        )

    @model_validator(mode="before")
    def validate_string(cls, value):
        if isinstance(value, str):
            chunks = value.split("+")
            if len(chunks) == 1:
                return {"model": value}
            elif len(chunks) == 2:
                return {"model": chunks[0], "lora": chunks[1]}
            raise RuntimeError(f"Can't parse {value}")
        return value

    @model_serializer()
    def serialize(self):
        if self.override_architecture is not None:
            return {
                "model": self.model,
                "lora": self.lora,
                "override_architecture": self.override_architecture,
            }
        res = str(self)
        if '"' in res or " " in res:
            return self
        return res

    @classmethod
    def parse(cls, value: str) -> "ModelReference":
        return ModelReference.model_validate(value)

    def __str__(self) -> str:
        if self.lora:
            return f"{str(self.model)}+{str(self.lora)}"
        return str(self.model)


# ---------------------------------------------------------------------------
# dtype helpers (MindSpore)
# ---------------------------------------------------------------------------

def dtype_from_name(name: Optional[str]) -> Optional[mindspore.dtype]:
    if not name:
        return None

    for prefix in ("torch.", "mindspore.", "ms."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    _MAP = {
        "bfloat16": mindspore.bfloat16,
        "float16": mindspore.float16,
        "float32": mindspore.float32,
        "float64": mindspore.float64,
        "int32": mindspore.int32,
        "int64": mindspore.int64,
    }
    if name in _MAP:
        return _MAP[name]
    raise RuntimeError(f'Unimplemented dtype "{name}"')


# ---------------------------------------------------------------------------
# parse_kmb
# ---------------------------------------------------------------------------

def parse_kmb(value: Union[str, int]) -> int:
    if isinstance(value, int):
        return value
    elif value.isnumeric():
        return int(value)
    elif value[-1].lower() == "k":
        return int(value[:-1]) * 1000
    elif value[-1].lower() == "m":
        return int(value[:-1]) * 1000 * 1000
    elif value[-1].lower() == "b":
        return int(value[:-1]) * 1000 * 1000 * 1000
    else:
        raise ValueError(value)


# ---------------------------------------------------------------------------
# ImmutableMap
# ---------------------------------------------------------------------------

T_K = TypeVar("T_K")
T_V = TypeVar("T_V")


class ImmutableMap(Generic[T_K, T_V]):
    data: immutables.Map

    def __init__(self, data: Mapping):
        self.data = data

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: Callable[[Any], core_schema.CoreSchema]
    ) -> core_schema.CoreSchema:
        instance_schema = core_schema.is_instance_schema(cls)
        args = get_args(source)
        if args:
            dict_schema = handler(Dict[args[0], args[1]])
        else:
            dict_schema = handler(Dict)
        non_instance_schema = core_schema.with_info_after_validator_function(
            lambda value, _info: immutables.Map(value), dict_schema
        )
        return core_schema.union_schema([instance_schema, non_instance_schema])

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()


# ---------------------------------------------------------------------------
# Auto model class detection (framework-agnostic via transformers)
# ---------------------------------------------------------------------------

ARCH_NAME_TO_AUTO_CLS: Dict[str, Any] = {}

try:
    import transformers
    import transformers.models.auto.modeling_auto as tf_auto
except ImportError:
    tf_auto = None

if tf_auto is not None:
    for map_name, cls_name in [
        ("MODEL_MAPPING_NAMES", "AutoModel"),
        (
            "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES",
            "AutoModelForAudioClassification",
        ),
        (
            "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES",
            "AutoModelForImageClassification",
        ),
        ("MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES", "AutoModelForSpeechSeq2Seq"),
        (
            "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES",
            "AutoModelForSequenceClassification",
        ),
        ("MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES", "AutoModelForSeq2SeqLM"),
        (
            "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES",
            "AutoModelForTokenClassification",
        ),
        (
            "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",
            "AutoModelForImageTextToText",
        ),
        ("MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES", "AutoModelForTextToWaveform"),
        ("MODEL_FOR_MASKED_LM_MAPPING_NAMES", "AutoModelForMaskedLM"),
        ("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", "AutoModelForCausalLM"),
    ]:
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            logging.info("Could not find %s in transformers", cls_name)
            continue
        if hasattr(tf_auto, map_name):
            name_to_arch_name = getattr(tf_auto, map_name)
            for arch_name in name_to_arch_name.values():
                ARCH_NAME_TO_AUTO_CLS[arch_name] = cls


class AutoClassProtocol(Protocol):
    def from_pretrained(
        self, pretrained_model_name_or_path: str, *model_args, **kwargs
    ): ...

    def from_config(self, config, *model_args, **kwargs): ...


def get_auto_cls(arch_name: str) -> AutoClassProtocol:
    if arch_name in ARCH_NAME_TO_AUTO_CLS:
        return ARCH_NAME_TO_AUTO_CLS[arch_name]

    if arch_name.endswith("ForMaskedLM"):
        auto_cls = transformers.AutoModelForMaskedLM
    elif arch_name.endswith("ForSequenceClassification"):
        auto_cls = transformers.AutoModelForSequenceClassification
    elif arch_name.endswith("ForTokenClassification"):
        auto_cls = transformers.AutoModelForTokenClassification
    else:
        if not arch_name.endswith("ForCausalLM") or arch_name.endswith(
            "LMHeadModel"
        ):
            logging.warning(
                "Unknown model type %s — assuming AutoModelForCausalLM",
                arch_name,
            )
        auto_cls = transformers.AutoModelForCausalLM
    return auto_cls


# ---------------------------------------------------------------------------
# Ascend NPU / accelerator helpers
# ---------------------------------------------------------------------------

def get_ascend_device_count() -> int:
    """Return the number of available Ascend NPU devices."""
    try:
        import acl  # pylint: disable=import-error  # Ascend Computing Language
        raw = acl.rt.get_device_count()
        count = _normalize_device_count(raw)
        if count > 0:
            return count
    except Exception as exc:
        LOG.debug(
            "ACL-based Ascend device probing failed (%s: %s)",
            type(exc).__name__,
            exc,
        )
    try:
        count = int(os.environ.get("ASCEND_DEVICE_NUM", "0"))
        if count > 0:
            return count
    except ValueError as exc:
        LOG.debug("Invalid ASCEND_DEVICE_NUM value (%s)", exc)
    try:
        device_list = mindspore.get_context("device_target")
        if device_list == "Ascend":
            return max(1, int(os.environ.get("RANK_SIZE", "1")))
    except Exception as exc:
        LOG.debug(
            "MindSpore context probing for Ascend failed (%s: %s)",
            type(exc).__name__,
            exc,
        )
    return 0


def get_accelerator_count(accelerator_name: Optional[str] = None) -> int:
    """Return device count for the specified (or default) accelerator."""
    if accelerator_name is not None:
        target, dev_id = _parse_accelerator(accelerator_name)
        if dev_id is not None:
            return 1
    else:
        target = _default_accelerator()

    if target == "CPU":
        return 1
    count = _probe_device_count(target)
    if count > 0:
        return count

    LOG.warning(
        "Could not determine device count for accelerator '%s'; defaulting to 1",
        target,
    )
    return 1


def get_accelerator_type(accelerator_name: Optional[str] = None) -> str:
    if accelerator_name is not None:
        return _parse_accelerator(accelerator_name)[0]
    return _default_accelerator()


def _parse_accelerator(spec: str) -> Tuple[str, Optional[int]]:
    parts = spec.split(":")
    target = parts[0]
    dev_id = int(parts[1]) if len(parts) > 1 else None
    return target, dev_id


def _default_accelerator() -> str:
    """Detect the default accelerator available on this system."""
    try:
        target = mindspore.get_context("device_target")
        if target and target != "CPU":
            return target
    except Exception as exc:
        LOG.debug(
            "MindSpore context accelerator probing failed (%s: %s)",
            type(exc).__name__,
            exc,
        )

    for candidate in ("Ascend", "GPU"):
        if _accelerator_available(candidate):
            return candidate

    if get_ascend_device_count() > 0:
        return "Ascend"
    return "CPU"


def _probe_device_count(target: str) -> int:
    normalized_target = target.upper()
    if normalized_target == "ASCEND":
        count = get_ascend_device_count()
        if count > 0:
            return count

    try:
        if hasattr(mindspore, "hal") and hasattr(mindspore.hal, "device_count"):
            count = _normalize_device_count(
                mindspore.hal.device_count(device_target=target)
            )
            if count > 0:
                return count
    except Exception as exc:
        LOG.debug(
            "MindSpore HAL device_count probing failed for %s (%s: %s)",
            target,
            type(exc).__name__,
            exc,
        )

    for env_key in (
        "RANK_SIZE",
        "WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "DEVICE_NUM",
        "ASCEND_DEVICE_NUM",
    ):
        raw = os.environ.get(env_key)
        if not raw:
            continue
        try:
            count = int(raw)
            if count > 0:
                return count
        except ValueError:
            LOG.debug("Invalid %s value=%r during device count probing", env_key, raw)

    return 0


def _accelerator_available(target: str) -> bool:
    try:
        if hasattr(mindspore, "hal") and hasattr(mindspore.hal, "is_available"):
            return bool(mindspore.hal.is_available(target))
    except Exception as exc:
        LOG.debug(
            "MindSpore HAL is_available probing failed for %s (%s: %s)",
            target,
            type(exc).__name__,
            exc,
        )
    return _probe_device_count(target) > 0


def _normalize_device_count(raw: Any) -> int:
    """Normalize vendor/runtime-specific device count return values."""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, (tuple, list)):
        for item in raw:
            if isinstance(item, int):
                return item
        return 0
    if isinstance(raw, dict):
        for key in ("count", "device_count", "num_devices"):
            value = raw.get(key)
            if isinstance(value, int):
                return value
        return 0
    try:
        return int(raw)
    except Exception:
        LOG.debug("Unsupported device count return value: %r", raw)
        return 0
