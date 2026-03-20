# Originally from MergeKit (https://github.com/arcee-ai/mergekit)
# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.

"""Configuration models and parameter readers for merge."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from .common import ModelReference
from .tokenizer.config import TokenizerConfig


class ConditionalParameter(BaseModel, frozen=True):
    """Conditionally-applied parameter value by tensor name filter."""

    value: Any
    filter: Optional[str] = None


ParameterSetting = Union[
    int,
    float,
    bool,
    str,
    ConditionalParameter,
    List[int],
    List[float],
    List[bool],
    List[str],
    List[ConditionalParameter],
]


def _filter_match(tensor_name: str, pattern: str) -> bool:
    if not pattern:
        return True
    if pattern == "*":
        return True
    try:
        return re.search(pattern, tensor_name) is not None
    except re.error:
        # Keep a safe fallback when users pass plain substrings.
        return pattern in tensor_name


def evaluate_setting(
    tensor_name: str,
    setting: Optional[ParameterSetting],
    *,
    t: float = 0.0,
) -> Any:
    """Resolve a parameter setting for a specific tensor and interpolation point."""
    if setting is None:
        return None

    if isinstance(setting, ConditionalParameter):
        if setting.filter and not _filter_match(tensor_name, setting.filter):
            return None
        return evaluate_setting(tensor_name, setting.value, t=t)

    if isinstance(setting, list):
        if not setting:
            return None

        if all(isinstance(x, ConditionalParameter) for x in setting):
            for item in setting:
                value = evaluate_setting(tensor_name, item, t=t)
                if value is not None:
                    return value
            return None

        if all(isinstance(x, (int, float)) for x in setting):
            if len(setting) == 1:
                return setting[0]
            pos = max(0.0, min(1.0, float(t))) * (len(setting) - 1)
            left = int(pos)
            right = min(left + 1, len(setting) - 1)
            if left == right:
                return setting[left]
            frac = pos - left
            return float(setting[left]) * (1.0 - frac) + float(setting[right]) * frac

        return setting[0]

    return setting


class InputModelDefinition(BaseModel, frozen=True):
    model: ModelReference
    parameters: Optional[Dict[str, ParameterSetting]] = None


class InputSliceDefinition(BaseModel, frozen=True):
    model: ModelReference
    layer_range: Tuple[int, int]
    parameters: Optional[Dict[str, ParameterSetting]] = None


class OutputSliceDefinition(BaseModel, frozen=True):
    sources: List[InputSliceDefinition]
    base_model: Optional[ModelReference] = None
    parameters: Optional[Dict[str, ParameterSetting]] = None


class OutputModuleDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: Optional[str] = None
    models: Optional[List[InputModelDefinition]] = None
    slices: Optional[List[OutputSliceDefinition]] = None
    parameters: Optional[Dict[str, ParameterSetting]] = None


class MergeConfiguration(BaseModel):
    """Top-level merge recipe."""

    model_config = ConfigDict(extra="allow")

    merge_method: str
    models: Optional[List[InputModelDefinition]] = None
    slices: Optional[List[Union[OutputSliceDefinition, InputSliceDefinition]]] = None
    modules: Optional[Dict[str, OutputModuleDefinition]] = None
    parameters: Optional[Dict[str, ParameterSetting]] = None
    base_model: Optional[ModelReference] = None
    tokenizer_source: Optional[Union[Literal["union"], Literal["base"], ModelReference]] = None
    tokenizer: Optional[TokenizerConfig] = None
    chat_template: Optional[str] = None
    dtype: Optional[str] = None
    out_dtype: Optional[str] = None
    # Extra compatibility fields used in architecture disambiguation.
    architectures: Optional[List[str]] = None
    model_type: Optional[str] = None

    @model_validator(mode="after")
    def _normalize(self):
        sources = int(bool(self.models)) + int(bool(self.slices)) + int(bool(self.modules))
        if sources != 1:
            raise RuntimeError(
                "Exactly one of models, slices, or modules must be specified"
            )

        if self.tokenizer_source is not None and self.tokenizer is not None:
            if self.tokenizer.source != self.tokenizer_source:
                raise RuntimeError(
                    "Cannot specify both tokenizer_source and tokenizer"
                )

        if self.slices and isinstance(self.slices[0], InputSliceDefinition):
            self.slices = [
                OutputSliceDefinition(sources=[s])  # type: ignore[arg-type]
                for s in self.slices  # type: ignore[assignment]
            ]

        if self.modules:
            normalized: Dict[str, OutputModuleDefinition] = {}
            for name, module in self.modules.items():
                if module.name is None:
                    module.name = name
                normalized[name] = module
            self.modules = normalized

        if self.tokenizer_source is not None and self.tokenizer is None:
            self.tokenizer = TokenizerConfig(source=self.tokenizer_source)

        model_count = len(self.models) if self.models else 0
        if model_count == 0 and not self.slices and not self.modules:
            raise RuntimeError("At least one model source must be provided")

        two_model_base_methods = {"slerp", "arcee_fusion", "nearswap"}
        if self.merge_method in two_model_base_methods:
            if self.base_model is None:
                raise RuntimeError(
                    f"merge_method '{self.merge_method}' requires base_model"
                )
        return self

    def referenced_models(self) -> List[ModelReference]:
        models: List[ModelReference] = []

        def _add(m: Optional[ModelReference]):
            if m is not None and m not in models:
                models.append(m)

        if self.models:
            for model_in in self.models:
                _add(model_in.model)
        if self.slices:
            for sl in self.slices:
                for src in sl.sources:
                    _add(src.model)
        if self.modules:
            for module in self.modules.values():
                if module.models:
                    for model_in in module.models:
                        _add(model_in.model)
                if module.slices:
                    for sl in module.slices:
                        for src in sl.sources:
                            _add(src.model)
        _add(self.base_model)
        if isinstance(self.tokenizer_source, ModelReference):
            _add(self.tokenizer_source)
        if self.tokenizer and isinstance(self.tokenizer.source, ModelReference):
            _add(self.tokenizer.source)
        return models

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json", exclude_none=True),
            allow_unicode=True,
            sort_keys=False,
        )


class ConfigReader(BaseModel):
    """Contextual parameter resolver with merge precedence rules."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    config: MergeConfiguration
    t: float = 0.0
    tensor_name: str = ""
    module: Optional[OutputModuleDefinition] = None
    slice_out: Optional[OutputSliceDefinition] = None

    @property
    def base_model(self) -> Optional[ModelReference]:
        if self.slice_out and self.slice_out.base_model is not None:
            return self.slice_out.base_model
        return self.config.base_model

    def with_t(self, t: float) -> "ConfigReader":
        return ConfigReader(
            config=self.config,
            t=t,
            tensor_name=self.tensor_name,
            module=self.module,
            slice_out=self.slice_out,
        )

    def for_tensor(self, tensor_name: str) -> "ConfigReader":
        return ConfigReader(
            config=self.config,
            t=self.t,
            tensor_name=tensor_name,
            module=self.module,
            slice_out=self.slice_out,
        )

    def for_out_slice(self, out_slice: OutputSliceDefinition) -> "ConfigReader":
        return ConfigReader(
            config=self.config,
            t=self.t,
            tensor_name=self.tensor_name,
            module=self.module,
            slice_out=out_slice,
        )

    def parameter(
        self,
        name: str,
        *,
        model: Optional[ModelReference] = None,
        required: bool = False,
        default: Any = None,
    ) -> Any:
        if model is not None and self.slice_out is not None:
            for source in self.slice_out.sources:
                if source.model == model and source.parameters and name in source.parameters:
                    value = evaluate_setting(
                        self.tensor_name,
                        source.parameters[name],
                        t=self.t,
                    )
                    if value is not None:
                        return value

        if self.slice_out and self.slice_out.parameters and name in self.slice_out.parameters:
            value = evaluate_setting(self.tensor_name, self.slice_out.parameters[name], t=self.t)
            if value is not None:
                return value

        if self.module and self.module.parameters and name in self.module.parameters:
            value = evaluate_setting(self.tensor_name, self.module.parameters[name], t=self.t)
            if value is not None:
                return value

        if self.config.parameters and name in self.config.parameters:
            value = evaluate_setting(self.tensor_name, self.config.parameters[name], t=self.t)
            if value is not None:
                return value

        if required and default is None:
            raise RuntimeError(f"Missing required parameter: {name}")
        return default


__all__ = [
    "ConditionalParameter",
    "ConfigReader",
    "InputModelDefinition",
    "InputSliceDefinition",
    "MergeConfiguration",
    "OutputModuleDefinition",
    "OutputSliceDefinition",
    "ParameterSetting",
    "evaluate_setting",
]
