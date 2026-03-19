# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

from abc import ABC, abstractmethod
from typing import List, Optional

import mindspore  # pylint: disable=import-error

from .config import MoEMergeConfig
from ..options import MergeOptions


class MoEOutputArchitecture(ABC):
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for the architecture."""

    @abstractmethod
    def supports_config(
        self,
        config: MoEMergeConfig,
        explain: bool = False,
        trust_remote_code: bool = False,
    ) -> bool:
        """Return whether this architecture supports the given config.

        If `explain` is True, log an explanation of why the config is not supported."""

    @abstractmethod
    def write_model(  # pylint: disable=too-many-positional-arguments
        self,
        out_path: str,
        config: MoEMergeConfig,
        merge_options: MergeOptions,
        router_weights: List[mindspore.Tensor],
        shared_router_weights: Optional[List[mindspore.Tensor]] = None,
    ):
        """Write the config and tensors for the output MoE to the given path."""
