# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#
#
import logging
from typing import Dict, List

from .base import MergeMethod
from ..sparsify import SparsificationMethod

LOG = logging.getLogger(__name__)

STATIC_MERGE_METHODS: List[MergeMethod] = []

try:
    from .linear import LinearMerge
    STATIC_MERGE_METHODS.append(LinearMerge())
except Exception as e:
    LOG.warning("Failed to register merge method linear", exc_info=e)

try:
    from .slerp import SlerpMerge
    STATIC_MERGE_METHODS.append(SlerpMerge())
except Exception as e:
    LOG.warning("Failed to register merge method slerp", exc_info=e)

try:
    from .nuslerp import NuSlerpMerge
    STATIC_MERGE_METHODS.append(NuSlerpMerge())
except Exception as e:
    LOG.warning("Failed to register merge method nuslerp", exc_info=e)

try:
    from .passthrough import PassthroughMerge
    STATIC_MERGE_METHODS.append(PassthroughMerge())
except Exception as e:
    LOG.warning("Failed to register merge method passthrough", exc_info=e)

try:
    from .model_stock import ModelStockMerge
    STATIC_MERGE_METHODS.append(ModelStockMerge())
except Exception as e:
    LOG.warning("Failed to register merge method model_stock", exc_info=e)

try:
    from .arcee_fusion import ArceeFusionMerge
    STATIC_MERGE_METHODS.append(ArceeFusionMerge())
except Exception as e:
    LOG.warning("Failed to register merge method arcee_fusion", exc_info=e)

try:
    from .karcher import KarcherMerge
    STATIC_MERGE_METHODS.append(KarcherMerge())
except Exception as e:
    LOG.warning("Failed to register merge method karcher", exc_info=e)

try:
    from .generalized_task_arithmetic import (
        ConsensusMethod,
        GeneralizedTaskArithmeticMerge,
    )
    # generalized task arithmetic methods
    STATIC_MERGE_METHODS.extend([
        GeneralizedTaskArithmeticMerge(
            consensus_method=None,
            sparsification_method=None,
            default_normalize=False,
            default_rescale=False,
            method_name="task_arithmetic",
            method_pretty_name="Task Arithmetic",
            method_reference_url="https://arxiv.org/abs/2212.04089",
        ),
        GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.magnitude,
            default_normalize=True,
            default_rescale=False,
            method_name="ties",
            method_pretty_name="TIES",
            method_reference_url="https://arxiv.org/abs/2306.01708",
        ),
        GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.random,
            default_normalize=False,
            default_rescale=True,
            method_name="dare_ties",
            method_pretty_name="DARE TIES",
            method_reference_url="https://arxiv.org/abs/2311.03099",
        ),
        GeneralizedTaskArithmeticMerge(
            consensus_method=None,
            sparsification_method=SparsificationMethod.random,
            default_normalize=False,
            default_rescale=True,
            method_name="dare_linear",
            method_pretty_name="Linear DARE",
            method_reference_url="https://arxiv.org/abs/2311.03099",
        ),
        GeneralizedTaskArithmeticMerge(
            consensus_method=None,
            sparsification_method=SparsificationMethod.magnitude_outliers,
            default_normalize=False,
            default_rescale=False,
            method_name="breadcrumbs",
            method_pretty_name="Model Breadcrumbs",
            method_reference_url="https://arxiv.org/abs/2312.06795",
        ),
        GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.magnitude_outliers,
            default_normalize=False,
            default_rescale=False,
            method_name="breadcrumbs_ties",
            method_pretty_name="Model Breadcrumbs with TIES",
            method_reference_url="https://arxiv.org/abs/2312.06795",
        ),
        GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.della_magprune,
            default_normalize=True,
            default_rescale=True,
            method_name="della",
            method_pretty_name="DELLA",
            method_reference_url="https://arxiv.org/abs/2406.11617",
        ),
        GeneralizedTaskArithmeticMerge(
            consensus_method=None,
            sparsification_method=SparsificationMethod.della_magprune,
            default_normalize=False,
            default_rescale=True,
            method_name="della_linear",
            method_pretty_name="Linear DELLA",
            method_reference_url="https://arxiv.org/abs/2406.11617",
        ),
    ])
except Exception as e:
    LOG.warning(
        "Failed to register generalized_task_arithmetic-derived merge methods",
        exc_info=e,
    )

REGISTERED_MERGE_METHODS: Dict[str, MergeMethod] = {
    method.name(): method for method in STATIC_MERGE_METHODS
}
