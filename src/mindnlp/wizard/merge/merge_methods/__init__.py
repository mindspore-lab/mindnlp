# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#
#
import logging

LOG = logging.getLogger(__name__)

try:
    from . import multislerp  # noqa: F401
except Exception as e:
    LOG.warning("Failed to import merge method module multislerp", exc_info=e)

try:
    from . import nearswap  # noqa: F401
except Exception as e:
    LOG.warning("Failed to import merge method module nearswap", exc_info=e)

try:
    from . import ram  # noqa: F401
except Exception as e:
    LOG.warning("Failed to import merge method module ram", exc_info=e)

try:
    from . import sce  # noqa: F401
except Exception as e:
    LOG.warning("Failed to import merge method module sce", exc_info=e)

from .base import MergeMethod  # pylint: disable=wrong-import-position
from .registry import REGISTERED_MERGE_METHODS  # pylint: disable=wrong-import-position

try:
    from .generalized_task_arithmetic import GeneralizedTaskArithmeticMerge
except Exception as e:
    LOG.warning(
        "Failed to import generalized_task_arithmetic merge methods", exc_info=e
    )
    GeneralizedTaskArithmeticMerge = None


def get(method: str) -> MergeMethod:
    if method in REGISTERED_MERGE_METHODS:
        return REGISTERED_MERGE_METHODS[method]
    raise RuntimeError(f"Unimplemented merge method {method}")


__all__ = [
    "MergeMethod",
    "get",
    "GeneralizedTaskArithmeticMerge",
    "REGISTERED_MERGE_METHODS",
]
