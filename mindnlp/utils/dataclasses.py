"""data classes for mindnlp"""
import enum


class DistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:
        - **MINDFORMERS** -- Using mindformers
        - **NO** -- Not a distributed environment, just a single process.
        - **MULTI_NPU_DATA_PARALLEL** -- Distributed data parallel on multiple NPUs.
    """

    MULTI_NPU_DATA_PARALLEL = "MULTI_NPU_DATA_PARALLEL"
    MINDFORMERS = "MINDFORMERS"
    NO = "NO"

