import enum

from dataclasses import dataclass, field


class DistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:
        - **MINDFORMERS** -- Using mindformers
    """

    MINDFORMERS = "MINDFORMERS"


@dataclass
class MindFormersLMPlugin:
    """
    Plugin for MindFormersLM to enable tensor, pipeline, sequence and data parallelism.
    """

    tp_degree: int = field(default=None, metadata={"help": "tensor parallelism degree."})
    pp_degree: int = field(default=None, metadata={"help": "pipeline parallelism degree."})
    num_micro_batches: int = field(default=None, metadata={"help": "number of micro-batches."})
