import dataclasses
from typing import List, Optional, Tuple
import subprocess
import re
from mindspore import context


@dataclasses.dataclass(frozen=True)
class CUDASpecs:
    cuda_version_string: str
    cuda_version_tuple: Tuple[int, int]


def get_cuda_version_tuple() -> Tuple[int, int]:
    result = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, text=True)
    match = re.search(r"V(\d+)\.(\d+)", result.stdout)
    if match:
        major, minor = map(int, match.groups())
        return major, minor
    return 0, 0


def get_cuda_version_string() -> str:
    major, minor = get_cuda_version_tuple()
    return f"{major}{minor}"


def get_cuda_specs() -> Optional[CUDASpecs]:
    if not context.get_context("device_target") == "GPU":
        return None

    return CUDASpecs(
        cuda_version_string=(get_cuda_version_string()),
        cuda_version_tuple=get_cuda_version_tuple(),
    )
