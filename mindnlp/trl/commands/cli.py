'''Copyright 2024 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

# pylint: disable= line-too-long


import os
import platform
import subprocess
import sys
from importlib.metadata import version
from subprocess import CalledProcessError

#库未完善：accelerate
# from ...accelerate.commands.config import default_config_file, load_config_from_file
from rich.console import Console

from .. import (
    __version__,
    is_deepspeed_available,
    is_diffusers_available,
    is_liger_kernel_available,
    is_llmblender_available,
)
from .cli_utils import get_git_commit_hash, init_zero_verbose


SUPPORTED_COMMANDS = ["sft", "dpo", "chat", "kto", "env"]


def print_env():
    '''打印当前环境'''
    #库未完善：accelerate
    # accelerate_config = accelerate_config_str = "not found"

    # Get the default from the config file.
    # if os.path.isfile(default_config_file):
    #     accelerate_config = load_config_from_file(default_config_file).to_dict()
    # accelerate_config_str = (
    #     "\n" + "\n".join([f"  - {prop}: {val}" for prop, val in accelerate_config.items()])
    #     if isinstance(accelerate_config, dict)
    #     else accelerate_config
    # )

    commit_hash = get_git_commit_hash("trl")

    info = {
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "Accelerate version": version("accelerate"),
        #库未完善：accelerate
        # "Accelerate config": accelerate_config_str,
        "Datasets version": version("datasets"),
        "HF Hub version": version("huggingface_hub"),
        "TRL version": f"{__version__}+{commit_hash[:7]}" if commit_hash else __version__,
        "DeepSpeed version": version("deepspeed") if is_deepspeed_available() else "not installed",
        "Diffusers version": version("diffusers") if is_diffusers_available() else "not installed",
        "Liger-Kernel version": version("liger_kernel") if is_liger_kernel_available() else "not installed",
        "LLM-Blender version": version("llm_blender") if is_llmblender_available() else "not installed",
    }

    info_str = "\n".join([f"- {prop}: {val}" for prop, val in info.items()])
    print(f"\nCopy-paste the following information when reporting an issue:\n\n{info_str}\n")  # noqa


def train(command_name):
    """
    使用指定的命令训练TRL（Transformers Reinforcement Learning）模型。  
  
    该函数初始化TRL CLI，构建执行训练脚本的命令，并使用subprocess.run来执行该命令。  
    它捕获输出，如果命令执行失败，则引发错误。  
  
    参数:  
        command_name (str): 要执行的命令（或脚本）的名称。这应该对应于scripts目录中的一个.py文件。  
  
    返回:  
        无。该函数执行一个子进程，不返回任何值。  
  
    抛出:  
        ValueError: 如果子进程执行失败，将引发此异常，并显示失败消息，建议检查上面的跟踪信息以获取详细信息。  
        CalledProcessError, ChildProcessError: 这些异常可能由subprocess.run引发，如果命令失败或子进程出现问题。  
  
    注意:  
        该函数假设`accelerate`命令在环境中可用，并且scripts目录包含一个有效的Python脚本，其名称与指定的command_name相对应。  
        它还假设sys.argv包含脚本所需的必要参数。另外，函数内部的`command_name = sys.argv[1]`这行代码可能是多余的，  
        因为`command_name`已经作为参数传递进来。如果需要从命令行参数中获取命令名称，请考虑移除函数参数`command_name`或更改相关逻辑。  
    """
    console = Console()
    # Make sure to import things locally to avoid verbose from third party libs.
    with console.status("[bold purple]Welcome! Initializing the TRL CLI..."):

        init_zero_verbose()
        command_name = sys.argv[1]
        trl_examples_dir = os.path.dirname(__file__)

    command = f"accelerate launch {trl_examples_dir}/scripts/{command_name}.py {' '.join(sys.argv[2:])}"

    try:
        subprocess.run(
            command.split(),
            text=True,
            check=True,
            encoding="utf-8",
            cwd=os.getcwd(),
            env=os.environ.copy(),
            capture_output=True,
        )
    except (CalledProcessError, ChildProcessError) as exc:
        console.log(f"TRL - {command_name.upper()} failed on ! See the logs above for further details.")
        raise ValueError("TRL CLI failed! Check the traceback above..") from exc


def chat():
    """  
    启动TRL的聊天功能。  
  
    此函数用于初始化TRL CLI，并构造及执行一个命令来启动聊天脚本。它使用subprocess.run来执行脚本，  
    并在执行失败时捕获异常并输出错误信息。  
  
    参数:  
        无。此函数不接受任何参数。  
  
    返回值:  
        无。此函数不返回任何值。  
  
    抛出异常:  
        ValueError: 如果子进程运行失败，此异常将被抛出，并带有指示失败的消息，建议检查上面的日志和回溯信息。  
        CalledProcessError, ChildProcessError: 这些异常可能由subprocess.run抛出，如果命令失败或子进程存在问题。  
  
    注意:  
        此函数假设`python`命令在环境中可用，且脚本目录（由`os.path.dirname(__file__)`指定）中包含一个名为`chat.py`的有效Python脚本。  
        它还假设`sys.argv`包含聊天脚本所需的任何额外参数。  
    """
    console = Console()
    # Make sure to import things locally to avoid verbose from third party libs.
    with console.status("[bold purple]Welcome! Initializing the TRL CLI..."):

        init_zero_verbose()
        trl_examples_dir = os.path.dirname(__file__)

    command = f"python {trl_examples_dir}/scripts/chat.py {' '.join(sys.argv[2:])}"

    try:
        subprocess.run(
            command.split(),
            text=True,
            check=True,
            encoding="utf-8",
            cwd=os.getcwd(),
            env=os.environ.copy(),
        )
    except (CalledProcessError, ChildProcessError) as exc:
        console.log("TRL - CHAT failed! See the logs above for further details.")
        raise ValueError("TRL CLI failed! Check the traceback above..") from exc


def main():
    """  
    程序的主入口函数。  
  
    此函数根据命令行参数中的第一个参数（即命令名称）来执行不同的操作。  
    它支持以下命令：  
        - "sft": 执行与"sft"相关的训练操作。  
        - "dpo": 执行与"dpo"相关的训练操作。  
        - "kto": 执行与"kto"相关的训练操作。  
        - "chat": 启动聊天功能。  
        - "env": 打印环境信息。  
  
    如果提供的命令名称不在支持的命令列表中，将抛出一个ValueError异常。  
  
    参数:  
        无。此函数不接受任何直接参数，但依赖于sys.argv来获取命令行参数。  
  
    返回值:  
        无。此函数不返回任何值。  
  
    抛出异常:  
        ValueError: 如果提供的命令名称不在支持的命令列表中，将抛出此异常。  
    """
    command_name = sys.argv[1]

    if command_name in ["sft", "dpo", "kto"]:
        train(command_name)
    elif command_name == "chat":
        chat()
    elif command_name == "env":
        print_env()
    else:
        raise ValueError(
            f"Please use one of the supported commands, got {command_name} - supported commands are {SUPPORTED_COMMANDS}"
        )

if __name__ == "__main__":
    main()
