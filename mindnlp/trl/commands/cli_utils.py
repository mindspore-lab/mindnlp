'''This file is a copy of trl/examples/scripts/sft.py so that we could
use it together with rich and the TRL CLI in a more customizable manner.
Copyright 2024 The HuggingFace Inc. team. All rights reserved.

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
# pylint: disable= too-many-instance-attributes
# pylint: disable= import-error
# pylint: disable= broad-exception-caught

import importlib
import inspect
import logging
import warnings
import os
import subprocess
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from rich.logging import RichHandler

import yaml
from ..extradatatools.hf_argparser import HfArgumentParser

from ..utils import ScriptArguments

logger = logging.getLogger(__name__)


class YamlConfigParser:
    """  
    用于解析YAML配置文件并设置环境变量的类。
    该类提供了读取YAML文件、提取配置设置、基于这些设置设置环境变量以及将配置设置转换为命令行字符串的方法。
    """
    def parse_and_set_env(self, config_path):
        """  
        解析YAML配置文件并设置环境变量。
        参数:
            config_path (str): YAML配置文件的路径。
        返回:
            dict: 包含解析后的配置设置的字典（排除了在'env'键下设置为环境变量的设置）。
        异常:
            ValueError: 如果YAML文件中的'env'字段不是字典。
            FileNotFoundError: 如果指定的配置文件不存在。
            yaml.YAMLError: 如果解析YAML文件时出现错误。
        """
        with open(config_path, 'r', encoding='utf-8') as yaml_file:
            config = yaml.safe_load(yaml_file)

        if "env" in config:
            env_vars = config.pop("env")
            if isinstance(env_vars, dict):
                for key, value in env_vars.items():
                    os.environ[key] = str(value)
            else:
                raise ValueError("`env` field should be a dict in the YAML file.")

        return config

    def to_string(self, config):
        """
        将配置设置转换为命令行字符串。
        该方法将给定的配置字典转换为命令行参数格式的字符串，格式为'--key value'。
        它跳过任何字典或列表值，因为这些值不适合直接用作命令行参数。  
        参数:
            config (dict): 包含配置设置的字典。        
        返回:
            str: 命令行参数的字符串。
        """
        final_string = ""
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                if len(value) != 0:
                    value = str(value)
                    value = value.replace("'", '"')
                    value = f"'{value}'"
                else:
                    continue

            final_string += f"--{key} {value} "
        return final_string


def init_zero_verbose():
    """
    Perform zero verbose init - use this method on top of the CLI modules to make
    """

    format_string = "%(message)s"
    logging.basicConfig(format=format_string, datefmt="[%X]", handlers=[RichHandler()], level=logging.ERROR)

    # Custom warning handler to redirect warnings to the logging system
    def warning_handler(message, category, filename, lineno):
        logging.warning("%s:%d: %s: %s", filename, lineno, category.__name__, message)

    # Add the custom warning handler - we need to do that before importing anything to make sure the loggers work well
    warnings.showwarning = warning_handler


@dataclass
class SFTScriptArguments(ScriptArguments):
    """  
    SFTScriptArguments 类（已弃用）  
  
    该类是 ScriptArguments 类的子类,但已被弃用,并将在未来的版本(v0.13)中移除。  
    建议使用 ScriptArguments 类代替。    
    """
    def __post_init__(self):
        logger.warning(
            "`SFTScriptArguments` is deprecated, and will be removed in v0.13. Please use "
            "`ScriptArguments` instead."
        )


@dataclass
class RewardScriptArguments(ScriptArguments):
    """  
    RewardScriptArguments 类（已弃用）
  
    该类是 ScriptArguments 类的子类，用于特定奖励脚本的参数配置。 
    然而，该类已被弃用，并将在未来的版本(v0.13)中移除.
    建议使用 ScriptArguments 类代替。
    """
    def __post_init__(self):
        logger.warning(
            "`RewardScriptArguments` is deprecated, and will be removed in v0.13. Please use "
            "`ScriptArguments` instead."
        )


@dataclass
class DPOScriptArguments(ScriptArguments):
    """  
    DPOScriptArguments 类（已弃用）  
  
    该类是 ScriptArguments 类的子类，原本用于DPO（委托权益证明）相关脚本的参数配置。  
    然而，该类已被弃用，并将在未来的版本（v0.13）中移除。  
    建议使用 ScriptArguments 类作为替代，因为它提供了更通用和灵活的参数配置。  
    """
    def __post_init__(self):
        logger.warning(
            "`DPOScriptArguments` is deprecated, and will be removed in v0.13. Please use "
            "`ScriptArguments` instead."
        )


@dataclass
class ChatArguments:
    """  
    聊天参数配置类  
  
    此类用于配置与聊天界面和模型生成相关的参数。  
  
    字段说明：  
  
    一般设置:  
    - model_name_or_path (str): 预训练模型的名称或路径。  
    - user (Optional[str]): 在聊天界面中显示的用户名。默认为 None。  
    - system_prompt (Optional[str]): 系统提示信息。默认为 None。  
    - save_folder (str): 保存聊天历史的文件夹路径。默认为 "./chat_history/"。   
    - config (str): 配置文件，用于设置配置。如果为 "default"，则使用 "examples/scripts/config/default_chat_config.yaml"。默认为 "default"。  
    - examples (Optional[str]): 空占位符，需要通过配置进行设置。默认为 None。  
  
    生成设置:  
    - max_new_tokens (int): 生成的最大令牌数。默认为 256。  
    - do_sample (bool): 在生成过程中是否采样输出。默认为 True。  
    - num_beams (int): 用于束搜索的束数。默认为 1。  
    - temperature (float): 生成的温度参数。默认为 1.0。  
    - top_k (int): 用于 top-k 采样的 k 值。默认为 50。  
    - top_p (float): 用于核采样（nucleus sampling）的 p 值。默认为 1.0。  
    - repetition_penalty (float): 重复惩罚。默认为 1.0。  
    - eos_tokens (Optional[str]): 用于停止生成的 EOS 令牌。如果有多个，应该用逗号分隔。默认为 None。  
    - eos_token_ids (Optional[str]): 用于停止生成的 EOS 令牌 ID。如果有多个，应该用逗号分隔。默认为 None。  
  
    模型加载:  
    - model_revision (str): 要使用的特定模型版本（可以是分支名称、标签名称或提交 ID）。默认为 "main"。    
    - trust_remote_code (bool): 在加载模型时是否信任远程代码。默认为 False。  
    - attn_implementation (Optional[str]): 要使用的注意力实现。例如，可以运行 --attn_implementation=flash_attention_2，但这需要手动安装，通过运行 `pip install flash-attn --no-build-isolation`。默认为 None。  
    - load_in_8bit (bool): 是否为基模型使用 8 位精度（仅与 LoRA 一起工作）。默认为 False。  
    - load_in_4bit (bool): 是否为基模型使用 4 位精度（仅与 LoRA 一起工作）。默认为 False。  
    - bnb_4bit_quant_type (str): 指定量化类型（fp4 或 nf4）。默认为 "nf4"。  
    - use_bnb_nested_quant (bool): 是否使用嵌套量化。默认为 False。  
    """
    # general settings
    model_name_or_path: str = field(metadata={"help": "Name of the pre-trained model"})
    user: str = field(default=None, metadata={"help": "Username to display in chat interface"})
    system_prompt: str = field(default=None, metadata={"help": "System prompt"})
    save_folder: str = field(default="./chat_history/", metadata={"help": "Folder to save chat history"})
    config: str = field(
        default="default",
        metadata={
            "help": "Config file used for setting the configs. If `default` uses examples/scripts/config/default_chat_config.yaml"
        },
    )
    examples: str = field(default=None, metadata={"help": "Empty placeholder needs to be set via config."})
    # generation settings
    max_new_tokens: int = field(default=256, metadata={"help": "Maximum number of tokens to generate"})
    do_sample: bool = field(default=True, metadata={"help": "Whether to sample outputs during generation"})
    num_beams: int = field(default=1, metadata={"help": "Number of beams for beam search"})
    temperature: float = field(default=1.0, metadata={"help": "Temperature parameter for generation"})
    top_k: int = field(default=50, metadata={"help": "Value of k for top-k sampling"})
    top_p: float = field(default=1.0, metadata={"help": "Value of p for nucleus sampling"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "Repetition penalty"})
    eos_tokens: str = field(
        default=None,
        metadata={"help": "EOS tokens to stop the generation. If multiple they should be comma separated"},
    )
    eos_token_ids: str = field(
        default=None,
        metadata={"help": "EOS token IDs to stop the generation. If multiple they should be comma separated"},
    )
    # model loading
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code when loading a model."})
    attn_implementation: str = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use; you can run --attn_implementation=flash_attention_2, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "use 8 bit precision for the base model - works only with LoRA"},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "use 4 bit precision for the base model - works only with LoRA"},
    )

    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"})
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})


class TrlParser(HfArgumentParser):
    """  
    TRL解析器解析一个解析器列表（例如TrainingArguments、trl.ModelConfig等），为传递了有效`config`字段的用户创建配置解析器，  
    并将配置中设置的值与处理过的解析器合并。  
    
    参数:  
        parsers (`List[argparse.ArgumentParser]`):  
            解析器列表。  
        ignore_extra_args (`bool`):  
            是否忽略配置中传递的额外参数并且不引发错误。  
    """
    def __init__(self, parsers, ignore_extra_args=False):
        super().__init__(parsers)
        self.yaml_parser = YamlConfigParser()
        self.ignore_extra_args = ignore_extra_args

    def post_process_dataclasses(self, dataclasses):
        '''Apply additional post-processing in case some arguments needs a special care'''
        training_args = trl_args = None
        training_args_index = None

        for i, dataclass_obj in enumerate(dataclasses):
            if dataclass_obj.__class__.__name__ == "TrainingArguments":
                training_args = dataclass_obj
                training_args_index = i
            elif dataclass_obj.__class__.__name__ in ("SFTScriptArguments", "DPOScriptArguments"):
                trl_args = dataclass_obj
            else:
                ...

        if trl_args is not None and training_args is not None:
            training_args.gradient_checkpointing_kwargs = {
                'use_reentrant': trl_args.gradient_checkpointing_use_reentrant
            }
            dataclasses[training_args_index] = training_args

        return dataclasses

    def parse_args_and_config(self, return_remaining_strings=False):
        """  
            解析命令行参数和配置文件。  
        
            此方法首先检查命令行参数中是否包含配置文件路径（通过 --config 标志指定）。  
            如果包含，它将解析该配置文件，并使用解析后的配置设置环境变量和默认值。  
            然后，它将解析剩余的命令行参数，并根据需要将它们转换为数据类实例。  
        
            参数:  
            return_remaining_strings (bool): 如果为 True，则返回剩余的字符串（包括命令行参数和配置文件中未使用的参数）。默认为 False。  
        
            返回:  
            tuple: 根据 `return_remaining_strings` 参数的值，返回不同的元组。  
                - 如果 `return_remaining_strings` 为 False，则返回一个包含解析后的数据类实例（或它们的列表）的元组。  
                - 如果 `return_remaining_strings` 为 True，则返回一个元组，其中包含除最后一个元素外的所有解析后的数据类实例（或它们的列表），  
                    以及一个字符串列表，该列表包含剩余的命令行字符串和配置文件中未使用的参数（以 "key: value" 的形式）。  
        
            抛出:  
            ValueError: 如果在 `return_remaining_strings` 为 True 时，解析后的命令行参数和配置文件中存在未使用的参数，并且 `ignore_extra_args` 为 False，  
                    则抛出此异常，并列出未使用的参数。  
        """
        yaml_config = None
        if "--config" in sys.argv:
            config_index = sys.argv.index("--config")

            _ = sys.argv.pop(config_index)  # --config
            config_path = sys.argv.pop(config_index)  # path to config
            yaml_config = self.yaml_parser.parse_and_set_env(config_path)

            self.set_defaults_with_config(**yaml_config)

        outputs = self.parse_args_into_dataclasses(return_remaining_strings=return_remaining_strings)

        if yaml_config is None:
            return outputs

        if return_remaining_strings:
            # if we have extra yaml config and command line strings
            # outputs[-1] is remaining command line strings
            # outputs[-2] is remaining yaml config as Namespace
            # combine them into remaining strings object
            remaining_strings = outputs[-1] + [f"{key}: {value}" for key, value in vars(outputs[-2]).items()]
            return outputs[:-2], remaining_strings

        # outputs[-1] is either remaining yaml config as Namespace or parsed config as Dataclass
        if isinstance(outputs[-1], Namespace) and not self.ignore_extra_args:
            remaining_args = vars(outputs[-1])
            raise ValueError(f"Some specified config arguments are not used by the TrlParser: {remaining_args}")

        return outputs

    def set_defaults_with_config(self, **kwargs):
        """Defaults we're setting with config allow us to change to required = False"""
        self._defaults.update(kwargs)

        # if these defaults match any existing arguments, replace
        # the previous default on the object with the new one
        for action in self._actions:
            if action.dest in kwargs:
                action.default = kwargs[action.dest]
                action.required = False


def get_git_commit_hash(package_name):
    """  
        Retrieve the current Git commit hash for the repository containing the specified Python package.  
    
        This function imports the specified Python package, determines its file path, and then  
        attempts to locate the root of the Git repository. Once the repository root is found, it  
        executes a `git rev-parse HEAD` command to retrieve the current commit hash.  
    
        Parameters:  
        package_name (str): The name of the Python package to inspect. This should be a string  
                            that can be imported using `importlib.import_module`.  
    
        Returns:  
        str: The current Git commit hash as a string, or `None` if the Git repository could not  
            be located or if the package does not appear to be part of a Git repository.  
    
        Raises:  
        This function does not raise exceptions directly. Instead, it returns a friendly error  
        message as a string if an error occurs during the process.  
    
        Examples:  
        >>> get_git_commit_hash('my_package')  
        '1a2b3c4d5e6f7g8h9i0jklmnopqrstuvwxyz'  
    
        Note:  
        - The function assumes that the Git repository is located directly above the package's  
        directory, or that the package's directory is within a subdirectory of the repository.  
        - If the `.git` directory cannot be found, the function will return `None`.  
        - If an error occurs during the import or file path determination, a friendly error  
        message will be returned instead of raising an exception.  
    """
    try:
        # Import the package to locate its path
        package = importlib.import_module(package_name)
        # Get the path to the package using inspect
        package_path = os.path.dirname(inspect.getfile(package))

        # Navigate up to the Git repository root if the package is inside a subdirectory
        git_repo_path = os.path.abspath(os.path.join(package_path, ".."))
        git_dir = os.path.join(git_repo_path, ".git")

        if os.path.isdir(git_dir):
            # Run the git command to get the current commit hash
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_repo_path).strip().decode("utf-8")
            )
            return commit_hash
        return None
    except Exception:
        return "Error: Unable to retrieve the Git commit hash. Please check your input and try again."
