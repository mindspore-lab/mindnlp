# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Workflow Class
"""

import sys
import mindspore

from .works.sentiment_analysis import SentimentAnalysisWork
from .works.information_extraction import UIEWork


WORKS = {
    "sentiment_analysis": {
        "models": {
            "bert": {
                "work_class": SentimentAnalysisWork,
                "work_flag": "sentiment_analysis-bert",
            }
        },
        "default": {"model": "bert"},
    },
    "information_extraction": {
        "models": {
            "uie-base": {
                "work_class": UIEWork,
                "hidden_size": 768,
                "work_flag": "information_extraction-uie-base",
            },
            "uie-medium": {
                "work_class": UIEWork,
                "hidden_size": 768,
                "work_flag": "information_extraction-uie-medium",
            },
            "uie-mini": {
                "work_class": UIEWork,
                "hidden_size": 384,
                "work_flag": "information_extraction-uie-mini",
            },
            "uie-micro": {
                "work_class": UIEWork,
                "hidden_size": 384,
                "work_flag": "information_extraction-uie-micro",
            },
            "uie-nano": {
                "work_class": UIEWork,
                "hidden_size": 312,
                "work_flag": "information_extraction-uie-nano",
            },
            "uie-tiny": {
                "work_class": UIEWork,
                "hidden_size": 768,
                "work_flag": "information_extraction-uie-tiny",
            },
            "uie-medical-base": {
                "work_class": UIEWork,
                "hidden_size": 768,
                "work_flag": "information_extraction-uie-medical-base",
            },
            "uie-base-en": {
                "work_class": UIEWork,
                "hidden_size": 768,
                "work_flag": "information_extraction-uie-base-en",
            },
        },
        "default": {"model": "uie-base"},
    },
}

support_schema_list = [
    "uie-base",
    "uie-medium",
    "uie-mini",
    "uie-micro",
    "uie-nano",
    "uie-tiny",
    "uie-base-en",
    "uie-senta-base",
    "uie-senta-medium",
    "uie-senta-mini",
    "uie-senta-micro",
    "uie-senta-nano",
]

support_argument_list = [
    "uie-base",
    "uie-medium",
    "uie-mini",
    "uie-micro",
    "uie-nano",
    "uie-tiny",
    "uie-medical-base",
    "uie-base-en",
]


class Workflow:
    """
    The Workflow is the end2end interface that could convert the raw text to model result,
    and decode the model result to work result. The main functions as follows:

    Args:
        work (str): The work name for the Workflow, and get the work class from the name.
        model (str, optional): The model name in the work, if set None, will use the default model.
        mode (str, optional): Select the mode of the work, only used in the works of
        word_segmentation and ner.
            If set None, will use the default mode.
        device_id (int, optional): The device id for the gpu, xpu and other devices,
        the defalut value is 0.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific work.
    """
    def __init__(
        self, work, model=None, mode=None, device_id=0, from_hf_hub=False, **kwargs
    ):
        """
        Initialize a new instance of Workflow.
        
        Args:
            self: The instance of the class.
            work (str): The name of the work to be performed. It should be a valid work name from the WORKS list.
            model (str): The specific model to be used for the work, default is None. It should be a valid model name from the corresponding WORKS list.
            mode (str): The mode to be used, default is None.
            device_id (int): The ID of the device to be used, default is 0. If set to -1 or device_target is 'CPU', PYNATIVE_MODE is used, else GRAPH_MODE is used.
            from_hf_hub (bool): Indicates if the model is loaded from the Hugging Face Hub, default is False.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None. This method initializes the Workflow instance.
        
        Raises:
            AssertionError: Raised if the work name provided is not in the WORKS list or if the model name is not in the corresponding WORKS list.
            KeyError: Raised if the specified 'tag' or 'ind_tag' is not found in the WORKS dictionary.
            KeyError: Raised if the 'work_class' key is not found in the configuration dictionary.
        """
        assert (
            work in WORKS
        ), f"The work name:{work} is not in Workflow list, \
            please check your work name."
        self.work = work

        device = mindspore.context.get_context("device_target")
        if device == "CPU" or device_id == -1:
            mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE)
        else:
            mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE)

        if self.work in ["word_segmentation", "ner"]:
            tag = "modes"
            ind_tag = "mode"
            self.model = mode
        else:
            tag = "models"
            ind_tag = "model"
            self.model = model

        if self.model is not None:
            assert self.model in set(
                WORKS[work][tag].keys()
            ), f"The {tag} name: {model} is not in work:[{work}]"
        else:
            self.model = WORKS[work]["default"][ind_tag]

        config_kwargs = WORKS[self.work][tag][self.model]
        kwargs["device_id"] = device_id
        kwargs.update(config_kwargs)
        self.kwargs = kwargs
        work_class = WORKS[self.work][tag][self.model]["work_class"]
        self.work_instance = work_class(
            model=self.model, work=self.work, from_hf_hub=from_hf_hub, **self.kwargs
        )
        work_list = WORKS.keys()
        Workflow.work_list = work_list

    def __call__(self, *inputs):
        """
        The main work function in the workflow.
        """
        results = self.work_instance(inputs)
        return results

    def help(self):
        """
        Return the work usage message.
        """
        return self.work_instance.help()

    def from_segments(self, *inputs):
        """
        dependency_parsing work special function.
        """
        results = self.work_instance.from_segments(inputs)
        return results

    def interactive_mode(self, max_turn):
        """
        dialogue work special function.
        """
        with self.work_instance.interactive_mode(max_turn):
            while True:
                human = input("[Human]:").strip()
                if human.lower() == "exit":
                    sys.exit()
                robot = self.work_instance(human)[0]
                print(f"[Bot]:{robot}")

    def set_schema(self, schema):
        """
        Set the schema for uie-based or wordtag work.
        """
        assert (
            self.work_instance.model in support_schema_list
        ), "This method can only be used by the work based on the model of uie or wordtag."
        self.work_instance.set_schema(schema)

    def set_argument(self, argument):
        """
        Set the argument for text-to-image generation,
        information extraction or zero-text-classification work.
        """
        assert self.work_instance.model in support_argument_list, (
            "This method can only be used by the work of \
                text-to-image generation, information extraction "
            "or zero-text-classification."
        )
        self.work_instance.set_argument(argument)
