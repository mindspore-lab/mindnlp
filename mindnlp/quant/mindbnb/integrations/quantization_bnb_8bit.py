# Copyright 2024 Huawei Technologies Co., Ltd
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
'''
    quantization bnb 8bit
'''
from .replace_modules import replace_with_bnb_linear


def quant_8bit(model, modules_to_not_convert=None, quantization_config=None):

    model = replace_with_bnb_linear(
        model,
        modules_to_not_convert=modules_to_not_convert,
        quantization_config=quantization_config,
    )

    return model
