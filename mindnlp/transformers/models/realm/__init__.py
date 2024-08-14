# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
REALM Model.
"""
from . import configuration_realm, retrieval_realm, modeling_realm, tokenization_realm, tokenization_realm_fast

from .configuration_realm import *
from .retrieval_realm import *
from .modeling_realm import *
from .tokenization_realm_fast import *
from .tokenization_realm import *

__all__ = []
__all__.extend(configuration_realm.__all__)
__all__.extend(modeling_realm.__all__)
__all__.extend(retrieval_realm.__all__)
__all__.extend(tokenization_realm.__all__)
__all__.extend(tokenization_realm_fast.__all__)
