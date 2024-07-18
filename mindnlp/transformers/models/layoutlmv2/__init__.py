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
Layoutlmv2 Model.
"""
from . import configuration_layoutlmv2, image_processing_layoutlmv2, modeling_layoutlmv2, \
    processing_layoutlmv2, tokenization_layoutlmv2, tokenization_layoutlmv2_fast

from .configuration_layoutlmv2 import *
from .image_processing_layoutlmv2 import *
from .modeling_layoutlmv2 import *
from .processing_layoutlmv2 import *
from .tokenization_layoutlmv2 import *
from .tokenization_layoutlmv2_fast import *

__all__ = []
__all__.extend(configuration_layoutlmv2.__all__)
__all__.extend(image_processing_layoutlmv2.__all__)
__all__.extend(modeling_layoutlmv2.__all__)
__all__.extend(processing_layoutlmv2.__all__)
__all__.extend(tokenization_layoutlmv2.__all__)
__all__.extend(tokenization_layoutlmv2_fast.__all__)
