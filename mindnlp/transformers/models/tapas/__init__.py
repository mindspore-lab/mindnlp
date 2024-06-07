# Copyright 2021 The HuggingFace Team. All rights reserved.
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

"""tapas model."""
from . import configuration_tapas, convert_tapas_original_tf_checkpoint_to_pytorch, modeling_tapas, modeling_tf_tapas, tokenization_tapas
from .configuration_tapas import *
from .convert_tapas_original_tf_checkpoint_to_pytorch import *
from .modeling_tapas import *
from .modeling_tf_tapas import *
from .tokenization_tapas import *


__all__ = []
__all__.extend(configuration_tapas.__all__)
__all__.extend(convert_tapas_original_tf_checkpoint_to_pytorch.__all__)
__all__.extend(modeling_tapas.__all__)
__all__.extend(modeling_tf_tapas.__all__)
__all__.extend(tokenization_tapas.__all__)