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
Encodec Model init
"""

<<<<<<< HEAD
from .import modeling_bark, configuration_bark, generation_bark
from .configuration_bark import *
from .generation_bark import *
from .modeling_bark import *

__all__ = []
__all__.extend(modeling_bark.__all__)
__all__.extend(configuration_bark.__all__)
# __all__.extend(generation_bark.__all__)
=======
from .import bark, bark_config, bark_generation
from .bark_config import *
from .bark_generation import *
from .bark import *

__all__ = []
__all__.extend(bark.__all__)
__all__.extend(bark_config.__all__)
# __all__.extend(bark_generation.__all__)
>>>>>>> 523e2f0cc3d30248fac725064d16125a9d23a063
