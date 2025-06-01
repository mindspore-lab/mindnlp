# Copyright 2024 Your Company. All rights reserved.
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

from mindnlp.utils import LazyModule
from mindnlp.utils.import_utils import define_import_structure

if TYPE_CHECKING:
    from .configuration_mimi import *
    from .modeling_mimi import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)