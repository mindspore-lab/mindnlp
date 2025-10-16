# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from transformers.utils.import_utils import _is_package_available


# Use same as transformers.utils.import_utils
_e2b_available = _is_package_available("e2b")


def is_e2b_available() -> bool:
    return _e2b_available


_morph_available = _is_package_available("morphcloud")


def is_morph_available() -> bool:
    return _morph_available
