# Copyright 2021 Huawei Technologies Co., Ltd
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
MindNLP compatibility layer - imports from mindhf for backward compatibility.
"""
import sys
import warnings

# Import everything from mindhf
import mindhf

# Re-export all public attributes from mindhf
__all__ = mindhf.__all__ if hasattr(mindhf, '__all__') else []

# Copy all attributes from mindhf to this module
for attr_name in dir(mindhf):
    if not attr_name.startswith('_'):
        setattr(sys.modules[__name__], attr_name, getattr(mindhf, attr_name))

# Copy version
__version__ = mindhf.__version__

# Issue deprecation warning
warnings.warn(
    "The 'mindnlp' package name is deprecated. Please use 'mindhf' instead. "
    "This compatibility layer will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)
