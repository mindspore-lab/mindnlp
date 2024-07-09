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
"""
TextLoader
"""

class TextLoader:
    """Load text file.

    Args:
        file_path: Path to the file to load.
    """

    def __init__(
        self,
        file_path: str,
    ):
        """Initialize with file path."""
        self.file_path = file_path

    def load(self) -> str:
        """Load from file path."""
        text = ""
        try:
            with open(self.file_path) as f:
                text = f.read()
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

        return text
