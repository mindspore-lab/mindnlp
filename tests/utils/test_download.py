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
Test Download
"""

import unittest
import os
from mindnlp.utils.download import get_cache_path, check_md5, get_filepath, match_file

class TestGetCachePath(unittest.TestCase):
    r"""
    Test get_cache_path
    """

    def setUp(self):
        self.input = None

    def test_get_cache_path(self):
        default_cache_path = get_cache_path()
        assert default_cache_path == (os.path.join(os.getcwd(), ".mindnlp"))

class TestCheckMd5(unittest.TestCase):
    r"""
    Test check_md5
    """

    def setUp(self):
        self.input = None

    def test_check_md5(self):
        filename = "test"
        check_md5_result = check_md5(filename)
        assert check_md5_result


class TestGetFilePath(unittest.TestCase):
    r"""
    Test get_file_path
    """

    def setUp(self):
        self.input = None

    def test_get_file_path(self):
        path = os.path.expanduser('~')
        get_filepath_result = get_filepath(path)
        assert get_filepath_result == (os.path.expanduser('~'))

class TestMatchFile(unittest.TestCase):
    r"""
    Test match_file
    """

    def setUp(self):
        self.input = None

    def test_match_file(self):
        name = 'aclImdb_v1.tar.gz'
        path = os.path.expanduser('~')
        match_file_result = match_file(name, path)
        assert match_file_result == ''
