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
homework two
This is the python file for homework two
step 1: pip install pytest
step 2: modify the function test2
step 3: Use pytest to check this function
"""

import math


def test2():
    """
    num1 is the first parameter
    num2 is the second parameter
    result is the result of this function
    use math.isclose to check the result of this function

    Formula for cyclic sequence:
        2 * 3 = 6
        6 - 3 = 3
        3 * 2 = 6
        6 / 3 = 2
    """
    num1 = 2
    num2 = 3
    result = num1 * num2
    result = result - num2
    result = result * num1
    result = result / num2
    result = result * num2
    result = result - num2
    result = result * num1
    result = result / num2
    result = result * num2
    result = result - num2
    result = result * num1
    result = result - num2
    result = result * num1
    result = result / num2
    result = result * num2
    result = result - num2

    assert math.isclose(result, 3, abs_tol=0.001)
