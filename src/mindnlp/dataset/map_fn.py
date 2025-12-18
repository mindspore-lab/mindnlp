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
Map fuction init
"""
class BaseMapFunction:

    r"""
    The BaseMapFunction class represents a base mapping function for processing input columns into output columns.
    
    Attributes:
        input_columns (list): A list of input columns to be processed.
        output_columns (list): A list of output columns to store the processed data.
    
    Methods:
        __call__(*args): Processes the input arguments and returns the result.
    
    """
    def __init__(self, input_colums, output_columns):
        r"""
        Initializes an instance of the BaseMapFunction class.
        
        Args:
            self: Reference to the current instance of the class.
            input_columns (list): List of input columns for the mapping function.
            output_columns (list): List of output columns for the mapping function.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        self.input_columns = input_colums
        self.output_columns = output_columns

    def __call__(self, *args):
        r"""
        The '__call__' method in the 'BaseMapFunction' class takes '1' parameter, which is 'self'.
        
        Args:
            - self: (object) The instance of the 'BaseMapFunction' class.
        
        Returns:
            - None: This method does not return any value.
        
        Raises:
            - None: This method does not raise any exceptions.
        """
        return args
