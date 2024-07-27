# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Customized mindspore.nn.Module Dict."""

from __future__ import absolute_import
from typing import Dict
from collections import OrderedDict, abc as container_abcs
from mindspore.nn.cell import Cell
from mindspore import Parameter
__all__ = ['CellDict','ParameterDict']

def _valid_index(cell_num, index, op_name=None):
    """Internal function, used to detect the value and type of index."""
    msg_prefix = f"For '{op_name}', the" if op_name else "The"
    if not isinstance(index, int):
        raise TypeError(f"{msg_prefix} type of 'index' must be int, but got {type(index).__name__}.")
    if not -cell_num <= index < cell_num:
        raise IndexError(f"{msg_prefix} value of 'index' must be a number in range [{-cell_num}, {cell_num}), "
                         f"but got {index}.")
    return index % cell_num


def _valid_cell(cell, op_name=None):
    """Internal function, used to check whether the input cell is a subclass of Cell."""
    if issubclass(cell.__class__, Cell):
        return True
    msg_prefix = f"For '{op_name}'," if op_name else ""
    raise TypeError(f'{msg_prefix} each cell must be subclass of Cell, but got {type(cell).__name__}.')


def _get_prefix_and_index(cells):
    """get prefix and index of parameter name in sequential cell or cell list."""
    prefix = ""
    index = 0
    if not cells:
        return prefix, index

    cell_list = list(cells.items())
    first_param, first_key = None, None
    second_param, second_key = None, None
    for key, cell in cell_list:
        try:
            _, param = next(cell.parameters_and_names())
        except StopIteration:
            continue
        if first_param is None:
            first_param = param
            first_key = key
            continue
        second_param = param
        second_key = key
        break

    if first_param is None:
        return prefix, index

    split_names = first_param.name.split(".")
    for idx, name in enumerate(split_names):
        if name == first_key:
            prefix = ".".join(split_names[:idx])
            prefix = prefix + "." if prefix else prefix
            index = idx
            if second_param is not None and second_param.name.split(".")[idx] == second_key:
                break
    return prefix, index


class CellDict(Cell):
    """nn.Module Dict"""
    _cells: Dict[str, Cell]

    def __init__(self, cells: Dict[str, Cell] = None):
        """Initialize CellDict."""
        super().__init__()
        if cells is not None:
            self.update(cells)

    def __getitem__(self, key: str) -> Cell:
        r"""
        Retrieve a cell from the CellDict by its key.
        
        Args:
            self (CellDict): The instance of the CellDict class.
            key (str): The key used to retrieve the cell. It should be a string representing the unique identifier of the cell.
        
        Returns:
            Cell: The cell associated with the provided key.
        
        Raises:
            KeyError: If the provided key does not exist in the CellDict.
        """
        return self._cells[key]

    def __setitem__(self, key: str, cell: Cell) -> None:
        r"""
        __setitem__
        
        Method for setting an item in the CellDict.
        
        Args:
            self (CellDict): The CellDict instance to operate on.
            key (str): The key to associate with the given cell.
            cell (Cell): The cell object to be inserted into the CellDict.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            - TypeError: If the key is not a string type.
            - ValueError: If the cell is not a valid Cell object.
        """
        self.insert_child_to_cell(key, cell)

    def __delitem__(self, key: str) -> None:
        r"""
        Deletes an item from the CellDict based on the provided key.
        
        Args:
            self (CellDict): The CellDict instance on which the item deletion operation is performed.
            key (str): The key of the item to be deleted from the CellDict.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            KeyError: If the specified key is not found in the CellDict.
        """
        del self._cells[key]

        # adjust OrderedDict
        prefix, key_index = _get_prefix_and_index(self._cells)
        temp_dict = OrderedDict()
        for idx, cell in enumerate(self._cells.values()):
            if self._auto_prefix:
                for _, param in cell.parameters_and_names():
                    param.name = prefix + str(idx) + "." + ".".join(param.name.split(".")[key_index+1:])
            temp_dict[str(idx)] = cell
        self._cells = temp_dict

    def __bool__(self):
        r"""
        Method '__bool__' in the class 'CellDict'.
        
        Args:
            self (object): The instance of the CellDict class.
                - This parameter represents the current instance of the CellDict class.
                - It is used to access the internal _cells attribute to determine if it is empty or not.
        
        Returns:
            NoneType: This method does not return a value directly. Instead, it returns a boolean value based on whether the _cells attribute is empty or not.
        
        Raises:
            None.
        """
        return len(self._cells) != 0

    def __len__(self):
        r"""
        This method '__len__' in the class 'CellDict' returns the length of the '_cells' attribute.
        
        Args:
            self (object): The instance of the 'CellDict' class.
                This parameter represents the instance of the 'CellDict' class for which the length is being calculated. 
        
        Returns:
            int: The method returns the length of the '_cells' attribute.
                The return value is an integer representing the number of elements in the '_cells' attribute.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        return len(self._cells)

    def __iter__(self):
        r"""
        Iterates over the CellDict object and returns an iterator.
        
        Args:
            self (CellDict): The instance of the CellDict class.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: No exceptions are raised by this method.
        """
        return iter(self._cells)

    def contains(self, key: str) -> bool:
        r"""
        Method Name: contains
        
        Description:
        This method checks if the specified key exists in the CellDict object.
        
        Args:
        - self: The instance of the CellDict class.
        - key (str): The key to be searched for in the CellDict object.
        
        Returns:
        bool: Returns True if the specified key exists in the CellDict object, otherwise returns False.
        
        Raises:
        - No specific exceptions are raised by this method.
        """
        return key in self._cells

    def keys(self):
        r"""
        Method 'keys' in the class 'CellDict' returns the keys in the CellDict object.
        
        Args:
            self: CellDict - The instance of the CellDict class.
                It represents the CellDict object for which keys are being retrieved.
        
        Returns:
            None - This method returns the keys of the CellDict object.
                The keys are returned as a list of keys present in the CellDict object.
        
        Raises:
            No exceptions are raised by this method.
        """
        return self._cells.keys()

    def values(self):
        r"""
        This method returns a view object that displays a list of all the values in the CellDict.
        
        Args:
            self (CellDict): An instance of the CellDict class.
            
        Returns:
            None: This method does not return any specific value but provides access to the values in the CellDict through the view object.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        return self._cells.values()

    def items(self):
        r"""
        Method 'items' in the class 'CellDict'.
        
        Args:
            self: CellDict object.
                Represents the instance of CellDict class.
        
        Returns:
            None.
            This method returns a dictionary view object that displays a list of key-value pairs in the CellDict.
        
        Raises:
            No exceptions are raised by this method.
        """
        return self._cells.items()

    def update(self, cells) -> None:
        """Update the `nn.ModuleDict`"""
        if not isinstance(cells, container_abcs.Iterable):
            raise TypeError("CellDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(cells).__name__)

        if isinstance(cells, (OrderedDict, CellDict, container_abcs.Mapping)):
            for key, cell in cells.items():
                self[key] = cell
        else:
            raise NotImplementedError
            # # modules here can be a list with two items
            # for j, m in enumerate(modules):
            #     if not isinstance(m, container_abcs.Iterable):
            #         raise TypeError("ModuleDict update sequence element "
            #                         "#" + str(j) + " should be Iterable; is" +
            #                         type(m).__name__)
            #     if not len(m) == 2:
            #         raise ValueError("ModuleDict update sequence element "
            #                          "#" + str(j) + " has length " + str(len(m)) +
            #                          "; 2 is required")
            #     # modules can be Mapping (what it's typed at), or a list: [(name1, module1), (name2, module2)]
            #     # that's too cumbersome to type correctly with overloads, so we add an ignore here
            #     self[m[0]] = m[1]  # type: ignore[assignment]

    def set_grad(self, flag=True):
        r"""
        Sets the gradient flag for the 'CellDict' object and all its cells.
        
        Args:
            self (CellDict): The 'CellDict' object itself.
            flag (bool): A flag indicating whether to enable or disable gradient calculation. 
                         If set to True, gradient calculation will be enabled for the 'CellDict' object and its cells.
                         If set to False, gradient calculation will be disabled for the 'CellDict' object and its cells.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        """
        self.requires_grad = flag
        for cell in self._cells.values():
            cell.set_grad(flag)

    def forward(self, *inputs):
        r"""
        Method to forward a CellDict object.
        
        Args:
            self (CellDict): The instance of the CellDict class.
            
        Returns:
            None. This method does not return any value.
        
        Raises:
            NotImplementedError: This exception is raised as the method is not implemented and should be overridden in subclasses.
        """
        raise NotImplementedError

class ParameterDict(Cell):

    r"""
    Represents a dictionary-like object that stores parameters and their corresponding values. Inherits from the Cell class.
    
    Provides methods for adding, accessing, updating, and deleting parameters within the dictionary. Supports key-based access and iteration over the parameters.
    
    Methods:
    - __init__: Initializes the ParameterDict with optional initial parameters.
    - __getitem__: Retrieves the parameter value associated with a given key.
    - __setitem__: Sets a parameter with the specified key and value.
    - __delitem__: Deletes the parameter with the specified key.
    - __len__: Returns the number of parameters in the ParameterDict.
    - __iter__: Returns an iterator over the keys of the parameters.
    - __contains__: Checks if a key is present in the ParameterDict.
    - clear: Removes all parameters from the ParameterDict.
    - pop: Removes and returns the parameter value associated with a given key.
    - keys: Returns a view of the keys in the ParameterDict.
    - items: Returns a view of key-value pairs in the ParameterDict.
    - values: Returns an iterable of the parameter values.
    - update: Updates the ParameterDict with new parameters from an iterable of key/value pairs.
    
    Note: The update method expects the input to be an iterable of key/value pairs. If the input is a mapping type, the keys and values are inserted into the ParameterDict. If the input is a sequence, each
element should be an iterable of length 2 representing a key/value pair.
    """
    _params: Dict[str, Parameter]

    def __init__(self, parameters: Dict[str, Parameter] = None):
        r"""
        Initializes a new instance of the ParameterDict class.
        
        Args:
            self: The current instance of the ParameterDict class.
            parameters (Dict[str, Parameter], optional): A dictionary of parameters. Defaults to None.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        
        This method is responsible for initializing a new instance of the ParameterDict class. It takes two parameters: self, which represents the current instance of the class, and parameters, which is an
optional dictionary of parameters.
        
        The 'self' parameter is required and is automatically passed when calling the method on an instance of the class.
        
        The 'parameters' parameter is an optional dictionary that contains string keys and Parameter values. If provided, the method will update the ParameterDict instance with the values from the 'parameters'
dictionary. If not provided, the instance will remain empty.
        
        The 'parameters' dictionary allows for easy initialization of the ParameterDict instance with pre-defined values. It should be a dictionary where the keys are strings representing parameter names, and
the values are instances of the Parameter class.
        
        Example usage:
            # Create a new instance of ParameterDict
            pd = ParameterDict()
        
            # Initialize with pre-defined parameters
            params = {
                'param1': Parameter('value1'),
                'param2': Parameter('value2')
            }
            pd.__init__(params)
            
            # Access the parameters
            pd['param1'].value  # Output: 'value1'
            pd['param2'].value  # Output: 'value2'
        """
        super(ParameterDict, self).__init__()
        if parameters is not None:
            self.update(parameters)

    def __getitem__(self, key):
        r"""
        __getitem__
        
        This method allows accessing the value associated with a given key in the ParameterDict.
        
        Args:
            self (ParameterDict): The instance of the ParameterDict class.
            key (hashable): The key whose associated value is to be retrieved. It should be a hashable type, such as a string or a number.
        
        Returns:
            The value associated with the provided key in the ParameterDict. If the key is not found, a KeyError is raised.
        
        Raises:
            KeyError: If the provided key is not found in the ParameterDict.
        """
        return self._params[key]

    def __setitem__(self, key, parameter):
        r"""
        Sets the value of a key in the ParameterDict object.
        
        Args:
            self (ParameterDict): The instance of the ParameterDict class.
            key (Any): The key to set the value for.
            parameter (Any): The value to be associated with the key.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        self.insert_param_to_cell(key, parameter)

    def __delitem__(self, key):
        r"""
        Deletes an item with the specified key from the ParameterDict instance.
        
        Args:
            self (ParameterDict): The ParameterDict instance.
            key: The key of the item to be deleted. It should be a valid key for the ParameterDict instance.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            KeyError: If the specified key does not exist in the ParameterDict instance.
        
        Note:
            This method modifies the ParameterDict instance by removing the item with the specified key. If the key is not found, a KeyError is raised.
        """
        del self._params[key]

    def __len__(self):
        r"""
        This method '__len__' in the class 'ParameterDict' returns the length of the '_params' attribute.
        
        Args:
            self (ParameterDict): The instance of the ParameterDict class for which the length is to be calculated.
            
        Returns:
            None. The method returns the length of the '_params' attribute within the ParameterDict instance.
        
        Raises:
            No exceptions are raised by this method.
        """
        return len(self._params)

    def __iter__(self):
        r"""
        Docstring for the '__iter__' method in the 'ParameterDict' class.
        
        Args:
            self (ParameterDict): An instance of the ParameterDict class.
                Represents the current instance of the ParameterDict class.
        
        Returns:
            None. The method returns an iterator object.
        
        Raises:
            No specific exceptions are raised by this method under normal circumstances.
        """
        return iter(self._params.keys())

    def __contains__(self, key):
        r"""
        Method '__contains__' in the class 'ParameterDict' checks if a given key is present in the ParameterDict instance.
        
        Args:
            self (ParameterDict): The ParameterDict instance in which to search for the key.
            key (any): The key to be checked for existence in the ParameterDict instance.
        
        Returns:
            None: This method returns None.
        
        Raises:
            - None
        """
        return key in self._params

    def clear(self):
        r"""
        This method clears all parameters in the ParameterDict.
        
        Args:
            self (ParameterDict): The instance of the ParameterDict class.
            
        Returns:
            None: This method does not return any value.
        
        Raises:
            No specific exceptions are documented for this method.
        """
        self._params.clear()

    def pop(self, key):
        r"""
        Method to remove and return the value associated with the specified key from the ParameterDict.
        
        Args:
            self (ParameterDict): The instance of the ParameterDict class.
            key (any): The key whose associated value is to be removed and returned.
        
        Returns:
            None: This method returns None.
        
        Raises:
            KeyError: If the specified key is not found in the ParameterDict.
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        r"""
        Returns a list of all the keys in the ParameterDict object.
        
        Args:
            self (ParameterDict): The ParameterDict object itself.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        return self._params.keys()

    def items(self):
        r"""
        Returns a list of key-value pairs in the ParameterDict object.
        
        Args:
            self (ParameterDict): The ParameterDict object on which the method is called.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        return self._params.items()

    def values(self):
        r"""Return an iterable of the ParameterDict values.
        """
        return self._params.values()

    def update(self, parameters):
        r"""Updates the ParameterDict object with the specified key-value pairs.
        
        Args:
            self (ParameterDict): The ParameterDict object to be updated.
            parameters (iterable): An iterable containing key-value pairs to update the ParameterDict object.
                It can be a mapping object or a sequence of tuples/lists representing key-value pairs.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            TypeError: If the parameters argument is not an iterable of key-value pairs.
            TypeError: If the elements of the parameters sequence are not iterable.
            ValueError: If the elements of the parameters sequence do not have a length of 2.
        """
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError("ParametersDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(parameters).__name__)

        if isinstance(parameters, container_abcs.Mapping):
            if isinstance(parameters, (OrderedDict, ParameterDict)):
                for key, parameter in parameters.items():
                    self[key] = parameter
            else:
                for key, parameter in sorted(parameters.items()):
                    self[key] = parameter
        else:
            for j, p in enumerate(parameters):
                if not isinstance(p, container_abcs.Iterable):
                    raise TypeError("ParameterDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(p).__name__)
                if not len(p) == 2:
                    raise ValueError("ParameterDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(p)) +
                                     "; 2 is required")
                self[p[0]] = p[1]
