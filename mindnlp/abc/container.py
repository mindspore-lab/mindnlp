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
"""Customized mindspore.nn.Cell Dict."""
# pylint: disable=C0116
# pylint: disable=W0237

from __future__ import absolute_import
from typing import Dict
from collections import OrderedDict, abc as container_abcs
from mindspore.nn.cell import Cell

__all__ = ['CellDict']

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
    """nn.Cell Dict"""
    _cells: Dict[str, Cell]

    def __init__(self, cells: Dict[str, Cell] = None):
        """Initialize CellDict."""
        super().__init__()
        if cells is not None:
            self.update(cells)

    def __getitem__(self, key: str) -> Cell:
        return self._cells[key]

    def __setitem__(self, key: str, cell: Cell) -> None:
        self.insert_child_to_cell(key, cell)

    def __delitem__(self, key: str) -> None:
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
        return len(self._cells) != 0

    def __len__(self):
        return len(self._cells)

    def __iter__(self):
        return iter(self._cells)

    def contains(self, key: str) -> bool:
        return key in self._cells

    def keys(self):
        return self._cells.keys()

    def values(self):
        return self._cells.values()

    def items(self):
        return self._cells.items()

    def update(self, cells) -> None:
        """Update the `nn.CellDict`"""
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
        self.requires_grad = flag
        for cell in self._cells.values():
            cell.set_grad(flag)

    def construct(self, *inputs):
        raise NotImplementedError
