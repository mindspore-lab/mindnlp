"""
计算图跟踪模块 - 用于路线2（GradOperation）的实现
记录操作历史，以便在 backward 时重建 forward 函数
"""
from typing import Optional, List, Tuple, Any, Callable
from collections import defaultdict
import weakref
from torch4ms.tensor import Tensor
from mindspore import Tensor as ms_Tensor
from mindspore import Parameter
from mindspore import ops


class OperationRecord:
    """记录单个操作的信息"""
    def __init__(self, op_name: str, op_func: Callable, inputs: Tuple, output: Tensor):
        self.op_name = op_name
        self.op_func = op_func
        self.inputs = inputs  # 输入 Tensor 的引用
        self.output = output  # 输出 Tensor
        
    def get_input_tensors(self) -> List[Tensor]:
        """获取输入 Tensor"""
        result = []
        for inp in self.inputs:
            if isinstance(inp, Tensor):
                result.append(inp)
            elif hasattr(inp, '__call__'):  # 弱引用
                inp_obj = inp()
                if inp_obj is not None:
                    result.append(inp_obj)
        return result


class ComputationGraph:
    """计算图跟踪器"""
    def __init__(self):
        # 从输出 Tensor 到操作记录的映射
        self.output_to_op: dict = {}
        # 从输入 Tensor 到使用它的操作的映射
        self.input_to_ops: dict = defaultdict(list)
    
    def record_operation(self, op_name: str, op_func: Callable, inputs: Tuple[Tensor, ...], output: Tensor):
        """记录一个操作"""
        # 直接保存输入 Tensor 的引用（不使用弱引用，因为我们需要在 backward 时访问）
        # 注意：这可能会导致循环引用，但在 backward 后会被清理
        input_refs = []
        for inp in inputs:
            if isinstance(inp, Tensor):
                input_refs.append(inp)
                # 记录这个输入被使用
                self.input_to_ops[id(inp)].append(output)
            else:
                input_refs.append(inp)
        
        record = OperationRecord(op_name, op_func, tuple(input_refs), output)
        self.output_to_op[id(output)] = record
    
    def get_forward_fn(self, output: Tensor, inputs: List[Tensor]) -> Optional[Callable]:
        """
        根据记录的操作历史，重建 forward 函数
        
        Args:
            output: 输出 Tensor
            inputs: 输入 Tensor 列表
            
        Returns:
            forward 函数，如果无法重建则返回 None
        """
        # 尝试从记录中重建
        output_id = id(output)
        if output_id not in self.output_to_op:
            return None
        
        record = self.output_to_op[output_id]
        input_tensors = record.get_input_tensors()
        
        # 检查输入数量是否匹配
        if len(input_tensors) != len(inputs):
            return None
        
        # 对于简单的二元运算，重建 forward 函数
        if record.op_name in ['add', 'sub', 'mul', 'div']:
            if len(inputs) == 2:
                def forward_fn(x, y):
                    return record.op_func(x, y)
                return forward_fn
            elif len(inputs) == 1:
                # 一元运算（虽然二元运算通常需要两个输入）
                def forward_fn(x):
                    return record.op_func(x, record.inputs[1] if len(record.inputs) > 1 else 0)
                return forward_fn
        
        # 对于 sum 操作
        if record.op_name == 'sum':
            if len(inputs) == 1:
                def forward_fn(x):
                    return ops.reduce_sum(x)
                return forward_fn
        
        return None


# 全局计算图跟踪器
_global_graph = ComputationGraph()


def get_computation_graph() -> ComputationGraph:
    """获取全局计算图跟踪器"""
    return _global_graph


def reset_computation_graph():
    """重置计算图（用于测试）"""
    global _global_graph
    _global_graph = ComputationGraph()

